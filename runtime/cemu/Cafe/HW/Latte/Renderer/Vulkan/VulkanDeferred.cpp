// BotW-Native: Vulkan command recording on its own thread. The GPU thread spent about a quarter of its time inside the
// driver recording commands. The vkCmd* function pointers (VulkanAPI.h) are swapped for thunks that copy their
// arguments into a ring; the "VkRecord" thread replays them in order. Command buffer begin/end/reset and submits go
// through the ring too. Present, acquire, wait-idle and command pool calls wait until the ring is drained, then run
// directly (present is queued like a submit); fence checks see a submit still in the ring as not yet signaled. The replay thread sleeps when caught
// up and is woken in batches (WakeReplay), so it does not spin through the gaps between draws.
// Only the thread that called VkDefer_SetProducerThread queues; any other thread calls the driver directly.
// BOTW_VKTHREAD=0 turns it off.
#include "Cafe/HW/Latte/Renderer/Vulkan/VulkanAPI.h"
#include "util/helpers/helpers.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <thread>
#include <immintrin.h>
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>

extern "C" void botw_place_thread(int kind);  // runtime/src/thread.cpp core plan
void LatteCP_idleUntilWrite(const void* addr, uint32 seen);  // MWAITX until addr is written (LatteCommandProcessor.cpp)

namespace
{
	// 1 MB, not 32: with 32 MB every record went to a cold cache line, so the store buffer filled with misses and the
	// fence in WakeNow waited them out (5% of LatteThread on the Deck). LatteThread and VkRecord share one physical
	// core and its L2 (core plan), so a small ring stays cached for both. ~2/3 of a frame of records; the largest
	// single record (a batched texture upload) is ~60 KB.
	constexpr uint64 kRingSize = 1ull << 20;
	uint8* s_ring;
	alignas(64) std::atomic<uint64> s_published{0};
	alignas(64) std::atomic<uint64> s_consumed{0};
	alignas(64) uint64 s_reserved = 0;  // producer only
	uint32 s_unnotified = 0;  // producer only: records published since the replay thread was last woken
	// The replay thread sleeps once it has caught up; the producer wakes it every kWakeBatch records, and at once
	// for submits, drains and a full ring. Waking it for every record kept it (and futex calls) busy all frame.
	// Short waits are MONITORX/MWAITX on the other side's counter (LatteCP_idleUntilWrite): the core idles at low power
	// and wakes on the store, no syscall. Only after ~1 ms idle (replay thread; ~5 ms producer) does a side sleep in a raw futex. History on the
	// Deck: a pause spin polled through the ~1 us gaps between records (a quarter of the thread); sleeping at once
	// (batch 28 via libstdc++ wait, batch 29 via futex) turned every gap into a sleep/wake syscall pair and took
	// LatteThread from ~10 to 16-20 ms a frame (30% of it in libc).
	alignas(64) std::atomic<bool> s_sleeping{false};
	alignas(64) std::atomic<uint32> s_wakeSeq{0};  // bumped to wake the replay thread
	alignas(64) std::atomic<bool> s_producerWaiting{false};
	alignas(64) std::atomic<uint32> s_consumedSeq{0};  // bumped to wake a producer waiting in WaitConsumed
	constexpr uint32 kWakeBatch = 128;

	void FutexWait(std::atomic<uint32>* w, uint32 seen)
	{
		syscall(SYS_futex, (uint32*)w, FUTEX_WAIT_PRIVATE, seen, nullptr, nullptr, 0);
	}

	void FutexWake(std::atomic<uint32>* w)
	{
		syscall(SYS_futex, (uint32*)w, FUTEX_WAKE_PRIVATE, 1, nullptr, nullptr, 0);
	}

	// pairs with the replay thread's seq_cst store of s_sleeping and reload of s_published
	void WakeNow()
	{
		std::atomic_thread_fence(std::memory_order_seq_cst);  // orders the release-stored s_published before the load
		if (s_sleeping.load(std::memory_order_seq_cst))
		{
			s_wakeSeq.fetch_add(1, std::memory_order_release);
			FutexWake(&s_wakeSeq);
		}
	}

	void WakeReplay()
	{
		s_unnotified = 0;
		WakeNow();
	}
	std::atomic<VkResult> s_submitError{VK_SUCCESS};
	bool s_installed = false;
	thread_local bool t_producer = false;

	struct Rec
	{
		void (*run)(Rec*);
		uint64 size;
	};
	static_assert(sizeof(Rec) == 16);

	constexpr uint64 Align16(uint64 v) { return (v + 15) & ~15ull; }

	void WaitConsumed(uint64 target)
	{
		for (int i = 0; i < 256; i++)  // each wait ends on the replay thread's next record or after ~20 us
		{
			const uint64 c = s_consumed.load(std::memory_order_acquire);
			if (c >= target)
				return;
			LatteCP_idleUntilWrite(&s_consumed, (uint32)c);
		}
		// announce the wait, then look again: the replay thread checks the flag after each caught-up batch
		while (true)
		{
			const uint32 seq = s_consumedSeq.load(std::memory_order_acquire);
			s_producerWaiting.store(true, std::memory_order_seq_cst);
			if (s_consumed.load(std::memory_order_seq_cst) >= target)
				break;
			FutexWait(&s_consumedSeq, seq);
		}
		s_producerWaiting.store(false, std::memory_order_relaxed);
	}

	void Drain()
	{
		if (!s_installed)
			return;
		WakeNow();
		WaitConsumed(s_published.load(std::memory_order_acquire));
	}

	Rec* Reserve(uint64 size)
	{
		if (size > kRingSize / 2)  // would wait for the ring forever
		{
			fputs("VkRecord: command record larger than half the ring\n", stderr);
			abort();
		}
		const uint64 off = s_reserved % kRingSize;
		const uint64 pad = off + size > kRingSize ? kRingSize - off : 0;
		// Full only when the replay thread is a whole ring behind. s_reserved never wraps, so the old test
		// (s_reserved + size > kRingSize) was true for every record after the first MB: an mfence + wake check per
		// record, ~7% of LatteThread on the Deck (b86 profile). s_consumedSeen is a lower bound of s_consumed (it only
		// grows), so it is safe to trust and spares a load of the replay thread's line per record.
		static uint64 s_consumedSeen = 0;  // producer only
		const uint64 need = s_reserved + pad + size;
		if (need > s_consumedSeen + kRingSize)
		{
			s_consumedSeen = s_consumed.load(std::memory_order_acquire);
			if (need > s_consumedSeen + kRingSize)
			{
				WakeReplay();
				WaitConsumed(need - kRingSize);
				s_consumedSeen = s_consumed.load(std::memory_order_acquire);
			}
		}
		if (pad)
		{
			Rec* p = (Rec*)(s_ring + off);
			p->run = nullptr;
			p->size = pad;
			s_reserved += pad;
		}
		Rec* r = (Rec*)(s_ring + s_reserved % kRingSize);
		r->size = size;
		s_reserved += size;
		return r;
	}

	// Submits still in the ring, per fence. vkQueueSubmit owns its fence until it returns, so a status check on such a
	// fence answers VK_NOT_READY without touching the driver, and waits and resets first let the ring pass the submit.
	// Entries are only added (by the producer); Cemu uses a fixed set of fences.
	struct FenceSlot
	{
		std::atomic<VkFence> fence{VK_NULL_HANDLE};
		std::atomic<uint32> pending{0};
	};
	FenceSlot s_fenceSlots[256];

	FenceSlot* FindFence(VkFence f, bool insert)
	{
		const uint32 start = (uint32)(((uint64)f >> 4) * 0x9E3779B1u) >> 24;
		for (uint32 i = 0; i < 256; i++)
		{
			FenceSlot& slot = s_fenceSlots[(start + i) & 255];
			const VkFence cur = slot.fence.load(std::memory_order_acquire);
			if (cur == f)
				return &slot;
			if (cur == VK_NULL_HANDLE)
			{
				if (!insert)
					return nullptr;
				slot.fence.store(f, std::memory_order_release);
				return &slot;
			}
		}
		return nullptr;
	}

	bool FencesPending(uint32_t n, const VkFence* fences)
	{
		for (uint32 k = 0; k < n; k++)
			if (FenceSlot* slot = FindFence(fences[k], false); slot && slot->pending.load(std::memory_order_acquire))
				return true;
		return false;
	}

	// make(uint8* extra) copies the pointed-to arguments into the record's extra bytes and returns the replay lambda
	template<typename Make>
	void Emit(uint64 extra, Make make)
	{
		using F = decltype(make((uint8*)nullptr));
		static_assert(alignof(F) <= 16);
		constexpr uint64 xOff = Align16(sizeof(Rec) + sizeof(F));
		Rec* r = Reserve(Align16(xOff + extra));
		new ((uint8*)r + sizeof(Rec)) F(make((uint8*)r + xOff));
		r->run = [](Rec* rec) {
			F* f = (F*)((uint8*)rec + sizeof(Rec));
			(*f)();
			f->~F();
		};
		// release only: a seq_cst store is a locked xchg on x86, ~15k a frame on the GPU thread. The StoreLoad
		// fence the sleep handshake needs sits in WakeNow; records published between wakes wait for the next one.
		s_published.store(s_reserved, std::memory_order_release);
		if (++s_unnotified >= kWakeBatch)
			WakeReplay();
	}

	template<typename T>
	constexpr uint64 Bytes(uint32 n) { return Align16(sizeof(T) * n); }

	template<typename T>
	T* Copy(uint8*& x, const T* src, uint32 n)
	{
		if (!src || n == 0)
			return nullptr;
		T* d = (T*)x;
		memcpy((void*)d, src, sizeof(T) * n);
		x += Bytes<T>(n);
		return d;
	}

	void ReplayThread()
	{
		SetThreadName("VkRecord");
		botw_place_thread(1);  // with LatteThread, its producer (runtime/src/thread.cpp core plan)
		uint64 pos = 0;
		while (true)
		{
			uint64 end = s_published.load(std::memory_order_acquire);
			// up to ~1 ms of MWAITX (the producer's store wakes it) before the futex sleep. b30-b100 slept after ~200 us:
			// the GPU thread's record bursts have longer gaps, and each one then cost a futex wake syscall + IPI on the
			// GPU thread (~2% of it in libc syscall() on the Deck). MWAITX idles this SMT thread beside the busy GPU thread.
			for (int i = 0; i < 50 && pos == end; i++)
			{
				LatteCP_idleUntilWrite(&s_published, (uint32)end);
				end = s_published.load(std::memory_order_acquire);
			}
			if (pos == end)
			{
				// announce the sleep, then look again: a producer that published before seeing the flag is caught here
				// (WakeNow); a wake between the seq read and the futex call makes the futex return at once
				const uint32 seq = s_wakeSeq.load(std::memory_order_acquire);
				s_sleeping.store(true, std::memory_order_seq_cst);
				end = s_published.load(std::memory_order_seq_cst);
				if (pos == end)
					FutexWait(&s_wakeSeq, seq);
				s_sleeping.store(false, std::memory_order_relaxed);
				continue;
			}
			while (pos != end)
			{
				Rec* r = (Rec*)(s_ring + pos % kRingSize);
				const uint64 size = r->size;
				if (r->run)
					r->run(r);
				pos += size;
				s_consumed.store(pos, std::memory_order_release);
			}
			// pairs with WaitConsumed's seq_cst flag store and reload of s_consumed
			s_consumed.store(pos, std::memory_order_seq_cst);
			if (s_producerWaiting.load(std::memory_order_seq_cst))
			{
				s_consumedSeq.fetch_add(1, std::memory_order_release);
				FutexWake(&s_consumedSeq);
			}
		}
	}

	// real driver entry points
#define VKDEFER_REAL(f) PFN_##f r_##f;
	VKDEFER_REAL(vkCmdBindPipeline) VKDEFER_REAL(vkCmdBindDescriptorSets) VKDEFER_REAL(vkCmdSetViewport) VKDEFER_REAL(vkCmdSetScissor)
	VKDEFER_REAL(vkCmdCopyImage) VKDEFER_REAL(vkCmdWriteTimestamp) VKDEFER_REAL(vkCmdCopyBufferToImage) VKDEFER_REAL(vkCmdClearDepthStencilImage)
	VKDEFER_REAL(vkCmdClearColorImage) VKDEFER_REAL(vkCmdEndRenderPass) VKDEFER_REAL(vkCmdDraw) VKDEFER_REAL(vkCmdBlitImage)
	VKDEFER_REAL(vkCmdCopyBuffer) VKDEFER_REAL(vkCmdBindVertexBuffers) VKDEFER_REAL(vkCmdBeginRenderPass) VKDEFER_REAL(vkCmdSetDepthBias)
	VKDEFER_REAL(vkCmdResetQueryPool) VKDEFER_REAL(vkCmdDrawIndexed) VKDEFER_REAL(vkCmdCopyImageToBuffer) VKDEFER_REAL(vkCmdBindIndexBuffer)
	VKDEFER_REAL(vkCmdSetBlendConstants) VKDEFER_REAL(vkCmdPushConstants) VKDEFER_REAL(vkCmdEndRenderingKHR) VKDEFER_REAL(vkCmdBeginRenderingKHR)
	VKDEFER_REAL(vkCmdEndQuery) VKDEFER_REAL(vkCmdBeginQuery) VKDEFER_REAL(vkCmdCopyQueryPoolResults) VKDEFER_REAL(vkCmdClearAttachments)
	VKDEFER_REAL(vkCmdPipelineBarrier) VKDEFER_REAL(vkBeginCommandBuffer) VKDEFER_REAL(vkEndCommandBuffer) VKDEFER_REAL(vkResetCommandBuffer)
	VKDEFER_REAL(vkQueueSubmit) VKDEFER_REAL(vkGetFenceStatus) VKDEFER_REAL(vkWaitForFences) VKDEFER_REAL(vkResetFences)
	VKDEFER_REAL(vkCmdWaitEvents) VKDEFER_REAL(vkCmdSetEvent) VKDEFER_REAL(vkCmdPipelineBarrier2KHR) VKDEFER_REAL(vkCmdSetAttachmentFeedbackLoopEnableEXT)
	VKDEFER_REAL(vkQueuePresentKHR) VKDEFER_REAL(vkDeviceWaitIdle) VKDEFER_REAL(vkFreeCommandBuffers)
	VKDEFER_REAL(vkAllocateCommandBuffers) VKDEFER_REAL(vkDestroyCommandPool) VKDEFER_REAL(vkAcquireNextImageKHR)
#undef VKDEFER_REAL

	// calls that need the ring empty, then go straight to the driver
	template<auto* Real, typename Sig>
	struct DrainThen;
	template<auto* Real, typename R, typename... A>
	struct DrainThen<Real, R (VKAPI_PTR*)(A...)>
	{
		static R VKAPI_PTR Call(A... a)
		{
			Drain();
			return (*Real)(a...);
		}
	};

// queue on the producer thread, call directly elsewhere
#define VKDEFER_DIRECT(f, ...) \
	if (!t_producer) \
		return r_##f(__VA_ARGS__);

	VKAPI_ATTR void VKAPI_CALL T_vkCmdBindPipeline(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipeline p)
	{
		VKDEFER_DIRECT(vkCmdBindPipeline, cb, bp, p);
		Emit(0, [=](uint8*) { return [=] { r_vkCmdBindPipeline(cb, bp, p); }; });
	}

	VKAPI_ATTR void VKAPI_CALL T_vkCmdBindDescriptorSets(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipelineLayout layout, uint32_t first, uint32_t n,
		const VkDescriptorSet* sets, uint32_t dn, const uint32_t* dofs)
	{
		VKDEFER_DIRECT(vkCmdBindDescriptorSets, cb, bp, layout, first, n, sets, dn, dofs);
		Emit(Bytes<VkDescriptorSet>(n) + Bytes<uint32_t>(dn), [&](uint8* x) {
			const VkDescriptorSet* s = Copy(x, sets, n);
			const uint32_t* o = Copy(x, dofs, dn);
			return [=] { r_vkCmdBindDescriptorSets(cb, bp, layout, first, n, s, dn, o); };
		});
	}

	VKAPI_ATTR void VKAPI_CALL T_vkCmdSetViewport(VkCommandBuffer cb, uint32_t first, uint32_t n, const VkViewport* v)
	{
		VKDEFER_DIRECT(vkCmdSetViewport, cb, first, n, v);
		Emit(Bytes<VkViewport>(n), [&](uint8* x) {
			const VkViewport* c = Copy(x, v, n);
			return [=] { r_vkCmdSetViewport(cb, first, n, c); };
		});
	}

	VKAPI_ATTR void VKAPI_CALL T_vkCmdSetScissor(VkCommandBuffer cb, uint32_t first, uint32_t n, const VkRect2D* s)
	{
		VKDEFER_DIRECT(vkCmdSetScissor, cb, first, n, s);
		Emit(Bytes<VkRect2D>(n), [&](uint8* x) {
			const VkRect2D* c = Copy(x, s, n);
			return [=] { r_vkCmdSetScissor(cb, first, n, c); };
		});
	}

	VKAPI_ATTR void VKAPI_CALL T_vkCmdCopyImage(VkCommandBuffer cb, VkImage s, VkImageLayout sl, VkImage d, VkImageLayout dl, uint32_t n, const VkImageCopy* r)
	{
		VKDEFER_DIRECT(vkCmdCopyImage, cb, s, sl, d, dl, n, r);
		Emit(Bytes<VkImageCopy>(n), [&](uint8* x) {
			const VkImageCopy* c = Copy(x, r, n);
			return [=] { r_vkCmdCopyImage(cb, s, sl, d, dl, n, c); };
		});
	}

	VKAPI_ATTR void VKAPI_CALL T_vkCmdWriteTimestamp(VkCommandBuffer cb, VkPipelineStageFlagBits stage, VkQueryPool pool, uint32_t q)
	{
		VKDEFER_DIRECT(vkCmdWriteTimestamp, cb, stage, pool, q);
		Emit(0, [=](uint8*) { return [=] { r_vkCmdWriteTimestamp(cb, stage, pool, q); }; });
	}

	VKAPI_ATTR void VKAPI_CALL T_vkCmdCopyBufferToImage(VkCommandBuffer cb, VkBuffer b, VkImage i, VkImageLayout l, uint32_t n, const VkBufferImageCopy* r)
	{
		VKDEFER_DIRECT(vkCmdCopyBufferToImage, cb, b, i, l, n, r);
		Emit(Bytes<VkBufferImageCopy>(n), [&](uint8* x) {
			const VkBufferImageCopy* c = Copy(x, r, n);
			return [=] { r_vkCmdCopyBufferToImage(cb, b, i, l, n, c); };
		});
	}

	VKAPI_ATTR void VKAPI_CALL T_vkCmdClearDepthStencilImage(VkCommandBuffer cb, VkImage i, VkImageLayout l, const VkClearDepthStencilValue* v, uint32_t n,
		const VkImageSubresourceRange* r)
	{
		VKDEFER_DIRECT(vkCmdClearDepthStencilImage, cb, i, l, v, n, r);
		Emit(Bytes<VkClearDepthStencilValue>(1) + Bytes<VkImageSubresourceRange>(n), [&](uint8* x) {
			const VkClearDepthStencilValue* cv = Copy(x, v, 1);
			const VkImageSubresourceRange* c = Copy(x, r, n);
			return [=] { r_vkCmdClearDepthStencilImage(cb, i, l, cv, n, c); };
		});
	}

	VKAPI_ATTR void VKAPI_CALL T_vkCmdClearColorImage(VkCommandBuffer cb, VkImage i, VkImageLayout l, const VkClearColorValue* v, uint32_t n,
		const VkImageSubresourceRange* r)
	{
		VKDEFER_DIRECT(vkCmdClearColorImage, cb, i, l, v, n, r);
		Emit(Bytes<VkClearColorValue>(1) + Bytes<VkImageSubresourceRange>(n), [&](uint8* x) {
			const VkClearColorValue* cv = Copy(x, v, 1);
			const VkImageSubresourceRange* c = Copy(x, r, n);
			return [=] { r_vkCmdClearColorImage(cb, i, l, cv, n, c); };
		});
	}

	VKAPI_ATTR void VKAPI_CALL T_vkCmdEndRenderPass(VkCommandBuffer cb)
	{
		VKDEFER_DIRECT(vkCmdEndRenderPass, cb);
		Emit(0, [=](uint8*) { return [=] { r_vkCmdEndRenderPass(cb); }; });
	}

	VKAPI_ATTR void VKAPI_CALL T_vkCmdDraw(VkCommandBuffer cb, uint32_t vc, uint32_t ic, uint32_t fv, uint32_t fi)
	{
		VKDEFER_DIRECT(vkCmdDraw, cb, vc, ic, fv, fi);
		Emit(0, [=](uint8*) { return [=] { r_vkCmdDraw(cb, vc, ic, fv, fi); }; });
	}

	VKAPI_ATTR void VKAPI_CALL T_vkCmdBlitImage(VkCommandBuffer cb, VkImage s, VkImageLayout sl, VkImage d, VkImageLayout dl, uint32_t n, const VkImageBlit* r,
		VkFilter filter)
	{
		VKDEFER_DIRECT(vkCmdBlitImage, cb, s, sl, d, dl, n, r, filter);
		Emit(Bytes<VkImageBlit>(n), [&](uint8* x) {
			const VkImageBlit* c = Copy(x, r, n);
			return [=] { r_vkCmdBlitImage(cb, s, sl, d, dl, n, c, filter); };
		});
	}

	VKAPI_ATTR void VKAPI_CALL T_vkCmdCopyBuffer(VkCommandBuffer cb, VkBuffer s, VkBuffer d, uint32_t n, const VkBufferCopy* r)
	{
		VKDEFER_DIRECT(vkCmdCopyBuffer, cb, s, d, n, r);
		Emit(Bytes<VkBufferCopy>(n), [&](uint8* x) {
			const VkBufferCopy* c = Copy(x, r, n);
			return [=] { r_vkCmdCopyBuffer(cb, s, d, n, c); };
		});
	}

	VKAPI_ATTR void VKAPI_CALL T_vkCmdBindVertexBuffers(VkCommandBuffer cb, uint32_t first, uint32_t n, const VkBuffer* b, const VkDeviceSize* o)
	{
		VKDEFER_DIRECT(vkCmdBindVertexBuffers, cb, first, n, b, o);
		Emit(Bytes<VkBuffer>(n) + Bytes<VkDeviceSize>(n), [&](uint8* x) {
			const VkBuffer* cbuf = Copy(x, b, n);
			const VkDeviceSize* co = Copy(x, o, n);
			return [=] { r_vkCmdBindVertexBuffers(cb, first, n, cbuf, co); };
		});
	}

	VKAPI_ATTR void VKAPI_CALL T_vkCmdBeginRenderPass(VkCommandBuffer cb, const VkRenderPassBeginInfo* info, VkSubpassContents contents)
	{
		VKDEFER_DIRECT(vkCmdBeginRenderPass, cb, info, contents);
		if (info->pNext)
		{
			Drain();
			return r_vkCmdBeginRenderPass(cb, info, contents);
		}
		Emit(Bytes<VkRenderPassBeginInfo>(1) + Bytes<VkClearValue>(info->clearValueCount), [&](uint8* x) {
			VkRenderPassBeginInfo* c = Copy(x, info, 1);
			c->pClearValues = Copy(x, info->pClearValues, info->clearValueCount);
			return [=] { r_vkCmdBeginRenderPass(cb, c, contents); };
		});
	}

	VKAPI_ATTR void VKAPI_CALL T_vkCmdSetDepthBias(VkCommandBuffer cb, float a, float b, float c)
	{
		VKDEFER_DIRECT(vkCmdSetDepthBias, cb, a, b, c);
		Emit(0, [=](uint8*) { return [=] { r_vkCmdSetDepthBias(cb, a, b, c); }; });
	}

	VKAPI_ATTR void VKAPI_CALL T_vkCmdResetQueryPool(VkCommandBuffer cb, VkQueryPool pool, uint32_t first, uint32_t n)
	{
		VKDEFER_DIRECT(vkCmdResetQueryPool, cb, pool, first, n);
		Emit(0, [=](uint8*) { return [=] { r_vkCmdResetQueryPool(cb, pool, first, n); }; });
	}

	VKAPI_ATTR void VKAPI_CALL T_vkCmdDrawIndexed(VkCommandBuffer cb, uint32_t ic, uint32_t inst, uint32_t fi, int32_t vo, uint32_t finst)
	{
		VKDEFER_DIRECT(vkCmdDrawIndexed, cb, ic, inst, fi, vo, finst);
		Emit(0, [=](uint8*) { return [=] { r_vkCmdDrawIndexed(cb, ic, inst, fi, vo, finst); }; });
	}

	VKAPI_ATTR void VKAPI_CALL T_vkCmdCopyImageToBuffer(VkCommandBuffer cb, VkImage i, VkImageLayout l, VkBuffer b, uint32_t n, const VkBufferImageCopy* r)
	{
		VKDEFER_DIRECT(vkCmdCopyImageToBuffer, cb, i, l, b, n, r);
		Emit(Bytes<VkBufferImageCopy>(n), [&](uint8* x) {
			const VkBufferImageCopy* c = Copy(x, r, n);
			return [=] { r_vkCmdCopyImageToBuffer(cb, i, l, b, n, c); };
		});
	}

	VKAPI_ATTR void VKAPI_CALL T_vkCmdBindIndexBuffer(VkCommandBuffer cb, VkBuffer b, VkDeviceSize o, VkIndexType t)
	{
		VKDEFER_DIRECT(vkCmdBindIndexBuffer, cb, b, o, t);
		Emit(0, [=](uint8*) { return [=] { r_vkCmdBindIndexBuffer(cb, b, o, t); }; });
	}

	VKAPI_ATTR void VKAPI_CALL T_vkCmdSetBlendConstants(VkCommandBuffer cb, const float c[4])
	{
		VKDEFER_DIRECT(vkCmdSetBlendConstants, cb, c);
		Emit(Bytes<float>(4), [&](uint8* x) {
			const float* cc = Copy(x, c, 4);
			return [=] { r_vkCmdSetBlendConstants(cb, cc); };
		});
	}

	VKAPI_ATTR void VKAPI_CALL T_vkCmdPushConstants(VkCommandBuffer cb, VkPipelineLayout layout, VkShaderStageFlags stages, uint32_t off, uint32_t size,
		const void* data)
	{
		VKDEFER_DIRECT(vkCmdPushConstants, cb, layout, stages, off, size, data);
		Emit(Bytes<uint8>(size), [&](uint8* x) {
			const uint8* d = Copy(x, (const uint8*)data, size);
			return [=] { r_vkCmdPushConstants(cb, layout, stages, off, size, d); };
		});
	}

	VKAPI_ATTR void VKAPI_CALL T_vkCmdEndRenderingKHR(VkCommandBuffer cb)
	{
		VKDEFER_DIRECT(vkCmdEndRenderingKHR, cb);
		Emit(0, [=](uint8*) { return [=] { r_vkCmdEndRenderingKHR(cb); }; });
	}

	VKAPI_ATTR void VKAPI_CALL T_vkCmdBeginRenderingKHR(VkCommandBuffer cb, const VkRenderingInfo* info)
	{
		VKDEFER_DIRECT(vkCmdBeginRenderingKHR, cb, info);
		bool chained = info->pNext != nullptr;
		for (uint32 i = 0; i < info->colorAttachmentCount; i++)
			chained |= info->pColorAttachments[i].pNext != nullptr;
		chained |= info->pDepthAttachment && info->pDepthAttachment->pNext;
		chained |= info->pStencilAttachment && info->pStencilAttachment->pNext;
		if (chained)
		{
			Drain();
			return r_vkCmdBeginRenderingKHR(cb, info);
		}
		Emit(Bytes<VkRenderingInfo>(1) + Bytes<VkRenderingAttachmentInfo>(info->colorAttachmentCount) + 2 * Bytes<VkRenderingAttachmentInfo>(1), [&](uint8* x) {
			VkRenderingInfo* c = Copy(x, info, 1);
			c->pColorAttachments = Copy(x, info->pColorAttachments, info->colorAttachmentCount);
			c->pDepthAttachment = Copy(x, info->pDepthAttachment, info->pDepthAttachment ? 1 : 0);
			c->pStencilAttachment = Copy(x, info->pStencilAttachment, info->pStencilAttachment ? 1 : 0);
			return [=] { r_vkCmdBeginRenderingKHR(cb, c); };
		});
	}

	VKAPI_ATTR void VKAPI_CALL T_vkCmdEndQuery(VkCommandBuffer cb, VkQueryPool pool, uint32_t q)
	{
		VKDEFER_DIRECT(vkCmdEndQuery, cb, pool, q);
		Emit(0, [=](uint8*) { return [=] { r_vkCmdEndQuery(cb, pool, q); }; });
	}

	VKAPI_ATTR void VKAPI_CALL T_vkCmdBeginQuery(VkCommandBuffer cb, VkQueryPool pool, uint32_t q, VkQueryControlFlags flags)
	{
		VKDEFER_DIRECT(vkCmdBeginQuery, cb, pool, q, flags);
		Emit(0, [=](uint8*) { return [=] { r_vkCmdBeginQuery(cb, pool, q, flags); }; });
	}

	VKAPI_ATTR void VKAPI_CALL T_vkCmdCopyQueryPoolResults(VkCommandBuffer cb, VkQueryPool pool, uint32_t first, uint32_t n, VkBuffer b, VkDeviceSize off,
		VkDeviceSize stride, VkQueryResultFlags flags)
	{
		VKDEFER_DIRECT(vkCmdCopyQueryPoolResults, cb, pool, first, n, b, off, stride, flags);
		Emit(0, [=](uint8*) { return [=] { r_vkCmdCopyQueryPoolResults(cb, pool, first, n, b, off, stride, flags); }; });
	}

	VKAPI_ATTR void VKAPI_CALL T_vkCmdClearAttachments(VkCommandBuffer cb, uint32_t n, const VkClearAttachment* a, uint32_t rn, const VkClearRect* r)
	{
		VKDEFER_DIRECT(vkCmdClearAttachments, cb, n, a, rn, r);
		Emit(Bytes<VkClearAttachment>(n) + Bytes<VkClearRect>(rn), [&](uint8* x) {
			const VkClearAttachment* ca = Copy(x, a, n);
			const VkClearRect* cr = Copy(x, r, rn);
			return [=] { r_vkCmdClearAttachments(cb, n, ca, rn, cr); };
		});
	}

	VKAPI_ATTR void VKAPI_CALL T_vkCmdPipelineBarrier(VkCommandBuffer cb, VkPipelineStageFlags s, VkPipelineStageFlags d, VkDependencyFlags f, uint32_t mc,
		const VkMemoryBarrier* m, uint32_t bc, const VkBufferMemoryBarrier* b, uint32_t ic, const VkImageMemoryBarrier* i)
	{
		VKDEFER_DIRECT(vkCmdPipelineBarrier, cb, s, d, f, mc, m, bc, b, ic, i);
		bool chained = false;
		for (uint32 k = 0; k < mc; k++)
			chained |= m[k].pNext != nullptr;
		for (uint32 k = 0; k < bc; k++)
			chained |= b[k].pNext != nullptr;
		for (uint32 k = 0; k < ic; k++)
			chained |= i[k].pNext != nullptr;
		if (chained)
		{
			Drain();
			return r_vkCmdPipelineBarrier(cb, s, d, f, mc, m, bc, b, ic, i);
		}
		Emit(Bytes<VkMemoryBarrier>(mc) + Bytes<VkBufferMemoryBarrier>(bc) + Bytes<VkImageMemoryBarrier>(ic), [&](uint8* x) {
			const VkMemoryBarrier* cm = Copy(x, m, mc);
			const VkBufferMemoryBarrier* cbm = Copy(x, b, bc);
			const VkImageMemoryBarrier* cim = Copy(x, i, ic);
			return [=] { r_vkCmdPipelineBarrier(cb, s, d, f, mc, cm, bc, cbm, ic, cim); };
		});
	}

	VKAPI_ATTR VkResult VKAPI_CALL T_vkBeginCommandBuffer(VkCommandBuffer cb, const VkCommandBufferBeginInfo* info)
	{
		VKDEFER_DIRECT(vkBeginCommandBuffer, cb, info);
		if (info->pNext || info->pInheritanceInfo)
		{
			Drain();
			return r_vkBeginCommandBuffer(cb, info);
		}
		const VkCommandBufferBeginInfo copy = *info;
		Emit(0, [=](uint8*) { return [=] { r_vkBeginCommandBuffer(cb, &copy); }; });
		return VK_SUCCESS;
	}

	VKAPI_ATTR VkResult VKAPI_CALL T_vkEndCommandBuffer(VkCommandBuffer cb)
	{
		VKDEFER_DIRECT(vkEndCommandBuffer, cb);
		Emit(0, [=](uint8*) { return [=] { r_vkEndCommandBuffer(cb); }; });
		return VK_SUCCESS;
	}

	VKAPI_ATTR VkResult VKAPI_CALL T_vkResetCommandBuffer(VkCommandBuffer cb, VkCommandBufferResetFlags flags)
	{
		VKDEFER_DIRECT(vkResetCommandBuffer, cb, flags);
		Emit(0, [=](uint8*) { return [=] { r_vkResetCommandBuffer(cb, flags); }; });
		return VK_SUCCESS;
	}

	// fences stay correct: the next fence check or wait simply sees the submit once the ring reaches it
	VKAPI_ATTR VkResult VKAPI_CALL T_vkQueueSubmit(VkQueue q, uint32_t n, const VkSubmitInfo* s, VkFence fence)
	{
		VKDEFER_DIRECT(vkQueueSubmit, q, n, s, fence);
		if (VkResult e = s_submitError.load(std::memory_order_relaxed); e != VK_SUCCESS)
			return e;
		uint64 extra = Bytes<VkSubmitInfo>(n);
		bool chained = false;
		for (uint32 k = 0; k < n; k++)
		{
			chained |= s[k].pNext != nullptr;
			extra += Bytes<VkSemaphore>(s[k].waitSemaphoreCount) + Bytes<VkPipelineStageFlags>(s[k].waitSemaphoreCount) +
				Bytes<VkCommandBuffer>(s[k].commandBufferCount) + Bytes<VkSemaphore>(s[k].signalSemaphoreCount);
		}
		if (chained)
		{
			Drain();
			return r_vkQueueSubmit(q, n, s, fence);
		}
		FenceSlot* fenceSlot = fence ? FindFence(fence, true) : nullptr;
		if (fenceSlot)
			fenceSlot->pending.fetch_add(1, std::memory_order_relaxed);
		Emit(extra, [&](uint8* x) {
			VkSubmitInfo* c = Copy(x, s, n);
			for (uint32 k = 0; k < n; k++)
			{
				c[k].pWaitSemaphores = Copy(x, s[k].pWaitSemaphores, s[k].waitSemaphoreCount);
				c[k].pWaitDstStageMask = Copy(x, s[k].pWaitDstStageMask, s[k].waitSemaphoreCount);
				c[k].pCommandBuffers = Copy(x, s[k].pCommandBuffers, s[k].commandBufferCount);
				c[k].pSignalSemaphores = Copy(x, s[k].pSignalSemaphores, s[k].signalSemaphoreCount);
			}
			return [=] {
				const VkResult r = r_vkQueueSubmit(q, n, c, fence);
				if (r != VK_SUCCESS)
					s_submitError.store(r, std::memory_order_relaxed);
				if (fenceSlot)
					fenceSlot->pending.fetch_sub(1, std::memory_order_release);
			};
		});
		WakeReplay();  // GPU work starts at the submit: record it now
		if (fence && !fenceSlot)
			Drain();  // fence table full: submit right away instead
		return VK_SUCCESS;
	}

	// Present goes through the ring like a submit: the GPU thread used to wait here for the ring to drain and for the
	// presentation engine (0.7 ms a frame, 1.7 ms in the frames that missed vsync at a 700 MHz pin, 2026-11-03) while
	// the GPU sat idle. The replay thread presents right after the frame's submit; queue and swapchain stay on one
	// thread (acquire still drains first). A failure (out of date, suboptimal, lost) comes back on the next present.
	std::atomic<VkResult> s_presentError{VK_SUCCESS};
	VKAPI_ATTR VkResult VKAPI_CALL T_vkQueuePresentKHR(VkQueue q, const VkPresentInfoKHR* p)
	{
		if (!t_producer || p->pNext || p->pResults)  // other threads, present ids (FIFO queue limit), per-swapchain results
		{
			Drain();
			return r_vkQueuePresentKHR(q, p);
		}
		const uint64 extra = Bytes<VkPresentInfoKHR>(1) + Bytes<VkSemaphore>(p->waitSemaphoreCount) +
			Bytes<VkSwapchainKHR>(p->swapchainCount) + Bytes<uint32_t>(p->swapchainCount);
		Emit(extra, [&](uint8* x) {
			VkPresentInfoKHR* c = Copy(x, p, 1);
			c->pWaitSemaphores = Copy(x, p->pWaitSemaphores, p->waitSemaphoreCount);
			c->pSwapchains = Copy(x, p->pSwapchains, p->swapchainCount);
			c->pImageIndices = Copy(x, p->pImageIndices, p->swapchainCount);
			return [=] {
				const VkResult r = r_vkQueuePresentKHR(q, c);
				if (r != VK_SUCCESS)
					s_presentError.store(r, std::memory_order_relaxed);
			};
		});
		WakeReplay();
		return s_presentError.exchange(VK_SUCCESS, std::memory_order_relaxed);
	}

	VKAPI_ATTR VkResult VKAPI_CALL T_vkGetFenceStatus(VkDevice dev, VkFence fence)
	{
		if (FencesPending(1, &fence))
			return VK_NOT_READY;
		return r_vkGetFenceStatus(dev, fence);
	}

	VKAPI_ATTR VkResult VKAPI_CALL T_vkWaitForFences(VkDevice dev, uint32_t n, const VkFence* fences, VkBool32 all, uint64_t timeout)
	{
		if (FencesPending(n, fences))
			Drain();
		return r_vkWaitForFences(dev, n, fences, all, timeout);
	}

	VKAPI_ATTR VkResult VKAPI_CALL T_vkResetFences(VkDevice dev, uint32_t n, const VkFence* fences)
	{
		if (FencesPending(n, fences))
			Drain();
		return r_vkResetFences(dev, n, fences);
	}
#undef VKDEFER_DIRECT
}  // namespace

void VkDefer_SetProducerThread()
{
	t_producer = s_installed;
}

void VkDefer_Install()
{
	if (const char* e = getenv("BOTW_VKTHREAD"); e && e[0] == '0')
		return;
	if (!vkCmdDrawIndexed || !vkQueueSubmit || !vkBeginCommandBuffer)
		return;
	s_ring = new (std::align_val_t(64)) uint8[kRingSize];
#define VKDEFER_SWAP(f) \
	if (f) \
	{ \
		r_##f = f; \
		f = T_##f; \
	}
#define VKDEFER_DRAIN(f) \
	if (f) \
	{ \
		r_##f = f; \
		f = DrainThen<&r_##f, PFN_##f>::Call; \
	}
	VKDEFER_SWAP(vkCmdBindPipeline) VKDEFER_SWAP(vkCmdBindDescriptorSets) VKDEFER_SWAP(vkCmdSetViewport) VKDEFER_SWAP(vkCmdSetScissor)
	VKDEFER_SWAP(vkCmdCopyImage) VKDEFER_SWAP(vkCmdWriteTimestamp) VKDEFER_SWAP(vkCmdCopyBufferToImage) VKDEFER_SWAP(vkCmdClearDepthStencilImage)
	VKDEFER_SWAP(vkCmdClearColorImage) VKDEFER_SWAP(vkCmdEndRenderPass) VKDEFER_SWAP(vkCmdDraw) VKDEFER_SWAP(vkCmdBlitImage)
	VKDEFER_SWAP(vkCmdCopyBuffer) VKDEFER_SWAP(vkCmdBindVertexBuffers) VKDEFER_SWAP(vkCmdBeginRenderPass) VKDEFER_SWAP(vkCmdSetDepthBias)
	VKDEFER_SWAP(vkCmdResetQueryPool) VKDEFER_SWAP(vkCmdDrawIndexed) VKDEFER_SWAP(vkCmdCopyImageToBuffer) VKDEFER_SWAP(vkCmdBindIndexBuffer)
	VKDEFER_SWAP(vkCmdSetBlendConstants) VKDEFER_SWAP(vkCmdPushConstants) VKDEFER_SWAP(vkCmdEndRenderingKHR) VKDEFER_SWAP(vkCmdBeginRenderingKHR)
	VKDEFER_SWAP(vkCmdEndQuery) VKDEFER_SWAP(vkCmdBeginQuery) VKDEFER_SWAP(vkCmdCopyQueryPoolResults) VKDEFER_SWAP(vkCmdClearAttachments)
	VKDEFER_SWAP(vkCmdPipelineBarrier) VKDEFER_SWAP(vkBeginCommandBuffer) VKDEFER_SWAP(vkEndCommandBuffer) VKDEFER_SWAP(vkResetCommandBuffer)
	VKDEFER_SWAP(vkQueueSubmit) VKDEFER_SWAP(vkGetFenceStatus) VKDEFER_SWAP(vkWaitForFences) VKDEFER_SWAP(vkResetFences)
	VKDEFER_DRAIN(vkCmdWaitEvents) VKDEFER_DRAIN(vkCmdSetEvent) VKDEFER_DRAIN(vkCmdPipelineBarrier2KHR) VKDEFER_DRAIN(vkCmdSetAttachmentFeedbackLoopEnableEXT)
	VKDEFER_SWAP(vkQueuePresentKHR) VKDEFER_DRAIN(vkDeviceWaitIdle) VKDEFER_DRAIN(vkFreeCommandBuffers)
	VKDEFER_DRAIN(vkAllocateCommandBuffers) VKDEFER_DRAIN(vkDestroyCommandPool)
	VKDEFER_DRAIN(vkAcquireNextImageKHR)  // keeps the acquire after every submit Cemu issued before it
#undef VKDEFER_SWAP
#undef VKDEFER_DRAIN
	s_installed = true;
	std::thread(ReplayThread).detach();
}
