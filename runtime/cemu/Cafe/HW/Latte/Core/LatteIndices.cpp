#include "Cafe/HW/Latte/Core/LatteConst.h"
#include "Cafe/HW/Latte/Renderer/Renderer.h"
#include "Cafe/HW/Latte/ISA/RegDefines.h"
#include "Cafe/HW/Latte/Core/LattePerformanceMonitor.h"
#include "Common/cpu_features.h"
#include "Cafe/HW/Latte/Core/LatteBufferCache.h"
#include "util/containers/robin_hood.h"
#include "telemetry.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <map>
#include <unordered_map>
#include <vector>

#if defined(ARCH_X86_64) && defined(__GNUC__)
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#endif

struct
{
	struct CacheEntry
	{
		// input data
		const void* lastPtr;
		uint32 lastCount;
		LattePrimitiveMode lastPrimitiveMode;
		LatteIndexType lastIndexType;
		uint64 lastUsed;
		// output
		uint32 indexMax;
		Renderer::INDEX_TYPE renderIndexType;
		uint32 outputCount;
		Renderer::IndexAllocation indexAllocation;
	};
	std::array<CacheEntry, 8> entry;
	uint64 currentUsageCounter{0};
}LatteIndexCache{};

// BotW-Native: cross-frame index cache. Upstream kept the converted (byte-swapped, quad-unpacked) indices of the
// last 8 draws and dropped them at every buffer cleanup, so with ~4500 draws a frame nearly every draw converted
// its indices again and churned the index heap (conversion + heap alloc/free ~9% of LatteThread on the Deck).
// Entries now live across frames and are dropped when the game flushes (DCFlush/GX2Invalidate) any byte of their
// source range - the rule Cemu's vertex/uniform buffer cache already relies on - or on a global flush.
// BOTW_INDEX_CACHE=0 restores the upstream behaviour.
namespace
{
	struct PersistentKey
	{
		const void* ptr;
		uint32 count;
		LattePrimitiveMode mode;
		LatteIndexType type;
		bool operator==(const PersistentKey& o) const { return ptr == o.ptr && count == o.count && mode == o.mode && type == o.type; }
	};
	struct PersistentKeyHash
	{
		size_t operator()(const PersistentKey& k) const
		{
			uint64 h = (uint64)(uintptr_t)k.ptr * 0x9E3779B97F4A7C15ull;
			h ^= ((uint64)k.count << 8 | (uint64)k.mode << 4 | (uint64)k.type) * 0xC2B2AE3D27D4EB4Full;
			return (size_t)(h ^ (h >> 29));
		}
	};
	struct PersistentEntry
	{
		uint64 lastUsed;
		uint32 inputBytes;
		uint32 outputBytes;
		uint32 indexMax;
		Renderer::INDEX_TYPE renderIndexType;
		uint32 outputCount;
		Renderer::IndexAllocation indexAllocation;
	};
	// converted index bytes kept (host-visible GTT memory); TEST: BOTW_INDEX_CACHE_KB=<n> to force trims
	const uint64 kPersistentBudget = getenv("BOTW_INDEX_CACHE_KB") ? atoll(getenv("BOTW_INDEX_CACHE_KB")) * 1024ull : 32ull * 1024 * 1024;
	// flat (entries inline): the node-based std::unordered_map lookup was 9.4% of LatteThread (PC call stacks, b76)
	using PersistentMap = robin_hood::unordered_flat_map<PersistentKey, PersistentEntry, PersistentKeyHash>;
	PersistentMap s_persist;
	std::multimap<uintptr_t, PersistentKey> s_persistByAddr;  // source start -> key, for range invalidation
	uint64 s_persistBytes = 0;
	uint32 s_persistMaxInput = 0;  // largest cached source range: bounds the invalidation lookback
	uintptr_t s_persistLo = UINTPTR_MAX, s_persistHi = 0;  // span of cached sources (quick reject)
	std::atomic<bool> s_persistGlobalFlush{false};
	uint64 s_statHits = 0, s_statMisses = 0, s_statInvalidated = 0, s_statGlobalClears = 0;

	void PersistentStats()
	{
		if ((s_statHits + s_statMisses) % 200000 != 0)
			return;
		cemuLog_log(LogType::Force, "index cache: {} hits {} misses {} invalidated {} global clears, {} entries {} KB",
		            s_statHits, s_statMisses, s_statInvalidated, s_statGlobalClears, s_persist.size(), s_persistBytes / 1024);
	}

	bool PersistentOn()
	{
		static const bool on = !(getenv("BOTW_INDEX_CACHE") && atoi(getenv("BOTW_INDEX_CACHE")) == 0);
		return on;
	}

	void PersistentErase(PersistentMap::iterator it)
	{
		auto range = s_persistByAddr.equal_range((uintptr_t)it->first.ptr);
		for (auto a = range.first; a != range.second; ++a)
		{
			if (a->second == it->first)
			{
				s_persistByAddr.erase(a);
				break;
			}
		}
		if (it->second.indexAllocation.rendererInternal)
			g_renderer->indexData_releaseIndexMemory(it->second.indexAllocation);
		s_persistBytes -= it->second.outputBytes;
		s_persist.erase(it);
	}

	void PersistentClear()
	{
		for (auto& [k, e] : s_persist)
			if (e.indexAllocation.rendererInternal)
				g_renderer->indexData_releaseIndexMemory(e.indexAllocation);
		s_persist.clear();
		s_persistByAddr.clear();
		s_persistBytes = 0;
		s_persistMaxInput = 0;
		s_persistLo = UINTPTR_MAX;
		s_persistHi = 0;
	}

	void PersistentInvalidate(uintptr_t lo, uintptr_t hi)
	{
		if (s_persist.empty() || hi <= s_persistLo || lo >= s_persistHi)
			return;
		auto a = s_persistByAddr.lower_bound(lo > s_persistMaxInput ? lo - s_persistMaxInput : 0);
		while (a != s_persistByAddr.end() && a->first < hi)
		{
			auto it = s_persist.find(a->second);
			++a;
			if (it != s_persist.end() && (uintptr_t)it->first.ptr + it->second.inputBytes > lo)
			{
				// PersistentErase removes the multimap node of this key; `a` already points past it
				PersistentErase(it);
				s_statInvalidated++;
			}
		}
	}

	// Over budget: drop the least recently used half (amortized; runs rarely). `keep` = the entry the current draw uses.
	void PersistentTrim(const PersistentKey& keep)
	{
		if (s_persistBytes <= kPersistentBudget)
			return;
		std::vector<uint64> ages;
		ages.reserve(s_persist.size());
		for (auto& [k, e] : s_persist)
			ages.push_back(e.lastUsed);
		std::nth_element(ages.begin(), ages.begin() + ages.size() / 2, ages.end());
		const uint64 cut = ages[ages.size() / 2];
		static uint32 s_trims = 0;
		if ((++s_trims & (s_trims - 1)) == 0)  // 1st, 2nd, 4th, 8th... trim
			cemuLog_log(LogType::Force, "index cache: trim #{} at {} KB, {} entries", s_trims, s_persistBytes / 1024, s_persist.size());
		// a flat map erase moves other entries: collect first, erase after
		std::vector<PersistentKey> drop;
		for (auto& [k, e] : s_persist)
			if (e.lastUsed <= cut && !(k == keep))
				drop.push_back(k);
		for (auto& k : drop)
			PersistentErase(s_persist.find(k));
	}
}

// DCFlush notifications (LatteBufferCache.cpp): called on LatteThread for every flushed page, before draws use them
void LatteIndices_invalidatePhysRange(uint32 physAddr, uint32 size)
{
	if (s_persist.empty())
		return;
	const uintptr_t lo = (uintptr_t)memory_getPointerFromVirtualOffset(physAddr);
	PersistentInvalidate(lo, lo + size);
}

// a global flush (address 0 or size ~0) from any thread: every cached entry is dropped at the next draw
void LatteIndices_notifyGlobalFlush()
{
	s_persistGlobalFlush.store(true, std::memory_order_release);
}

void LatteIndices_invalidate(const void* memPtr, uint32 size)
{
	for(auto& entry : LatteIndexCache.entry)
	{
		if (entry.lastPtr >= memPtr && (entry.lastPtr < ((uint8*)memPtr + size)) )
		{
			if(entry.lastPtr != nullptr)
				g_renderer->indexData_releaseIndexMemory(entry.indexAllocation);
			entry.lastPtr = nullptr;
			entry.lastCount = 0;
		}
	}
	PersistentInvalidate((uintptr_t)memPtr, (uintptr_t)memPtr + size);
}

// buffer cleanup after finished command buffers: upstream dropped the cache here; persistent entries stay
void LatteIndices_invalidateAll()
{
	if (PersistentOn())
		return;
	for(auto& entry : LatteIndexCache.entry)
	{
		if (entry.lastPtr != nullptr)
			g_renderer->indexData_releaseIndexMemory(entry.indexAllocation);
		entry.lastPtr = nullptr;
		entry.lastCount = 0;
	}
}

uint64 LatteIndices_GetNextUsageIndex()
{
	return LatteIndexCache.currentUsageCounter++;
}

uint32 LatteIndices_calculateIndexOutputSize(LattePrimitiveMode primitiveMode, LatteIndexType indexType, uint32 count)
{
	if (primitiveMode == LattePrimitiveMode::QUADS)
	{
		sint32 numQuads = count / 4;
		if (indexType == LatteIndexType::AUTO)
		{
			if(count <= 0xFFFF)
				return numQuads * 6 * sizeof(uint16);
			return numQuads * 6 * sizeof(uint32);
		}
		if (indexType == LatteIndexType::U16_BE || indexType == LatteIndexType::U16_LE)
			return numQuads * 6 * sizeof(uint16);
		if (indexType == LatteIndexType::U32_BE || indexType == LatteIndexType::U32_LE)
			return numQuads * 6 * sizeof(uint32);
		cemu_assert_suspicious();
		return 0;
	}
	else if (primitiveMode == LattePrimitiveMode::QUAD_STRIP)
	{
		if (count <= 3)
		{
			return 0;
		}
		sint32 numQuads = (count-2) / 2;
		if (indexType == LatteIndexType::AUTO)
		{
			if (count <= 0xFFFF)
				return numQuads * 6 * sizeof(uint16);
			return numQuads * 6 * sizeof(uint32);
		}
		if (indexType == LatteIndexType::U16_BE || indexType == LatteIndexType::U16_LE)
			return numQuads * 6 * sizeof(uint16);
		if (indexType == LatteIndexType::U32_BE || indexType == LatteIndexType::U32_LE)
			return numQuads * 6 * sizeof(uint32);
		cemu_assert_suspicious();
		return 0;
	}
	else if (primitiveMode == LattePrimitiveMode::LINE_LOOP)
	{
		count++; // one extra vertex to reconnect the LINE_STRIP to the beginning
		if (indexType == LatteIndexType::AUTO)
		{
			if (count <= 0xFFFF)
				return count * sizeof(uint16);
			return count * sizeof(uint32);
		}
		if (indexType == LatteIndexType::U16_BE || indexType == LatteIndexType::U16_LE)
			return count * sizeof(uint16);
		if (indexType == LatteIndexType::U32_BE || indexType == LatteIndexType::U32_LE)
			return count * sizeof(uint32);
		cemu_assert_suspicious();
		return 0;
	}
	else if (primitiveMode == LattePrimitiveMode::TRIANGLE_FAN && g_renderer->GetType() == RendererAPI::Metal)
	{
		if (indexType == LatteIndexType::AUTO)
		{
			if (count <= 0xFFFF)
				return count * sizeof(uint16);
			return count * sizeof(uint32);
		}
		if (indexType == LatteIndexType::U16_BE || indexType == LatteIndexType::U16_LE)
			return count * sizeof(uint16);
		if (indexType == LatteIndexType::U32_BE || indexType == LatteIndexType::U32_LE)
			return count * sizeof(uint32);
		cemu_assert_suspicious();
		return 0;
	}
	else if(indexType == LatteIndexType::AUTO)
		return 0;
	else if (indexType == LatteIndexType::U16_BE || indexType == LatteIndexType::U16_LE)
		return count * sizeof(uint16);
	else if (indexType == LatteIndexType::U32_BE || indexType == LatteIndexType::U32_LE)
		return count * sizeof(uint32);
	return 0;
}

template<typename T>
void LatteIndices_convertBE(const void* indexDataInput, void* indexDataOutput, uint32 count, uint32& indexMax)
{
	const betype<T>* src = (betype<T>*)indexDataInput;
	T* dst = (T*)indexDataOutput;
	for (uint32 i = 0; i < count; i++)
	{
		T v = *src;
		*dst = v;
		indexMax = std::max(indexMax, (uint32)v);
		dst++;
		src++;
	}
}

template<typename T>
void LatteIndices_convertLE(const void* indexDataInput, void* indexDataOutput, uint32 count, uint32& indexMax)
{
	const T* src = (T*)indexDataInput;
	T* dst = (T*)indexDataOutput;
	for (uint32 i = 0; i < count; i++)
	{
		T v = *src;
		*dst = v;
		indexMax = std::max(indexMax, (uint32)v);
		dst++;
		src++;
	}
}

template<typename T>
void LatteIndices_unpackQuadsAndConvert(const void* indexDataInput, void* indexDataOutput, uint32 count, uint32& indexMax)
{
	sint32 numQuads = count / 4;
	const betype<T>* src = (betype<T>*)indexDataInput;
	T* dst = (T*)indexDataOutput;
	for (sint32 i = 0; i < numQuads; i++)
	{
		T idx0 = src[0];
		T idx1 = src[1];
		T idx2 = src[2];
		T idx3 = src[3];
		indexMax = std::max(indexMax, (uint32)idx0);
		indexMax = std::max(indexMax, (uint32)idx1);
		indexMax = std::max(indexMax, (uint32)idx2);
		indexMax = std::max(indexMax, (uint32)idx3);
		dst[0] = idx0;
		dst[1] = idx1;
		dst[2] = idx2;
		dst[3] = idx0;
		dst[4] = idx2;
		dst[5] = idx3;
		src += 4;
		dst += 6;
	}
}

template<typename T>
void LatteIndices_generateAutoQuadIndices(const void* indexDataInput, void* indexDataOutput, uint32 count, uint32& indexMax)
{
	sint32 numQuads = count / 4;
	const betype<T>* src = (betype<T>*)indexDataInput;
	T* dst = (T*)indexDataOutput;
	for (sint32 i = 0; i < numQuads; i++)
	{
		T idx0 = i * 4 + 0;
		T idx1 = i * 4 + 1;
		T idx2 = i * 4 + 2;
		T idx3 = i * 4 + 3;
		dst[0] = idx0;
		dst[1] = idx1;
		dst[2] = idx2;
		dst[3] = idx0;
		dst[4] = idx2;
		dst[5] = idx3;
		src += 4;
		dst += 6;
	}
	indexMax = std::max(count, 1u) - 1;
}

template<typename T>
void LatteIndices_unpackQuadStripAndConvert(const void* indexDataInput, void* indexDataOutput, uint32 count, uint32& indexMax)
{
	if (count <= 3)
		return;
	sint32 numQuads = (count - 2) / 2;
	const betype<T>* src = (betype<T>*)indexDataInput;
	T* dst = (T*)indexDataOutput;
	for (sint32 i = 0; i < numQuads; i++)
	{
		T idx0 = src[0];
		T idx1 = src[1];
		T idx2 = src[2];
		T idx3 = src[3];
		indexMax = std::max(indexMax, (uint32)idx0);
		indexMax = std::max(indexMax, (uint32)idx1);
		indexMax = std::max(indexMax, (uint32)idx2);
		indexMax = std::max(indexMax, (uint32)idx3);
		dst[0] = idx0;
		dst[1] = idx1;
		dst[2] = idx2;
		dst[3] = idx2;
		dst[4] = idx1;
		dst[5] = idx3;
		src += 2;
		dst += 6;
	}
}

template<typename T>
void LatteIndices_unpackLineLoopAndConvert(const void* indexDataInput, void* indexDataOutput, uint32 count, uint32& indexMax)
{
	if (count <= 0)
		return;
	const betype<T>* src = (betype<T>*)indexDataInput;
	T firstIndex = *src;
	T* dst = (T*)indexDataOutput;
	for (sint32 i = 0; i < (sint32)count; i++)
	{
		T idx = *src;
		indexMax = std::max(indexMax, (uint32)idx);
		*dst = idx;
		src++;
		dst++;
	}
	*dst = firstIndex;
}

template<typename T>
void LatteIndices_generateAutoQuadStripIndices(void* indexDataOutput, uint32 count, uint32& indexMax)
{
	if (count <= 3)
		return;
	sint32 numQuads = (count - 2) / 2;
	T* dst = (T*)indexDataOutput;
	for (sint32 i = 0; i < numQuads; i++)
	{
		T idx0 = i * 2 + 0;
		T idx1 = i * 2 + 1;
		T idx2 = i * 2 + 2;
		T idx3 = i * 2 + 3;
		dst[0] = idx0;
		dst[1] = idx1;
		dst[2] = idx2;
		dst[3] = idx2;
		dst[4] = idx1;
		dst[5] = idx3;
		dst += 6;
	}
	indexMax = std::max(count, 1u) - 1;
}


template<typename T>
void LatteIndices_generateAutoLineLoopIndices(void* indexDataOutput, uint32 count, uint32& indexMax)
{
	if (count == 0)
		return;
	T* dst = (T*)indexDataOutput;
	for (sint32 i = 0; i < (sint32)count; i++)
	{
		*dst = (T)i;
		dst++;
	}
	*dst = 0;
	dst++;
	indexMax = std::max(count, 1u) - 1;
}

template<typename T>
void LatteIndices_unpackTriangleFanAndConvert(const void* indexDataInput, void* indexDataOutput, uint32 count, uint32& indexMax)
{
	const betype<T>* src = (betype<T>*)indexDataInput;
	T* dst = (T*)indexDataOutput;
	// TODO: check this
	for (sint32 i = 0; i < count; i++)
	{
	    uint32 i0;
		if (i % 2 == 0)
		    i0 = i / 2;
        else
            i0 = count - 1 - i / 2;
        T idx = src[i0];
		indexMax = std::max(indexMax, (uint32)idx);
		dst[i] = idx;
	}
}

template<typename T>
void LatteIndices_generateAutoTriangleFanIndices(const void* indexDataInput, void* indexDataOutput, uint32 count, uint32& indexMax)
{
	const betype<T>* src = (betype<T>*)indexDataInput;
	T* dst = (T*)indexDataOutput;
	for (sint32 i = 0; i < count; i++)
	{
		T idx = i;
		if (idx % 2 == 0)
            idx = idx / 2;
        else
            idx = count - 1 - idx / 2;
		dst[i] = idx;
	}
	indexMax = std::max(count, 1u) - 1;
}

#if defined(ARCH_X86_64)
ATTRIBUTE_AVX2
void LatteIndices_fastConvertU16_AVX2(const void* indexDataInput, void* indexDataOutput, uint32 count, uint32& indexMax)
{
	// using AVX + AVX2 we can process 16 indices at a time
	const uint16* indicesU16BE = (const uint16*)indexDataInput;
	uint16* indexOutput = (uint16*)indexDataOutput;
	sint32 count16 = count >> 4;
	sint32 countRemaining = count & 15;
	if (count16)
	{
		__m256i mMin = _mm256_set_epi16((sint16)0xFFFF, (sint16)0xFFFF, (sint16)0xFFFF, (sint16)0xFFFF, (sint16)0xFFFF, (sint16)0xFFFF, (sint16)0xFFFF, (sint16)0xFFFF,
						           		(sint16)0xFFFF, (sint16)0xFFFF, (sint16)0xFFFF, (sint16)0xFFFF, (sint16)0xFFFF, (sint16)0xFFFF, (sint16)0xFFFF, (sint16)0xFFFF);
		__m256i mMax = _mm256_set_epi16(0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000);
		__m256i mShuffle16Swap = _mm256_set_epi8(30, 31, 28, 29, 26, 27, 24, 25, 22, 23, 20, 21, 18, 19, 16, 17, 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1);

		do
		{
			__m256i mIndexData = _mm256_loadu_si256((const __m256i*)indicesU16BE);
			indicesU16BE += 16;
			_mm_prefetch((const char*)indicesU16BE, _MM_HINT_T0);
			// endian swap
			mIndexData = _mm256_shuffle_epi8(mIndexData, mShuffle16Swap);
			_mm256_store_si256((__m256i*)indexOutput, mIndexData);
			mMax = _mm256_max_epu16(mIndexData, mMax);
			indexOutput += 16;
		} while (--count16);

		// fold 32 to 16 byte
		mMax = _mm256_max_epu16(mMax, _mm256_permute2x128_si256(mMax, mMax, 1));
		// fold 16 to 8 byte
		mMax = _mm256_max_epu16(mMax, _mm256_shuffle_epi32(mMax, (2 << 0) | (3 << 2) | (2 << 4) | (3 << 6)));

		uint16* mMaxU16 = (uint16*)&mMax;

		indexMax = std::max(indexMax, (uint32)mMaxU16[0]);
		indexMax = std::max(indexMax, (uint32)mMaxU16[1]);
		indexMax = std::max(indexMax, (uint32)mMaxU16[2]);
		indexMax = std::max(indexMax, (uint32)mMaxU16[3]);
	}
	// process remaining indices
	uint32 _maxIndex = 0;
	for (sint32 i = countRemaining; (--i) >= 0;)
	{
		uint16 idx = _swapEndianU16(*indicesU16BE);
		*indexOutput = idx;
		indexOutput++;
		indicesU16BE++;
		_maxIndex = std::max(_maxIndex, (uint32)idx);
	}
	// update max
	indexMax = std::max(indexMax, _maxIndex);
}

ATTRIBUTE_SSE41
void LatteIndices_fastConvertU16_SSE41(const void* indexDataInput, void* indexDataOutput, uint32 count, uint32& indexMax)
{
	// SSSE3 & SSE4.1 optimized decoding
	const uint16* indicesU16BE = (const uint16*)indexDataInput;
	uint16* indexOutput = (uint16*)indexDataOutput;
	sint32 count8 = count >> 3;
	sint32 countRemaining = count & 7;
	if (count8)
	{
		__m128i mMax = _mm_set_epi16(0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000);
		__m128i mTemp;
		__m128i* mRawIndices = (__m128i*)indicesU16BE;
		indicesU16BE += count8 * 8;
		__m128i* mOutputIndices = (__m128i*)indexOutput;
		indexOutput += count8 * 8;
		__m128i shufmask = _mm_set_epi8(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1);
		while (count8--)
		{
			mTemp = _mm_loadu_si128(mRawIndices);
			mRawIndices++;
			mTemp = _mm_shuffle_epi8(mTemp, shufmask);
			mMax = _mm_max_epu16(mMax, mTemp);
			_mm_store_si128(mOutputIndices, mTemp);
			mOutputIndices++;
		}

		uint16* mMaxU16 = (uint16*)&mMax;

		indexMax = std::max(indexMax, (uint32)mMaxU16[0]);
		indexMax = std::max(indexMax, (uint32)mMaxU16[1]);
		indexMax = std::max(indexMax, (uint32)mMaxU16[2]);
		indexMax = std::max(indexMax, (uint32)mMaxU16[3]);
		indexMax = std::max(indexMax, (uint32)mMaxU16[4]);
		indexMax = std::max(indexMax, (uint32)mMaxU16[5]);
		indexMax = std::max(indexMax, (uint32)mMaxU16[6]);
		indexMax = std::max(indexMax, (uint32)mMaxU16[7]);
	}
	uint32 _maxIndex = 0;
	for (sint32 i = countRemaining; (--i) >= 0;)
	{
		uint16 idx = _swapEndianU16(*indicesU16BE);
		*indexOutput = idx;
		indexOutput++;
		indicesU16BE++;
		_maxIndex = std::max(_maxIndex, (uint32)idx);
	}
	indexMax = std::max(indexMax, _maxIndex);
}

ATTRIBUTE_AVX2
void LatteIndices_fastConvertU32_AVX2(const void* indexDataInput, void* indexDataOutput, uint32 count, uint32& indexMax)
{
	// using AVX + AVX2 we can process 8 indices at a time
	const uint32* indicesU32BE = (const uint32*)indexDataInput;
	uint32* indexOutput = (uint32*)indexDataOutput;
	sint32 count8 = count >> 3;
	sint32 countRemaining = count & 7;
	if (count8)
	{
		__m256i mMax = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, 0);
		__m256i mShuffle32Swap = _mm256_set_epi8(28,29,30,31,
			24,25,26,27,
			20,21,22,23,
			16,17,18,19,
			12,13,14,15,
			8,9,10,11,
			4,5,6,7,
			0,1,2,3);
		// unaligned
		do
		{
			__m256i mIndexData = _mm256_loadu_si256((const __m256i*)indicesU32BE);
			indicesU32BE += 8;
			_mm_prefetch((const char*)indicesU32BE, _MM_HINT_T0);
			// endian swap
			mIndexData = _mm256_shuffle_epi8(mIndexData, mShuffle32Swap);
			_mm256_store_si256((__m256i*)indexOutput, mIndexData);
			mMax = _mm256_max_epu32(mIndexData, mMax);
			indexOutput += 8;
		} while (--count8);

		// fold 32 to 16 byte
		mMax = _mm256_max_epu32(mMax, _mm256_permute2x128_si256(mMax, mMax, 1));
		// fold 16 to 8 byte
		mMax = _mm256_max_epu32(mMax, _mm256_shuffle_epi32(mMax, (2 << 0) | (3 << 2) | (2 << 4) | (3 << 6)));

		uint32* mMaxU32 = (uint32*)&mMax;
		indexMax = std::max(indexMax, (uint32)mMaxU32[0]);
		indexMax = std::max(indexMax, (uint32)mMaxU32[1]);
	}
	// process remaining indices
	uint32 _maxIndex = 0;
	for (sint32 i = countRemaining; (--i) >= 0;)
	{
		uint32 idx = _swapEndianU32(*indicesU32BE);
		*indexOutput = idx;
		indexOutput++;
		indicesU32BE++;
		_maxIndex = std::max(_maxIndex, (uint32)idx);
	}
	// update min/max
	indexMax = std::max(indexMax, _maxIndex);
}
#elif defined(__aarch64__)

void LatteIndices_fastConvertU16_NEON(const void* indexDataInput, void* indexDataOutput, uint32 count, uint32& indexMax)
{
	const uint16* indicesU16BE = (const uint16*)indexDataInput;
	uint16* indexOutput = (uint16*)indexDataOutput;
	sint32 count8 = count >> 3;
	sint32 countRemaining = count & 7;

	if (count8)
	{
		uint16x8_t mMax = vdupq_n_u16(0x0000);
		uint16x8_t mTemp;
		uint16x8_t* mRawIndices = (uint16x8_t*) indicesU16BE;
		indicesU16BE += count8 * 8;
		uint16x8_t* mOutputIndices = (uint16x8_t*) indexOutput;
		indexOutput += count8 * 8;

		while (count8--)
		{
			mTemp = vld1q_u16((uint16*)mRawIndices);
			mRawIndices++;
			mTemp = vrev16q_u8(mTemp);
			mMax = vmaxq_u16(mMax, mTemp);
			vst1q_u16((uint16*)mOutputIndices, mTemp);
			mOutputIndices++;
		}

		uint16* mMaxU16 = (uint16*)&mMax;

		for (int i = 0; i < 8; ++i) {
			indexMax = std::max(indexMax, (uint32)mMaxU16[i]);
		}
	}
	// process remaining indices
	uint32 _maxIndex = 0;
	for (sint32 i = countRemaining; (--i) >= 0;)
	{
		uint16 idx = _swapEndianU16(*indicesU16BE);
		*indexOutput = idx;
		indexOutput++;
		indicesU16BE++;
		_maxIndex = std::max(_maxIndex, (uint32)idx);
	}
	// update min/max
	indexMax = std::max(indexMax, _maxIndex);
}

void LatteIndices_fastConvertU32_NEON(const void* indexDataInput, void* indexDataOutput, uint32 count, uint32& indexMax)
{
	const uint32* indicesU32BE = (const uint32*)indexDataInput;
	uint32* indexOutput = (uint32*)indexDataOutput;
	sint32 count8 = count >> 2;
	sint32 countRemaining = count & 3;

	if (count8)
	{
		uint32x4_t mMax = vdupq_n_u32(0x00000000);
		uint32x4_t mTemp;
		uint32x4_t* mRawIndices = (uint32x4_t*) indicesU32BE;
		indicesU32BE += count8 * 4;
		uint32x4_t* mOutputIndices = (uint32x4_t*) indexOutput;
		indexOutput += count8 * 4;

		while (count8--)
		{
			mTemp = vld1q_u32((uint32*)mRawIndices);
			mRawIndices++;
			mTemp = vrev32q_u8(mTemp);
			mMax = vmaxq_u32(mMax, mTemp);
			vst1q_u32((uint32*)mOutputIndices, mTemp);
			mOutputIndices++;
		}

		uint32* mMaxU32 = (uint32*)&mMax;

		for (int i = 0; i < 4; ++i) {
			indexMax = std::max(indexMax, mMaxU32[i]);
		}
	}
	// process remaining indices
	uint32 _maxIndex = 0;
	for (sint32 i = countRemaining; (--i) >= 0;)
	{
		uint32 idx = _swapEndianU32(*indicesU32BE);
		*indexOutput = idx;
		indexOutput++;
		indicesU32BE++;
		_maxIndex = std::max(_maxIndex, idx);
	}
	// update min/max
	indexMax = std::max(indexMax, _maxIndex);
}

#endif

template<typename T>
void _LatteIndices_alternativeCalculateIndexMax(const void* indexData, uint32 count, uint32 primitiveRestartIndex, uint32& indexMax)
{
	cemu_assert_debug(count != 0);
	const betype<T>* idxPtrT = (betype<T>*)indexData;
	T _indexMax = *idxPtrT;
	cemu_assert_debug(primitiveRestartIndex <= std::numeric_limits<T>::max());
	T restartIndexT = (T)primitiveRestartIndex;
	while (count)
	{
		T idx = *idxPtrT;
		if (idx != restartIndexT)
		{
			_indexMax = std::max(_indexMax, idx);
		}
		idxPtrT++;
		count--;
	}
	indexMax = _indexMax;
}

// calculate min and max index while taking primitive restart into account
// fallback implementation in case the fast path gives us invalid results
void LatteIndices_alternativeCalculateIndexMax(const void* indexData, LatteIndexType indexType, uint32 count, uint32& indexMax)
{
	if (count == 0)
	{
		indexMax = 0;
		return;
	}
	uint32 primitiveRestartIndex = LatteGPUState.contextNew.VGT_MULTI_PRIM_IB_RESET_INDX.get_RESTART_INDEX();

	if (indexType == LatteIndexType::U16_BE)
	{
		_LatteIndices_alternativeCalculateIndexMax<uint16>(indexData, count, primitiveRestartIndex, indexMax);
	}
	else if (indexType == LatteIndexType::U32_BE)
	{
		_LatteIndices_alternativeCalculateIndexMax<uint32>(indexData, count, primitiveRestartIndex, indexMax);
	}
	else
	{
		cemu_assert_debug(false);
	}
}

void LatteIndices_decode(const void* indexData, LatteIndexType indexType, uint32 count, LattePrimitiveMode primitiveMode, uint32& indexMax, Renderer::INDEX_TYPE& renderIndexType, uint32& outputCount, Renderer::IndexAllocation& indexAllocation)
{
	// what this should do:
	// [x] use fast SIMD-based index decoding
	// [x] unpack QUAD indices to triangle indices
	// [x] calculate min and max index, be careful about primitive restart index
	// [x] decode data directly into coherent memory buffer?
	// [ ] better cache implementation, allow to cache across frames

	const bool persistent = PersistentOn();
	if (persistent)
	{
		if (s_persistGlobalFlush.load(std::memory_order_relaxed) && s_persistGlobalFlush.exchange(false, std::memory_order_acq_rel))
		{
			PersistentClear();
			s_statGlobalClears++;
		}
		// flushes the game made before submitting this draw invalidate cached indices first (upstream processed the
		// queue only after the index decode, in LatteBufferCache_Sync)
		LatteBufferCache_processDCFlushQueue();
		auto hit = s_persist.find(PersistentKey{indexData, count, primitiveMode, indexType});
		if (hit != s_persist.end())
		{
			indexMax = hit->second.indexMax;
			renderIndexType = hit->second.renderIndexType;
			outputCount = hit->second.outputCount;
			indexAllocation = hit->second.indexAllocation;
			hit->second.lastUsed = LatteIndices_GetNextUsageIndex();
			if (telem::on())  // statistics: telemetry only
			{
				s_statHits++;
				PersistentStats();
			}
			return;
		}
		if (telem::on())
		{
			s_statMisses++;
			PersistentStats();
		}
	}
	// reuse from cache if data didn't change
	auto cacheEntry = persistent ? LatteIndexCache.entry.end() : std::find_if(LatteIndexCache.entry.begin(), LatteIndexCache.entry.end(), [indexData, count, primitiveMode, indexType](const auto& entry)
	{
		return entry.lastPtr == indexData && entry.lastCount == count && entry.lastPrimitiveMode == primitiveMode && entry.lastIndexType == indexType;
	});
	if (cacheEntry != LatteIndexCache.entry.end())
	{
		indexMax = cacheEntry->indexMax;
		renderIndexType = cacheEntry->renderIndexType;
		outputCount = cacheEntry->outputCount;
		indexAllocation = cacheEntry->indexAllocation;
		cacheEntry->lastUsed = LatteIndices_GetNextUsageIndex();
		return;
	}

	outputCount = 0;
	if (indexType == LatteIndexType::AUTO)
		renderIndexType = Renderer::INDEX_TYPE::NONE;
	else if (indexType == LatteIndexType::U16_BE || indexType == LatteIndexType::U16_LE)
		renderIndexType = Renderer::INDEX_TYPE::U16;
	else if (indexType == LatteIndexType::U32_BE)
		renderIndexType = Renderer::INDEX_TYPE::U32;
	else
		cemu_assert_debug(false);

	uint32 primitiveRestartIndex = LatteGPUState.contextNew.VGT_MULTI_PRIM_IB_RESET_INDX.get_RESTART_INDEX();

	// calculate index output size
	uint32 indexOutputSize = LatteIndices_calculateIndexOutputSize(primitiveMode, indexType, count);
	if (indexOutputSize == 0)
	{
		outputCount = count;
		indexMax = std::max(count, 1u)-1;
		renderIndexType = Renderer::INDEX_TYPE::NONE;
		indexAllocation = {};
		return; // no indices
	}
	// query index buffer from renderer
	indexAllocation = g_renderer->indexData_reserveIndexMemory(indexOutputSize);
	void* indexOutputPtr = indexAllocation.mem;

	// decode indices
	indexMax = std::numeric_limits<uint32>::min();
	if (primitiveMode == LattePrimitiveMode::QUADS)
	{
		// unpack quads into triangles
		if (indexType == LatteIndexType::AUTO)
		{
			if (count <= 0xFFFF)
			{
				LatteIndices_generateAutoQuadIndices<uint16>(indexData, indexOutputPtr, count, indexMax);
				renderIndexType = Renderer::INDEX_TYPE::U16;
			}
			else
			{
				LatteIndices_generateAutoQuadIndices<uint32>(indexData, indexOutputPtr, count, indexMax);
				renderIndexType = Renderer::INDEX_TYPE::U32;
			}
		}
		else if (indexType == LatteIndexType::U16_BE)
			LatteIndices_unpackQuadsAndConvert<uint16>(indexData, indexOutputPtr, count, indexMax);
		else if (indexType == LatteIndexType::U32_BE)
			LatteIndices_unpackQuadsAndConvert<uint32>(indexData, indexOutputPtr, count, indexMax);
		else
			cemu_assert_debug(false);
		outputCount = count / 4 * 6;
	}
	else if (primitiveMode == LattePrimitiveMode::QUAD_STRIP)
	{
		// unpack quad strip into triangles
		if (indexType == LatteIndexType::AUTO)
		{
			if (count <= 0xFFFF)
			{
				LatteIndices_generateAutoQuadStripIndices<uint16>(indexOutputPtr, count, indexMax);
				renderIndexType = Renderer::INDEX_TYPE::U16;
			}
			else
			{
				LatteIndices_generateAutoQuadStripIndices<uint32>(indexOutputPtr, count, indexMax);
				renderIndexType = Renderer::INDEX_TYPE::U32;
			}
		}
		else if (indexType == LatteIndexType::U16_BE)
			LatteIndices_unpackQuadStripAndConvert<uint16>(indexData, indexOutputPtr, count, indexMax);
		else if (indexType == LatteIndexType::U32_BE)
			LatteIndices_unpackQuadStripAndConvert<uint32>(indexData, indexOutputPtr, count, indexMax);
		else
			cemu_assert_debug(false);
		if (count >= 2)
			outputCount = (count - 2) / 2 * 6;
		else
			outputCount = 0;
	}
	else if (primitiveMode == LattePrimitiveMode::LINE_LOOP)
	{
		// unpack line loop into line strip with extra reconnecting vertex
		if (indexType == LatteIndexType::AUTO)
		{
			if (count <= 0xFFFF)
			{
				LatteIndices_generateAutoLineLoopIndices<uint16>(indexOutputPtr, count, indexMax);
				renderIndexType = Renderer::INDEX_TYPE::U16;
			}
			else
			{
				LatteIndices_generateAutoLineLoopIndices<uint32>(indexOutputPtr, count, indexMax);
				renderIndexType = Renderer::INDEX_TYPE::U32;
			}
		}
		else if (indexType == LatteIndexType::U16_BE)
			LatteIndices_unpackLineLoopAndConvert<uint16>(indexData, indexOutputPtr, count, indexMax);
		else if (indexType == LatteIndexType::U32_BE)
			LatteIndices_unpackLineLoopAndConvert<uint32>(indexData, indexOutputPtr, count, indexMax);
		else
			cemu_assert_debug(false);
		outputCount = count + 1;
	}
	else if (primitiveMode == LattePrimitiveMode::TRIANGLE_FAN && g_renderer->GetType() == RendererAPI::Metal)
	{
        if (indexType == LatteIndexType::AUTO)
    	{
    		if (count <= 0xFFFF)
    		{
    			LatteIndices_generateAutoTriangleFanIndices<uint16>(indexData, indexOutputPtr, count, indexMax);
    			renderIndexType = Renderer::INDEX_TYPE::U16;
    		}
    		else
    		{
    			LatteIndices_generateAutoTriangleFanIndices<uint32>(indexData, indexOutputPtr, count, indexMax);
    			renderIndexType = Renderer::INDEX_TYPE::U32;
    		}
    	}
    	else if (indexType == LatteIndexType::U16_BE)
    		LatteIndices_unpackTriangleFanAndConvert<uint16>(indexData, indexOutputPtr, count, indexMax);
    	else if (indexType == LatteIndexType::U32_BE)
    		LatteIndices_unpackTriangleFanAndConvert<uint32>(indexData, indexOutputPtr, count, indexMax);
    	else
    		cemu_assert_debug(false);
    	outputCount = count;
	}
	else
	{
		if (indexType == LatteIndexType::U16_BE)
		{
#if defined(ARCH_X86_64)
			if (g_CPUFeatures.x86.avx2)
				LatteIndices_fastConvertU16_AVX2(indexData, indexOutputPtr, count, indexMax);
			else if (g_CPUFeatures.x86.sse4_1 && g_CPUFeatures.x86.ssse3)
				LatteIndices_fastConvertU16_SSE41(indexData, indexOutputPtr, count, indexMax);
			else
				LatteIndices_convertBE<uint16>(indexData, indexOutputPtr, count, indexMax);
#elif defined(__aarch64__)
			LatteIndices_fastConvertU16_NEON(indexData, indexOutputPtr, count, indexMax);
#else
			LatteIndices_convertBE<uint16>(indexData, indexOutputPtr, count, indexMax);
#endif
		}
		else if (indexType == LatteIndexType::U32_BE)
		{
#if defined(ARCH_X86_64)
			if (g_CPUFeatures.x86.avx2)
				LatteIndices_fastConvertU32_AVX2(indexData, indexOutputPtr, count, indexMax);
			else
				LatteIndices_convertBE<uint32>(indexData, indexOutputPtr, count, indexMax);
#elif defined(__aarch64__)
			LatteIndices_fastConvertU32_NEON(indexData, indexOutputPtr, count, indexMax);
#else
			LatteIndices_convertBE<uint32>(indexData, indexOutputPtr, count, indexMax);
#endif
		}
		else if (indexType == LatteIndexType::U16_LE)
		{
			LatteIndices_convertLE<uint16>(indexData, indexOutputPtr, count, indexMax);
		}
		else if (indexType == LatteIndexType::U32_LE)
		{
			LatteIndices_convertLE<uint32>(indexData, indexOutputPtr, count, indexMax);
		}
		else
			cemu_assert_debug(false);
		outputCount = count;
	}
	// the above algorithms use a fast approach to get indexMax which does not filter out indices matching primitiveRestartIndex
	// here we use a fallback in case the determined index equals the primitive restart index
	if (primitiveRestartIndex == indexMax)
	{
		// recalculate index range but filter out primitive restart index
		LatteIndices_alternativeCalculateIndexMax(indexData, indexType, count, indexMax);
	}
	g_renderer->indexData_uploadIndexMemory(indexAllocation);
	performanceMonitor.cycle[performanceMonitor.cycleIndex].indexDataUploaded += indexOutputSize;
	if (persistent)
	{
		const bool hasSource = indexType != LatteIndexType::AUTO;  // AUTO: generated indices, no guest data to watch
		const uint32 inputBytes = !hasSource ? 0 : count * (indexType == LatteIndexType::U32_BE || indexType == LatteIndexType::U32_LE ? 4 : 2);
		const PersistentKey key{indexData, count, primitiveMode, indexType};
		s_persist[key] = PersistentEntry{LatteIndices_GetNextUsageIndex(), inputBytes, indexOutputSize, indexMax,
		                                 renderIndexType, outputCount, indexAllocation};
		if (hasSource)
		{
			s_persistByAddr.emplace((uintptr_t)indexData, key);
			s_persistMaxInput = std::max(s_persistMaxInput, inputBytes);
			s_persistLo = std::min(s_persistLo, (uintptr_t)indexData);
			s_persistHi = std::max(s_persistHi, (uintptr_t)indexData + inputBytes);
		}
		s_persistBytes += indexOutputSize;
		PersistentTrim(key);
		return;
	}
	// get least recently used cache entry
	auto lruEntry = std::min_element(LatteIndexCache.entry.begin(), LatteIndexCache.entry.end(), [](const auto& a, const auto& b)
	{
		return a.lastUsed < b.lastUsed;
	});
	// invalidate previous allocation
	if(lruEntry->lastPtr != nullptr)
		g_renderer->indexData_releaseIndexMemory(lruEntry->indexAllocation);
	// update cache
	lruEntry->lastPtr = indexData;
	lruEntry->lastCount = count;
	lruEntry->lastPrimitiveMode = primitiveMode;
	lruEntry->lastIndexType = indexType;
	lruEntry->indexMax = indexMax;
	lruEntry->renderIndexType = renderIndexType;
	lruEntry->outputCount = outputCount;
	lruEntry->indexAllocation = indexAllocation;
	lruEntry->lastUsed = LatteIndices_GetNextUsageIndex();
}
