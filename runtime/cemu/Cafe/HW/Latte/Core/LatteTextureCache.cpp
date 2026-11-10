#include "Cafe/HW/Latte/Core/Latte.h"
#include "Cafe/HW/Latte/Core/LatteDraw.h"
#include "Cafe/HW/Latte/Core/LatteTexture.h"
#include "Cafe/HW/Latte/Renderer/Renderer.h"
#include "Common/cpu_features.h"

#include <bit>
#include <algorithm>
#include <cstdlib>
#if defined(__x86_64__)
#include <xmmintrin.h>
#endif

std::unordered_set<LatteTexture*> g_allTextures;

void LatteTC_Init()
{
	cemu_assert_debug(g_allTextures.empty());
}

void LatteTC_RegisterTexture(LatteTexture* tex)
{
	g_allTextures.emplace(tex);
}

void LatteTC_UnregisterTexture(LatteTexture* tex)
{
	g_allTextures.erase(tex);
}

// sample few uint64s uniformly over memory range
uint32 _quickStochasticHash(void* texData, uint32 memRange, bool is16ByteFormat)
{
	uint64* texDataU64 = (uint64*)texData;

	uint64 hashVal = 0;
	memRange /= sizeof(uint64);

	uint32 memStep = memRange / 37; // use prime here to avoid memStep aligning nicely with pitch of texture, leading to sampling only along the border of a texture

	if (!is16ByteFormat)
	{
		for (sint32 i = 0; i < 37; i++)
		{
			hashVal += *texDataU64;
			hashVal = (hashVal << 3) | (hashVal >> 61);
			texDataU64 += memStep;
		}
	}
	else
	{
		for (sint32 i = 0; i < 37; i++)
		{
			hashVal += *texDataU64 ^ *(texDataU64 + 1); 
			hashVal = (hashVal << 3) | (hashVal >> 61);
			texDataU64 += memStep; 
		}

	}
	return (uint32)hashVal ^ (uint32)(hashVal >> 32);
}

// BotW-Native: the add-rotate hash over every step-th word, in four independent chains. The single chain was latency
// bound (each sample waited for the previous rotate). The words sampled, and so which changes are seen, are the same.
// Each texture in use is hashed once a frame and the small/medium ones touch every cache line, so these reads came from
// DRAM (19% of the GPU thread's DRAM fills, b107 Deck event sampling) and evicted the main thread's data from the shared
// 4 MB L3 (main's L2 misses came from DRAM more often than from L3). Non-temporal prefetches ~2 KB ahead fetch the same
// words without installing them in L2/L3. Same words, same hash. BOTW_TEXHASH_NTA=0 off.
#if defined(__x86_64__)
static const bool s_texHashNta = !(getenv("BOTW_TEXHASH_NTA") && getenv("BOTW_TEXHASH_NTA")[0] == '0');
#endif

template<typename T, int R>
static T hashStrided(const T* p, uint32 count, uint32 step)
{
	T h0 = 0, h1 = 0, h2 = 0, h3 = 0;
	uint32 i = 0;
#if defined(__x86_64__)
	if (s_texHashNta)
	{
		const uint32 ahead = 4 * step * std::max<uint32>(2, 2048 / (4 * step * sizeof(T)));  // elements, whole iterations
		for (; i + 4 <= count; i += 4, p += 4 * step)
		{
			_mm_prefetch((const char*)(p + ahead), _MM_HINT_NTA);
			_mm_prefetch((const char*)(p + ahead + step), _MM_HINT_NTA);
			_mm_prefetch((const char*)(p + ahead + 2 * step), _MM_HINT_NTA);
			_mm_prefetch((const char*)(p + ahead + 3 * step), _MM_HINT_NTA);
			h0 = std::rotl<T>(h0 + p[0], R);
			h1 = std::rotl<T>(h1 + p[step], R);
			h2 = std::rotl<T>(h2 + p[2 * step], R);
			h3 = std::rotl<T>(h3 + p[3 * step], R);
		}
	}
#endif
	for (; i + 4 <= count; i += 4, p += 4 * step)
	{
		h0 = std::rotl<T>(h0 + p[0], R);
		h1 = std::rotl<T>(h1 + p[step], R);
		h2 = std::rotl<T>(h2 + p[2 * step], R);
		h3 = std::rotl<T>(h3 + p[3 * step], R);
	}
	for (; i < count; i++, p += step)
		h0 = std::rotl<T>(h0 + p[0], R);
	constexpr int bits = sizeof(T) * 8;
	return h0 ^ std::rotl<T>(h1, bits / 4) ^ std::rotl<T>(h2, bits / 2) ^ std::rotl<T>(h3, bits / 4 * 3);
}

#if BOOST_OS_WINDOWS
// The intrinsics are always_inline and require AVX2. A target attribute compiles only this
// function for AVX2; the rest of the file stays on the baseline so a CPU without AVX2 can run.
#if defined(__GNUC__)
__attribute__((target("avx2")))
#endif
static uint32 hash_huge_tex_avx2(const uint32* texDataU32, uint32 memRange)
{
	__m256i h256 = _mm256_setzero_si256();
	const __m256i* readPtr = reinterpret_cast<const __m256i*>(texDataU32);
	memRange /= 288;
	while (memRange--)
	{
		__m256i temp = _mm256_load_si256(readPtr);
		readPtr += (288 / 32);
		h256 = _mm256_xor_si256(h256, temp);
	}
	alignas(32) uint32 lanes[8];
	_mm256_store_si256(reinterpret_cast<__m256i*>(lanes), h256);
	return lanes[0] + lanes[1] + lanes[2] + lanes[3] + lanes[4] + lanes[5] + lanes[6] + lanes[7];
}
#endif

uint32 LatteTexture_CalculateTextureDataHash(LatteTexture* hostTexture)
{
	if( hostTexture->texDataPtrHigh == hostTexture->texDataPtrLow )
	{
		return 0;
	}
	if (hostTexture->format == Latte::E_GX2SURFFMT::R11_G11_B10_FLOAT)
	{
		// this is an exotic format that usually isn't generated or updated CPU-side
		// therefore as an optimization we can risk to only check a minimal amount of bytes at the beginning of the texture data
		// updates which change the entire texture should still be detected this way
		// this also helps with a bug in BotW which seems to fill the empty areas of the textures with other data which causes unnecessary invalidations and texture reloads

		// Wonderful 101 generates this format in a 8x8x8 3D texture using tiling aperture
		if (hostTexture->tileMode == Latte::E_HWTILEMODE::TM_1D_TILED_THICK && hostTexture->depth == 8 && hostTexture->width == 8 && hostTexture->height == 8)
		{
			// special case for Wonderful 101
			uint32* texDataU32 = (uint32*)memory_getPointerFromPhysicalOffset(hostTexture->texDataPtrLow);
			return texDataU32[0] ^ texDataU32[0x100/4] ^ texDataU32[0x200/4] ^ texDataU32[0x300/4]; // check the first thick slice (each slice has 0x400 bytes, with 0x100 bytes between layers)
		}
		uint32* texDataU32 = (uint32*)memory_getPointerFromPhysicalOffset(hostTexture->texDataPtrLow);
		return texDataU32[0] ^ texDataU32[1] ^ texDataU32[2] ^ texDataU32[3];
	}

	uint32 memRange = hostTexture->texDataPtrHigh - hostTexture->texDataPtrLow;
	uint32* texDataU32 = (uint32*)memory_getPointerFromPhysicalOffset(hostTexture->texDataPtrLow);
	uint32 hashVal = 0;
	uint32 pixelCount = hostTexture->width*hostTexture->height;

	bool isCompressedFormat = hostTexture->IsCompressedFormat();
	if (isCompressedFormat || hostTexture->useLightHash)
	{
		// check only 32 samples of the texture
		if (memRange < 256)
		{
			memRange /= sizeof(uint32);
			while (memRange--)
			{
				hashVal += *texDataU32;
				hashVal = (hashVal << 3) | (hashVal >> 29);
				texDataU32++;
			}
		}
		else
		{
			bool is16ByteFormat = (hostTexture->format == Latte::E_GX2SURFFMT::BC2_UNORM || 
			                       hostTexture->format == Latte::E_GX2SURFFMT::BC2_SRGB || 
			                       hostTexture->format == Latte::E_GX2SURFFMT::BC3_UNORM || 
			                       hostTexture->format == Latte::E_GX2SURFFMT::BC3_SRGB);

			hashVal = _quickStochasticHash(texDataU32, memRange, is16ByteFormat);
		}
		return hashVal;
	}


	if( pixelCount <= (700*700) )
	{
		// small texture size
		bool isCompressedFormat = hostTexture->IsCompressedFormat();
		if( isCompressedFormat == false || memRange < 0x200 )
			hashVal = hashStrided<uint32, 3>(texDataU32, memRange / (4 * sizeof(uint32)), 4);
		else
			hashVal = hashStrided<uint32, 3>(texDataU32, memRange / (32 * sizeof(uint32)), 32);
	}
	else if( pixelCount <= (1200*1200) )
	{
		// medium texture size
		bool isCompressedFormat = hostTexture->IsCompressedFormat();
		if( isCompressedFormat == false )
			hashVal = hashStrided<uint32, 3>(texDataU32, memRange / (12 * sizeof(uint32)), 12);
		else
			hashVal = hashStrided<uint32, 3>(texDataU32, memRange / (96 * sizeof(uint32)), 96);
	}
	else
	{
		// huge texture size
		bool isCompressedFormat = hostTexture->IsCompressedFormat();
		if( isCompressedFormat == false )
		{
#if BOOST_OS_WINDOWS
			if (g_CPUFeatures.x86.avx2)
				hashVal = hash_huge_tex_avx2(texDataU32, memRange);
#else
			if( false ) {}
#endif
			else
			{
				const uint64 h64 = hashStrided<uint64, 3>((const uint64*)texDataU32, memRange / (32 * sizeof(uint64)), 32);
				hashVal = (h64 & 0xFFFFFFFF) + (h64 >> 32);
			}
		}
		else
			hashVal = hashStrided<uint32, 3>(texDataU32, memRange / (512 * sizeof(uint32)), 512);
	}

	return hashVal;
}

uint64 _botwLargeTexHax = 0;

// BotW-Native: the reset hash of a reload whose decode runs on the texture decode helpers is computed there too (it
// read the whole texture on the GPU thread, ~15% of the worst load-in frames, b175 Deck profile). Like the decode, it
// reads the guest memory before the next GPU progress point the game can observe (those join all helper jobs).
// Whatever reads texDataHash2 or changes what the hash covers resolves the pending hash first: it takes the helper's
// value or, when no helper has started it yet, cancels the job and hashes here as before.
struct LatteTextureHashJob
{
	std::atomic<uint32> state{1};  // 1 queued, 2 running, 3 done, 0 cancelled
	std::atomic<uint32> refs{2};  // held by the texture and by the queued job
	uint32 value{};
	LatteTexture* tex{};
};

void LatteTextureLoader_QueueHashJob(LatteTextureHashJob* job);  // LatteTextureLoader.cpp

static void HashJobRelease(LatteTextureHashJob* job)
{
	if (job->refs.fetch_sub(1, std::memory_order_acq_rel) == 1)
		delete job;
}

// decode helper (or the GPU thread joining)
void LatteTC_RunHashJob(LatteTextureHashJob* job)
{
	uint32 expected = 1;
	if (job->state.compare_exchange_strong(expected, 2, std::memory_order_acquire))
	{
		job->value = LatteTexture_CalculateTextureDataHash(job->tex);
		job->state.store(3, std::memory_order_release);
		job->state.notify_all();
	}
	HashJobRelease(job);
}

// apply = false: the texture is being deleted, only make sure no helper still reads it
void LatteTC_ResolvePendingHash(LatteTexture* tex, bool apply)
{
	LatteTextureHashJob* job = tex->pendingHash;
	if (!job)
		return;
	tex->pendingHash = nullptr;
	uint32 expected = 1;
	if (job->state.compare_exchange_strong(expected, 0, std::memory_order_acquire))
	{
		if (apply)
			tex->texDataHash2 = LatteTexture_CalculateTextureDataHash(tex);
	}
	else
	{
		for (uint32 s; (s = job->state.load(std::memory_order_acquire)) != 3;)
			job->state.wait(s, std::memory_order_acquire);
		if (apply)
			tex->texDataHash2 = job->value;  // what the reset's LatteTC_HasTextureChanged(tex, true) stores
	}
	HashJobRelease(job);
}

void LatteTC_ResolvePendingHash(LatteTexture* tex)
{
	LatteTC_ResolvePendingHash(tex, true);
}

// LatteTC_ResetTextureChangeTracker(tex, true) with the hash queued for the decode helpers
void LatteTC_ResetTextureChangeTrackerAsync(LatteTexture* tex)
{
	LatteTC_ResolvePendingHash(tex, true);
	if (tex->depth == 83 && tex->width == 1024 && tex->height == 1024)
	{
		// BotW's terrain array: its _botwLargeTexHax bookkeeping (LatteTC_HasTextureChanged) is done at once
		LatteTC_ResetTextureChangeTracker(tex, true);
		return;
	}
	// what the forced reset does before hashing
	tex->forceInvalidate = false;
	tex->lastDataUpdateFrameCounter = LatteGPUState.frameCounter;
	auto* job = new LatteTextureHashJob();
	job->tex = tex;
	tex->pendingHash = job;
	LatteTextureLoader_QueueHashJob(job);
}

bool LatteTC_HasTextureChanged(LatteTexture* hostTexture, bool force)
{
	if (hostTexture->forceInvalidate)
	{
		force = true;
		debug_printf("Force invalidate 0x%08x\n", hostTexture->physAddress);
		hostTexture->forceInvalidate = false;
	}
	// only check each texture for updates once a frame
	// todo: Instead of relying on frames, it would be better to recheck only after any GPU wait operation occurred.
	if( hostTexture->lastDataUpdateFrameCounter == LatteGPUState.frameCounter && force == false)
		return false;
	hostTexture->lastDataUpdateFrameCounter = LatteGPUState.frameCounter;
	// we assume that certain texture properties indicate that the texture will never be written by the CPU
	if (hostTexture->width == 1280 && hostTexture->format != Latte::E_GX2SURFFMT::R8_UNORM && hostTexture->format != Latte::E_GX2SURFFMT::BC3_UNORM && force == false)
	{
		// todo - remove this or find a better way to handle excluded texture invalidation checks (maybe via game profile?)
		return false;
	}
	// workaround for corrupted terrain texture in BotW after video playback
	// probably would be fixed if we added support for invalidating individual slices/mips of a texture
	LatteTC_ResolvePendingHash(hostTexture, true);
	uint32 texDataHash = LatteTexture_CalculateTextureDataHash(hostTexture);
	if( texDataHash != hostTexture->texDataHash2 )
	{
		hostTexture->texDataHash2 = texDataHash;
		if (hostTexture->depth == 83 && hostTexture->width == 1024 && hostTexture->height == 1024)
		{
			_botwLargeTexHax = LatteGPUState.frameCounter;
		}
		return true;
	}
	if (_botwLargeTexHax != 0 && hostTexture->depth == 83 && hostTexture->width == 1024 && hostTexture->height == 1024 && _botwLargeTexHax != LatteGPUState.frameCounter)
	{
		_botwLargeTexHax = 0;
		return true;
	}
	return false;
}

// For GPU updated textures we use a lighter hash algorithm. This also resets the current hash and loses any previous untracked modifications
void LatteTC_SwitchToLightHash(LatteTexture* hostTexture)
{
	if (hostTexture->useLightHash)
		return;
	LatteTC_ResolvePendingHash(hostTexture, true);
	hostTexture->useLightHash = true;
	hostTexture->texDataHash2 = LatteTexture_CalculateTextureDataHash(hostTexture);
}

void LatteTC_FlagSliceAsGPUUpdated(LatteTexture* hostTexture, uint32 sliceIndex, uint32 mipIndex)
{
	// todo - track gpu updated state per slice
	hostTexture->isUpdatedOnGPU = true;
	LatteTC_SwitchToLightHash(hostTexture);
}

void LatteTC_ResetTextureChangeTracker(LatteTexture* hostTexture, bool force)
{
	if( hostTexture->lastDataUpdateFrameCounter == LatteGPUState.frameCounter && force == false)
		return;
	hostTexture->lastDataUpdateFrameCounter = LatteGPUState.frameCounter;
	LatteTC_HasTextureChanged(hostTexture, true);
}

/*
 * This function should be called whenever the texture is still used in some form (any kind of access counts)
 * The purpose of this function is to prevent garbage collection of textures that are still actively used
 */
void LatteTC_MarkTextureStillInUse(LatteTexture* texture)
{
	texture->lastAccessTick = LatteGPUState.currentDrawCallTick;
	texture->lastAccessFrameCount = LatteGPUState.frameCounter;
}

// check if a texture has been overwritten by another texture using GPU-writes
bool LatteTC_IsTextureDataOverwritten(LatteTexture* texture)
{
	// check overlaps
	sint32 mipLevels = texture->mipLevels;
	sint32 sliceCount = texture->depth;
	mipLevels = std::min(mipLevels, 3); // only check first 3 mip levels
	for (sint32 mipIndex = 0; mipIndex < mipLevels; mipIndex++)
	{
		sint32 mipSliceCount;
		if (texture->Is3DTexture())
			mipSliceCount = std::max(1, sliceCount >> mipIndex);
		else
			mipSliceCount = sliceCount;
		for (sint32 sliceIndex = 0; sliceIndex < mipSliceCount; sliceIndex++)
		{
			LatteTextureSliceMipInfo* sliceMipInfo = texture->sliceMipInfo + texture->GetSliceMipArrayIndex(sliceIndex, mipIndex);
			bool isSliceMipOutdated = false;
			for (auto& overlapData : sliceMipInfo->list_dataOverlap)
			{
				if (sliceMipInfo->lastDynamicUpdate < overlapData.destMipSliceInfo->lastDynamicUpdate)
				{
					isSliceMipOutdated = true;
					break;
				}
			}
			if (isSliceMipOutdated == false)
				return false;
		}
	}
	return true;
}

void LatteTexture_Delete(LatteTexture* texture)
{
	LatteTC_ResolvePendingHash(texture, false);
	LatteTC_UnregisterTexture(texture);
	LatteMRT::NotifyTextureDeletion(texture);
	LatteTextureReadback_NotifyTextureDeletion(texture);
	LatteTexture_DeleteTextureRelations(texture);
	// delete views
	while (!texture->views.empty())
		delete texture->views[0];
	cemu_assert_debug(texture->views.empty());
	cemu_assert_debug(texture->baseView == nullptr);
	// free data overlap tracking
	LatteTexture_DeleteDataOverlapTracking(texture);
	// remove from lists
	LatteTexture_UnregisterTextureMemoryOccupancy(texture);
	// free memory
	if (texture->sliceMipInfo)
	{
		delete[] texture->sliceMipInfo;
		texture->sliceMipInfo = nullptr;
	}
	delete texture;
}

/*
 * Checks if the texture can be dropped from the cache and if yes, delete it
 * Returns true if the texture was deleted
 */
// BotW-Native: VRAM pressure. RAM-backed textures stayed until unused for 120 s and 2000 frames, so the Deck's 1 GB VRAM
// carve-out held everything shown in the last two minutes (990 of 1024 MB in play) and the kernel moved memory around
// mid-frame. Above 75% of the carve-out they go after 15 s and 900 frames instead. Opt-in (BOTW_VRAM_TRIM=1): with
// texture memory moving to system memory once the carve-out is nearly full (VKRMemoryManager), the carve-out no longer
// overflows, and the early frees tripled texture re-uploads at 8 W (166 vs 50-64 a second) for no gain.
bool BotW_ReadVramCarveout(uint64& used, uint64& total);  // VKRMemoryManager.cpp
static bool s_vramPressure = false;

static void LatteTC_UpdateVramPressure(uint32 currentTick)
{
	static const bool s_on = [] { const char* e = getenv("BOTW_VRAM_TRIM"); return e && e[0] == '1'; }();
	static uint32 s_lastCheck = 0;
	if (!s_on || currentTick - s_lastCheck < 1000)
		return;
	s_lastCheck = currentTick;
	uint64 used, total;
	s_vramPressure = BotW_ReadVramCarveout(used, total) && total > 0 && total <= (4ull << 30) && used * 4 > total * 3;
}

bool LatteTC_CleanupCheckTexture(LatteTexture* texture, uint32 currentTick)
{
	uint32 currentFrameCount = LatteGPUState.frameCounter;
	uint32 ticksSinceLastAccess = currentTick - texture->lastAccessTick;
	uint32 framesSinceLastAccess = currentFrameCount - texture->lastAccessFrameCount;
	if( !texture->isUpdatedOnGPU )
	{
		// RAM-only textures are safe to be deleted since we can always restore them from RAM
		const uint32 tickLimit = s_vramPressure ? 15 * 1000 : 120 * 1000;
		const uint32 frameLimit = s_vramPressure ? 900 : 2000;
		if( ticksSinceLastAccess >= tickLimit && framesSinceLastAccess >= frameLimit )
		{
			LatteTexture_Delete(texture);
			return true;
		}
	}

	if ((LatteGPUState.currentDrawCallTick - texture->lastAccessTick) >= 100 && 
		LatteTC_IsTextureDataOverwritten(texture))
	{
		LatteTexture_Delete(texture);
		return true;
	}
	// if unused for more than 5 seconds, start deleting views since they are cheap to recreate
	if (ticksSinceLastAccess >= 5 * 1000 && framesSinceLastAccess >= 30)
	{
		for (sint32 i = 0; i < 3; i++)
		{
			if (texture->views.size() <= 1)
				break;
			LatteTextureView* view = texture->views[0];
			if (view == texture->baseView)
				view = texture->views[1];
			delete view;
		}
	}
	return false;
}

void LatteTexture_RefreshInfoCache();

/*
 * Scans for unused textures and deletes them
 * Called at the end of every frame
 */
void LatteTC_CleanupUnusedTextures()
{
	static size_t currentScanIndex = 0;
	uint32 currentTick = GetTickCount();
	LatteTC_UpdateVramPressure(currentTick);
	sint32 maxDelete = 10;
	std::vector<LatteTexture*>& allTextures = LatteTexture::GetAllTextures();
	if (!allTextures.empty())
	{
		for (sint32 c = 0; c < 25; c++)
		{
			if (currentScanIndex >= allTextures.size())
				currentScanIndex = 0;
			LatteTexture* texItr = allTextures[currentScanIndex];
			currentScanIndex++;
			if (!texItr)
				continue;
			if (LatteTC_CleanupCheckTexture(texItr, currentTick))
			{
				maxDelete--;
				if (maxDelete <= 0)
					break; // deleting can be an expensive operation, dont delete too many at once to avoid micro stutter
				if (allTextures.empty())
					break;
			}
		}
	}
	LatteTexture_RefreshInfoCache(); // find a better place to call this from?
}

std::vector<LatteTexture*> LatteTC_GetDeleteableTextures()
{
	std::vector<LatteTexture*> texList;
	uint32 currentFrameCount = LatteGPUState.frameCounter;

	for (auto& itr : g_allTextures)
	{
		if(itr->lastAccessFrameCount == 0)
			continue; // not initialized
		uint32 framesSinceLastAccess = currentFrameCount - itr->lastAccessFrameCount;
		if(framesSinceLastAccess < 3)
			continue;
		if (itr->isUpdatedOnGPU)
		{
			if (LatteTC_IsTextureDataOverwritten(itr))
				texList.emplace_back(itr);
		}
		else
		{
			texList.emplace_back(itr);
		}
	}

	return texList;
}

void LatteTC_UnloadAllTextures()
{
	std::vector<LatteTexture*> allTexturesCopy = LatteTexture::GetAllTextures();
	for (auto& itr : allTexturesCopy)
	{
		if(itr)
			LatteTexture_Delete(itr);
	}
	LatteRenderTarget_unloadAll();
}
