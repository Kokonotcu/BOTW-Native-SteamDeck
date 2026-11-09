#include "Cafe/HW/Latte/ISA/RegDefines.h"
#include "Cafe/HW/Latte/Core/Latte.h"
#include "Cafe/HW/Latte/Core/LatteShader.h"
#include "Cafe/HW/Latte/LatteAddrLib/LatteAddrLib.h"

#include "Cafe/HW/Latte/Renderer/Renderer.h"

#ifdef ENABLE_OPENGL
#include "Cafe/HW/Latte/Renderer/OpenGL/OpenGLRenderer.h"
#include "Cafe/HW/Latte/Renderer/OpenGL/LatteTextureGL.h"
#include "Cafe/HW/Latte/Renderer/OpenGL/LatteTextureViewGL.h"
#endif

struct TexScaleXY
{
	float xy[2];
};

struct
{
	TexScaleXY perUnit[Latte::GPU_LIMITS::NUM_TEXTURES_PER_STAGE]; // stores actualResolution/effectiveResolution ratio for each texture
}LatteTextureScale[static_cast<size_t>(LatteConst::ShaderType::TotalCount)] = { };

float* LatteTexture_getEffectiveTextureScale(LatteConst::ShaderType shaderType, sint32 texUnit)
{
	cemu_assert_debug(texUnit >= 0 && texUnit < Latte::GPU_LIMITS::NUM_TEXTURES_PER_STAGE);
	return LatteTextureScale[static_cast<size_t>(shaderType)].perUnit[texUnit].xy;
}

void LatteTexture_setEffectiveTextureScale(LatteConst::ShaderType shaderType, sint32 texUnit, float u, float v)
{
	cemu_assert_debug(texUnit >= 0 && texUnit < Latte::GPU_LIMITS::NUM_TEXTURES_PER_STAGE);
	float* t = LatteTextureScale[static_cast<size_t>(shaderType)].perUnit[texUnit].xy;
	t[0] = u;
	t[1] = v;
}

void LatteTextureLoader_UpdateTextureSliceData(LatteTexture* tex, uint32 sliceIndex, uint32 mipIndex, MPTR physImagePtr, MPTR physMipPtr, Latte::E_DIM dim, uint32 width, uint32 height, uint32 depth, uint32 mipLevels, uint32 pitch, Latte::E_HWTILEMODE tileMode, uint32 swizzle, bool dumpTex);
void LatteTextureLoader_UpdateTextureSlices(LatteTexture* tex, const std::vector<std::pair<uint32, uint32>>& sliceMips);

// BotW-Native: one sample per KiB of a slice's mip-0 data (the whole-texture change check samples far less)
static uint64 LatteTexture_HashSlice(MPTR addr, uint32 size)
{
	const uint64* p = (const uint64*)memory_getPointerFromPhysicalOffset(addr);
	uint64 h = 0;
	for (uint32 i = 0; i < size / 8; i += 128)
		h = (h ^ p[i]) * 0x9E3779B97F4A7C15ull;
	return h;
}

// BotW-Native: which slices of a 2D array to load; empty = all. BotW streams terrain into single slices of 83-slice
// arrays, and every change reloaded all 83 slices x 11 mips (then again a frame later, _botwLargeTexHax).
// Changed slices are loaded once more at the next reload, as the game may still have been writing them.
static std::vector<uint8> LatteTexture_SlicesToReload(LatteTexture* tex, bool changedSlicesOnly)
{
	std::vector<uint8> reload;
	if (tex->dim != Latte::E_DIM::DIM_2D_ARRAY || tex->depth <= 1 || tex->overwriteInfo.hasFormatOverwrite || tex->overwriteInfo.hasResolutionOverwrite)
		return reload;
	LatteAddrLib::AddrSurfaceInfo_OUT info;
	LatteAddrLib::GX2CalculateSurfaceInfo(tex->format, tex->width, tex->height, tex->depth, tex->dim, Latte::MakeGX2TileMode(tex->tileMode), 0, 0, &info);
	const auto tm = info.hwTileMode;
	const bool thick = tm == Latte::E_HWTILEMODE::TM_1D_TILED_THICK || tm == Latte::E_HWTILEMODE::TM_2D_TILED_THICK ||
		tm == Latte::E_HWTILEMODE::TM_2B_TILED_THICK || tm == Latte::E_HWTILEMODE::TM_3D_TILED_THICK;
	if (thick || (uint64)info.sliceSize * tex->depth != info.surfSize)
	{
		tex->sliceHash.clear();
		return reload;
	}
	std::vector<uint64> hashes(tex->depth);
	for (uint32 s = 0; s < tex->depth; s++)
		hashes[s] = LatteTexture_HashSlice(tex->physAddress + s * info.sliceSize, info.sliceSize);
	std::vector<uint32> changed;
	if (changedSlicesOnly && tex->sliceHash.size() == hashes.size())
	{
		for (uint32 s = 0; s < tex->depth; s++)
			if (hashes[s] != tex->sliceHash[s])
				changed.push_back(s);
		if (!changed.empty() || !tex->sliceRecheck.empty())
		{
			reload.assign(tex->depth, 0);
			for (uint32 s : changed)
				reload[s] = 1;
			for (uint32 s : tex->sliceRecheck)
				reload[s] = 1;
		}
		// else: the texture check saw a change these samples miss, load everything
	}
	tex->sliceHash = std::move(hashes);
	tex->sliceRecheck = std::move(changed);
	return reload;
}

void LatteTexture_ReloadData(LatteTexture* tex, bool changedSlicesOnly)
{
	tex->reloadCount++;
	struct UploadBatch
	{
		UploadBatch() { g_renderer->texture_beginUploadBatch(); }
		~UploadBatch() { g_renderer->texture_endUploadBatch(); }
	} uploadBatch;
	const std::vector<uint8> reloadSlice = LatteTexture_SlicesToReload(tex, changedSlicesOnly);
	// (slice, mip) in load order; LatteTextureLoader_UpdateTextureSlices decodes them on several threads when large
	std::vector<std::pair<uint32, uint32>> sliceMips;
	for(sint32 mip=0; mip<tex->mipLevels; mip++)
	{
		if(tex->dim == Latte::E_DIM::DIM_2D_ARRAY ||
			tex->dim == Latte::E_DIM::DIM_2D_ARRAY_MSAA )
		{
			sint32 numSlices = std::max(tex->depth, 1);
			for(sint32 s=0; s<numSlices; s++)
			{
				if (!reloadSlice.empty() && !reloadSlice[s])
					continue;
				sliceMips.emplace_back(s, mip);
			}
		}
		else if( tex->dim == Latte::E_DIM::DIM_CUBEMAP )
		{
			cemu_assert_debug((tex->depth % 6) == 0);
			sint32 numFullCubeMaps = tex->depth/6; // number of cubemaps (if numFullCubeMaps is >1 then this texture is a cubemap array)
			for(sint32 s=0; s<numFullCubeMaps*6; s++)
				sliceMips.emplace_back(s, mip);
		}
		else if( tex->dim == Latte::E_DIM::DIM_3D )
		{
			sint32 mipDepth = std::max(tex->depth>>mip, 1);
			for(sint32 s=0; s<mipDepth; s++)
				sliceMips.emplace_back(s, mip);
		}
		else
		{
			// load slice 0
			sliceMips.emplace_back(0, mip);
		}
	}
	LatteTextureLoader_UpdateTextureSlices(tex, sliceMips);
	tex->lastUpdateEventCounter = LatteTexture_getNextUpdateEventCounter();
}

LatteTextureView* LatteTexture_CreateTexture(Latte::E_DIM dim, MPTR physAddress, MPTR physMipAddress, Latte::E_GX2SURFFMT format, uint32 width, uint32 height, uint32 depth, uint32 pitch, uint32 mipLevels, uint32 swizzle, Latte::E_HWTILEMODE tileMode, bool isDepth)
{
	const auto tex = g_renderer->texture_createTextureEx(dim, physAddress, physMipAddress, format, width, height, depth, pitch, mipLevels, swizzle, tileMode, isDepth);

	// init slice/mip info array
	LatteTexture_InitSliceAndMipInfo(tex);
	LatteTexture_RegisterTextureMemoryOccupancy(tex);
	cemu_assert_debug(mipLevels != 0);

	LatteTexture_ReloadData(tex);
	LatteTC_MarkTextureStillInUse(tex);
	LatteTC_RegisterTexture(tex);

	// create initial view that maps to the whole texture
	tex->baseView = tex->GetOrCreateView(0, tex->mipLevels, 0, tex->depth);
	return tex->baseView;
}

Latte::E_GX2SURFFMT LatteTexture_ReconstructGX2Format(const Latte::LATTE_SQ_TEX_RESOURCE_WORD1_N& texUnitWord1, const Latte::LATTE_SQ_TEX_RESOURCE_WORD4_N& texUnitWord4)
{
	Latte::E_GX2SURFFMT gx2Format = (Latte::E_GX2SURFFMT)texUnitWord1.get_DATA_FORMAT();
	auto nfa = texUnitWord4.get_NUM_FORM_ALL();
	if (nfa == Latte::LATTE_SQ_TEX_RESOURCE_WORD4_N::E_NUM_FORMAT_ALL::NUM_FORMAT_SCALED)
		gx2Format |= Latte::E_GX2SURFFMT::FMT_BIT_FLOAT;
	else if (nfa == Latte::LATTE_SQ_TEX_RESOURCE_WORD4_N::E_NUM_FORMAT_ALL::NUM_FORMAT_INT)
		gx2Format |= Latte::E_GX2SURFFMT::FMT_BIT_INT;

	if(texUnitWord4.get_FORCE_DEGAMMA())
		gx2Format |= Latte::E_GX2SURFFMT::FMT_BIT_SRGB;

	if (texUnitWord4.get_FORMAT_COMP_X() == Latte::LATTE_SQ_TEX_RESOURCE_WORD4_N::E_FORMAT_COMP::COMP_SIGNED)
		gx2Format |= Latte::E_GX2SURFFMT::FMT_BIT_SIGNED;

	return gx2Format;
}

extern uint32 g_texViewLookupGen;  // LatteTextureView.cpp

void LatteTexture_updateTexturesForStage(LatteDecompilerShader* shaderContext, uint32 glBackendBaseTexUnit, _LatteRegisterSetTextureUnit* texRegBase)
{
	for (sint32 z = 0; z < shaderContext->textureUnitListCount; z++)
	{
		sint32 textureIndex = shaderContext->textureUnitList[z];
		const auto& texRegister = texRegBase[textureIndex];

		// get physical address of texture data
		MPTR physAddr = (texRegister.word2.get_BASE_ADDRESS() << 8);
		if (physAddr == MPTR_NULL)
			continue; // invalid data
		MPTR physMipAddr = (texRegister.word3.get_MIP_ADDRESS() << 8);

		// word0
		const auto word0 = texRegister.word0;
		auto dim = word0.get_DIM();
		uint32 pitch = (word0.get_PITCH() + 1) << 3;
		uint32 width = word0.get_WIDTH() + 1;
		auto tileMode = word0.get_TILE_MODE();
		// word1
		const auto word1 = texRegister.word1;
		uint32 depth = word1.get_DEPTH();
		if (dim == Latte::E_DIM::DIM_2D_ARRAY || dim == Latte::E_DIM::DIM_3D || dim == Latte::E_DIM::DIM_2D_ARRAY_MSAA || dim == Latte::E_DIM::DIM_1D_ARRAY)
		{
			depth = depth + 1;
		}
		else
		{
			if (dim == Latte::E_DIM::DIM_CUBEMAP)
				depth = 6 * (depth + 1);
			if (depth == 0)
				depth = 1;
		}
		uint32 height = word1.get_HEIGHT() + 1;
		if (dim == Latte::E_DIM::DIM_1D || dim == Latte::E_DIM::DIM_1D_ARRAY)
			height = 1;
		if (Latte::IsCompressedFormat(word1.get_DATA_FORMAT()))
			pitch /= 4;
		// view slice
		const auto word4 = texRegister.word4;
		const auto word5 = texRegister.word5;

		uint32 viewFirstSlice = word5.get_BASE_ARRAY();
		uint32 viewNumSlices = word5.get_LAST_ARRAY() + 1 - viewFirstSlice;

		uint32 viewFirstMip = word4.get_BASE_LEVEL();
		uint32 viewNumMips = word5.get_LAST_LEVEL() + 1 - viewFirstMip;

		cemu_assert_debug(viewNumMips != 0);

		Latte::E_GX2SURFFMT format = LatteTexture_ReconstructGX2Format(word1, word4);

		// todo - AA
		if (dim == Latte::E_DIM::DIM_2D_MSAA)
		{
			// MSAA only supports one mip level?
			// without this we encounter a crash in The Mysterious Cities of Gold: Secret Paths due to it setting mip count to 2 and leaving mip pointer on an invalid uninitialized value
			viewFirstMip = 0;
			viewNumMips = 1;
		}

		// swizzle
		uint32 swizzle = 0;
		if (Latte::TM_IsMacroTiled(tileMode))
		{
			// extract swizzle bits from pointer if macro-tiled
			swizzle = (physAddr & 0x700);
			physAddr &= ~0x700;
		}

		bool isDepthSampler = shaderContext->textureUsesDepthCompare[textureIndex];
		// look for already existing texture
		// BotW-Native: consecutive draws mostly bind the same texture to a unit; the view found for the same register
		// words (everything the lookup reads: base address, words 0/1/4/5, depth compare) is reused while no view
		// entered or left the lookup cache (g_texViewLookupGen). BOTW_TEXUNIT_CACHE=0 off.
		struct UnitCacheEntry { uint32 w0, w1, w2, w4, w5; uint32 gen; bool depth; LatteTextureView* view; };
		static UnitCacheEntry s_unitCache[256];
		static const bool s_unitCacheOn = !(getenv("BOTW_TEXUNIT_CACHE") && atoi(getenv("BOTW_TEXUNIT_CACHE")) == 0);
		UnitCacheEntry& uc = s_unitCache[(glBackendBaseTexUnit + textureIndex) & 255];
		const uint32 rw0 = texRegister.word0.getRawValue(), rw1 = texRegister.word1.getRawValue(), rw2 = texRegister.word2.getRawValue(),
		             rw4 = texRegister.word4.getRawValue(), rw5 = texRegister.word5.getRawValue();
		LatteTextureView* textureView = nullptr;
		if (s_unitCacheOn && uc.view && uc.gen == g_texViewLookupGen && uc.depth == isDepthSampler && uc.w0 == rw0 && uc.w1 == rw1 &&
		    uc.w2 == rw2 && uc.w4 == rw4 && uc.w5 == rw5)
			textureView = uc.view;
		else if (!isDepthSampler)
			textureView = LatteTextureViewLookupCache::lookup(physAddr, width, height, depth, pitch, viewFirstMip, viewNumMips, viewFirstSlice, viewNumSlices, format, dim);
		else
			textureView = LatteTextureViewLookupCache::lookupWithColorOrDepthType(physAddr, width, height, depth, pitch, viewFirstMip, viewNumMips, viewFirstSlice, viewNumSlices, format, dim, true);
		if (s_unitCacheOn && textureView)
			uc = {rw0, rw1, rw2, rw4, rw5, g_texViewLookupGen, isDepthSampler, textureView};
		if (!textureView)
		{
			// view not found, create a new mapping which will also create a new texture if necessary
			textureView = LatteTexture_CreateMapping(physAddr, physMipAddr, width, height, depth, pitch, tileMode, swizzle, viewFirstMip, viewNumMips, viewFirstSlice, viewNumSlices, format, dim, dim, isDepthSampler);
			if (textureView == nullptr)
				continue;
			LatteGPUState.repeatTextureInitialization = true;
		}

#ifdef ENABLE_OPENGL
		if (g_renderer->GetType() == RendererAPI::OpenGL)
		{
			// on OpenGL, texture views and sampler parameters are tied together (we are avoiding sampler objects due to driver bugs)
			// in order to emulate different sampler parameters when a texture is bound multiple times we create extra views
			OpenGLRenderer* rendererGL = static_cast<OpenGLRenderer*>(g_renderer.get());

			// if this texture is bound multiple times then use alternative views
			if (textureView->lastTextureBindIndex == LatteGPUState.textureBindCounter)
			{
				LatteTextureViewGL* textureViewGL = (LatteTextureViewGL*)textureView;
				// get next unused alternative texture view
				while (true)
				{
					textureViewGL = textureViewGL->GetAlternativeView();
					if (textureViewGL->lastTextureBindIndex != LatteGPUState.textureBindCounter)
						break;
				}
				textureView = textureViewGL;
		}
			textureView->lastTextureBindIndex = LatteGPUState.textureBindCounter;
			rendererGL->renderstate_updateTextureSettingsGL(shaderContext, textureView, textureIndex + glBackendBaseTexUnit, word4, textureIndex, isDepthSampler);
		}
#endif

		g_renderer->texture_setLatteTexture(textureView, textureIndex + glBackendBaseTexUnit);
		// update if data changed
		bool swizzleChanged = false;
		if (textureView->baseTexture->swizzle != swizzle)
		{
			debug_printf("BaseSwizzle diff prev %08x new %08x rt %08x tm %d\n", textureView->baseTexture->swizzle, swizzle, textureView->baseTexture->lastRenderTargetSwizzle, textureView->baseTexture->tileMode);
			if (swizzle == textureView->baseTexture->lastRenderTargetSwizzle)
			{
				// last render to texture updated the swizzle and we can assume the texture data is still valid
				textureView->baseTexture->swizzle = textureView->baseTexture->lastRenderTargetSwizzle;
			}
			else
			{
				// reload texture
				swizzleChanged = true;
			}
		}
		else if ((viewFirstMip + viewNumMips) > 1 && (textureView->baseTexture->physMipAddress != physMipAddr))
		{
			debug_printf("MipPhys/Swizzle change diff prev %08x new %08x tm %d\n", textureView->baseTexture->physMipAddress, physMipAddr, textureView->baseTexture->tileMode);
			swizzleChanged = true;
			cemu_assert_debug(physMipAddr != MPTR_NULL);
		}
		// check for changes
		if (LatteTC_HasTextureChanged(textureView->baseTexture) || swizzleChanged)
		{
			debug_printf("Reload texture 0x%08x res %dx%d memRange %08x-%08x SwizzleChange: %s\n", textureView->baseTexture->physAddress, textureView->baseTexture->width, textureView->baseTexture->height, textureView->baseTexture->texDataPtrLow, textureView->baseTexture->texDataPtrHigh, swizzleChanged ? "yes" : "no");
			// update swizzle / changed mip address
			if (swizzleChanged)
			{
				textureView->baseTexture->swizzle = swizzle;
				if ((viewFirstMip + viewNumMips) > 1)
				{
					textureView->baseTexture->physMipAddress = physMipAddr;
				}
			}
			debug_printf("Reload reason: Data-change when bound as texture (new hash 0x%08x)\n", textureView->baseTexture->texDataHash2);
			LatteTexture_ReloadData(textureView->baseTexture, !swizzleChanged);
		}
		LatteTexture* baseTexture = textureView->baseTexture;
		if (baseTexture->reloadFromDynamicTextures)
		{
			LatteTexture_UpdateCacheFromDynamicTextures(baseTexture);
			baseTexture->reloadFromDynamicTextures = false;
		}
		LatteTC_MarkTextureStillInUse(baseTexture);

		// check if barrier is necessary
		if ((sint32)(LatteGPUState.drawCallCounter - baseTexture->lastUnflushedRTDrawcallIndex) < 2)
		{
			LatteGPUState.requiresTextureBarrier = true;
			baseTexture->lastUnflushedRTDrawcallIndex = 0;
		}
		// update scale
		float texScaleU, texScaleV;
		if (baseTexture->overwriteInfo.hasResolutionOverwrite == false)
		{
			texScaleU = 1.0f;
			texScaleV = 1.0f;
		}
		else
		{
			texScaleU = (float)baseTexture->overwriteInfo.width / (float)baseTexture->width;
			texScaleV = (float)baseTexture->overwriteInfo.height / (float)baseTexture->height;
		}
		LatteTexture_setEffectiveTextureScale(shaderContext->shaderType, textureIndex, texScaleU, texScaleV);
	}
}

// initialize textures used by the current drawcall
// Sets LatteGPUState.repeatTextureInitialization to true if a new texture mapping was created (indicating that this function must be called again)
// also sets LatteGPUState.requiresTextureBarrier to true if texture barrier is required
void LatteTexture_updateTextures()
{
	LatteGPUState.textureBindCounter++;
	// pixel shader
	LatteDecompilerShader* pixelShader = LatteSHRC_GetActivePixelShader();
	if (pixelShader)
		LatteTexture_updateTexturesForStage(pixelShader, LATTE_CEMU_PS_TEX_UNIT_BASE, LatteGPUState.contextNew.SQ_TEX_START_PS);
	// vertex shader
	LatteDecompilerShader* vertexShader = LatteSHRC_GetActiveVertexShader();
	cemu_assert_debug(vertexShader != nullptr);
	LatteTexture_updateTexturesForStage(vertexShader, LATTE_CEMU_VS_TEX_UNIT_BASE, LatteGPUState.contextNew.SQ_TEX_START_VS);
	// geometry shader
	LatteDecompilerShader* geometryShader = LatteSHRC_GetActiveGeometryShader();
	if (geometryShader)
		LatteTexture_updateTexturesForStage(geometryShader, LATTE_CEMU_GS_TEX_UNIT_BASE, LatteGPUState.contextNew.SQ_TEX_START_GS);
}

sint32 LatteTexture_getEffectiveWidth(LatteTexture* texture)
{
	if (texture->overwriteInfo.hasResolutionOverwrite)
		return texture->overwriteInfo.width;
	return texture->width;
}

// returns true if the two textures have the same rescale factor
bool LatteTexture_doesEffectiveRescaleRatioMatch(LatteTexture* texture1, sint32 mipLevel1, LatteTexture* texture2, sint32 mipLevel2)
{
	double widthRatio1 = (double)LatteTexture_getEffectiveWidth(texture1) / (double)texture1->width;
	double widthRatio2 = (double)LatteTexture_getEffectiveWidth(texture2) / (double)texture2->width;
	// the difference between the factors must be less than 5%
	double diff = widthRatio1 / widthRatio2;
	if (abs(1.0 - diff) > 0.05)
	{
		return false;
	}
	return true;
}

void LatteTexture_scaleToEffectiveSize(LatteTexture* texture, sint32* x, sint32* y, sint32 mipLevel)
{
	if( texture->overwriteInfo.hasResolutionOverwrite == false )
		return;
	*x = *x * std::max(1,texture->overwriteInfo.width>>mipLevel) / std::max(1,texture->width>>mipLevel);
	*y = *y * std::max(1,texture->overwriteInfo.height>>mipLevel) / std::max(1, texture->height>>mipLevel);
}

uint64 _textureUpdateEventCounter = 1;

uint64 LatteTexture_getNextUpdateEventCounter()
{
	uint64 counter = _textureUpdateEventCounter;
	_textureUpdateEventCounter++;
	return counter;
}

void LatteTexture_init()
{
}
