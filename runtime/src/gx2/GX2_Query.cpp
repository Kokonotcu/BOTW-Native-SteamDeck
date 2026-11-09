// Vendored from Cemu (src/Cafe/OS/libs/gx2/GX2_Query.cpp), MPL-2.0. Occlusion queries and conditional rendering;
// packets identical to Cemu. Dropped: Cemu's Xenoblade X title-ID workaround in _BeginOcclusionQuery (never BotW).
// Exports registered with CAFE_EXPORT instead of cafeExportRegister (GX2QueryInit dropped).
#include "../compat/cemu_export.h"
#include "../latte/isa/RegDefines.h"
#include "GX2.h"
#include "GX2_Query.h"

#define LATTE_GC_NUM_RB							2
#define _QUERY_REG_COUNT						8 // each reg/result is 64bits, little endian

#include <chrono>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
void log_write(const char* fmt, ...);

// TEST ONLY: BOTW_QPROBE=1 logs how long occlusion results take to reach the game (GX2QueryEnd -> first ready poll)
// and how many polls see "not ready"; every ~5 s.
namespace
{
	bool qprobe_on()
	{
		static const bool v = getenv("BOTW_QPROBE") && atoi(getenv("BOTW_QPROBE")) != 0;
		return v;
	}
	std::mutex s_qlock;
	std::unordered_map<uint32, uint64> s_qEnd;  // query -> end time (us)
	uint64 s_qPolls, s_qNotReady, s_qZero, s_qHist[7], s_qLast;
	uint64 qnow()
	{
		return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
	}
	void qprobe_end(uint32 q)
	{
		std::lock_guard<std::mutex> l(s_qlock);
		s_qEnd[q] = qnow();
	}
	void qprobe_poll(uint32 q, bool ready, uint64 samples)
	{
		std::lock_guard<std::mutex> l(s_qlock);
		const uint64 t = qnow();
		s_qPolls++;
		if (!ready)
			s_qNotReady++;
		else
		{
			if (samples == 0)
				s_qZero++;
			if (auto it = s_qEnd.find(q); it != s_qEnd.end())
			{
				const uint64 ms = (t - it->second) / 1000;
				static const uint64 edges[6] = {17, 34, 50, 67, 100, 200};
				int b = 0;
				while (b < 6 && ms >= edges[b])
					b++;
				s_qHist[b]++;
				s_qEnd.erase(it);
			}
		}
		if (t - s_qLast < 5000000)
			return;
		s_qLast = t;
		log_write("qprobe polls %llu notready %llu zero %llu | ready after ms <17:%llu <34:%llu <50:%llu <67:%llu <100:%llu <200:%llu more:%llu",
			(unsigned long long)s_qPolls, (unsigned long long)s_qNotReady, (unsigned long long)s_qZero,
			(unsigned long long)s_qHist[0], (unsigned long long)s_qHist[1], (unsigned long long)s_qHist[2], (unsigned long long)s_qHist[3],
			(unsigned long long)s_qHist[4], (unsigned long long)s_qHist[5], (unsigned long long)s_qHist[6]);
	}
}

namespace GX2
{
	struct GX2Query
	{
		// 4*2 sets of uint64 results
		uint32 reg[_QUERY_REG_COUNT * 2];
	};

	static_assert(sizeof(GX2Query) == 0x40);

	void _BeginOcclusionQuery(GX2Query* queryInfo, bool isGPUQuery)
	{
		if (isGPUQuery)
		{
			// BotWRecomp: Cemu's XCX (title 00050000101c4c00/101c4d00/10116100) branch that writes a fake result instead
			// of the packets below is dropped; for every other title Cemu takes this path.
			{
				GX2ReserveCmdSpace(5 * _QUERY_REG_COUNT);
				MPTR queryInfoPhys = memory_virtualToPhysical(memory_getVirtualOffsetFromPointer(queryInfo));
				for (sint32 i = 0; i < _QUERY_REG_COUNT; i++)
				{
					gx2WriteGather_submitU32AsBE(pm4HeaderType3(IT_MEM_WRITE, 4));
					gx2WriteGather_submitU32AsBE((queryInfoPhys + i * 8) | 0x2);
					gx2WriteGather_submitU32AsBE(0x20000); // 0x20000 -> ?
					uint32 v = 0;
					if (i >= LATTE_GC_NUM_RB * 2)
						v |= 0x80000000;
					gx2WriteGather_submitU32AsBE(0);
					gx2WriteGather_submitU32AsBE(v);
				}
			}
		}
		else
		{
			memset(queryInfo, 0, 0x10); // size maybe GPU7_GC_NUM_RB*2*4 ?
			queryInfo->reg[LATTE_GC_NUM_RB * 4 + 0] = 0;
			queryInfo->reg[LATTE_GC_NUM_RB * 4 + 1] = _swapEndianU32('OCPU');
		}
		// todo: Set mmDB_RENDER_CONTROL
	}

	void GX2QueryBegin(uint32 queryType, GX2Query* query)
	{
		if (queryType == GX2_QUERY_TYPE_OCCLUSION_CPU)
		{
			_BeginOcclusionQuery(query, false);
		}
		else if (queryType == GX2_QUERY_TYPE_OCCLUSION_GPU)
		{
			_BeginOcclusionQuery(query, true);
		}
		else
		{
			debug_printf("GX2QueryBegin(): Unsupported type %d\n", queryType);
			debugBreakpoint();
			return;
		}
		// HLE packet
		GX2ReserveCmdSpace(2);
		gx2WriteGather_submitU32AsBE(pm4HeaderType3(IT_HLE_BEGIN_OCCLUSION_QUERY, 1));
		gx2WriteGather_submitU32AsBE(MEMPTR<GX2Query>(query).GetMPTR());
	}

	void GX2QueryEnd(uint32 queryType, GX2Query* query)
	{
		GX2ReserveCmdSpace(2);
		if (queryType == GX2_QUERY_TYPE_OCCLUSION_CPU || queryType == GX2_QUERY_TYPE_OCCLUSION_GPU)
		{
			// HLE packet
			gx2WriteGather_submitU32AsBE(pm4HeaderType3(IT_HLE_END_OCCLUSION_QUERY, 1));
			gx2WriteGather_submitU32AsBE(MEMPTR<GX2Query>(query).GetMPTR());
			if (qprobe_on())
				qprobe_end(MEMPTR<GX2Query>(query).GetMPTR());
		}
		else
		{
			debug_printf("GX2QueryBegin(): Unsupported %d\n", queryType);
			debugBreakpoint();
			return;
		}
	}

	uint32 GX2QueryGetOcclusionResult(GX2Query* query, uint64be* resultOut)
	{
		const uint32 qm = MEMPTR<GX2Query>(query).GetMPTR();
		if (query->reg[LATTE_GC_NUM_RB * 4 + 1] == _swapEndianU32('OCPU') && query->reg[LATTE_GC_NUM_RB * 4 + 0] == 0)
		{
			// CPU query result not ready
			if (qprobe_on())
				qprobe_poll(qm, false, 0);
			return GX2_FALSE;
		}

		uint64 startValue = *(uint64*)(query->reg + 0);
		uint64 endValue = *(uint64*)(query->reg + 2);
		if ((startValue & 0x8000000000000000ULL) || (endValue & 0x8000000000000000ULL))
		{
			if (qprobe_on())
				qprobe_poll(qm, false, 0);
			return GX2_FALSE;
		}
		*resultOut = endValue - startValue;
		if (qprobe_on())
			qprobe_poll(qm, true, endValue - startValue);
		return GX2_TRUE;
	}

	void GX2QueryBeginConditionalRender(uint32 queryType, GX2Query* query, uint32 dontWaitBool, uint32 pixelsMustPassBool)
	{
		GX2ReserveCmdSpace(3);

		uint32 flags = 0;
		if (pixelsMustPassBool)
			flags |= (1<<31);
		if (queryType == GX2_QUERY_TYPE_OCCLUSION_GPU)
			flags |= (1 << 13);
		else
			flags |= (2 << 13);

		flags |= ((dontWaitBool != 0) << 19);

		gx2WriteGather_submitU32AsBE(pm4HeaderType3(IT_SET_PREDICATION, 2));
		gx2WriteGather_submitU32AsBE(memory_virtualToPhysical(MEMPTR<GX2Query>(query).GetMPTR()));
		gx2WriteGather_submitU32AsBE(flags);
	}

	void GX2QueryEndConditionalRender()
	{
		GX2ReserveCmdSpace(3);

		gx2WriteGather_submitU32AsBE(pm4HeaderType3(IT_SET_PREDICATION, 2));
		gx2WriteGather_submitU32AsBE(MPTR_NULL);
		gx2WriteGather_submitU32AsBE(0); // unknown / todo
	}

};

CAFE_EXPORT(gx2, GX2QueryBegin, GX2::GX2QueryBegin);
CAFE_EXPORT(gx2, GX2QueryEnd, GX2::GX2QueryEnd);
CAFE_EXPORT(gx2, GX2QueryGetOcclusionResult, GX2::GX2QueryGetOcclusionResult);
CAFE_EXPORT(gx2, GX2QueryBeginConditionalRender, GX2::GX2QueryBeginConditionalRender);
CAFE_EXPORT(gx2, GX2QueryEndConditionalRender, GX2::GX2QueryEndConditionalRender);
