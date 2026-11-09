#include "Cafe/HW/Latte/Renderer/Vulkan/VulkanRenderer.h"

class LatteQueryObjectVk : public LatteQueryObject
{
	friend class VulkanRenderer;

	LatteQueryObjectVk(VulkanRenderer* rendererVk) : m_rendererVk(rendererVk) 
	{

	};

	bool getResult(uint64& numSamplesPassed) override;
	void begin() override;
	void end() override;
	void beginFragment();
	void endFragment();
	void handleFinishedFragments();

	uint32 acquireQueryIndex();
	void releaseQueryIndex(uint32 queryIndex);

private:
	struct queryFragment 
	{
		uint32 queryIndex;
		uint64 m_finishCommandBuffer;
		bool isFinished;
	};

	VulkanRenderer* m_rendererVk;
	//sint32 m_queryIndex;
	std::vector<queryFragment> list_queryFragments;
	bool m_vkQueryEnded{};
	bool m_hasActiveQuery{};
	bool m_hasActiveFragment{};
	uint64 m_finishCommandBuffer;
	uint64 m_acccumulatedSum;
};

bool LatteQueryObjectVk::getResult(uint64& numSamplesPassed)
{
	if (!m_vkQueryEnded)
		return false;
	if (!m_rendererVk->HasCommandBufferFinished(m_finishCommandBuffer))
		return false;
	handleFinishedFragments();
	cemu_assert_debug(list_queryFragments.empty());
	numSamplesPassed = m_acccumulatedSum;
	//numSamplesPassed = m_rendererVk->m_occlusionQueries.ptrQueryResults[m_queryIndex];
	return true;
}

// BotW-Native: fragments run inside render passes only (all draws are there), so beginning or ending a query no longer
// splits a render pass: a fragment ends with its pass, and the next pass begins a new one
void LatteQueryObjectVk::beginFragment()
{
	if (!m_rendererVk->m_state.activeRenderpassFBO)
		return;  // occlusionQuery_notifyBeginRenderPass
	auto& resetIndices = m_rendererVk->m_occlusionQueries.list_resetIndices;
	if (resetIndices.empty())
	{
		// No reset query left (resets happen outside render passes, see occlusionQuery_outsideRenderPass). Ending the
		// pass here would leave the draw that is being set up outside any render pass: this pass goes uncounted and
		// the next pass begins the fragment.
		return;
	}

	handleFinishedFragments();
	uint32 newQueryIndex = resetIndices.back();
	resetIndices.pop_back();

	queryFragment qf{};
	qf.queryIndex = newQueryIndex;
	qf.isFinished = false;
	qf.m_finishCommandBuffer = 0;
	list_queryFragments.emplace_back(qf);


	vkCmdBeginQuery(m_rendererVk->m_state.currentCommandBuffer, m_rendererVk->m_occlusionQueries.queryPool, newQueryIndex, VK_QUERY_CONTROL_PRECISE_BIT);
	// todo - we already synchronize with command buffers, should we also set wait bits?

	m_hasActiveFragment = true;
}

void LatteQueryObjectVk::begin()
{
	m_vkQueryEnded = false;
	m_hasActiveQuery = true;
	beginFragment();
}

void LatteQueryObjectVk::endFragment()
{
	cemu_assert_debug(m_hasActiveFragment);
	uint32 queryIndex = list_queryFragments.back().queryIndex;
	vkCmdEndQuery(m_rendererVk->m_state.currentCommandBuffer, m_rendererVk->m_occlusionQueries.queryPool, queryIndex);
	// the result copy follows outside the render pass, still in this command buffer (occlusionQuery_outsideRenderPass)
	m_rendererVk->m_occlusionQueries.list_pendingResultCopies.push_back(queryIndex);
	list_queryFragments.back().m_finishCommandBuffer = m_rendererVk->GetCurrentCommandBufferId();
	list_queryFragments.back().isFinished = true;
	m_hasActiveFragment = false;
}

void LatteQueryObjectVk::handleFinishedFragments()
{
	// remove finished fragments and add to m_acccumulatedSum
	while (!list_queryFragments.empty())
	{
		auto& it = list_queryFragments.front();
		if (!it.isFinished)
			break;
		if (!m_rendererVk->HasCommandBufferFinished(it.m_finishCommandBuffer))
			break;
		m_acccumulatedSum += m_rendererVk->m_occlusionQueries.ptrQueryResults[it.queryIndex];
		releaseQueryIndex(it.queryIndex);
		list_queryFragments.erase(list_queryFragments.begin());
	}
}

uint32 LatteQueryObjectVk::acquireQueryIndex()
{
	if (m_rendererVk->m_occlusionQueries.list_availableQueryIndices.empty())
	{
		cemuLog_log(LogType::Force, "Vulkan-Error: Exhausted query pool");
		assert_dbg();
	}
	uint32 queryIndex = m_rendererVk->m_occlusionQueries.list_availableQueryIndices.back();
	m_rendererVk->m_occlusionQueries.list_availableQueryIndices.pop_back();
	return queryIndex;
}

void LatteQueryObjectVk::releaseQueryIndex(uint32 queryIndex)
{
	m_rendererVk->m_occlusionQueries.list_availableQueryIndices.emplace_back(queryIndex);
}

void LatteQueryObjectVk::end()
{
	cemu_assert_debug(!list_queryFragments.empty());
	if(m_hasActiveFragment)
		endFragment();
	m_vkQueryEnded = true;
	m_hasActiveQuery = false;
	m_finishCommandBuffer = m_rendererVk->GetCurrentCommandBufferId();
	m_rendererVk->m_occlusionQueries.m_lastCommandBuffer = m_finishCommandBuffer;
	// BotW-Native: no RequestSubmitSoon. Every query end cut the command buffer within 10 draws, and every submit
	// ends with the GPU drained (RADV waits for its query writes at command buffer end). The result still arrives
	// with the submit at idle, the draw threshold or the present; a guest waiting on it submits at once
	// (WaitCommandBufferFinished).
	m_rendererVk->RequestSubmitOnIdle();
}

LatteQueryObject* VulkanRenderer::occlusionQuery_create()
{
	// create query pool if it doesn't already exist
	if(m_occlusionQueries.queryPool == VK_NULL_HANDLE)
	{ 
		VkQueryPoolCreateInfo queryPoolCreateInfo{};
		queryPoolCreateInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
		queryPoolCreateInfo.flags = 0;
		queryPoolCreateInfo.queryType = VK_QUERY_TYPE_OCCLUSION;
		queryPoolCreateInfo.queryCount = OCCLUSION_QUERY_POOL_SIZE;
		queryPoolCreateInfo.pipelineStatistics = 0;
		auto r = vkCreateQueryPool(m_logicalDevice, &queryPoolCreateInfo, nullptr, &m_occlusionQueries.queryPool);
		if (r != VK_SUCCESS)
		{
			cemuLog_log(LogType::Force, "Vulkan-Error: Failed to create query pool with error {}", (sint32)r);
			return nullptr;
		}
	}
	LatteQueryObjectVk* queryObjVk = nullptr;
	if (m_occlusionQueries.list_cachedQueries.empty())
	{
		queryObjVk = new LatteQueryObjectVk(this);
	}
	else
	{
		queryObjVk = m_occlusionQueries.list_cachedQueries.front();
		m_occlusionQueries.list_cachedQueries.erase(m_occlusionQueries.list_cachedQueries.begin()+0);
	}
	queryObjVk->queryEnded = false;
	queryObjVk->queryEventStart = 0;
	queryObjVk->queryEventEnd = 0;
	queryObjVk->m_vkQueryEnded = false;
	queryObjVk->m_acccumulatedSum = 0;
	cemu_assert_debug(queryObjVk->list_queryFragments.empty()); // query fragment list should always be cleared in _destroy()
	m_occlusionQueries.list_currentlyActiveQueries.emplace_back(queryObjVk);
	return queryObjVk;
}

void VulkanRenderer::occlusionQuery_destroy(LatteQueryObject* queryObj)
{
	LatteQueryObjectVk* queryObjVk = static_cast<LatteQueryObjectVk*>(queryObj);
	m_occlusionQueries.list_currentlyActiveQueries.erase(std::remove(m_occlusionQueries.list_currentlyActiveQueries.begin(), m_occlusionQueries.list_currentlyActiveQueries.end(), queryObj), m_occlusionQueries.list_currentlyActiveQueries.end());
	m_occlusionQueries.list_cachedQueries.emplace_back(queryObjVk);
	for (auto& it : queryObjVk->list_queryFragments)
		queryObjVk->releaseQueryIndex(it.queryIndex);
	queryObjVk->list_queryFragments.clear();
}

void VulkanRenderer::occlusionQuery_flush()
{
	WaitCommandBufferFinished(m_occlusionQueries.m_lastCommandBuffer);
}

void VulkanRenderer::occlusionQuery_updateState()
{
	// check for finished command buffers here since query states are tied to buffers
	ProcessFinishedCommandBuffers();
}

void VulkanRenderer::occlusionQuery_notifyEndCommandBuffer()
{
	for (auto& it : m_occlusionQueries.list_currentlyActiveQueries)
		if(it->m_hasActiveFragment)
			it->endFragment();
	occlusionQuery_outsideRenderPass();
}

void VulkanRenderer::occlusionQuery_notifyBeginCommandBuffer()
{
	occlusionQuery_outsideRenderPass();
}

void VulkanRenderer::occlusionQuery_notifyBeginRenderPass()
{
	for (auto& it : m_occlusionQueries.list_currentlyActiveQueries)
		if (it->m_hasActiveQuery && !it->m_hasActiveFragment)
			it->beginFragment();
}

// called before the render pass ends: a query begun inside a render pass must end inside it
void VulkanRenderer::occlusionQuery_notifyEndRenderPass()
{
	for (auto& it : m_occlusionQueries.list_currentlyActiveQueries)
		if (it->m_hasActiveFragment)
			it->endFragment();
}

void VulkanRenderer::occlusionQuery_outsideRenderPass()
{
	auto& oq = m_occlusionQueries;
	if (oq.queryPool == VK_NULL_HANDLE)
		return;
	for (uint16 i : oq.list_pendingResultCopies)
		vkCmdCopyQueryPoolResults(m_state.currentCommandBuffer, oq.queryPool, i, 1, oq.bufferQueryResults, i * sizeof(uint64), 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
	oq.list_pendingResultCopies.clear();
	if (oq.list_resetIndices.size() >= 24)
		return;
	while (oq.list_resetIndices.size() < 64 && !oq.list_availableQueryIndices.empty())
	{
		const uint16 i = oq.list_availableQueryIndices.back();
		oq.list_availableQueryIndices.pop_back();
		vkCmdResetQueryPool(m_state.currentCommandBuffer, oq.queryPool, i, 1);
		oq.list_resetIndices.push_back(i);
	}
}
