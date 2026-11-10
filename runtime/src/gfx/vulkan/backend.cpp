#ifndef VK_ENABLE_BETA_EXTENSIONS
#define VK_ENABLE_BETA_EXTENSIONS
#endif
#define WWHD_CREATE_WINDOW_SURFACE SDL_Vulkan_CreateSurface
#include "backend.h"
#include "bc_decode.h"
#include "buffer_cache.h"
#include "present.h"
#include "present_worker.h"
#include "presentation_queue.h"
#include "gfx/display.h"
#include "gfx/display_modes.h"
#include "gx2/gx2.h"
#include <SDL3/SDL_vulkan.h>
#include "hle.h"
#include "os.h"
#include "shaders.h"
#include "settings.h"
#include "gfx/capture_schedule.h"
#include "sparse_hash_memo.h"
#include "gfx/write_watch.h"
#include <algorithm>
#include <functional>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <future>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <time.h>
#include <vulkan/vulkan_beta.h>

namespace gx2 { uint64_t flips_presented(); void checkpoint_vulkan_caches(); }

namespace gfxvk {
Renderer R;
const VkFormatProperties& format_properties(VkFormat format) {
  auto [entry, inserted] = R.formatProperties.try_emplace(format);
  if (inserted) vkGetPhysicalDeviceFormatProperties(R.physicalDevice,format,&entry->second);
  return entry->second;
}
namespace {
bool perf_enabled() {
  static const bool enabled = std::getenv("WWHD_VK_STATS") != nullptr;
  return enabled;
}
bool cpu_only_stats_enabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("WWHD_VK_CPU_ONLY_STATS");
    return value && !std::strcmp(value, "1");
  }();
  return enabled;
}
struct WaitTiming { uint64_t count = 0, ns = 0; };
struct PresentTiming { WaitTiming acquire, compose, present, idle; };
WaitTiming submitWait, presentSubmitWait, queueSubmitTiming;
WaitTiming fenceStatusTiming, fenceResetTiming, retirementCleanupTiming;
WaitTiming commandEndTiming;
std::array<WaitTiming,5> swapPhaseTiming;
PresentTiming screenTiming[2];
constexpr size_t maxPipelineCacheBytes = 64u * 1024u * 1024u;
std::string pipelineCachePath;
uint64_t pipelineCacheAttemptFrame = 0;
bool compatible_pipeline_cache(const std::vector<uint8_t>& bytes) {
  // Vulkan cache header fields are stored little-endian, independent of host.
  if (bytes.size() < 16 + VK_UUID_SIZE) return false;
  auto word = [&](size_t offset) {
    return uint32_t(bytes[offset]) | uint32_t(bytes[offset+1]) << 8 |
           uint32_t(bytes[offset+2]) << 16 | uint32_t(bytes[offset+3]) << 24;
  };
  uint32_t headerSize=word(0);
  return headerSize >= 16+VK_UUID_SIZE && headerSize <= bytes.size() &&
         word(4)==VK_PIPELINE_CACHE_HEADER_VERSION_ONE &&
         word(8)==R.properties.vendorID && word(12)==R.properties.deviceID &&
         !std::memcmp(bytes.data()+16,R.properties.pipelineCacheUUID,VK_UUID_SIZE);
}
// strictMul changes translated shaders; discard old driver pipelines before the 64 MB cap.
constexpr std::array<uint8_t,8> pipelineCacheMagic{'W','W','V','K','P','C','0','2'};
constexpr size_t pipelineCacheWrapperSize=24;
uint64_t pipeline_cache_checksum(const uint8_t* bytes,size_t size) {
  uint64_t hash=14695981039346656037ull;
  for (size_t i=0;i<size;++i) hash=(hash^bytes[i])*1099511628211ull;
  return hash;
}
std::array<uint8_t,pipelineCacheWrapperSize> pipeline_cache_wrapper(const std::vector<uint8_t>& bytes) {
  std::array<uint8_t,pipelineCacheWrapperSize> header{};
  std::copy(pipelineCacheMagic.begin(),pipelineCacheMagic.end(),header.begin());
  uint64_t size=bytes.size(),checksum=pipeline_cache_checksum(bytes.data(),bytes.size());
  for (unsigned i=0;i<8;++i) {
    header[8+i]=uint8_t(size>>(i*8));header[16+i]=uint8_t(checksum>>(i*8));
  }
  return header;
}
bool unpack_pipeline_cache(std::vector<uint8_t>& file) {
  if (file.size()<pipelineCacheWrapperSize ||
      !std::equal(pipelineCacheMagic.begin(),pipelineCacheMagic.end(),file.begin())) return false;
  auto word64=[&](size_t offset) {
    uint64_t value=0;
    for (unsigned i=0;i<8;++i) value|=uint64_t(file[offset+i])<<(i*8);
    return value;
  };
  uint64_t size=word64(8);
  if (size<16+VK_UUID_SIZE || size>maxPipelineCacheBytes ||
      size!=file.size()-pipelineCacheWrapperSize) return false;
  if (word64(16)!=pipeline_cache_checksum(file.data()+pipelineCacheWrapperSize,size)) return false;
  file.erase(file.begin(),file.begin()+pipelineCacheWrapperSize);
  return compatible_pipeline_cache(file);
}
void init_pipeline_cache() try {
  if (const char* explicitPath=std::getenv("WWHD_VK_PIPELINE_CACHE")) {
    if (std::strcmp(explicitPath,"0")) {
      pipelineCachePath=explicitPath;
    }
  } else if (const char* shaderPath=std::getenv("WWHD_SHADER_CACHE");
             !shaderPath || std::strcmp(shaderPath,"0")) {
    char ids[32];
    std::snprintf(ids,sizeof ids,"vulkan-%08x-%08x-",R.properties.vendorID,R.properties.deviceID);
    // a WWHD_SHADER_CACHE file (test runs, separate setups) keeps the pipeline cache next to it,
    // so such runs never write the user's own cache in the config folder
    pipelineCachePath=shaderPath ? std::string(shaderPath)+"."+ids
                                 : g_user_root+"/shadercache/"+ids;
    for (uint8_t byte : R.properties.pipelineCacheUUID) {
      char hex[3]; std::snprintf(hex,sizeof hex,"%02x",byte); pipelineCachePath+=hex;
    }
    pipelineCachePath+=".bin";
  }
  std::vector<uint8_t> bytes;
  if (!pipelineCachePath.empty()) {
    std::error_code ec;
    uintmax_t size=std::filesystem::file_size(pipelineCachePath,ec);
    if (!ec && size>=pipelineCacheWrapperSize+16+VK_UUID_SIZE &&
        size<=maxPipelineCacheBytes+pipelineCacheWrapperSize) {
      bytes.resize(size);
      FILE* file=std::fopen(pipelineCachePath.c_str(),"rb");
      if (file) {
        if (std::fread(bytes.data(),1,bytes.size(),file)!=bytes.size() ||
            std::fgetc(file)!=EOF || std::ferror(file)) bytes.clear();
        std::fclose(file);
        if (!unpack_pipeline_cache(bytes)) bytes.clear();
      } else bytes.clear();
    }
  }
  VkPipelineCacheCreateInfo ci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
  ci.initialDataSize=bytes.size();ci.pInitialData=bytes.empty()?nullptr:bytes.data();
  VkResult result=vkCreatePipelineCache(R.device,&ci,nullptr,&R.pipelineCache);
  if (result!=VK_SUCCESS && !bytes.empty()) {
    ci.initialDataSize=0;ci.pInitialData=nullptr;
    result=vkCreatePipelineCache(R.device,&ci,nullptr,&R.pipelineCache);
    bytes.clear();
  }
  if (result!=VK_SUCCESS) {
    R.pipelineCache=VK_NULL_HANDLE;
    LOG("[vulkan cache] creation failed (%d); continuing without cache",result);
  } else if (perf_enabled()) {
    LOG("[vulkan cache] loaded %zu bytes; persistence %s",bytes.size(),
        pipelineCachePath.empty()?"disabled":pipelineCachePath.c_str());
  }
} catch (const std::exception& error) {
  pipelineCachePath.clear();
  VkPipelineCacheCreateInfo ci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
  if (vkCreatePipelineCache(R.device,&ci,nullptr,&R.pipelineCache)!=VK_SUCCESS)
    R.pipelineCache=VK_NULL_HANDLE;
  LOG("[vulkan cache] load disabled after error: %s",error.what());
}
template<class F> VkResult timed_call(WaitTiming& timing, F&& call) {
  if (!perf_enabled()) return call();
  auto start = std::chrono::steady_clock::now();
  VkResult result = call();
  const uint64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now() - start).count();
  if (perf_enabled()) { ++timing.count; timing.ns += ns; }
  return result;
}
} // namespace
namespace {
struct PipelineCacheWriteResult {
  bool written = false;
  uint64_t revision = 0, changedFrame = 0;
  size_t size = 0;
  double elapsedMs = 0, checksumMs = 0;
  std::string error;
};
std::future<PipelineCacheWriteResult> pipelineCacheWrite;
// File work owns immutable bytes and a copied path. It never accesses Vulkan
// objects, renderer state, or render-thread logging.
PipelineCacheWriteResult write_pipeline_cache(std::vector<uint8_t> bytes,
                                              std::string path,
                                              uint64_t revision,
                                              uint64_t changedFrame) {
  PipelineCacheWriteResult result;
  result.revision=revision;result.changedFrame=changedFrame;result.size=bytes.size();
  auto start=std::chrono::steady_clock::now();
  std::string temporary;
  try {
    auto wrapper=pipeline_cache_wrapper(bytes);
    result.checksumMs=std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now()-start).count();
    std::error_code ec;
    auto parent=std::filesystem::path(path).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent,ec);
    if (ec) throw std::runtime_error(ec.message());
    temporary=path+".tmp-"+std::to_string(start.time_since_epoch().count());
    FILE* file=std::fopen(temporary.c_str(),"wb");
    if (file) {
      bool written=std::fwrite(wrapper.data(),1,wrapper.size(),file)==wrapper.size();
      if (written) written=std::fwrite(bytes.data(),1,bytes.size(),file)==bytes.size();
      if (std::fclose(file)) written=false;
      if (written) { std::error_code renameError; std::filesystem::rename(temporary,path,renameError); result.written=!renameError; }
    }
    if (!result.written) {
      std::filesystem::remove(temporary,ec);
      result.error="file write or replacement failed";
    }
  } catch (const std::exception& error) {
    result.error=error.what();
    if (!temporary.empty()) {
      std::error_code ec;std::filesystem::remove(temporary,ec);
    }
  }
  result.elapsedMs=std::chrono::duration<double,std::milli>(
      std::chrono::steady_clock::now()-start).count();
  return result;
}
void poll_pipeline_cache_write(bool wait) {
  if (!pipelineCacheWrite.valid()) return;
  if (!wait && pipelineCacheWrite.wait_for(std::chrono::seconds(0)) !=
                   std::future_status::ready) return;
  auto result=pipelineCacheWrite.get();
  // A newer pipeline, even in the same guest frame, must remain dirty.
  if (result.written && result.revision==R.pipelineCreates &&
      result.changedFrame==R.pipelineCacheChangedFrame)
    R.pipelineCacheDirty=false;
  if (result.written && perf_enabled())
    LOG("[vulkan cache] persisted %zu payload bytes in %.2f ms (checksum %.2f ms)",
        result.size,result.elapsedMs,result.checksumMs);
  else if (!result.written)
    LOG("[vulkan cache] save skipped: %s",result.error.c_str());
}
void start_pipeline_cache_write() {
  if (pipelineCacheWrite.valid() || !R.pipelineCache ||
      !R.pipelineCacheDirty || pipelineCachePath.empty()) return;
  // This is the only Vulkan cache read, serialized with pipeline creation on
  // the render thread. Worker serialization/checksum never holds that thread.
  size_t size=0;
  if (vkGetPipelineCacheData(R.device,R.pipelineCache,&size,nullptr)!=VK_SUCCESS ||
      size<16+VK_UUID_SIZE || size>maxPipelineCacheBytes) return;
  std::vector<uint8_t> bytes(size);
  if (vkGetPipelineCacheData(R.device,R.pipelineCache,&size,bytes.data())!=VK_SUCCESS) return;
  bytes.resize(size);
  if (!compatible_pipeline_cache(bytes)) return;
  pipelineCacheWrite=std::async(std::launch::async,write_pipeline_cache,
      std::move(bytes),pipelineCachePath,R.pipelineCreates,R.pipelineCacheChangedFrame);
}
void checkpoint_pipeline_cache() try {
  poll_pipeline_cache_write(false);
  if (R.pipelineCacheDirty && R.frame-R.pipelineCacheChangedFrame>=120 &&
      R.frame-pipelineCacheAttemptFrame>=120 && !pipelineCacheWrite.valid()) {
    pipelineCacheAttemptFrame=R.frame;
    start_pipeline_cache_write();
  }
} catch (const std::exception& error) {
  LOG("[vulkan cache] checkpoint skipped after error: %s",error.what());
}
} // namespace
void save_pipeline_cache() try {
  // Explicit shutdown/checkpoint waits before _Exit, then persists any newer
  // revision created while the periodic job was running.
  poll_pipeline_cache_write(true);
  start_pipeline_cache_write();
  poll_pipeline_cache_write(true);
} catch (const std::exception& error) {
  LOG("[vulkan cache] save skipped after error: %s",error.what());
}
class VulkanFailure : public std::runtime_error {
public:
  VkResult result;
  VulkanFailure(VkResult value,const char* operation)
      : std::runtime_error(std::string(operation)+": Vulkan result "+std::to_string(value)),result(value) {}
};
void vk_check(VkResult r, const char *op) {
  if (r != VK_SUCCESS) {
    VulkanFailure failure(r, op);
    LOG("[gfx] %s", failure.what());  // logged like every runtime error (exception_report.h)
    throw failure;
  }
}
uint32_t memory_type(uint32_t bits, VkMemoryPropertyFlags flags) {
  VkPhysicalDeviceMemoryProperties p;
  vkGetPhysicalDeviceMemoryProperties(R.physicalDevice, &p);
  for (uint32_t i = 0; i < p.memoryTypeCount; i++)
    if ((bits & (1u << i)) && (p.memoryTypes[i].propertyFlags & flags) == flags)
      return i;
  throw std::runtime_error("No compatible Vulkan memory type");
}
Buffer create_buffer(VkDeviceSize size, VkBufferUsageFlags usage,
                     VkMemoryPropertyFlags flags, VkMemoryPropertyFlags preferred) {
  Buffer b;
  b.size = std::max<VkDeviceSize>(size, 16);
  VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  ci.size = b.size;
  ci.usage = usage;
  ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  vk_check(vkCreateBuffer(R.device, &ci, nullptr, &b.buffer), "create buffer");
  VkMemoryRequirements req;
  vkGetBufferMemoryRequirements(R.device, b.buffer, &req);
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = req.size;
  ai.memoryTypeIndex = memory_type(req.memoryTypeBits, flags);
  if (preferred) {
    VkPhysicalDeviceMemoryProperties p;
    vkGetPhysicalDeviceMemoryProperties(R.physicalDevice, &p);
    for (uint32_t i = 0; i < p.memoryTypeCount; i++)
      if ((req.memoryTypeBits & (1u << i)) &&
          (p.memoryTypes[i].propertyFlags & (flags | preferred)) == (flags | preferred)) {
        ai.memoryTypeIndex = i;
        break;
      }
  }
  {
    VkPhysicalDeviceMemoryProperties p;
    vkGetPhysicalDeviceMemoryProperties(R.physicalDevice, &p);
    b.properties = p.memoryTypes[ai.memoryTypeIndex].propertyFlags;
  }
  vk_check(vkAllocateMemory(R.device, &ai, nullptr, &b.memory),
           "allocate buffer memory");
  vk_check(vkBindBufferMemory(R.device, b.buffer, b.memory, 0),
           "bind buffer memory");
  if (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
    vk_check(vkMapMemory(R.device, b.memory, 0, VK_WHOLE_SIZE, 0, &b.mapped),
             "map buffer");
  return b;
}
// GPU -> CPU copies (captures, overlay signatures): the CPU reads these, so host-cached memory where
// the device has it (uncached reads of the plain host-visible type are very slow on discrete GPUs).
Buffer create_readback_buffer(VkDeviceSize size) {
  return create_buffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                       VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
}
// The per-submission upload arena (and the buffer cache's blocks, buffer_cache.cpp) is written by the
// CPU and read by the GPU only. The memory is host-visible but usually not host-cached: uncached or
// write-combined system memory, or device-local BAR memory on discrete GPUs, where CPU reads are about
// 100 times slower than cached ones. Rule: the CPU never reads mapped upload memory. Reuse checks and
// index scans use CPU copies kept beside the slices (vertex_snapshot_history.h, uniform_snapshot.h,
// draw.cpp's index paths, the buffer cache's index shadows); see docs/vulkan.md. The one exception is
// the buffer cache's opt-in verify mode (WWHD_VK_BUFFER_CACHE_VERIFY=1, a diagnostic).
// Unless the memory is host-cached: if the arena's memory type is HOST_CACHED and HOST_COHERENT (Apple
// silicon/MoltenVK has only cached types; many UMA drivers too), reads cost what heap reads cost and the
// copies only add time, so R.uploadReadsDirect lets the reuse caches and the native index scan read the
// slices. CACHED without COHERENT never counts: the arena requires COHERENT (no flush/invalidate).
// WWHD_VK_UPLOAD_READS=auto (default) | shadow (always keep CPU copies) | direct (always read slices).
namespace {
enum class UploadReads { Auto, Shadow, Direct };
UploadReads upload_reads_mode() {
  static const UploadReads mode = [] {
    const char* e = std::getenv("WWHD_VK_UPLOAD_READS");
    if (!e || !*e || !std::strcmp(e, "auto")) return UploadReads::Auto;
    if (!std::strcmp(e, "shadow")) return UploadReads::Shadow;
    if (!std::strcmp(e, "direct")) return UploadReads::Direct;
    LOG("[vulkan] WWHD_VK_UPLOAD_READS=%s unknown (auto|shadow|direct): auto", e);
    return UploadReads::Auto;
  }();
  return mode;
}
void note_upload_block(const Buffer& buffer) {
  const VkMemoryPropertyFlags readable =
      VK_MEMORY_PROPERTY_HOST_CACHED_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  const bool first = R.uploadAllocations == 0;
  const bool cached = (buffer.properties & readable) == readable && (first || R.uploadCached);
  const auto mode = upload_reads_mode();
  const bool direct = mode == UploadReads::Direct || (mode == UploadReads::Auto && cached);
  if (first || direct != R.uploadReadsDirect)
    LOG("[vulkan] upload arena memory: %s; reuse checks %s (WWHD_VK_UPLOAD_READS=%s)",
        cached ? "host-cached" : "not host-cached", direct ? "read the slices" : "keep CPU copies",
        mode == UploadReads::Auto ? "auto" : mode == UploadReads::Shadow ? "shadow" : "direct");
  R.uploadCached = cached;
  R.uploadReadsDirect = direct;
}
}  // namespace
UploadSlice allocate_upload(VkDeviceSize size, VkDeviceSize alignment) {
  size = std::max<VkDeviceSize>(size,16);
  alignment = std::max<VkDeviceSize>(alignment,4);
  auto slice = [&](Renderer::UploadBlock& block) {
    VkDeviceSize offset = ((block.used + alignment - 1) / alignment) * alignment;
    if (offset > block.buffer.size || size > block.buffer.size - offset)
      return UploadSlice{};
    block.used = offset + size;
    R.uploadBytes += size;
    return UploadSlice{block.buffer.buffer,offset,size,
                       static_cast<uint8_t*>(block.buffer.mapped)+offset};
  };
  for (auto& block : R.uploadBlocks) {
    auto result = slice(block);
    if (result.buffer) return result;
  }
  auto buffer = create_buffer(std::max<VkDeviceSize>(32ull<<20,size),
      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  note_upload_block(buffer);
  R.uploadBlocks.push_back({buffer,0});
  ++R.uploadAllocations;
  return slice(R.uploadBlocks.back());
}
void defer_buffer(Buffer b) { R.garbageBuffers.push_back(b); }
void defer_surface_image(VkImage image, VkDeviceMemory memory,
                         std::vector<VkImageView> views) {
  R.garbageImages.push_back({image, memory, std::move(views)});
}
void forget_texture_views() {
} // descriptor sets are rebuilt for every draw and retired on submit
namespace {
struct GpuScopeBucket {
  GpuScopeMetadata metadata{};
  double intervalNs=0,maxNs=0;
  uint64_t count=0,draws=0;
};
std::array<GpuScopeBucket,256> gpuScopeBuckets{};
uint32_t gpuScopeBucketCount=0;
struct GpuScopeTotals { uint64_t available=0,unavailable=0,zero=0,dropped=0,bucketDrops=0; } gpuScopeTotals;
bool same_gpu_scope(const GpuScopeMetadata& a,const GpuScopeMetadata& b) {
  return a.kind==b.kind && a.colors==b.colors && a.depth==b.depth &&
      a.extent.width==b.extent.width && a.extent.height==b.extent.height && a.extent.depth==b.extent.depth &&
      a.mips==b.mips && a.layers==b.layers && a.aspects==b.aspects && a.colorCount==b.colorCount &&
      a.slices==b.slices && a.depthSlice==b.depthSlice && a.depthOnly==b.depthOnly;
}
void add_gpu_scope(const Renderer::Submission::GpuScope& scope,double ns) {
  ++gpuScopeTotals.available;
  gpuScopeTotals.zero += ns==0;
  uint32_t i=0;
  while(i<gpuScopeBucketCount && !same_gpu_scope(gpuScopeBuckets[i].metadata,scope.metadata)) ++i;
  if(i==gpuScopeBucketCount) {
    if(i==gpuScopeBuckets.size()) { ++gpuScopeTotals.bucketDrops; return; }
    gpuScopeBuckets[i]={};gpuScopeBuckets[i].metadata=scope.metadata;++gpuScopeBucketCount;
  }
  auto& bucket=gpuScopeBuckets[i];
  bucket.intervalNs+=ns;bucket.maxNs=std::max(bucket.maxNs,ns);
  ++bucket.count;bucket.draws+=scope.draws;
}
} // namespace
static bool timestamp_capable(uint32_t bits, double period) {
  return bits > 0 && bits <= 64 && period > 0 && std::isfinite(period);
}
static uint64_t timestamp_elapsed_ticks(uint64_t start, uint64_t end, uint32_t bits) {
  const uint64_t mask = bits == 64 ? UINT64_MAX : (uint64_t{1} << bits) - 1;
  return (end - start) & mask;
}
// Overlay queries live independently of reusable submission slots: DrawDone can recycle a slot
// several times in the SAME frame. Keep its query pool until a later frame; never wait for a result.
struct OverlayGpuQuery {
  VkQueryPool pool = VK_NULL_HANDLE;
  uint64_t frame = 0, token = 0;
  bool pending = false;
};
static std::array<OverlayGpuQuery, 256> overlayGpuQueries{};
static void destroy_overlay_gpu_queries() {
  // Called only after the existing submission drain at shutdown.
  for (auto& q : overlayGpuQueries) {
    if (q.pool) vkDestroyQueryPool(R.device, q.pool, nullptr);
    q = {};
  }
}
static bool overlay_timestamp_capable() {
  return timestamp_capable(R.gpuTimestampValidBits, R.properties.limits.timestampPeriod);
}
static void begin_overlay_gpu_queries(VkCommandBuffer, Renderer::Submission& submission) {
  submission.perfQuery = UINT32_MAX;
}
static void destroy_gpu_timestamp_queries() {
  // Caller has already drained submissions; never destroy in-flight queries.
  destroy_overlay_gpu_queries();
  R.gpuTimestampsEnabled = false;
  R.gpuPassTimestampsEnabled = false;
  for (auto& slot : R.submissions) {
    if (slot.timestampQueries) vkDestroyQueryPool(R.device, slot.timestampQueries, nullptr);
    slot.timestampQueries = VK_NULL_HANDLE;
    slot.timestampRecorded = false;
    slot.gpuScopeCount = 0; slot.activeRenderScope = UINT32_MAX;
  }
}
static void init_gpu_timestamp_queries() {
  const char* requested = std::getenv("WWHD_VK_GPU_TIMESTAMPS");
  const char* passes = std::getenv("WWHD_VK_GPU_PASS_TIMESTAMPS");
  const bool passRequested = passes && !std::strcmp(passes, "1");
  if ((!requested || std::strcmp(requested, "1")) && !passRequested) return;
  if (!timestamp_capable(R.gpuTimestampValidBits, R.properties.limits.timestampPeriod)) {
    LOG("[vulkan GPU timestamps] unsupported: valid bits %u, period %.9g ns",
        R.gpuTimestampValidBits, double(R.properties.limits.timestampPeriod));
    return;
  }
  for (auto& slot : R.submissions) {
    VkQueryPoolCreateInfo info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    info.queryType = VK_QUERY_TYPE_TIMESTAMP;
    info.queryCount = passRequested ? 2 + 2 * Renderer::Submission::maxGpuScopes : 2;
    const VkResult result = vkCreateQueryPool(R.device, &info, nullptr, &slot.timestampQueries);
    if (result != VK_SUCCESS) {
      // Initialization has submitted no work, so partial pools are safe to destroy.
      slot.timestampQueries = VK_NULL_HANDLE;
      bc_decode_shutdown();
  destroy_gpu_timestamp_queries();
      LOG("[vulkan GPU timestamps] optional query pools unavailable (%d); disabled", int(result));
      return;
    }
  }
  R.gpuTimestampsEnabled = true;
  R.gpuPassTimestampsEnabled = passRequested;
  if (passRequested) LOG("[vulkan GPU pass timestamps] enabled: 256 bounded scopes/submission; stage intervals may overlap or include stalls; NOT GPU busy time");
  LOG("[vulkan GPU timestamps] enabled: %u valid bits, %.9g ns/tick; submission intervals, NOT GPU busy time",
      R.gpuTimestampValidBits, double(R.properties.limits.timestampPeriod));
}
static void collect_gpu_timestamp_queries(Renderer::Submission& slot) {
  if (!slot.timestampRecorded) return;
  slot.timestampRecorded = false; // Consume this submission exactly once.
  if (!R.gpuTimestampsEnabled) return;
  struct Result { uint64_t ticks, available; } values[2]{};
  const VkResult result = vkGetQueryPoolResults(R.device, slot.timestampQueries, 0, 2,
      sizeof(values), values, sizeof(Result),
      VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
  if (result == VK_NOT_READY || (result == VK_SUCCESS &&
      (!values[0].available || !values[1].available))) {
    ++R.gpuTimestampStats.unavailable;
    return;
  }
  if (result != VK_SUCCESS) {
    ++R.gpuTimestampStats.unavailable;
    R.gpuTimestampsEnabled = false; // Retain pools until the shutdown drain.
    LOG("[vulkan GPU timestamps] optional results failed (%d); disabled", int(result));
    return;
  }
  const double ns = double(timestamp_elapsed_ticks(values[0].ticks, values[1].ticks,
      R.gpuTimestampValidBits)) * double(R.properties.limits.timestampPeriod);
  // Reporting resets the rolling window. Acceptance measurements need counters
  // that remain monotonic across those reports and across timing phases.
  for (auto* stats : {&R.gpuTimestampStats, &R.gpuTimestampLifetimeStats}) {
    ++stats->submissions;
    stats->intervalNs += ns;
    stats->maxIntervalNs = std::max(stats->maxIntervalNs, ns);
    stats->zeroIntervals += ns == 0;
  }
}
static void collect_gpu_pass_queries(Renderer::Submission& slot) {
  const uint32_t count=slot.gpuScopeCount;
  slot.gpuScopeCount=0;slot.activeRenderScope=UINT32_MAX;
  if(!count || !R.gpuTimestampsEnabled || !R.gpuPassTimestampsEnabled) return;
  struct Result { uint64_t ticks,available; };
  std::array<Result,2*Renderer::Submission::maxGpuScopes> values{};
  const VkResult result=vkGetQueryPoolResults(R.device,slot.timestampQueries,2,2*count,
      2*count*sizeof(Result),values.data(),sizeof(Result),
      VK_QUERY_RESULT_64_BIT|VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
  if(result!=VK_SUCCESS && result!=VK_NOT_READY) {
    gpuScopeTotals.unavailable+=count;R.gpuTimestampsEnabled=false;
    LOG("[vulkan GPU pass timestamps] optional results failed (%d); disabled",int(result));
    return;
  }
  for(uint32_t i=0;i<count;++i) {
    if(!slot.gpuScopes[i].ended || !values[2*i].available || !values[2*i+1].available) {
      ++gpuScopeTotals.unavailable;continue;
    }
    const double ns=double(timestamp_elapsed_ticks(values[2*i].ticks,values[2*i+1].ticks,
        R.gpuTimestampValidBits))*double(R.properties.limits.timestampPeriod);
    add_gpu_scope(slot.gpuScopes[i],ns);
  }
}
static void report_gpu_pass_timestamps(double frames) {
  if(!R.gpuPassTimestampsEnabled) return;
  LOG("[vulkan GPU pass timestamps] %llu available; %llu unavailable; %llu zero; %llu capacity drops; %llu bucket drops; stage intervals NOT GPU busy time%s",
      (unsigned long long)gpuScopeTotals.available,(unsigned long long)gpuScopeTotals.unavailable,
      (unsigned long long)gpuScopeTotals.zero,(unsigned long long)gpuScopeTotals.dropped,
      (unsigned long long)gpuScopeTotals.bucketDrops,
      gpuScopeTotals.available && gpuScopeTotals.available==gpuScopeTotals.zero
          ? "; all-zero/emulated timing UNUSABLE" : "");
  std::array<uint32_t,256> order{};
  for(uint32_t i=0;i<gpuScopeBucketCount;++i) order[i]=i;
  std::sort(order.begin(),order.begin()+gpuScopeBucketCount,[](uint32_t a,uint32_t b) {
    return gpuScopeBuckets[a].intervalNs>gpuScopeBuckets[b].intervalNs;
  });
  for(uint32_t rank=0;rank<std::min(gpuScopeBucketCount,8u);++rank) {
    const auto& b=gpuScopeBuckets[order[rank]];const auto& m=b.metadata;
    char formats[128]{};size_t used=0;
    for(uint32_t i=0;i<m.colors.size();++i) if(m.colors[i]!=VK_FORMAT_UNDEFINED)
      used+=size_t(std::snprintf(formats+used,sizeof(formats)-used,"%s%u:%u",used?",":"",i,uint32_t(m.colors[i])));
    LOG("[vulkan GPU scope %u] %s %ux%ux%u; colors [%s] depth %u; mips %u layers %u aspects 0x%x depth-only %u; %.3f ms/frame %.3f mean %.3f max ms; %.2f scopes/frame %.2f draws/frame",
        rank+1,m.kind?"feedback":"render",m.extent.width,m.extent.height,m.extent.depth,
        formats,uint32_t(m.depth),m.mips,m.layers,m.aspects,m.depthOnly,
        b.intervalNs/1e6/frames,b.intervalNs/1e6/double(b.count),b.maxNs/1e6,
        b.count/frames,b.draws/frames);
  }
  gpuScopeBuckets={};gpuScopeBucketCount=0;gpuScopeTotals={};
}
static void report_gpu_timestamps() {
  static uint64_t previousFrame = 0;
  if (!R.gpuTimestampsEnabled || R.frame - previousFrame < 120) return;
  const auto& stats = R.gpuTimestampStats;
  const double frames = double(R.frame - previousFrame);
  LOG("[vulkan GPU timestamps] submission interval sum %.3f ms/frame; max %.3f ms/submission; %llu submissions; %llu unavailable; %llu zero; NOT GPU busy time%s",
      stats.intervalNs / 1e6 / frames, stats.maxIntervalNs / 1e6,
      (unsigned long long)stats.submissions, (unsigned long long)stats.unavailable,
      (unsigned long long)stats.zeroIntervals,
      stats.submissions && stats.zeroIntervals == stats.submissions
          ? "; all-zero/emulated timing UNUSABLE" : "");
  report_gpu_pass_timestamps(frames);
  R.gpuTimestampStats = {};
  previousFrame = R.frame;
}
VkCommandBuffer command_buffer() {
  if (!R.recording) {
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vk_check(vkBeginCommandBuffer(R.cmd, &bi), "begin command buffer");
    R.recording = true;
    begin_overlay_gpu_queries(R.cmd, R.submissions[R.activeSubmission]);
    if (R.gpuTimestampsEnabled) {
      auto& slot = R.submissions[R.activeSubmission];
      vkCmdResetQueryPool(R.cmd, slot.timestampQueries, 0,
          R.gpuPassTimestampsEnabled ? 2 + 2 * Renderer::Submission::maxGpuScopes : 2);
      vkCmdWriteTimestamp(R.cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, slot.timestampQueries, 0);
      slot.timestampRecorded = true;
      slot.timestampFrame = R.frame;
    }
  }
  return R.cmd;
}
static GpuScopeToken begin_gpu_scope(const GpuScopeMetadata& metadata) {
  if(!R.gpuTimestampsEnabled || !R.gpuPassTimestampsEnabled) return {};
  if(R.rendering) { ++gpuScopeTotals.dropped;return {}; }
  const VkCommandBuffer cmd=command_buffer();
  auto& slot=R.submissions[R.activeSubmission];
  if(slot.gpuScopeCount==Renderer::Submission::maxGpuScopes) { ++gpuScopeTotals.dropped;return {}; }
  const uint32_t index=slot.gpuScopeCount++;
  slot.gpuScopes[index]={metadata,0,R.submissionGeneration,false};
  vkCmdWriteTimestamp(cmd,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,slot.timestampQueries,2+2*index);
  return {R.submissionGeneration,index};
}
static void end_gpu_scope(GpuScopeToken token) {
  if(token.index==UINT32_MAX || token.generation!=R.submissionGeneration) return;
  auto& slot=R.submissions[R.activeSubmission];
  if(token.index>=slot.gpuScopeCount || slot.gpuScopes[token.index].ended ||
      slot.gpuScopes[token.index].generation!=token.generation) return;
  if(R.rendering) { ++gpuScopeTotals.dropped;return; }
  vkCmdWriteTimestamp(R.cmd,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,slot.timestampQueries,3+2*token.index);
  slot.gpuScopes[token.index].ended=true;
}
void gpu_begin_render_scope(const std::array<Surface*,8>& colors,Surface* depth,
                            const uint32_t* slices,uint32_t depthSlice,uint32_t width,uint32_t height) {
  if(!R.gpuTimestampsEnabled || !R.gpuPassTimestampsEnabled) return;
  GpuScopeMetadata metadata;metadata.extent={width,height,1};metadata.depthOnly=true;
  Surface* target=depth;
  for(uint32_t i=0;i<colors.size();++i) if(colors[i]) {
    metadata.colors[i]=colors[i]->fmt.pixel;metadata.slices[i]=slices[i];
    ++metadata.colorCount;metadata.depthOnly=false;if(target==depth) target=colors[i];
  }
  if(depth) { metadata.depth=depth->fmt.pixel;metadata.depthSlice=depthSlice; }
  if(target) { metadata.mips=target->mips;metadata.layers=target->arrayLayers;metadata.aspects=target->aspect; }
  auto token=begin_gpu_scope(metadata);
  R.submissions[R.activeSubmission].activeRenderScope=token.index;
}
void gpu_count_render_draw() {
  if(!R.gpuTimestampsEnabled || !R.gpuPassTimestampsEnabled) return;
  auto& slot=R.submissions[R.activeSubmission];
  if(slot.activeRenderScope<slot.gpuScopeCount) {
    auto& scope=slot.gpuScopes[slot.activeRenderScope];
    if(scope.generation==R.submissionGeneration && !scope.ended) ++scope.draws;
  }
}
GpuScopeToken gpu_begin_feedback_scope(const Surface& source) {
  if(!R.gpuTimestampsEnabled || !R.gpuPassTimestampsEnabled) return {};
  GpuScopeMetadata metadata;metadata.kind=1;metadata.extent=source.extent;
  metadata.mips=source.mips;metadata.layers=source.arrayLayers;metadata.aspects=source.aspect;
  if(source.fmt.depth) { metadata.depth=source.fmt.pixel;metadata.depthOnly=true; }
  else { metadata.colors[0]=source.fmt.pixel;metadata.colorCount=1; }
  return begin_gpu_scope(metadata);
}
void gpu_end_feedback_scope(GpuScopeToken token) { end_gpu_scope(token); }
void end_encoder() {
  if (R.rendering) {
    vkCmdEndRendering(R.cmd);
    R.rendering = false;
    R.passTracked = false;
    if(R.gpuPassTimestampsEnabled) {
      auto& slot=R.submissions[R.activeSubmission];
      end_gpu_scope({R.submissionGeneration,slot.activeRenderScope});
      slot.activeRenderScope=UINT32_MAX;
    }
  }
}
bool narrow_barriers() {
  static const bool enabled = [] {
    const char* e = std::getenv("WWHD_VK_NARROW_BARRIERS");
    // Precise resource dependencies: on by default on Android (tile-based GPUs, where the gain is
    // expected; device numbers pending), opt-in on desktop until native Windows/Linux drivers have
    // been checked (neutral on MoltenVK). WWHD_VK_NARROW_BARRIERS=1|0 overrides it everywhere.
    if (e && *e) return std::strcmp(e, "0") != 0;
    return false;
  }();
  return enabled;
}
void transition_buffer(Buffer& buffer, VkPipelineStageFlags stage, VkAccessFlags access) {
  auto d = derive_dependency(buffer.use, stage, access);
  if (!d.needed) return;
  end_encoder();
  VkBufferMemoryBarrier b{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
  b.srcAccessMask = d.sourceAccess;
  b.dstAccessMask = access;
  b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.buffer = buffer.buffer;
  b.size = VK_WHOLE_SIZE;
  vkCmdPipelineBarrier(command_buffer(), d.source, d.destination, 0, 0, nullptr, 1, &b, 0, nullptr);
}
void transition_image(Surface *s, VkImageLayout layout,
                      VkPipelineStageFlags stage, VkAccessFlags access) {
  const bool conservativePassBreak = s->layout != layout ||
      layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL || access != VK_ACCESS_SHADER_READ_BIT;
  if (narrow_barriers() && stage == VK_PIPELINE_STAGE_ALL_COMMANDS_BIT &&
      access == (VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT) &&
      layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
    stage = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    access = VK_ACCESS_SHADER_READ_BIT;
  }
  const bool changed = s->layout != layout;
  const auto dependency = derive_dependency(s->use, stage, access, changed);
  if (narrow_barriers()) {
    if (!dependency.needed) {
      // Barrier elision must not also change dynamic-rendering pass boundaries.
      if (conservativePassBreak) end_encoder();
      return;
    }
  } else if (!changed && layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL &&
             access == VK_ACCESS_SHADER_READ_BIT) return;
  end_encoder();
  VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  b.oldLayout = s->layout;
  b.newLayout = layout;
  b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.image = s->image;
  b.subresourceRange = {s->aspect, 0, s->mips, 0, s->arrayLayers};
  b.srcAccessMask = narrow_barriers() ? dependency.sourceAccess :
      (s->layout == VK_IMAGE_LAYOUT_UNDEFINED ? 0 : VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
  b.dstAccessMask = access;
  vkCmdPipelineBarrier(command_buffer(), narrow_barriers() ? dependency.source :
      (s->layout == VK_IMAGE_LAYOUT_UNDEFINED ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_ALL_COMMANDS_BIT),
      stage, 0, 0, nullptr, 0, nullptr, 1, &b);
  s->layout = layout;
}
static void cleanup_submission(Renderer::Submission& slot) {
  for (auto& complete : slot.completions) complete();
  slot.completions.clear();
  // The submit fence has completed; slices can now be overwritten safely.
  for (auto& block : slot.uploadBlocks) block.used = 0;
  for (auto b : slot.garbageBuffers) {
    if (b.mapped)
      vkUnmapMemory(R.device, b.memory);
    vkDestroyBuffer(R.device, b.buffer, nullptr);
    vkFreeMemory(R.device, b.memory, nullptr);
  }
  slot.garbageBuffers.clear();
  buffer_cache_free(slot.garbageCacheRegions);
  slot.garbageCacheRegions.clear();
  for (auto &i : slot.garbageImages) {
    for (auto v : i.views)
      if (v)
        vkDestroyImageView(R.device, v, nullptr);
    if (i.image)
      vkDestroyImage(R.device, i.image, nullptr);
    if (i.memory)
      vkFreeMemory(R.device, i.memory, nullptr);
  }
  slot.garbageImages.clear();
  vk_check(vkResetCommandPool(R.device, slot.commandPool, 0),
           "reset command pool");
  vk_check(vkResetDescriptorPool(R.device, slot.descriptorPool, 0),
           "reset descriptor pool");
  ++R.submissionGeneration;
}
static void retire_submission(Renderer::Submission& slot, WaitTiming& timing=submitWait) {
  if (!slot.pending) return;
  VkResult status=timed_call(fenceStatusTiming,[&] { return vkGetFenceStatus(R.device,slot.fence); });
  if (status==VK_NOT_READY) {
    vk_check(timed_call(timing,[&] {
      return vkWaitForFences(R.device,1,&slot.fence,VK_TRUE,UINT64_MAX);
    }),"wait submission retirement");
  } else vk_check(status,"submission fence status");
  timed_call(retirementCleanupTiming,[&] {
    collect_gpu_timestamp_queries(slot);
    if(R.gpuPassTimestampsEnabled) collect_gpu_pass_queries(slot);
    cleanup_submission(slot);
    return VK_SUCCESS;
  });
  slot.pending=false;
}
static void activate_submission(size_t index) {
  auto& slot=R.submissions[index];
  retire_submission(slot);
  R.activeSubmission=index;
  slot.gpuScopeCount=0;slot.activeRenderScope=UINT32_MAX;
  ++R.submissionGeneration;
  R.commandPool=slot.commandPool;R.cmd=slot.cmd;R.fence=slot.fence;
  R.descriptorPool=slot.descriptorPool;
  R.uploadBlocks=std::move(slot.uploadBlocks);
  R.garbageBuffers=std::move(slot.garbageBuffers);
  R.garbageImages=std::move(slot.garbageImages);
  R.garbageCacheRegions=std::move(slot.garbageCacheRegions);
  R.recording=false;R.rendering=false;R.passTracked=false;
}
static void drain_submissions() {
  // A fence signal covers earlier submissions on this same graphics queue.
  // Wait once at the newest pending fence rather than waking at each older
  // fence in ring-array order. Keep each slot's status check and fenced cleanup.
  Renderer::Submission* newest=nullptr;
  for (auto& slot:R.submissions)
    if (slot.pending && (!newest || slot.serial>newest->serial)) newest=&slot;
  if (newest) retire_submission(*newest);
  for (auto& slot:R.submissions) retire_submission(slot);
}
static void submit(VkSemaphore wait = VK_NULL_HANDLE,
                   VkSemaphore signal = VK_NULL_HANDLE, bool asynchronous=false) {
  if (!R.recording)
    return;
  end_encoder();
  auto& recordingSlot = R.submissions[R.activeSubmission];
  if (recordingSlot.perfQuery != UINT32_MAX) {
    auto& q = overlayGpuQueries[recordingSlot.perfQuery];
    vkCmdWriteTimestamp(R.cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, q.pool, 1);
    recordingSlot.perfQuery = UINT32_MAX;
  }
  if (recordingSlot.timestampRecorded)
    vkCmdWriteTimestamp(R.cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                        recordingSlot.timestampQueries, 1);
  vk_check(timed_call(commandEndTiming,[&] { return vkEndCommandBuffer(R.cmd); }), "end command buffer");
  vk_check(timed_call(fenceResetTiming,[&] { return vkResetFences(R.device, 1, &R.fence); }), "reset fence");
  std::array<VkPipelineStageFlags,2> stages{VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT};
  std::array<VkSemaphore,2> waits{};
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &R.cmd;
  if(wait)waits[si.waitSemaphoreCount++]=wait;
  si.pWaitSemaphores=waits.data();si.pWaitDstStageMask=stages.data();
  std::array<VkSemaphore,2> signals{};
  if(signal)signals[si.signalSemaphoreCount++]=signal;
  si.pSignalSemaphores=signals.data();
  vk_check(timed_call(queueSubmitTiming, [&] {
    return vkQueueSubmit(R.queue, 1, &si, R.fence);
  }), "submit graphics");
  R.recording=false;
  auto& slot=R.submissions[R.activeSubmission];
  static uint64_t nextSubmissionSerial=0;
  slot.serial=++nextSubmissionSerial;
  slot.uploadBlocks=std::move(R.uploadBlocks);
  slot.completions=std::move(R.completions);
  slot.garbageBuffers=std::move(R.garbageBuffers);
  slot.garbageImages=std::move(R.garbageImages);
  slot.garbageCacheRegions=std::move(R.garbageCacheRegions);
  slot.pending=true;
  if (asynchronous) {
    activate_submission((R.activeSubmission+1)%R.submissions.size());
  } else {
    retire_submission(slot,wait ? presentSubmitWait : submitWait);
    activate_submission(R.activeSubmission);
  }
}
void flush_async() {
  // Deferred objects may reference earlier queued work even if this slot has
  // no draw commands. Submit an empty command buffer to retire them in order.
  if (!R.recording && (!R.garbageBuffers.empty() || !R.garbageImages.empty() ||
                       !R.garbageCacheRegions.empty())) command_buffer();
  submit(VK_NULL_HANDLE,VK_NULL_HANDLE,true);
}
void flush() {
  if (!R.recording && (!R.garbageBuffers.empty() || !R.garbageImages.empty() ||
                       !R.garbageCacheRegions.empty())) command_buffer();
  submit();
  drain_submissions();
}
static VkResult device_wait_idle() {
  return vkDeviceWaitIdle(R.device);
}
void wait_idle() {
  flush();
  vk_check(device_wait_idle(), "device idle");
}
void with_autorelease_pool(void (*fn)()) { fn(); }
uint64_t frames_completed() { return R.completed.load(); }
uint64_t frame_count() { return std::atomic_ref<uint64_t>(R.frame).load(); }
static bool has_extension(const std::vector<VkExtensionProperties> &es,
                          const char *name) {
  return std::any_of(es.begin(), es.end(),
                     [&](auto &e) { return !strcmp(e.extensionName, name); });
}
static void make_swapchain(Screen &s) {
  vk_check(device_wait_idle(), "resize device idle");
  VkSurfaceCapabilitiesKHR caps;
  vk_check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(R.physicalDevice,
                                                     s.surface, &caps),
           "surface capabilities");
  uint32_t n = 0;
  vk_check(vkGetPhysicalDeviceSurfaceFormatsKHR(R.physicalDevice, s.surface, &n,
                                       nullptr),"surface format count");
  std::vector<VkSurfaceFormatKHR> fs(n);
  const VkResult formats=vkGetPhysicalDeviceSurfaceFormatsKHR(R.physicalDevice,s.surface,&n,fs.data());
  if(formats!=VK_INCOMPLETE)vk_check(formats,"surface formats");
  fs.resize(n);
  if (fs.empty())
    throw std::runtime_error("No presentation formats");
  // Presentation uses image blits, so both advertised surface support and
  // optimal-tiling BLIT_DST format support must hold.
  if (fs.size() == 1 && fs[0].format == VK_FORMAT_UNDEFINED)
    fs[0].format = VK_FORMAT_B8G8R8A8_SRGB;
  VkSurfaceFormatKHR format{};
  bool found = false;
  const VkFormat preferred = VK_FORMAT_B8G8R8A8_SRGB;
  for (auto f : fs) {
    VkFormatProperties props;
    vkGetPhysicalDeviceFormatProperties(R.physicalDevice, f.format, &props);
    if (!(props.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT))
      continue;
    if (!found || f.format == preferred) {
      format = f;
      found = true;
    }
    if (f.format == preferred &&
        f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
      break;
  }
  if (!found)
    throw std::runtime_error("No swapchain format supports image blits");
  VkExtent2D extent = caps.currentExtent;
  if (extent.width == UINT32_MAX)
    extent = {std::clamp<uint32_t>(std::max(s.width.load(), 1),
                                   caps.minImageExtent.width,
                                   caps.maxImageExtent.width),
              std::clamp<uint32_t>(std::max(s.height.load(), 1),
                                   caps.minImageExtent.height,
                                   caps.maxImageExtent.height)};
  const VkSurfaceTransformFlagBitsKHR transform = caps.currentTransform;
  if (!(caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT))
    throw std::runtime_error("Swapchain cannot receive scan-buffer blits");
  VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
  ci.surface = s.surface;
  ci.minImageCount = std::max(caps.minImageCount, 2u);
  if (caps.maxImageCount)
    ci.minImageCount = std::min(ci.minImageCount, caps.maxImageCount);
  ci.imageFormat = format.format;
  ci.imageColorSpace = format.colorSpace;
  ci.imageExtent = extent;
  ci.imageArrayLayers = 1;
  VkFormatProperties presentationFormat;
  vkGetPhysicalDeviceFormatProperties(R.physicalDevice,format.format,&presentationFormat);
  const bool shaderPresentation = (caps.supportedUsageFlags & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) &&
      (presentationFormat.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT);
  const bool captureTransfer = present_capture_requested() &&
      (caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) &&
      (presentationFormat.optimalTilingFeatures & VK_FORMAT_FEATURE_TRANSFER_SRC_BIT);
  ci.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT |
      (shaderPresentation ? VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT : 0) |
      (captureTransfer ? VK_IMAGE_USAGE_TRANSFER_SRC_BIT : 0);
  ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  ci.preTransform = transform;
  ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  for (auto flag : {VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
                    VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
                    VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
                    VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR})
    if (caps.supportedCompositeAlpha & flag) {
      ci.compositeAlpha = flag;
      break;
    }
  uint32_t modeCount = 0;
  vk_check(vkGetPhysicalDeviceSurfacePresentModesKHR(R.physicalDevice,
      s.surface, &modeCount, nullptr), "presentation mode count");
  std::vector<VkPresentModeKHR> modes(modeCount);
  vk_check(vkGetPhysicalDeviceSurfacePresentModesKHR(R.physicalDevice,
      s.surface, &modeCount, modes.data()), "presentation modes");
  // Presentation (Graphics > Presentation in the settings overlay, WWHD_VK_PRESENT_MODE): FIFO (vsync)
  // by default; mailbox (low latency) or immediate (may tear) when chosen and the surface offers them.
  // Guest GX2 pacing still controls game flips in every mode.
  static const VkPresentModeKHR kModes[kPresentModes] = {VK_PRESENT_MODE_FIFO_KHR, VK_PRESENT_MODE_MAILBOX_KHR,
                                                         VK_PRESENT_MODE_IMMEDIATE_KHR};
  unsigned offered = 0;
  std::string offeredNames;
  for (int m = 0; m < kPresentModes; m++)
    if (std::find(modes.begin(), modes.end(), kModes[m]) != modes.end()) {
      offered |= 1u << m;
      offeredNames += std::string(offeredNames.empty() ? "" : ", ") + present_mode_name(m);
    }
  if (&s == &R.tv) set_present_modes_offered(offered);
  const int wanted = effective_present_mode();
  const int chosen = offered >> wanted & 1 ? wanted : kPresentFifo;
  ci.presentMode = kModes[chosen];
  if (chosen != s.presentMode || wanted != s.presentWanted)
    LOG("[vulkan] %s present mode %s (available: %s)%s", &s == &R.tv ? "TV" : "GamePad", present_mode_name(chosen),
        offeredNames.c_str(), chosen != wanted ? " - the requested mode is not offered" : "");
  s.presentMode = chosen;
  s.presentWanted = wanted;
  ci.clipped = VK_TRUE;
  ci.oldSwapchain = s.swapchain;
  VkSwapchainKHR sc;
  vk_check(vkCreateSwapchainKHR(R.device, &ci, nullptr, &sc),
           "create swapchain");
  {
    reset_present_screen(s); // The compatibility path drained the device above.
    if (s.swapchain) vkDestroySwapchainKHR(R.device, s.swapchain, nullptr);
  }
  s.swapchain = sc;
  ++s.swapchainGeneration;
  s.swapFormat = format.format;
  s.swapExtent = extent;
  vkGetSwapchainImagesKHR(R.device, sc, &n, nullptr);
  s.images.resize(n);
  vkGetSwapchainImagesKHR(R.device, sc, &n, s.images.data());
  s.layouts.assign(n, VK_IMAGE_LAYOUT_UNDEFINED);
  s.imageUses.assign(n, {});
  prepare_present_screen(s,shaderPresentation,captureTransfer);
  s.resize = false;
}
// Asynchronous presentation (the default on every platform since 2026-10-07; WWHD_VK_ASYNC_PRESENT=0
// restores the waiting path): the presentation submission goes into the four-slot ring like GX2Flush
// work instead of waiting for the GPU, and swap() does not drain the queue, so the render thread
// records frame N+1 while the GPU draws frame N. Each frame in flight has its own acquire semaphore
// (reused only after the submission that waited on it retired) and each swapchain image its own
// render-finished semaphore. Captures and frame dumps keep the waiting path.
static bool async_present() {
  static const bool on = [] {
    const char *e = std::getenv("WWHD_VK_ASYNC_PRESENT");
    return !e || std::atoi(e) != 0;
  }();
  return on;
}
struct AsyncPresentState {
  std::array<VkSemaphore, 3> acquire{};
  std::array<uint64_t, 3> serial{};  // submission that waited on acquire[k]
  std::array<size_t, 3> slot{};
  unsigned next = 0;
  std::vector<VkSemaphore> finished;  // per swapchain image
};
static AsyncPresentState asyncPresent[2];  // TV, GamePad
static VkSemaphore new_semaphore() {
  VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  VkSemaphore sem;
  vk_check(vkCreateSemaphore(R.device, &si, nullptr, &sem), "create presentation semaphore");
  return sem;
}
static bool screen_has_window(const Screen& s) {
  return s.window != nullptr;
}
static void present(Screen &s) {
  constexpr bool workerAcquired=false;
  // Presentation changed (settings overlay): a new swapchain, as for a resize (also for a window that
  // is not shown right now, so the next frame it shows uses the new mode)
  if (!workerAcquired && screen_has_window(s) && s.swapchain && s.presentWanted != effective_present_mode())
    make_swapchain(s);
  Surface* scan = presentation_source_screen(s).scan.get();
  if (!screen_has_window(s) || !s.visible || s.width <= 0 || s.height <= 0 || !scan ||
      !scan->image)
    return;
  const auto& sourceFormat = format_properties(scan->fmt.pixel);
  if (!(sourceFormat.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT))
    throw std::runtime_error(
        "Scan-buffer format does not support presentation blits");
  if (scan->fmt.kind != FormatInfo::FLOAT)
    throw std::runtime_error("Integer scan buffers cannot be blitted to "
                             "normalized presentation images");
  VkFilter filter = (sourceFormat.optimalTilingFeatures &
                     VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)
                        ? VK_FILTER_LINEAR
                        : VK_FILTER_NEAREST;
  if (s.resize || !s.swapchain)
    make_swapchain(s);
  uint32_t index;
  auto& timing = screenTiming[&s == &R.tv ? 0 : 1];
  bool async = async_present() && !present_capture_requested();
  // The GamePad may run at a different refresh rate or stop consuming images
  // entirely. Never wait for its compositor on the main render thread.
  const bool secondary = &s == &R.drc;
  AsyncPresentState &ap = asyncPresent[&s == &R.tv ? 0 : 1];
  VkSemaphore acquireSemaphore = s.acquired;
  unsigned acquireIndex = 0;
  if (async || workerAcquired) {
    acquireIndex = ap.next;
    if(!workerAcquired)ap.next = (ap.next + 1) % ap.acquire.size();
    if (!ap.acquire[acquireIndex])
      ap.acquire[acquireIndex] = new_semaphore();
    // the submission that last waited on this semaphore must be done with it
    auto &previous = R.submissions[ap.slot[acquireIndex]];
    if (ap.serial[acquireIndex] && previous.pending && previous.serial == ap.serial[acquireIndex]) {
      if (secondary) {
        const VkResult ready = timed_call(fenceStatusTiming,[&] { return vkGetFenceStatus(R.device, previous.fence); });
        if (ready == VK_NOT_READY)
          return;
        vk_check(ready, "secondary acquire semaphore completion");
      }
      retire_submission(previous);
    }
    acquireSemaphore = ap.acquire[acquireIndex];
  }
  VkResult ar;
  ar = timed_call(timing.acquire, [&] {
    return vkAcquireNextImageKHR(R.device, s.swapchain, secondary ? 0 : UINT64_MAX,
                                acquireSemaphore, VK_NULL_HANDLE, &index);
  });
  // No image was acquired and the semaphore remains unsignalled. Leave it
  // available for another frame; submitting a wait here would deadlock.
  if (ar == VK_NOT_READY || ar == VK_TIMEOUT)
    return;
  if (ar == VK_ERROR_OUT_OF_DATE_KHR) {
    s.resize = true;
    return;
  }
  if (ar != VK_SUBOPTIMAL_KHR)
    vk_check(ar, "acquire scan image");
  bool composed=false;
  timed_call(timing.compose, [&] {
    composed=draw_present_screen(s,index);
    return VK_SUCCESS;
  });
  if (!composed) {
    transition_image(scan, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                     VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.oldLayout = s.layouts[index];
    b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = s.images[index];
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    auto cmd = command_buffer();
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &b);
    VkClearColorValue black{};
    vkCmdClearColorImage(cmd, s.images[index],
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1,
                         &b.subresourceRange);
    // Clear and blit overlap the swap image. Order the two transfer writes.
    b.oldLayout = b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b.srcAccessMask = b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &b);
    const float ratio =
        std::min(float(s.swapExtent.width) / scan->extent.width,
                 float(s.swapExtent.height) / scan->extent.height);
    int w = int(scan->extent.width * ratio),
        h = int(scan->extent.height * ratio);
    int x = (s.swapExtent.width - w) / 2, y = (s.swapExtent.height - h) / 2;
    VkImageBlit blit{};
    blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.srcOffsets[1] = {int(scan->extent.width), int(scan->extent.height),
                          1};
    blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.dstOffsets[0] = {x, y, 0};
    blit.dstOffsets[1] = {x + w, y + h, 1};
    vkCmdBlitImage(cmd, scan->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   s.images[index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                   &blit, filter);
    b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.dstAccessMask = 0;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &b);
    s.layouts[index] = b.newLayout;
    s.imageUses[index] = {};
    derive_dependency(s.imageUses[index], VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
  }
  record_present_capture(s,index);
  VkSemaphore finishedSemaphore = s.finished;
  if (async || workerAcquired) {
    if (ap.finished.size() != s.images.size()) {  // new swapchain
      vk_check(device_wait_idle(), "presentation semaphores idle");
      for (VkSemaphore f : ap.finished)
        vkDestroySemaphore(R.device, f, nullptr);
      ap.finished.clear();
      for (size_t i = 0; i < s.images.size(); i++)
        ap.finished.push_back(new_semaphore());
    }
    finishedSemaphore = ap.finished[index];
    const size_t slot = R.activeSubmission;
    submit(acquireSemaphore, finishedSemaphore, async);
    ap.slot[acquireIndex] = slot;
    ap.serial[acquireIndex] = R.submissions[slot].serial;
  } else {
    submit(s.acquired, s.finished);
  }
  finish_present_capture(s);
  VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
  pi.waitSemaphoreCount = 1;
  pi.pWaitSemaphores = &finishedSemaphore;
  pi.swapchainCount = 1;
  pi.pSwapchains = &s.swapchain;
  pi.pImageIndices = &index;
  VkResult pr = timed_call(timing.present, [&] {
    VkResult result=vkQueuePresentKHR(R.queue, &pi);
    return result;
  });
  if (pr == VK_SUCCESS || pr == VK_SUBOPTIMAL_KHR)
    ++s.presented;
  if (pr == VK_ERROR_OUT_OF_DATE_KHR || pr == VK_SUBOPTIMAL_KHR)
    s.resize = true;
  else
    vk_check(pr, "present scan buffer");
  if (!async)
    vk_check(timed_call(timing.idle, [&] { return vkQueueWaitIdle(R.queue); }),
             "present completion");
}
void copy_to_scan(uint32_t cb, uint32_t target) {
  Surface *src = surface_from_color_buffer(cb);
  if (!src)
    return;
  Screen &s = target == 4 ? R.drc : R.tv;
  if (!s.scan || s.scan->extent.width != src->extent.width ||
      s.scan->extent.height != src->extent.height ||
      s.scan->fmt.pixel != src->fmt.pixel) {
    if (s.scan)
      destroy_surface_image(s.scan.get());
    s.scan = std::make_unique<Surface>();
    auto *d = s.scan.get();
    d->width = src->extent.width;
    d->height = src->extent.height;
    d->format = src->format;
    d->fmt = src->fmt;
    d->scale = 1;
    create_surface_image(d, false);
  }
  resample(src, s.scan.get(), 1);
  if (target & 1)
    draw_mod_overlay(*s.scan);  // HUD of gameplay mods (stamina wheel), as the Metal renderer
  mark_gpu_written(s.scan.get());
}
// debug: WWHD_DUMP_FRAMES=100,300 writes the TV image of those frames to frame_<n>.png (and
// frame_<n>_drc.png; WWHD_DUMP_PRESENT=1 adds the composed window, frame_<n>_present.png), as Metal
static void dump_scan(Screen &s, const std::string &path) {
  if (!s.scan || !s.scan->image)
    return;
  try {
    write_rgba_png(path, s.scan->extent.width, s.scan->extent.height,
                   read_surface_rgba(*s.scan, s.srgb.load()));
    LOG("[gfx] wrote %s (%ux%u)", path.c_str(), s.scan->extent.width, s.scan->extent.height);
  } catch (const std::exception &e) {
    LOG("[gfx] cannot write %s: %s", path.c_str(), e.what());
  }
}
// the composed windows (what present() puts on the screen), written after the frame (swap)
static void request_present_dump(const std::string& path) { gfx::request_present_dump(path); }
static std::atomic<bool> captureRequested{false};
void request_capture() { captureRequested = true; }
static void frame_dumps(uint64_t frame) {
  static const auto frames = gfx::capture_schedule::parse(getenv("WWHD_DUMP_FRAMES"));
  if (gfx::capture_schedule::contains(frames, frame)) {
    dump_scan(R.tv, "frame_" + std::to_string(frame) + ".png");
    dump_scan(R.drc, "frame_" + std::to_string(frame) + "_drc.png");
    if (getenv("WWHD_DUMP_PRESENT"))
      request_present_dump("frame_" + std::to_string(frame) + "_present.png");
  }
  // P / F12 (Graphics menu): the pictures of this frame in captures/<time>/ (WWHD_CAPTURE=<frame>
  // scripts it when WWHD_CAPTURE_PATH is not used); the Metal renderer also writes a draw log
  static const uint64_t scripted = getenv("WWHD_CAPTURE") && !getenv("WWHD_CAPTURE_PATH")
                                       ? strtoull(getenv("WWHD_CAPTURE"), nullptr, 10) : ~0ull;
  if (captureRequested.exchange(false) || frame == scripted) {
    char dir[64];
    time_t t = time(nullptr);
    strftime(dir, sizeof dir, "captures/%Y%m%d-%H%M%S", localtime(&t));
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    dump_scan(R.tv, std::string(dir) + "/tv.png");
    dump_scan(R.drc, std::string(dir) + "/gamepad.png");
    request_present_dump(std::string(dir) + "/present.png");
    LOG("[gfx] capture of frame %llu written to %s", (unsigned long long)frame, dir);
  }
}
// Screenshots (screenshot.h): recorded into the frame's command buffer at the swap, read back once
// that submission's fence has signalled (checked at later swaps: no wait), encoded on the screenshot
// thread; the readback buffers come back from it to be freed on this thread.
namespace {
struct PendingShot {
  std::string path;
  bool tv = false;
  uint64_t frame = 0;
  Buffer buffer{};
  uint32_t width = 0, height = 0;
  bool bgra = false;
  size_t slot = 0;
  uint64_t serial = 0;  // 0: not submitted yet
};
std::vector<PendingShot> pendingShots;
std::mutex shotBuffersMu;
std::vector<Buffer> shotBuffersDone;
}  // namespace
void swap() {
  const auto swapPhaseStarted=std::chrono::steady_clock::now();
  service_captures();
  frame_dumps(R.frame + 1);
  // the layout of both pictures (GamePad window, picture-in-picture, automatic overlay, GamePad only)
  // comes from gfx/display_modes.cpp, which both window hosts and the Metal renderer share
  Surface *tvScan = R.tv.scan && R.tv.scan->image ? R.tv.scan.get() : nullptr;
  Surface *drcScan = R.drc.scan && R.drc.scan->image ? R.drc.scan.get() : nullptr;
  gfx::g_filter = scale_filter();  // the SDL host keeps the scaling filter with the Vulkan settings
  const float layerW = float(R.tv.width.load()), layerH = float(R.tv.height.load());
  gfx::PresentPlan plan = gfx::display_plan(
      tvScan, tvScan ? float(tvScan->extent.width) : 0, tvScan ? float(tvScan->extent.height) : 0,
      drcScan, drcScan ? float(drcScan->extent.width) : 0, drcScan ? float(drcScan->extent.height) : 0,
      layerW, layerH, R.frame + 1);
  set_present_plan(&plan);
  set_overlay_draw(nullptr);  // no overlay drawn yet (the host menu will be)
  compose_headless_diagnostic(R.tv);
  bool sampled[2] = {};
  if (plan.sample_auto && drcScan) {
    sampled[0] = record_signature(0, *drcScan, R.drc.srgb.load());
    sampled[1] = sampled[0] && tvScan && record_signature(1, *tvScan, R.tv.srgb.load());
  }
  if(perf_enabled()) {
    ++swapPhaseTiming[0].count;
    swapPhaseTiming[0].ns+=std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now()-swapPhaseStarted).count();
  }
  timed_call(swapPhaseTiming[1],[&] { present(R.tv);return VK_SUCCESS; });
  if (plan.drc_window)
    timed_call(swapPhaseTiming[2],[&] { present(R.drc);return VK_SUCCESS; });
  // asynchronous presentation: queued like GX2Flush work, the ring's fences retire it (the automatic
  // overlay's signatures are read back right away, so those frames wait; present dumps and captures
  // read back through flush()). Both window hosts: the AppKit host presents to its CAMetalLayers
  // through the same swapchain path.
  timed_call(swapPhaseTiming[3],[&] {
    if (async_present() && !sampled[0]) flush_async();
    else flush();
    return VK_SUCCESS;
  });
  const auto swapBookkeepingStarted=std::chrono::steady_clock::now();
  if (sampled[0]) {
    std::vector<float> d = read_signature(0), t = sampled[1] ? read_signature(1) : std::vector<float>{};
    gfx::display_auto_signature(d, sampled[1] ? &t : nullptr, R.frame + 1);
  }
  for (auto &path : gfx::display_take_present_dumps()) {
    if (!tvScan || plan.dw < 1 || plan.dh < 1)
      continue;
    try {
      write_rgba_png(path, uint32_t(plan.dw), uint32_t(plan.dh),
                     compose_offscreen(R.tv, uint32_t(plan.dw), uint32_t(plan.dh), R.tv.srgb.load()));
      // the GamePad window's size: its swapchain, or (SDL host test runs with hidden windows, which
      // present nothing) the window's pixel size
      uint32_t drcW = R.drc.swapchain ? R.drc.swapExtent.width : 0, drcH = R.drc.swapchain ? R.drc.swapExtent.height : 0;
      if (!R.drc.swapchain && R.drc.window)
        drcW = uint32_t(std::max(0, R.drc.width.load())), drcH = uint32_t(std::max(0, R.drc.height.load()));
      if (plan.drc_window && drcScan && drcW && drcH) {
        std::string p = path;
        size_t dot = p.rfind(".png");
        p.insert(dot == std::string::npos ? p.size() : dot, "_drc");
        write_rgba_png(p, drcW, drcH, compose_offscreen(R.drc, drcW, drcH, R.drc.srgb.load()));
      }
      gfx::display_log_present_dump(path, plan, float(tvScan->extent.width), float(tvScan->extent.height));
    } catch (const std::exception &e) {
      LOG("[gfx] present dump %s failed: %s", path.c_str(), e.what());
    }
  }
  set_present_plan(nullptr);
  std::atomic_ref<uint64_t>(R.frame).fetch_add(1);
  R.completed = R.frame;
  buffer_cache_end_frame();
  report_gpu_timestamps();
  checkpoint_pipeline_cache();
  vk::checkpoint_shader_cache(R.frame);
  latch_res_scale();
  if(perf_enabled()) {
    ++swapPhaseTiming[4].count;
    swapPhaseTiming[4].ns+=std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now()-swapBookkeepingStarted).count();
  }
  if (perf_enabled() || cpu_only_stats_enabled()) {
    static auto start = std::chrono::steady_clock::now();
    static auto previousSwap = start;
    static bool havePreviousSwap = false;
    static std::array<double, 120> swapIntervals{};
    static size_t intervalCount = 0, slowIntervals = 0;
    static uint64_t frame=0,draws=0,bytes=0,allocs=0,passes=0;
    static uint64_t descriptorLookups=0,descriptorHits=0,descriptorAllocations=0,descriptorFastHits=0;
    static uint64_t vertexDeclaredBytes=0,vertexCopiedBytes=0;
    static uint64_t vertexReuseChecks=0,vertexReuseHits=0,vertexReuseBytes=0,vertexReuseCompareNs=0;
    static uint64_t batchSubmissions=0,fetchLookups=0,fetchLastHits=0;
    static uint64_t stateHashLookups=0,stateHashMemoHits=0,stateHashBytes=0;
    static uint64_t vertexBindCalls=0,vertexBindSkips=0;
    static uint64_t vertexHistoryRequests=0,vertexHistoryMatches=0,vertexHistoryBytes=0;
    static uint64_t vertexHistoryReuseChecks=0,vertexHistoryReuseHits=0,vertexHistoryReuseBytes=0;
    static std::array<uint64_t,7> vertexHistoryDistances{};
    static Renderer::CpuPreparationStats previousPreparation;
    static SparseHashStats previousSparse;
    static gx2::ShaderKeyDirtyStats previousShaderDirty;
    auto threadCpuNs = []() -> uint64_t {
      timespec time{};
      if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &time)) return 0;
      return uint64_t(time.tv_sec) * 1000000000ull + uint64_t(time.tv_nsec);
    };
    static uint64_t previousThreadCpu = threadCpuNs();
    auto now=std::chrono::steady_clock::now();
    if (havePreviousSwap && intervalCount < swapIntervals.size()) {
      double ms = std::chrono::duration<double, std::milli>(now-previousSwap).count();
      swapIntervals[intervalCount++] = ms;
      if (ms > 40.0) ++slowIntervals;
    }
    previousSwap=now;havePreviousSwap=true;
    if (R.frame-frame>=120) {
      double seconds=std::chrono::duration<double>(now-start).count();
      const uint64_t currentThreadCpu = threadCpuNs();
      if (previousThreadCpu && currentThreadCpu >= previousThreadCpu)
        LOG("[vulkan thread CPU] %.3f ms/frame (excludes sleeping and GPU waits)",
            double(currentThreadCpu - previousThreadCpu) / 1e6 / (R.frame - frame));
      previousThreadCpu = currentThreadCpu;
      LOG("[vulkan perf] frame %llu %.2f fps, %.1f draws/frame, %.2f MiB uploads/frame, %llu new upload blocks, %.1f passes/frame",
          (unsigned long long)R.frame,(R.frame-frame)/seconds,
          double(R.drawCount-draws)/(R.frame-frame),double(R.uploadBytes-bytes)/(R.frame-frame)/(1<<20),
          (unsigned long long)(R.uploadAllocations-allocs), double(R.renderPassCount-passes)/(R.frame-frame));
      if (intervalCount) {
        std::sort(swapIntervals.begin(), swapIntervals.begin()+intervalCount);
        // Nearest-rank percentiles; the first report contains 119 intervals.
        size_t p50=(intervalCount+1)/2-1, p95=(intervalCount*95+99)/100-1;
        LOG("[vulkan pacing] %llu intervals p50 %.2f ms p95 %.2f ms max %.2f ms; >40 ms %llu",
            (unsigned long long)intervalCount,swapIntervals[p50],swapIntervals[p95],
            swapIntervals[intervalCount-1],(unsigned long long)slowIntervals);
      }
      intervalCount=0;slowIntervals=0;
      auto ss=vk::shader_stats();
      LOG("[vulkan shaders] lookups %llu last hits %llu variants %llu compiles %llu total compile %.1f ms; %llu programs, %llu keys, %llu shaders (%llu keys shared), %llu sharing a module",
          (unsigned long long)ss.lookups,(unsigned long long)ss.lastHits,(unsigned long long)ss.variantHits,(unsigned long long)ss.compiles,ss.compileNs/1e6,
          (unsigned long long)ss.programs,(unsigned long long)ss.variantKeys,(unsigned long long)ss.shaders,(unsigned long long)ss.variantAliases,(unsigned long long)ss.moduleAliases);
      if (ss.verifyChecks || ss.verifySplitKeys)
        LOG("[vulkan shader key verify] %llu pre-narrowing keys checked, %llu violations, %llu split; %.1f ms",
            (unsigned long long)ss.verifyChecks,(unsigned long long)ss.verifyViolations,
            (unsigned long long)ss.verifySplitKeys,ss.verifyNs/1e6);
      LOG("[vulkan shader disk] hits %llu; memory reuses %llu; SPIR-V compiles %llu %.1f ms; decompile %.1f ms; loads %llu %.1f ms; saves %llu worker %.1f ms %.2f MiB; snapshots %.1f ms",
          (unsigned long long)ss.diskHits,(unsigned long long)ss.spirvReuseHits,(unsigned long long)ss.spirvCompiles,ss.spirvCompileNs/1e6,
          ss.decompileNs/1e6,(unsigned long long)ss.diskLoads,ss.diskLoadNs/1e6,
          (unsigned long long)ss.diskSaves,ss.diskSaveNs/1e6,ss.diskSavedBytes/double(1<<20),ss.diskSnapshotNs/1e6);
      LOG("[vulkan pipelines] creates %llu total %.1f ms",
          (unsigned long long)R.pipelineCreates,R.pipelineCreateNs/1e6);
      {
        // texture change detection (surfaces.cpp upload_surface, write_watch.h)
        extern uint64_t g_stat_full_checks, g_stat_uploads;
        static uint64_t checks=0, uploads=0;
        uint64_t faults, protectedPages;
        wwatch::take_stats(faults, protectedPages);
        LOG("[vulkan textures] %llu full checks, %llu uploads, %llu page write faults, %llu pages protected",
            (unsigned long long)(g_stat_full_checks-checks),(unsigned long long)(g_stat_uploads-uploads),
            (unsigned long long)faults,(unsigned long long)protectedPages);
        checks=g_stat_full_checks;uploads=g_stat_uploads;
      }
      buffer_cache_report(double(R.frame-frame));
      // CPU-only reports leave per-draw counters/comparison clocks disabled.
      if (perf_enabled()) {
      double frames = double(R.frame-frame);
      LOG("[vulkan fetch memo] %.1f lookups/frame %.1f last hits/frame",
          (ss.fetchLookups-fetchLookups)/frames,(ss.fetchLastHits-fetchLastHits)/frames);
      fetchLookups=ss.fetchLookups;fetchLastHits=ss.fetchLastHits;
      LOG("[vulkan state hash memo] %.1f lookups/frame %.1f hits/frame %.3f MiB hashed/frame",
          (ss.stateHashLookups-stateHashLookups)/frames,
          (ss.stateHashMemoHits-stateHashMemoHits)/frames,
          (ss.stateHashBytes-stateHashBytes)/frames/(1<<20));
      stateHashLookups=ss.stateHashLookups;
      stateHashMemoHits=ss.stateHashMemoHits;stateHashBytes=ss.stateHashBytes;
      uint64_t currentBatchSubmissions=draw_batch_submissions();
      LOG("[vulkan draw batches] %.2f extra submissions/frame",
          (currentBatchSubmissions-batchSubmissions)/frames);
      batchSubmissions=currentBatchSubmissions;
      LOG("[vulkan vertex snapshots] declared %.2f MiB/frame; prepared %.2f MiB/frame",
          (R.vertexDeclaredBytes-vertexDeclaredBytes)/frames/(1<<20),
          (R.vertexCopiedBytes-vertexCopiedBytes)/frames/(1<<20));
      vertexDeclaredBytes=R.vertexDeclaredBytes;vertexCopiedBytes=R.vertexCopiedBytes;
      LOG("[vulkan vertex binds] %.1f actual/frame %.1f skipped/frame",
          (R.vertexBindCalls-vertexBindCalls)/frames,
          (R.vertexBindSkips-vertexBindSkips)/frames);
      vertexBindCalls=R.vertexBindCalls;vertexBindSkips=R.vertexBindSkips;
      LOG("[vulkan vertex history] %.1f requests/frame %.1f nonconsecutive keys/frame %.3f MiB potential/frame; distances 1..7 %.1f %.1f %.1f %.1f %.1f %.1f %.1f",
          (R.vertexHistoryRequests-vertexHistoryRequests)/frames,
          (R.vertexHistoryMatches-vertexHistoryMatches)/frames,
          (R.vertexHistoryBytes-vertexHistoryBytes)/frames/(1<<20),
          (R.vertexHistoryDistances[0]-vertexHistoryDistances[0])/frames,
          (R.vertexHistoryDistances[1]-vertexHistoryDistances[1])/frames,
          (R.vertexHistoryDistances[2]-vertexHistoryDistances[2])/frames,
          (R.vertexHistoryDistances[3]-vertexHistoryDistances[3])/frames,
          (R.vertexHistoryDistances[4]-vertexHistoryDistances[4])/frames,
          (R.vertexHistoryDistances[5]-vertexHistoryDistances[5])/frames,
          (R.vertexHistoryDistances[6]-vertexHistoryDistances[6])/frames);
      vertexHistoryRequests=R.vertexHistoryRequests;vertexHistoryMatches=R.vertexHistoryMatches;
      vertexHistoryBytes=R.vertexHistoryBytes;vertexHistoryDistances=R.vertexHistoryDistances;
      LOG("[vulkan vertex history reuse] %.1f checks/frame %.1f hits/frame %.3f MiB reused/frame",
          (R.vertexHistoryReuseChecks-vertexHistoryReuseChecks)/frames,
          (R.vertexHistoryReuseHits-vertexHistoryReuseHits)/frames,
          (R.vertexHistoryReuseBytes-vertexHistoryReuseBytes)/frames/(1<<20));
      vertexHistoryReuseChecks=R.vertexHistoryReuseChecks;
      vertexHistoryReuseHits=R.vertexHistoryReuseHits;vertexHistoryReuseBytes=R.vertexHistoryReuseBytes;
      LOG("[vulkan vertex reuse] %.1f checks/frame; %.1f hits/frame; %.2f MiB reused/frame; compare %.3f ms/frame",
          (R.vertexReuseChecks-vertexReuseChecks)/frames,
          (R.vertexReuseHits-vertexReuseHits)/frames,
          (R.vertexReuseBytes-vertexReuseBytes)/frames/(1<<20),
          (R.vertexReuseCompareNs-vertexReuseCompareNs)/frames/1e6);
      vertexReuseChecks=R.vertexReuseChecks;vertexReuseHits=R.vertexReuseHits;
      vertexReuseBytes=R.vertexReuseBytes;vertexReuseCompareNs=R.vertexReuseCompareNs;
      uint64_t descriptorChecks=R.descriptorLookups-descriptorLookups;
      uint64_t descriptorReuses=R.descriptorCacheHits-descriptorHits;
      LOG("[vulkan descriptors] %.1f lookups/frame; %.1f allocations/frame; %.1f%% reused; %.1f%% consecutive",
          descriptorChecks/frames,(R.descriptorAllocations-descriptorAllocations)/frames,
          descriptorChecks ? 100.0*descriptorReuses/descriptorChecks : 0.0,
          descriptorChecks ? 100.0*(R.descriptorFastHits-descriptorFastHits)/descriptorChecks : 0.0);
      descriptorLookups=R.descriptorLookups;descriptorHits=R.descriptorCacheHits;
      descriptorAllocations=R.descriptorAllocations;descriptorFastHits=R.descriptorFastHits;
      const auto& prep=R.cpuPreparation;
      const auto sparse=sparse_hash_stats();
      LOG("[vulkan sparse hash memo] %.1f checks/frame %.1f hits/frame %.3f MiB fresh samples/frame %.1f mixed words/frame %.1f overflow/frame",
          (sparse.checks-previousSparse.checks)/frames,
          (sparse.memoHits-previousSparse.memoHits)/frames,
          (sparse.sampleBytes-previousSparse.sampleBytes)/frames/(1<<20),
          (sparse.mixerWords-previousSparse.mixerWords)/frames,
          (sparse.overflows-previousSparse.overflows)/frames);
      previousSparse=sparse;
      const auto dirty=gx2::shader_key_dirty_stats();
      LOG("[vulkan shader key dirty] %.1f changed batches/frame %.1f baseline bumps/frame %.1f actual bumps/frame %.1f avoided/frame %.1f masked words/frame",
          (dirty.changedBatches-previousShaderDirty.changedBatches)/frames,
          (dirty.baselineWouldBumps-previousShaderDirty.baselineWouldBumps)/frames,
          (dirty.actualBumps-previousShaderDirty.actualBumps)/frames,
          (dirty.avoidedBumps-previousShaderDirty.avoidedBumps)/frames,
          (dirty.maskedWords-previousShaderDirty.maskedWords)/frames);
      previousShaderDirty=dirty;
      LOG("[vulkan sampler memo] %.1f requests/frame %.1f hits/frame %.1f map lookups/frame",
          (prep.samplerRequests-previousPreparation.samplerRequests)/frames,
          (prep.samplerMemoHits-previousPreparation.samplerMemoHits)/frames,
          (prep.samplerMapLookups-previousPreparation.samplerMapLookups)/frames);
      LOG("[vulkan uniform preparation] %.1f packs/frame %.3f MiB/frame; %llu scratch growths %.3f KiB added; %.1f snapshot requests/frame %.3f MiB/frame",
          (prep.packCalls-previousPreparation.packCalls)/frames,
          (prep.packBytes-previousPreparation.packBytes)/frames/(1<<20),
          (unsigned long long)(prep.packCapacityGrowths-previousPreparation.packCapacityGrowths),
          (prep.packCapacityGrowthBytes-previousPreparation.packCapacityGrowthBytes)/1024.0,
          (prep.uniformSnapshotCalls-previousPreparation.uniformSnapshotCalls)/frames,
          (prep.uniformSnapshotBytes-previousPreparation.uniformSnapshotBytes)/frames/(1<<20));
      LOG("[vulkan uniform reuse] %.1f checks/frame %.1f comparisons/frame %.1f hits/frame %.3f MiB avoided/frame",
          (prep.uniformReuseChecks-previousPreparation.uniformReuseChecks)/frames,
          (prep.uniformReuseComparisons-previousPreparation.uniformReuseComparisons)/frames,
          (prep.uniformReuseHits-previousPreparation.uniformReuseHits)/frames,
          (prep.uniformReuseBytes-previousPreparation.uniformReuseBytes)/frames/(1<<20));
      const uint64_t candidates=prep.descriptorBindCandidates-previousPreparation.descriptorBindCandidates;
      const double percentage=candidates?100.0/candidates:0.0;
      LOG("[vulkan bind opportunities] %.1f candidates/frame; exact whole %.2f%% VS %.2f%% PS %.2f%%; %.1f actual binds/frame %.1f skipped sets/frame",
          candidates/frames,(prep.descriptorWholeMatches-previousPreparation.descriptorWholeMatches)*percentage,
          (prep.descriptorStageMatches[0]-previousPreparation.descriptorStageMatches[0])*percentage,
          (prep.descriptorStageMatches[1]-previousPreparation.descriptorStageMatches[1])*percentage,
          (prep.descriptorBindCalls-previousPreparation.descriptorBindCalls)/frames,
          (prep.descriptorSkippedSets-previousPreparation.descriptorSkippedSets)/frames);
      LOG("[vulkan pipeline lookup] %.1f requests/frame; %.1f last hits; %.1f lookaside hits; %.1f map lookups",
          (prep.pipelineLookups-previousPreparation.pipelineLookups)/frames,
          (prep.pipelineLastHits-previousPreparation.pipelineLastHits)/frames,
          (prep.pipelineLookasideHits-previousPreparation.pipelineLookasideHits)/frames,
          (prep.pipelineMapLookups-previousPreparation.pipelineMapLookups)/frames);
      previousPreparation=prep;
      LOG("[vulkan waits] submit %.2f/frame %.2f ms/frame; present submit %.2f/frame %.2f ms/frame",
          submitWait.count/frames, submitWait.ns/1e6/frames,
          presentSubmitWait.count/frames, presentSubmitWait.ns/1e6/frames);
      LOG("[vulkan queue submit] %.2f/frame %.2f ms/frame",
          queueSubmitTiming.count/frames, queueSubmitTiming.ns/1e6/frames);
      LOG("[vulkan swap phases] setup %.2f ms/frame; primary %.2f; secondary %.2f; flush %.2f; bookkeeping %.2f",
          swapPhaseTiming[0].ns/1e6/frames,swapPhaseTiming[1].ns/1e6/frames,
          swapPhaseTiming[2].ns/1e6/frames,swapPhaseTiming[3].ns/1e6/frames,swapPhaseTiming[4].ns/1e6/frames);
      swapPhaseTiming={};
      LOG("[vulkan command end] %.2f/frame %.2f ms/frame",
          commandEndTiming.count/frames,commandEndTiming.ns/1e6/frames);
      commandEndTiming={};
      LOG("[vulkan retirement CPU] fence status %.2f/frame %.2f ms/frame; reset %.2f/frame %.2f ms/frame; cleanup %.2f/frame %.2f ms/frame",
          fenceStatusTiming.count/frames,fenceStatusTiming.ns/1e6/frames,
          fenceResetTiming.count/frames,fenceResetTiming.ns/1e6/frames,
          retirementCleanupTiming.count/frames,retirementCleanupTiming.ns/1e6/frames);
      fenceStatusTiming={};fenceResetTiming={};retirementCleanupTiming={};
      for (int screen = 0; screen < 2; ++screen) {
        auto& t = screenTiming[screen];
        LOG("[vulkan compose %s] %.2f/frame %.2f ms/frame",
            screen ? "DRC" : "TV", t.compose.count/frames, t.compose.ns/1e6/frames);
        LOG("[vulkan waits %s] acquire %.2f/frame %.2f ms/frame; present %.2f/frame %.2f ms/frame; queue idle %.2f/frame %.2f ms/frame",
            screen ? "DRC" : "TV", t.acquire.count/frames, t.acquire.ns/1e6/frames,
            t.present.count/frames, t.present.ns/1e6/frames,
            t.idle.count/frames, t.idle.ns/1e6/frames);
        t = {};
      }
      submitWait = {}; presentSubmitWait = {}; queueSubmitTiming = {};
      }
      start=now;frame=R.frame;draws=R.drawCount;bytes=R.uploadBytes;allocs=R.uploadAllocations;passes=R.renderPassCount;
    }
  }
}
void set_tv_format(uint32_t format, bool tv) {
  (tv ? R.tv : R.drc).srgb = (format & 0x400) != 0;
}
static VKAPI_ATTR VkBool32 VKAPI_CALL debug_message(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity, VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT *data, void *) {
  LOG("[vulkan validation %s] %s: %s",
          severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT ? "error" : "warning",
          data->pMessageIdName ? data->pMessageIdName : "validation", data->pMessage);
  return VK_FALSE;
}
// Test aids for the paths of older drivers (docs/vulkan.md): WWHD_VK_FORCE_API=1.2 treats every GPU as
// if it reported that Vulkan version (Vulkan 1.3 GPUs then take the VK_KHR_dynamic_rendering path);
// WWHD_VK_HIDE_EXTENSIONS=VK_KHR_dynamic_rendering,... hides device extensions from the renderer.
static uint32_t device_api_version(const VkPhysicalDeviceProperties &p) {
  static const uint32_t forced = [] {
    unsigned major, minor;
    const char *e = getenv("WWHD_VK_FORCE_API");
    return e && sscanf(e, "%u.%u", &major, &minor) == 2 ? VK_MAKE_API_VERSION(0, major, minor, 0) : ~0u;
  }();
  return forced < VK_MAKE_API_VERSION(0, VK_API_VERSION_MAJOR(p.apiVersion), VK_API_VERSION_MINOR(p.apiVersion), 0)
             ? forced : p.apiVersion;
}
static std::vector<VkExtensionProperties> device_extensions(VkPhysicalDevice device) {
  uint32_t n = 0;
  vkEnumerateDeviceExtensionProperties(device, nullptr, &n, nullptr);
  std::vector<VkExtensionProperties> des(n);
  vkEnumerateDeviceExtensionProperties(device, nullptr, &n, des.data());
  des.resize(n);
  if (const char *hide = getenv("WWHD_VK_HIDE_EXTENSIONS"))
    std::erase_if(des, [&](const VkExtensionProperties &e) {
      for (const char *p = hide; *p;) {
        size_t len = strcspn(p, ",");
        if (len == strlen(e.extensionName) && !strncmp(p, e.extensionName, len))
          return true;
        p += len + (p[len] == ',');
      }
      return false;
    });
  return des;
}
static std::string version_text(uint32_t v) {
  return std::to_string(VK_API_VERSION_MAJOR(v)) + "." + std::to_string(VK_API_VERSION_MINOR(v)) + "." +
         std::to_string(VK_API_VERSION_PATCH(v));
}
// driverVersion: vendor-specific packing, decoded as vulkaninfo does (report_header.cpp)
static constexpr bool kWindows = false;
static std::string driver_version_text(const VkPhysicalDeviceProperties &p) {
  char text[32];  // the standard Vulkan version encoding (Mesa's RADV uses it for its driver version)
  std::snprintf(text, sizeof text, "%u.%u.%u", VK_API_VERSION_MAJOR(p.driverVersion), VK_API_VERSION_MINOR(p.driverVersion),
                VK_API_VERSION_PATCH(p.driverVersion));
  return text;
}
// the GPU line of the performance report header: name, driver, Vulkan version ("" before init)
std::string device_description() {
  if (!R.properties.deviceName[0]) return "";
  char api[32];
  std::snprintf(api, sizeof api, "%u.%u", VK_API_VERSION_MAJOR(R.properties.apiVersion), VK_API_VERSION_MINOR(R.properties.apiVersion));
  return std::string(R.properties.deviceName) + ", driver " + driver_version_text(R.properties) + ", Vulkan " + api;
}
// How a GPU provides dynamic rendering, the renderer's only way of drawing: Vulkan 1.3 core, or
// VK_KHR_dynamic_rendering on a Vulkan 1.1 / 1.2 driver. Shaders are SPIR-V 1.3 (Vulkan 1.1).
enum class DynamicRendering { None, Core, KHR };
static DynamicRendering dynamic_rendering(VkPhysicalDevice device, const VkPhysicalDeviceProperties &properties,
                                          const std::vector<VkExtensionProperties> &des) {
  const uint32_t api = device_api_version(properties);
  if (api < VK_API_VERSION_1_1)
    return DynamicRendering::None;
  if (api >= VK_API_VERSION_1_3) {
    VkPhysicalDeviceVulkan13Features f13{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.pNext = &f13;
    vkGetPhysicalDeviceFeatures2(device, &f2);
    if (f13.dynamicRendering)
      return DynamicRendering::Core;
  }
  // the extension requires VK_KHR_depth_stencil_resolve (core in 1.2), which requires
  // VK_KHR_create_renderpass2 (core in 1.2)
  if (!has_extension(des, VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME) ||
      (api < VK_API_VERSION_1_2 && (!has_extension(des, VK_KHR_DEPTH_STENCIL_RESOLVE_EXTENSION_NAME) ||
                                    !has_extension(des, VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME))))
    return DynamicRendering::None;
  VkPhysicalDeviceDynamicRenderingFeaturesKHR fdr{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR};
  VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  f2.pNext = &fdr;
  vkGetPhysicalDeviceFeatures2(device, &f2);
  return fdr.dynamicRendering ? DynamicRendering::KHR : DynamicRendering::None;
}
static const char kUpdateDriver[] =
    "Install the newest graphics driver for this GPU (on Linux: an up-to-date Mesa, or the vendor's "
    "driver) and start the game again. "
    "If no newer driver exists, this GPU cannot run the game's Vulkan renderer.";
// The Vulkan layers the loader offers, once in the log (so also in a crash log's last lines).
// Overlays install implicit layers that load into every Vulkan program (Steam, Discord, RivaTuner,
// OBS, Overwolf, ReShade...); a crash inside one then shows up in the crash log's module names, and
// this line says which were around. The list holds the explicit layers as well; implicit ones are
// active unless their disable variable is set.
static void log_instance_layers() {
  if (!vkEnumerateInstanceLayerProperties)
    return;
  uint32_t count = 0;
  if (vkEnumerateInstanceLayerProperties(&count, nullptr) != VK_SUCCESS)
    return;
  std::vector<VkLayerProperties> layers(count);
  if (count && vkEnumerateInstanceLayerProperties(&count, layers.data()) < VK_SUCCESS)
    return;
  layers.resize(count);
  // one log line holds 240 characters: several lines of up to 200 (each layer name is <= 256)
  std::string line;
  int lines = 0;
  for (const VkLayerProperties &l : layers) {
    char one[300];
    snprintf(one, sizeof one, "%s%s (%u.%u.%u)", line.empty() ? "" : ", ", l.layerName,
             VK_API_VERSION_MAJOR(l.specVersion), VK_API_VERSION_MINOR(l.specVersion),
             VK_API_VERSION_PATCH(l.specVersion));
    if (!line.empty() && line.size() + strlen(one) > 200) {
      LOG("[vulkan] layers: %s,", line.c_str());
      line.clear();
      if (++lines == 4) {
        LOG("[vulkan] layers: ... %u in all", count);
        return;
      }
      snprintf(one, sizeof one, "%s (%u.%u.%u)", l.layerName, VK_API_VERSION_MAJOR(l.specVersion),
               VK_API_VERSION_MINOR(l.specVersion), VK_API_VERSION_PATCH(l.specVersion));
    }
    line += one;
  }
  LOG("[vulkan] layers: %s", count ? line.c_str() : "none");
}

// Instance, device, submission slots and swapchains. `extensions`: the window system's instance
// extensions; `create_surfaces` makes R.tv.surface / R.drc.surface once the instance exists.
// The host has loaded the Vulkan loader's global functions (load_global_functions).
static void init_device(std::vector<const char *> extensions,
                        const std::function<void()> &create_surfaces) {
  reset_pipeline_lookup_cache();
  uint32_t n;
  uint32_t loaderVersion = VK_API_VERSION_1_0;
  if (vkEnumerateInstanceVersion)
    vkEnumerateInstanceVersion(&loaderVersion);
  if (loaderVersion < VK_API_VERSION_1_1)
    throw std::runtime_error("The Vulkan runtime on this computer supports only Vulkan " +
                             version_text(loaderVersion) + "; the game needs Vulkan 1.1 or newer.\n\n" +
                             kUpdateDriver);
  log_instance_layers();
  uint32_t en = 0;
  vkEnumerateInstanceExtensionProperties(nullptr, &en, nullptr);
  std::vector<VkExtensionProperties> ies(en);
  vkEnumerateInstanceExtensionProperties(nullptr, &en, ies.data());
  if (!has_extension(ies, VK_KHR_SURFACE_EXTENSION_NAME))
    throw std::runtime_error("No Vulkan driver found (MoltenVK on a Mac: the release ships it next to the "
                             "game as libMoltenVK.dylib with MoltenVK_icd.json; source builds: brew install "
                             "molten-vk; or point VK_DRIVER_FILES at a MoltenVK_icd.json)");
  for (const char *e : extensions)
    if (!has_extension(ies, e))
      throw std::runtime_error(std::string("the Vulkan driver lacks instance extension ") + e);
  VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  if (has_extension(ies, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
    extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    ci.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
  }
  VkDebugUtilsMessengerCreateInfoEXT debug{
      VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
  const char *layer = "VK_LAYER_KHRONOS_validation";
  VkValidationFeatureEnableEXT syncFeature = VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT;
  VkValidationFeaturesEXT validation{VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT};
  validation.enabledValidationFeatureCount = 1;
  validation.pEnabledValidationFeatures = &syncFeature;
  if (getenv("WWHD_VK_VALIDATION")) {
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    extensions.push_back(VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME);
    ci.enabledLayerCount = 1;
    ci.ppEnabledLayerNames = &layer;
    debug.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    debug.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                        VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    debug.pfnUserCallback = debug_message;
    validation.pNext = &debug;
    ci.pNext = &validation;
  }
  VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  ai.pApplicationName = "WWHD Vulkan";
  ai.apiVersion = VK_API_VERSION_1_3;
  ci.pApplicationInfo = &ai;
  ci.enabledExtensionCount = extensions.size();
  ci.ppEnabledExtensionNames = extensions.data();
  vk_check(vkCreateInstance(&ci, nullptr, &R.instance),
           "create Vulkan instance");
  debug.pNext = nullptr;
  load_instance_functions(R.instance);
  if (getenv("WWHD_VK_VALIDATION")) {
    auto create = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
        R.instance, "vkCreateDebugUtilsMessengerEXT");
    VkDebugUtilsMessengerEXT messenger;
    if (create)
      vk_check(create(R.instance, &debug, nullptr, &messenger),
               "create validation messenger");
  }
  create_surfaces();
  vkEnumeratePhysicalDevices(R.instance, &n, nullptr);
  std::vector<VkPhysicalDevice> devices(n);
  vkEnumeratePhysicalDevices(R.instance, &n, devices.data());
  devices.resize(n);
  // First the GPUs with Vulkan 1.3 (in the driver's order), then those with VK_KHR_dynamic_rendering
  std::string unsuitable;  // the GPUs passed over, for the message when none is left
  for (DynamicRendering want : {DynamicRendering::Core, DynamicRendering::KHR}) {
    for (auto device : devices) {
      VkPhysicalDeviceProperties properties;
      vkGetPhysicalDeviceProperties(device, &properties);
      const DynamicRendering have = dynamic_rendering(device, properties, device_extensions(device));
      if (have != want) {
        if (want == DynamicRendering::KHR && have == DynamicRendering::None)
          unsuitable += std::string("\n") + properties.deviceName + ": Vulkan " +
                        version_text(device_api_version(properties)) + ", driver " +
                        driver_version_text(properties) +
                        (device_api_version(properties) < VK_API_VERSION_1_1 ? "; older than Vulkan 1.1"
                                                                             : "; VK_KHR_dynamic_rendering missing");
        continue;
      }
      uint32_t qn;
      vkGetPhysicalDeviceQueueFamilyProperties(device, &qn, nullptr);
      std::vector<VkQueueFamilyProperties> qs(qn);
      vkGetPhysicalDeviceQueueFamilyProperties(device, &qn, qs.data());
      for (uint32_t q = 0; q < qn; q++) {
        VkBool32 tv = 0, drc = VK_TRUE;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, q, R.tv.surface, &tv);
        if (R.drc.surface)
          vkGetPhysicalDeviceSurfaceSupportKHR(device, q, R.drc.surface, &drc);
        if ((qs[q].queueFlags & VK_QUEUE_GRAPHICS_BIT) && tv && drc) {
          R.physicalDevice = device;
          R.formatProperties.clear();
          R.queueFamily = q;
          R.computeQueue = (qs[q].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0;
          R.gpuTimestampValidBits = qs[q].timestampValidBits;
          R.dynamicRenderingKHR = have == DynamicRendering::KHR;
          break;
        }
      }
      if (R.physicalDevice)
        break;
      unsuitable += std::string("\n") + properties.deviceName + ": cannot draw to the game window";
    }
    if (R.physicalDevice)
      break;
  }
  if (!R.physicalDevice)
    throw std::runtime_error(
        devices.empty()
            ? std::string("No graphics card with a Vulkan driver was found.\n\n") + kUpdateDriver
            : "The graphics driver does not support the Vulkan features the game needs: Vulkan 1.3, or "
              "Vulkan 1.1 / 1.2 with the VK_KHR_dynamic_rendering extension.\n" + unsuitable +
                  "\n\n" + kUpdateDriver);
  vkGetPhysicalDeviceProperties(R.physicalDevice, &R.properties);
  const uint32_t deviceApi = device_api_version(R.properties);
  LOG("[vulkan] device: %s (Vulkan %s, driver %s%s)", R.properties.deviceName,
      version_text(R.properties.apiVersion).c_str(), driver_version_text(R.properties).c_str(),
      R.dynamicRenderingKHR ? "; dynamic rendering through VK_KHR_dynamic_rendering" : "");
  if (perf_enabled()) {
    const auto& limits=R.properties.limits;
    LOG("[vulkan uniforms] dynamic/set %u; uniform/stage %u; uniform/set %u; resources/stage %u; alignment %llu; range %u",
        limits.maxDescriptorSetUniformBuffersDynamic,limits.maxPerStageDescriptorUniformBuffers,
        limits.maxDescriptorSetUniformBuffers,limits.maxPerStageResources,
        (unsigned long long)limits.minUniformBufferOffsetAlignment,limits.maxUniformBufferRange);
  }
  const std::vector<VkExtensionProperties> des = device_extensions(R.physicalDevice);
  std::vector<const char *> de{VK_KHR_SWAPCHAIN_EXTENSION_NAME};
  if (R.dynamicRenderingKHR) {
    de.push_back(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);
    if (deviceApi < VK_API_VERSION_1_2) {
      de.push_back(VK_KHR_DEPTH_STENCIL_RESOLVE_EXTENSION_NAME);
      de.push_back(VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME);
    }
  }
  if (has_extension(des, "VK_KHR_portability_subset"))
    de.push_back("VK_KHR_portability_subset");
  R.portabilitySubset =
      has_extension(des, VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);
  VkPhysicalDevicePortabilitySubsetFeaturesKHR portability{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PORTABILITY_SUBSET_FEATURES_KHR};
  VkPhysicalDeviceVulkan12Features available12{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  VkPhysicalDeviceFeatures2 features{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  // VkPhysicalDeviceVulkan12Features: Vulkan 1.2 devices and newer
  const bool vulkan12 = deviceApi >= VK_API_VERSION_1_2;
  void **tail = &features.pNext;
  if (vulkan12) {
    *tail = &available12;
    tail = &available12.pNext;
  }
  if (R.portabilitySubset) {
    *tail = &portability;tail=&portability.pNext;
  }
  vkGetPhysicalDeviceFeatures2(R.physicalDevice, &features);
  if (R.portabilitySubset) {
    R.imageViewSwizzle = portability.imageViewFormatSwizzle;
    R.imageViewReinterpretation = portability.imageViewFormatReinterpretation;
    R.imageView2DOn3DImage = portability.imageView2DOn3DImage;
    R.samplerMipLodBias = portability.samplerMipLodBias;
    R.separateStencilMaskRef = portability.separateStencilMaskRef;
    R.constantAlphaColorBlendFactors =
        portability.constantAlphaColorBlendFactors;
    R.vertexAttributeAccessBeyondStride =
        portability.vertexAttributeAccessBeyondStride;
  }
  VkPhysicalDeviceVulkan12Features enabled12{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  enabled12.samplerMirrorClampToEdge = available12.samplerMirrorClampToEdge;
  if (!vulkan12 && has_extension(des, VK_KHR_SAMPLER_MIRROR_CLAMP_TO_EDGE_EXTENSION_NAME)) {
    de.push_back(VK_KHR_SAMPLER_MIRROR_CLAMP_TO_EDGE_EXTENSION_NAME);  // Vulkan 1.1: the extension
    enabled12.samplerMirrorClampToEdge = VK_TRUE;
  }
  R.samplerMirrorClampToEdge = enabled12.samplerMirrorClampToEdge;
  VkPhysicalDeviceFeatures available;
  vkGetPhysicalDeviceFeatures(R.physicalDevice, &available);
  VkPhysicalDeviceFeatures enabled{};
  // The game's BC1-BC5 textures need this core feature enabled, not just supported (Vulkan spec);
  // devices without it (many Mali/PowerVR GPUs) decode BC uploads with a compute shader (bc_decode).
  enabled.textureCompressionBC = available.textureCompressionBC;
  if (!available.textureCompressionBC) LOG("[vulkan] device has no BC texture support; compute upload decoder enabled");
  enabled.samplerAnisotropy = available.samplerAnisotropy;
  enabled.independentBlend = available.independentBlend;
  enabled.depthClamp = available.depthClamp;
  enabled.depthBiasClamp = available.depthBiasClamp;
  enabled.shaderClipDistance = available.shaderClipDistance;
  enabled.shaderCullDistance = available.shaderCullDistance;
  enabled.fillModeNonSolid = available.fillModeNonSolid;
  enabled.largePoints = available.largePoints;
  enabled.dualSrcBlend = available.dualSrcBlend;
  enabled.logicOp = available.logicOp;
  R.enabledFeatures = enabled;
  portability.pNext=nullptr;
  VkPhysicalDeviceVulkan13Features f13{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
  VkPhysicalDeviceDynamicRenderingFeaturesKHR fdr{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR};
  f13.dynamicRendering = fdr.dynamicRendering = VK_TRUE;
  void *chain = R.dynamicRenderingKHR ? static_cast<void *>(&fdr) : &f13;
  tail = R.dynamicRenderingKHR ? &fdr.pNext : &f13.pNext;
  if (vulkan12) {
    *tail = &enabled12;
    tail = &enabled12.pNext;
  }
  if (R.portabilitySubset)
    *tail = &portability;
  float priorities[2]={1.0f,0.5f};
  VkDeviceQueueCreateInfo queues[2]={{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO},
                                   {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}};
  queues[0].queueFamilyIndex=R.queueFamily;
  queues[0].queueCount=1;
  queues[0].pQueuePriorities=priorities;
  uint32_t requestedFamilies=1;
  R.primaryRequestedQueues=queues[0].queueCount;
  VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  di.pNext = chain;
  di.queueCreateInfoCount = requestedFamilies;
  di.pQueueCreateInfos = queues;
  di.pEnabledFeatures = &enabled;
  di.enabledExtensionCount = de.size();
  di.ppEnabledExtensionNames = de.data();
  vk_check(vkCreateDevice(R.physicalDevice, &di, nullptr, &R.device),
           "create Vulkan device");
  load_device_functions(R.device, R.dynamicRenderingKHR);
  vkGetDeviceQueue(R.device, R.queueFamily, 0, &R.queue);
  init_pipeline_cache();
  for (auto& slot:R.submissions) {
  VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pool.queueFamilyIndex = R.queueFamily;
  pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  vk_check(vkCreateCommandPool(R.device, &pool, nullptr, &slot.commandPool),
           "create command pool");
  VkCommandBufferAllocateInfo ca{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  ca.commandPool = slot.commandPool;
  ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  ca.commandBufferCount = 1;
  vk_check(vkAllocateCommandBuffers(R.device, &ca, &slot.cmd),
           "allocate command buffer");
  VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
  vk_check(vkCreateFence(R.device, &fi, nullptr, &slot.fence),
           "create frame fence");
  VkDescriptorPoolSize sizes[] = {
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 65536},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 32768},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 32768},
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 65536}};
  VkDescriptorPoolCreateInfo dp{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dp.maxSets = 32768;
  dp.poolSizeCount = std::size(sizes);
  dp.pPoolSizes = sizes;
  vk_check(vkCreateDescriptorPool(R.device, &dp, nullptr, &slot.descriptorPool),
           "create descriptor pool");
  }
  init_gpu_timestamp_queries();
  activate_submission(0);
  for (Screen *s : {&R.tv, &R.drc})
    if (s->window) {
      VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
      vk_check(vkCreateSemaphore(R.device, &si, nullptr, &s->acquired),
               "create acquire semaphore");
      vk_check(vkCreateSemaphore(R.device, &si, nullptr, &s->finished),
               "create present semaphore");
      make_swapchain(*s);
    }
  vk::select_renderer();
  set_graphics_feature_available(GraphicsFeature::AO);
  set_graphics_feature_available(GraphicsFeature::AOHires);
  set_graphics_feature_available(GraphicsFeature::Anisotropy,
                                 R.enabledFeatures.samplerAnisotropy);
  if (const char* path = getenv("WWHD_CAPTURE_PATH")) {
    const char* frame = getenv("WWHD_CAPTURE");
    request_tv_dump(path, frame ? std::max(0, atoi(frame)) : 120);
  }
}

static bool hidden_windows() {
  static const bool hidden = [] { const char* e = getenv("WWHD_HIDDEN_WINDOWS"); return e && *e && strcmp(e, "0"); }();
  return hidden;
}
// The game's own icon on the windows (title bar, taskbar): meta/iconTex.tga of the game folder, an
// uncompressed 32-bit TGA (128x128 in Wind Waker HD). Nothing happens without it.
void init(SDL_Window* window) {
  // the Vulkan loader (libvulkan.so.1), loaded here rather than imported (loader.h); the window uses the same one
  if (!SDL_Vulkan_LoadLibrary(nullptr))
    throw std::runtime_error(std::string("Vulkan is not installed on this computer: the Vulkan runtime "
                                         "could not be loaded (") + SDL_GetError() + ").\n\n" + kUpdateDriver);
  auto gipa = reinterpret_cast<PFN_vkGetInstanceProcAddr>(SDL_Vulkan_GetVkGetInstanceProcAddr());
  load_global_functions(gipa);
  R.tv.window = window;  // the TV picture; no GamePad window (the host shows one screen)
  uint32_t n;
  const char *const *se = SDL_Vulkan_GetInstanceExtensions(&n);
  if (!se)
    throw std::runtime_error(SDL_GetError());
  for (Screen *s : {&R.tv, &R.drc})
    if (s->window) {
      int width = 0, height = 0;
      if (!SDL_GetWindowSizeInPixels(s->window, &width, &height))
        throw std::runtime_error(SDL_GetError());
      s->width = width;
      s->height = height;
    }
  init_device(std::vector<const char *>(se, se + n), [] {
    if (!SDL_Vulkan_CreateSurface(R.tv.window, R.instance, nullptr, &R.tv.surface))
      throw std::runtime_error(SDL_GetError());
  });
}
void save_renderer_caches() {
  // Called during orderly shutdown under the renderer execution lock.
  reset_feedback_images();
  wait_idle();
  bc_decode_shutdown();
  destroy_gpu_timestamp_queries();
  vk::save_shader_cache();
  save_pipeline_cache();
}

} // namespace gfxvk
