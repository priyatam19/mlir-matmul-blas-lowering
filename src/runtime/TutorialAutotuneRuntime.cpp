#include "TutorialAutotuneRuntime.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr int WarmupCalls = 2;
constexpr int TimedCalls = 5;

struct TuneState {
  explicit TuneState(int64_t count)
      : candidateCount(count), samples(static_cast<size_t>(count)) {}

  int64_t candidateCount;
  int64_t candidate = 0;
  int callsForCandidate = 0;
  int64_t winner = -1;
  std::vector<std::vector<float>> samples;
};

struct PendingMeasurement {
  std::string fullKey;
  int64_t candidate = 0;
  bool active = false;
  bool timed = false;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
};

std::mutex runtimeMutex;
std::unordered_map<std::string, int64_t> persistedWinners;
std::unordered_map<std::string, TuneState> tuningStates;
std::once_flag initializeOnce;
thread_local PendingMeasurement pending;

bool environmentFlag(const char *name) {
  const char *value = std::getenv(name);
  return value && value[0] != '\0' && std::string(value) != "0";
}

bool debugEnabled() { return environmentFlag("TUTORIAL_AUTOTUNE_DEBUG"); }

void checkCuda(cudaError_t status, const char *operation) {
  if (status == cudaSuccess)
    return;
  std::fprintf(stderr, "tutorial autotune: %s failed: %s\n", operation,
               cudaGetErrorString(status));
  std::abort();
}

std::filesystem::path cachePath() {
  if (const char *overridePath = std::getenv("TUTORIAL_AUTOTUNE_CACHE"))
    return overridePath;
  const char *home = std::getenv("HOME");
  return std::filesystem::path(home ? home : ".") / ".cache" /
         "mlir-tutorial" / "gpu-autotune-v1.csv";
}

std::string deviceIdentity() {
  static std::string identity = [] {
    int device = 0;
    int driver = 0;
    cudaDeviceProp properties{};
    checkCuda(cudaGetDevice(&device), "cudaGetDevice");
    checkCuda(cudaGetDeviceProperties(&properties, device),
              "cudaGetDeviceProperties");
    checkCuda(cudaDriverGetVersion(&driver), "cudaDriverGetVersion");

    std::ostringstream stream;
    stream << "uuid=";
    for (unsigned char byte : properties.uuid.bytes)
      stream << std::hex << std::setw(2) << std::setfill('0')
             << static_cast<unsigned>(byte);
    stream << std::dec << ";cc=" << properties.major << properties.minor
           << ";driver=" << driver << ";compiler="
#ifdef MLIR_TUTORIAL_COMPILER_REVISION
           << MLIR_TUTORIAL_COMPILER_REVISION;
#else
           << "unknown";
#endif
    return stream.str();
  }();
  return identity;
}

std::string makeFullKey(int64_t key) {
  return deviceIdentity() + ";kernel=" + std::to_string(key);
}

void loadCache() {
  std::filesystem::path path = cachePath();
  if (environmentFlag("TUTORIAL_AUTOTUNE_RESET")) {
    std::error_code error;
    std::filesystem::remove(path, error);
    return;
  }

  std::ifstream input(path);
  std::string line;
  std::getline(input, line);
  while (std::getline(input, line)) {
    size_t comma = line.rfind(',');
    if (comma == std::string::npos)
      continue;
    try {
      persistedWinners[line.substr(0, comma)] =
          std::stoll(line.substr(comma + 1));
    } catch (const std::exception &) {
      if (debugEnabled())
        std::fprintf(stderr, "tutorial autotune: ignored malformed cache row\n");
    }
  }
}

void persistCache() {
  if (environmentFlag("TUTORIAL_AUTOTUNE_READ_ONLY"))
    return;
  std::filesystem::path path = cachePath();
  std::error_code error;
  std::filesystem::create_directories(path.parent_path(), error);
  std::filesystem::path temporary = path;
  temporary += ".tmp." + std::to_string(static_cast<unsigned long long>(
                            std::hash<std::thread::id>{}(
                                std::this_thread::get_id())));
  {
    std::ofstream output(temporary, std::ios::trunc);
    output << "key,candidate\n";
    std::vector<std::pair<std::string, int64_t>> rows(
        persistedWinners.begin(), persistedWinners.end());
    std::sort(rows.begin(), rows.end());
    for (const auto &[key, candidate] : rows)
      output << key << ',' << candidate << '\n';
  }
  std::filesystem::rename(temporary, path, error);
  if (error) {
    std::filesystem::remove(path, error);
    error.clear();
    std::filesystem::rename(temporary, path, error);
  }
  if (error && debugEnabled())
    std::fprintf(stderr, "tutorial autotune: could not persist %s: %s\n",
                 path.c_str(), error.message().c_str());
}

float median(std::vector<float> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

int64_t selectWinner(TuneState &state) {
  int64_t winner = 0;
  float best = median(state.samples[0]);
  for (int64_t candidate = 1; candidate < state.candidateCount; ++candidate) {
    float candidateMedian = median(state.samples[candidate]);
    if (candidateMedian < best) {
      best = candidateMedian;
      winner = candidate;
    }
  }
  return winner;
}

void initializeRuntime() { loadCache(); }

} // namespace

extern "C" int64_t tutorial_autotune_begin(int64_t key,
                                            int64_t candidateCount) {
  pending = PendingMeasurement{};
  if (candidateCount <= 0)
    return 0;
  if (environmentFlag("TUTORIAL_AUTOTUNE_DISABLE"))
    return 0;
  std::call_once(initializeOnce, initializeRuntime);
  std::string fullKey = makeFullKey(key);
  std::lock_guard<std::mutex> lock(runtimeMutex);

  auto cached = persistedWinners.find(fullKey);
  if (cached != persistedWinners.end() && cached->second < candidateCount)
    return cached->second;
  if (environmentFlag("TUTORIAL_AUTOTUNE_READ_ONLY"))
    return 0;

  auto [iterator, inserted] =
      tuningStates.try_emplace(fullKey, candidateCount);
  TuneState &state = iterator->second;
  if (!inserted && state.candidateCount != candidateCount) {
    state = TuneState(candidateCount);
  }
  if (state.winner >= 0)
    return state.winner;

  pending.fullKey = fullKey;
  pending.candidate = state.candidate;
  pending.active = true;
  pending.timed = state.callsForCandidate >= WarmupCalls;
  if (pending.timed) {
    checkCuda(cudaEventCreate(&pending.start), "cudaEventCreate(start)");
    checkCuda(cudaEventCreate(&pending.stop), "cudaEventCreate(stop)");
    checkCuda(cudaEventRecord(pending.start), "cudaEventRecord(start)");
  }
  return state.candidate;
}

extern "C" void tutorial_autotune_end(int64_t key, int64_t candidate) {
  (void)key;
  if (!pending.active)
    return;
  float elapsed = 0.0f;
  if (pending.timed) {
    checkCuda(cudaEventRecord(pending.stop), "cudaEventRecord(stop)");
    checkCuda(cudaEventSynchronize(pending.stop),
              "cudaEventSynchronize(stop)");
    checkCuda(cudaEventElapsedTime(&elapsed, pending.start, pending.stop),
              "cudaEventElapsedTime");
    cudaEventDestroy(pending.start);
    cudaEventDestroy(pending.stop);
  }

  std::lock_guard<std::mutex> lock(runtimeMutex);
  auto iterator = tuningStates.find(pending.fullKey);
  if (iterator == tuningStates.end() || iterator->second.winner >= 0 ||
      candidate != pending.candidate) {
    pending = PendingMeasurement{};
    return;
  }
  TuneState &state = iterator->second;
  if (pending.timed)
    state.samples[static_cast<size_t>(candidate)].push_back(elapsed);
  ++state.callsForCandidate;
  if (state.callsForCandidate < WarmupCalls + TimedCalls) {
    pending = PendingMeasurement{};
    return;
  }

  state.callsForCandidate = 0;
  ++state.candidate;
  if (state.candidate < state.candidateCount) {
    pending = PendingMeasurement{};
    return;
  }

  state.winner = selectWinner(state);
  persistedWinners[pending.fullKey] = state.winner;
  persistCache();
  if (debugEnabled())
    std::fprintf(stderr,
                 "tutorial autotune: key=%s candidates=%lld winner=%lld\n",
                 pending.fullKey.c_str(),
                 static_cast<long long>(state.candidateCount),
                 static_cast<long long>(state.winner));
  pending = PendingMeasurement{};
}
