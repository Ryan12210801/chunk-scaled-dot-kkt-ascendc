#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "acl/acl.h"
#include "aclnn_chunk_scaled_dot_kkt.h"

namespace {

constexpr int64_t kChunkSize = 64;
constexpr float kAbsTolerance = 1.0e-6F;
constexpr float kRelTolerance = 1.0e-5F;
constexpr int32_t kOutputPoisonByte = 0xA5;

#define CHECK_ACL(expr)                                                            \
    do {                                                                           \
        const int _ret = (expr);                                                    \
        if (_ret != ACL_SUCCESS) {                                                  \
            std::printf("ACL call failed: %s, ret=%d, file=%s, line=%d\n",        \
                        #expr, _ret, __FILE__, __LINE__);                           \
            return false;                                                          \
        }                                                                          \
    } while (0)

enum class GatePattern {
    MONOTONIC_CUMSUM,
    NON_MONOTONIC,
    ROW_ZERO_ACTIVE,
    NEAR_ZERO_DIFF,
    ALL_EQUAL,
    WIDE_RANGE,
};

enum class Suite {
    CORRECTNESS,
    PERFORMANCE,
    STRESS,
    PROFILE
};

enum class VerifyMode {
    FULL,
    SAMPLED,
    NONE,
};

struct TestCase {
    std::string name;
    Suite suite;
    int64_t t;
    int64_t h;
    int64_t hg;
    int64_t k;
    std::vector<int32_t> chunkOffsets;
    GatePattern gatePattern;
    VerifyMode verifyMode;
    int warmup;
    int repeat;
    int sampledRows;
};

struct Options {
    std::string suite = "correctness";
    std::string caseName;
    int32_t deviceId = 0;
    int warmupOverride = -1;
    int repeatOverride = -1;
    int sampledRowsOverride = -1;
    bool disableCheck = false;
    bool listOnly = false;
    bool showHelp = false;
};

struct HostInputs {
    std::vector<uint16_t> k;
    std::vector<uint16_t> beta;
    std::vector<float> g;
};

struct DeviceTensor {
    aclTensor* tensor = nullptr;
    void* addr = nullptr;
    size_t bytes = 0;

    DeviceTensor() = default;
    DeviceTensor(const DeviceTensor&) = delete;
    DeviceTensor& operator=(const DeviceTensor&) = delete;

    ~DeviceTensor()
    {
        Reset();
    }

    void Reset()
    {
        if (tensor != nullptr) {
            aclDestroyTensor(tensor);
            tensor = nullptr;
        }
        if (addr != nullptr) {
            aclrtFree(addr);
            addr = nullptr;
        }
        bytes = 0;
    }
};

struct EventPair {
    aclrtEvent start = nullptr;
    aclrtEvent end = nullptr;

    EventPair() = default;
    EventPair(const EventPair&) = delete;
    EventPair& operator=(const EventPair&) = delete;

    ~EventPair()
    {
        if (start != nullptr) {
            aclrtDestroyEvent(start);
        }
        if (end != nullptr) {
            aclrtDestroyEvent(end);
        }
    }
};

uint16_t FloatToBFloat16(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t lsb = (bits >> 16U) & 1U;
    bits += 0x7FFFU + lsb;
    return static_cast<uint16_t>(bits >> 16U);
}

float BFloat16ToFloat(uint16_t value)
{
    const uint32_t bits = static_cast<uint32_t>(value) << 16U;
    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

bool CheckedShapeSize(const std::vector<int64_t>& shape, int64_t& result)
{
    result = 1;
    for (const int64_t dim : shape) {
        if (dim <= 0 || result > std::numeric_limits<int64_t>::max() / dim) {
            return false;
        }
        result *= dim;
    }
    return true;
}

std::vector<int64_t> BuildContiguousStrides(const std::vector<int64_t>& shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] =
            shape[static_cast<size_t>(i + 1)] * strides[static_cast<size_t>(i + 1)];
    }
    return strides;
}

template <typename T>
bool CreateTensorFromHost(const std::vector<T>& hostData,
                          const std::vector<int64_t>& shape,
                          aclDataType dataType,
                          DeviceTensor& output)
{
    int64_t elementCount = 0;
    if (!CheckedShapeSize(shape, elementCount) ||
        static_cast<uint64_t>(elementCount) != hostData.size()) {
        std::printf("Invalid tensor shape or host data size\n");
        return false;
    }

    output.bytes = hostData.size() * sizeof(T);
    CHECK_ACL(aclrtMalloc(&output.addr, output.bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(output.addr,
                         output.bytes,
                         hostData.data(),
                         output.bytes,
                         ACL_MEMCPY_HOST_TO_DEVICE));

    const std::vector<int64_t> strides = BuildContiguousStrides(shape);
    output.tensor = aclCreateTensor(shape.data(),
                                    shape.size(),
                                    dataType,
                                    strides.data(),
                                    0,
                                    ACL_FORMAT_ND,
                                    shape.data(),
                                    shape.size(),
                                    output.addr);
    if (output.tensor == nullptr) {
        std::printf("aclCreateTensor failed\n");
        return false;
    }
    return true;
}

bool CreateOutputTensor(const std::vector<int64_t>& shape, DeviceTensor& output)
{
    int64_t elementCount = 0;
    if (!CheckedShapeSize(shape, elementCount)) {
        std::printf("Invalid output shape\n");
        return false;
    }

    output.bytes = static_cast<size_t>(elementCount) * sizeof(float);
    CHECK_ACL(aclrtMalloc(&output.addr, output.bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemset(output.addr,
                         output.bytes,
                         kOutputPoisonByte,
                         output.bytes));

    const std::vector<int64_t> strides = BuildContiguousStrides(shape);
    output.tensor = aclCreateTensor(shape.data(),
                                    shape.size(),
                                    ACL_FLOAT,
                                    strides.data(),
                                    0,
                                    ACL_FORMAT_ND,
                                    shape.data(),
                                    shape.size(),
                                    output.addr);
    if (output.tensor == nullptr) {
        std::printf("aclCreateTensor for output failed\n");
        return false;
    }
    return true;
}

std::vector<int32_t> BuildChunkOffsets(const std::vector<int32_t>& cuSeqlens)
{
    std::vector<int32_t> offsets;
    for (size_t seq = 0; seq + 1 < cuSeqlens.size(); ++seq) {
        const int32_t begin = cuSeqlens[seq];
        const int32_t end = cuSeqlens[seq + 1];
        for (int32_t start = begin; start < end;
             start += static_cast<int32_t>(kChunkSize)) {
            offsets.push_back(start);
        }
    }
    offsets.push_back(cuSeqlens.back());
    return offsets;
}

bool ValidateCase(const TestCase& tc)
{
    if (tc.t <= 0 || tc.h <= 0 || tc.hg <= 0 || tc.h % tc.hg != 0 ||
        (tc.k != 128 && tc.k != 256) || tc.chunkOffsets.size() < 2 ||
        tc.chunkOffsets.front() != 0 || tc.chunkOffsets.back() != tc.t ||
        tc.warmup < 0 || tc.repeat <= 0 || tc.sampledRows < 0) {
        return false;
    }

    for (size_t i = 0; i + 1 < tc.chunkOffsets.size(); ++i) {
        const int32_t begin = tc.chunkOffsets[i];
        const int32_t end = tc.chunkOffsets[i + 1];
        if (begin < 0 || end <= begin || end > tc.t || end - begin > kChunkSize) {
            return false;
        }
    }
    return true;
}

void FillInputs(const TestCase& tc, HostInputs& inputs)
{
    inputs.k.resize(static_cast<size_t>(tc.t * tc.hg * tc.k));
    inputs.beta.resize(static_cast<size_t>(tc.h * tc.t));
    inputs.g.assign(static_cast<size_t>(tc.h * tc.t), 0.0F);

    for (int64_t token = 0; token < tc.t; ++token) {
        for (int64_t kv = 0; kv < tc.hg; ++kv) {
            for (int64_t d = 0; d < tc.k; ++d) {
                const int64_t raw =
                    (token * 17 + kv * 13 + d * 7 + (token * d) % 19) % 37 - 18;
                const float value = static_cast<float>(raw) * 0.015625F;
                const size_t index = static_cast<size_t>(
                    (token * tc.hg + kv) * tc.k + d);
                inputs.k[index] = FloatToBFloat16(value);
            }
        }
    }

    for (int64_t head = 0; head < tc.h; ++head) {
        for (int64_t token = 0; token < tc.t; ++token) {
            const int64_t raw = (head * 11 + token * 5) % 23 - 11;
            const float value = static_cast<float>(raw) * 0.09375F;
            inputs.beta[static_cast<size_t>(head * tc.t + token)] =
                FloatToBFloat16(value);
        }
    }

    for (size_t chunk = 0; chunk + 1 < tc.chunkOffsets.size(); ++chunk) {
        const int32_t begin = tc.chunkOffsets[chunk];
        const int32_t end = tc.chunkOffsets[chunk + 1];

        for (int64_t head = 0; head < tc.h; ++head) {
            float cumulative = 0.0F;
            for (int32_t token = begin; token < end; ++token) {
                const int32_t local = token - begin;
                float value = 0.0F;

                switch (tc.gatePattern) {
                    case GatePattern::MONOTONIC_CUMSUM: {
                        const float increment =
                            -0.0125F - 0.001F * static_cast<float>((head + local * 3) % 7);
                        cumulative += increment;
                        value = cumulative;
                        break;
                    }
                    case GatePattern::NON_MONOTONIC: {
                        const int32_t raw =
                            (local * 7 + static_cast<int32_t>(head) * 3) % 13 - 6;
                        value = static_cast<float>(raw) * 0.075F;
                        break;
                    }
                    case GatePattern::ROW_ZERO_ACTIVE: {
                        value = local == 0
                            ? -0.75F - 0.01F * static_cast<float>(head)
                            : 0.35F - 0.045F * static_cast<float>((local + head) % 9);
                        break;
                    }
                    case GatePattern::NEAR_ZERO_DIFF: {
                        static const float values[8] = {
                            0.0F, -1.0e-6F, -1.0e-6F, 1.0e-6F,
                            0.0F, -2.0e-6F, 2.0e-6F, 0.0F};
                        value = values[(local + static_cast<int32_t>(head)) % 8];
                        break;
                    }
                    case GatePattern::ALL_EQUAL: {
                        value = -0.125F * static_cast<float>(head % 3);
                        break;
                    }
                    case GatePattern::WIDE_RANGE: {
                        const int32_t raw =
                            (local * 29 + static_cast<int32_t>(head) * 17) % 41 - 20;
                        value = static_cast<float>(raw) * 0.5F;
                        break;
                    }
                }

                inputs.g[static_cast<size_t>(head * tc.t + token)] = value;
            }
        }
    }
}

size_t FindChunkIndex(const TestCase& tc, int64_t token)
{
    const auto it = std::upper_bound(tc.chunkOffsets.begin(),
                                     tc.chunkOffsets.end(),
                                     static_cast<int32_t>(token));
    if (it == tc.chunkOffsets.begin() || it == tc.chunkOffsets.end()) {
        return tc.chunkOffsets.size();
    }
    return static_cast<size_t>((it - tc.chunkOffsets.begin()) - 1);
}

float ComputeExpectedAt(const TestCase& tc,
                        const HostInputs& inputs,
                        int64_t head,
                        int64_t token,
                        int64_t localJ)
{
    const size_t chunk = FindChunkIndex(tc, token);
    if (chunk + 1 >= tc.chunkOffsets.size()) {
        return 0.0F;
    }

    const int64_t chunkBegin = tc.chunkOffsets[chunk];
    const int64_t chunkEnd = tc.chunkOffsets[chunk + 1];
    const int64_t j = chunkBegin + localJ;
    if (j >= chunkEnd) {
        return 0.0F;
    }

    const float gi = inputs.g[static_cast<size_t>(head * tc.t + token)];
    const float gj = inputs.g[static_cast<size_t>(head * tc.t + j)];
    const float diff = gi - gj;
    if (!(diff < 0.0F)) {
        return 0.0F;
    }

    const int64_t headsPerKv = tc.h / tc.hg;
    const int64_t kv = head / headsPerKv;
    float dot = 0.0F;
    for (int64_t d = 0; d < tc.k; ++d) {
        const size_t iIndex = static_cast<size_t>(
            (token * tc.hg + kv) * tc.k + d);
        const size_t jIndex = static_cast<size_t>(
            (j * tc.hg + kv) * tc.k + d);
        dot += BFloat16ToFloat(inputs.k[iIndex]) *
               BFloat16ToFloat(inputs.k[jIndex]);
    }

    const float beta = BFloat16ToFloat(
        inputs.beta[static_cast<size_t>(head * tc.t + token)]);
    return beta * std::exp(diff) * dot;
}

struct CheckStats {
    size_t total = 0;
    size_t mismatch = 0;
    size_t paddingMismatch = 0;
    size_t nonFinite = 0;
    float maxAbs = 0.0F;
    float maxRel = 0.0F;
    int64_t maxHead = 0;
    int64_t maxToken = 0;
    int64_t maxLocalJ = 0;
};

bool ValuesMatch(float actual, float expected, float& absError, float& relError)
{
    if (!std::isfinite(actual)) {
        absError = std::numeric_limits<float>::infinity();
        relError = std::numeric_limits<float>::infinity();
        return false;
    }

    absError = std::fabs(actual - expected);
    relError = absError / std::max(std::fabs(expected), 1.0e-30F);
    return absError <= kAbsTolerance || relError <= kRelTolerance;
}

void UpdateCheckStats(CheckStats& stats,
                      float actual,
                      float expected,
                      int64_t head,
                      int64_t token,
                      int64_t localJ,
                      bool padding,
                      size_t& printed)
{
    ++stats.total;

    float absError = 0.0F;
    float relError = 0.0F;
    const bool match = ValuesMatch(actual, expected, absError, relError);

    if (!std::isfinite(actual)) {
        ++stats.nonFinite;
    }

    if (absError > stats.maxAbs) {
        stats.maxAbs = absError;
        stats.maxRel = relError;
        stats.maxHead = head;
        stats.maxToken = token;
        stats.maxLocalJ = localJ;
    }

    if (match) {
        return;
    }

    ++stats.mismatch;
    if (padding) {
        ++stats.paddingMismatch;
    }

    if (printed < 8) {
        std::printf("  mismatch[%zu]: h=%ld t=%ld jLocal=%ld region=%s "
                    "actual=%.9g expected=%.9g abs=%.3e rel=%.3e\n",
                    printed,
                    head,
                    token,
                    localJ,
                    padding ? "padding" : "valid",
                    actual,
                    expected,
                    absError,
                    relError);
        ++printed;
    }
}

bool ValidateFullOutput(const TestCase& tc,
                        const HostInputs& inputs,
                        const DeviceTensor& output)
{
    const size_t outputElements =
        static_cast<size_t>(tc.h * tc.t * kChunkSize);
    std::vector<float> actual(outputElements, 0.0F);
    CHECK_ACL(aclrtMemcpy(actual.data(),
                         actual.size() * sizeof(float),
                         output.addr,
                         output.bytes,
                         ACL_MEMCPY_DEVICE_TO_HOST));

    CheckStats stats;
    size_t printed = 0;

    for (int64_t head = 0; head < tc.h; ++head) {
        for (int64_t token = 0; token < tc.t; ++token) {
            const size_t chunk = FindChunkIndex(tc, token);
            const int64_t chunkBegin = tc.chunkOffsets[chunk];
            const int64_t chunkEnd = tc.chunkOffsets[chunk + 1];

            for (int64_t localJ = 0; localJ < kChunkSize; ++localJ) {
                const size_t index = static_cast<size_t>(
                    (head * tc.t + token) * kChunkSize + localJ);
                const float expected =
                    ComputeExpectedAt(tc, inputs, head, token, localJ);
                const bool padding = chunkBegin + localJ >= chunkEnd;
                UpdateCheckStats(stats,
                                 actual[index],
                                 expected,
                                 head,
                                 token,
                                 localJ,
                                 padding,
                                 printed);
            }
        }
    }

    std::printf("  check: total=%zu mismatch=%zu paddingMismatch=%zu "
                "nonFinite=%zu maxAbs=%.9g maxRel=%.9g "
                "at(h=%ld,t=%ld,j=%ld)\n",
                stats.total,
                stats.mismatch,
                stats.paddingMismatch,
                stats.nonFinite,
                stats.maxAbs,
                stats.maxRel,
                stats.maxHead,
                stats.maxToken,
                stats.maxLocalJ);
    return stats.mismatch == 0;
}

std::vector<std::pair<int64_t, int64_t>> BuildSampleRows(const TestCase& tc,
                                                          int targetRows)
{
    targetRows = static_cast<int>(std::min<int64_t>(
        targetRows, tc.h * tc.t));
    std::vector<std::pair<int64_t, int64_t>> rows;
    std::unordered_set<uint64_t> seen;

    const auto addRow = [&](int64_t head, int64_t token) {
        if (head < 0 || head >= tc.h || token < 0 || token >= tc.t) {
            return;
        }
        const uint64_t key =
            (static_cast<uint64_t>(head) << 32U) | static_cast<uint32_t>(token);
        if (seen.insert(key).second) {
            rows.emplace_back(head, token);
        }
    };

    addRow(0, 0);
    addRow(tc.h - 1, tc.t - 1);
    addRow(tc.h / 2, tc.t / 2);

    for (size_t chunk = 0;
         chunk + 1 < tc.chunkOffsets.size() &&
         static_cast<int>(rows.size()) < targetRows;
         ++chunk) {
        const int64_t begin = tc.chunkOffsets[chunk];
        const int64_t end = tc.chunkOffsets[chunk + 1];
        addRow(0, begin);
        addRow(tc.h - 1, end - 1);
    }

    uint64_t state = 0x9E3779B97F4A7C15ULL ^
                     static_cast<uint64_t>(tc.t * 131 + tc.h * 17 + tc.k);
    while (static_cast<int>(rows.size()) < targetRows) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        const int64_t head = static_cast<int64_t>(state % static_cast<uint64_t>(tc.h));
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        const int64_t token = static_cast<int64_t>(state % static_cast<uint64_t>(tc.t));
        addRow(head, token);
    }

    return rows;
}

bool ValidateSampledOutput(const TestCase& tc,
                           const HostInputs& inputs,
                           const DeviceTensor& output,
                           int sampledRows)
{
    if (sampledRows <= 0) {
        return true;
    }

    const std::vector<std::pair<int64_t, int64_t>> rows =
        BuildSampleRows(tc, sampledRows);
    std::vector<float> actualRow(static_cast<size_t>(kChunkSize), 0.0F);
    CheckStats stats;
    size_t printed = 0;

    for (const auto& row : rows) {
        const int64_t head = row.first;
        const int64_t token = row.second;
        const size_t rowElementOffset = static_cast<size_t>(
            (head * tc.t + token) * kChunkSize);
        const void* deviceRow = static_cast<const void*>(
            static_cast<const char*>(output.addr) + rowElementOffset * sizeof(float));

        CHECK_ACL(aclrtMemcpy(actualRow.data(),
                             actualRow.size() * sizeof(float),
                             deviceRow,
                             actualRow.size() * sizeof(float),
                             ACL_MEMCPY_DEVICE_TO_HOST));

        const size_t chunk = FindChunkIndex(tc, token);
        const int64_t chunkBegin = tc.chunkOffsets[chunk];
        const int64_t chunkEnd = tc.chunkOffsets[chunk + 1];

        for (int64_t localJ = 0; localJ < kChunkSize; ++localJ) {
            const float expected =
                ComputeExpectedAt(tc, inputs, head, token, localJ);
            const bool padding = chunkBegin + localJ >= chunkEnd;
            UpdateCheckStats(stats,
                             actualRow[static_cast<size_t>(localJ)],
                             expected,
                             head,
                             token,
                             localJ,
                             padding,
                             printed);
        }
    }

    std::printf("  sampled check: rows=%zu values=%zu mismatch=%zu "
                "paddingMismatch=%zu nonFinite=%zu maxAbs=%.9g maxRel=%.9g "
                "at(h=%ld,t=%ld,j=%ld)\n",
                rows.size(),
                stats.total,
                stats.mismatch,
                stats.paddingMismatch,
                stats.nonFinite,
                stats.maxAbs,
                stats.maxRel,
                stats.maxHead,
                stats.maxToken,
                stats.maxLocalJ);
    return stats.mismatch == 0;
}

std::vector<int32_t> BuildCuSeqlens(int chunkSize, int numChunks)
            {
            std::vector<int32_t> v;
            v.reserve(numChunks + 1);
            for (int i = 0; i <= numChunks; ++i)
            v.push_back(i * chunkSize);
        return v;
            }

const char* SuiteName(Suite suite)
{
    switch (suite) {
        case Suite::CORRECTNESS:
            return "correctness";
        case Suite::PERFORMANCE:
            return "performance";
        case Suite::STRESS:
            return "stress";
        case Suite::PROFILE:
            return "profile";
    }
    return "unknown";
}

std::vector<TestCase> BuildTestCases()
{
    return {
        // Small cases: full golden comparison and edge-condition regression.
        {"full64_monotonic_k128", Suite::CORRECTNESS,
         64, 1, 1, 128, {0, 64}, GatePattern::MONOTONIC_CUMSUM,
         VerifyMode::FULL, 1, 3, 0},

        {"full64_nonmonotonic_row0_k128", Suite::CORRECTNESS,
         64, 2, 1, 128, {0, 64}, GatePattern::ROW_ZERO_ACTIVE,
         VerifyMode::FULL, 1, 3, 0},

        {"full64_numrepeat1_k256", Suite::CORRECTNESS,
         64, 4, 4, 256, {0, 64}, GatePattern::NON_MONOTONIC,
         VerifyMode::FULL, 1, 3, 0},

        {"tail63_k128", Suite::CORRECTNESS,
         63, 4, 2, 128, {0, 63}, GatePattern::NON_MONOTONIC,
         VerifyMode::FULL, 1, 3, 0},

        {"tail33_k256", Suite::CORRECTNESS,
         33, 4, 2, 256, {0, 33}, GatePattern::WIDE_RANGE,
         VerifyMode::FULL, 1, 3, 0},

        {"tail32_all_equal_k128", Suite::CORRECTNESS,
         32, 8, 1, 128, {0, 32}, GatePattern::ALL_EQUAL,
         VerifyMode::FULL, 1, 3, 0},

        {"single_token_tail_k128", Suite::CORRECTNESS,
         1, 1, 1, 128, {0, 1}, GatePattern::ROW_ZERO_ACTIVE,
         VerifyMode::FULL, 1, 3, 0},

        {"near_zero_diff_k256", Suite::CORRECTNESS,
         64, 4, 2, 256, {0, 64}, GatePattern::NEAR_ZERO_DIFF,
         VerifyMode::FULL, 1, 3, 0},

        {"mixed_full_tail_multiseq_k128", Suite::CORRECTNESS,
         300, 8, 2, 128,
         BuildChunkOffsets({0, 65, 164, 229, 300}),
         GatePattern::MONOTONIC_CUMSUM,
         VerifyMode::FULL, 1, 3, 0},

        {"many_heads_high_reuse_small_k128", Suite::CORRECTNESS,
         129, 128, 1, 128, {0, 64, 128, 129},
         GatePattern::NON_MONOTONIC,
         VerifyMode::FULL, 1, 3, 0},

        {"max_hg_small_k256", Suite::CORRECTNESS,
         129, 32, 32, 256, {0, 64, 128, 129},
         GatePattern::MONOTONIC_CUMSUM,
         VerifyMode::FULL, 1, 3, 0},

        // Performance cases: realistic sizes, warmup, repeated execution,
        // and sampled post-run verification outside the timed region.
        {"perf_official_like_k128", Suite::PERFORMANCE,
         10016, 8, 2, 128,
         BuildChunkOffsets({0, 10016}),
         GatePattern::MONOTONIC_CUMSUM,
         VerifyMode::SAMPLED, 5, 20, 16},

        {"perf_official_like_k256", Suite::PERFORMANCE,
         10016, 8, 2, 256,
         BuildChunkOffsets({0, 10016}),
         GatePattern::NON_MONOTONIC,
         VerifyMode::SAMPLED, 5, 20, 16},

        {"perf_large_t_multiseq_k128", Suite::PERFORMANCE,
         65536, 8, 2, 128,
         BuildChunkOffsets({0, 16384, 32768, 49152, 65536}),
         GatePattern::MONOTONIC_CUMSUM,
         VerifyMode::SAMPLED, 5, 20, 20},

        {"perf_high_reuse_h128_hg1", Suite::PERFORMANCE,
         4096, 128, 1, 128,
         BuildChunkOffsets({0, 4096}),
         GatePattern::NON_MONOTONIC,
         VerifyMode::SAMPLED, 5, 20, 24},

        {"perf_many_kv_h32_hg32_k256", Suite::PERFORMANCE,
         8192, 32, 32, 256,
         BuildChunkOffsets({0, 8192}),
         GatePattern::MONOTONIC_CUMSUM,
         VerifyMode::SAMPLED, 5, 15, 20},

        {"perf_varlen_4seq_k256", Suite::PERFORMANCE,
         32768, 16, 4, 256,
         BuildChunkOffsets({0, 8065, 16384, 24641, 32768}),
         GatePattern::WIDE_RANGE,
         VerifyMode::SAMPLED, 5, 20, 20},


        // Large memory/throughput pressure. Run explicitly with --suite stress.
        {"stress_max_t_light_heads", Suite::STRESS,
         262144, 8, 2, 128,
         BuildChunkOffsets({0, 65536, 131072, 196608, 262144}),
         GatePattern::MONOTONIC_CUMSUM,
         VerifyMode::SAMPLED, 3, 10, 12},
         {
        "trace_ratio1_full64_k128",Suite::PROFILE,
        1280, 32, 32, 128,
        BuildChunkOffsets({0,1280}),
        GatePattern::MONOTONIC_CUMSUM,
        VerifyMode::SAMPLED, 3, 10, 12},

    {
        "trace_ratio2_full64_k128",Suite::PROFILE,
        1280, 64, 32, 128,
        BuildChunkOffsets({0,1280}),
        GatePattern::MONOTONIC_CUMSUM,
        VerifyMode::SAMPLED, 3, 10, 12},
    {
        "trace_ratio3_full64_k128",Suite::PROFILE,
        1280, 96, 32, 128,
        BuildChunkOffsets({0, 1280}),
        GatePattern::MONOTONIC_CUMSUM,
        VerifyMode::SAMPLED, 3, 10, 12},
    {
        "trace_ratio4_full64_k128",Suite::PROFILE,
        1280, 128, 32, 128,
        BuildChunkOffsets({0, 1280}),
        GatePattern::MONOTONIC_CUMSUM,
        VerifyMode::SAMPLED, 3, 10, 12},
};
}

void PrintUsage(const char* program)
{
    std::printf(
        "Usage: %s [options]\n"
        "\n"
        "Options:\n"
        "  --suite correctness|perf|stress|all  Select suite (default: correctness)\n"
        "  --case NAME                         Run one named case\n"
        "  --device ID                         Device ID (default: 0)\n"
        "  --warmup N                          Override warmup count\n"
        "  --repeat N                          Override measured repeat count\n"
        "  --samples N                         Override sampled validation rows\n"
        "  --no-check                          Skip output validation\n"
        "  --list                              List available cases\n"
        "  --help                              Show this help\n"
        "\n"
        "Examples:\n"
        "  %s --suite correctness\n"
        "  %s --suite perf\n"
        "  %s --case perf_official_like_k128\n"
        "  %s --case perf_official_like_k128 --warmup 1 --repeat 1 --no-check\n",
        program, program, program, program, program);
}

bool ParseNonNegativeInt(const char* text, int& value)
{
    if (text == nullptr || *text == '\0') {
        return false;
    }
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (*end != '\0' || parsed < 0 || parsed > std::numeric_limits<int>::max()) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

bool ParseOptions(int argc, char** argv, Options& options)
{
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto requireValue = [&](const char* option) -> const char* {
            if (i + 1 >= argc) {
                std::printf("Missing value for %s\n", option);
                return nullptr;
            }
            return argv[++i];
        };

        if (arg == "--suite") {
            const char* value = requireValue("--suite");
            if (value == nullptr) {
                return false;
            }
            options.suite = value;
        } else if (arg == "--case") {
            const char* value = requireValue("--case");
            if (value == nullptr) {
                return false;
            }
            options.caseName = value;
        } else if (arg == "--device") {
            const char* value = requireValue("--device");
            int parsed = 0;
            if (value == nullptr || !ParseNonNegativeInt(value, parsed)) {
                std::printf("Invalid --device value\n");
                return false;
            }
            options.deviceId = parsed;
        } else if (arg == "--warmup") {
            const char* value = requireValue("--warmup");
            if (value == nullptr ||
                !ParseNonNegativeInt(value, options.warmupOverride)) {
                std::printf("Invalid --warmup value\n");
                return false;
            }
        } else if (arg == "--repeat") {
            const char* value = requireValue("--repeat");
            if (value == nullptr ||
                !ParseNonNegativeInt(value, options.repeatOverride) ||
                options.repeatOverride == 0) {
                std::printf("Invalid --repeat value\n");
                return false;
            }
        } else if (arg == "--samples") {
            const char* value = requireValue("--samples");
            if (value == nullptr ||
                !ParseNonNegativeInt(value, options.sampledRowsOverride)) {
                std::printf("Invalid --samples value\n");
                return false;
            }
        } else if (arg == "--no-check") {
            options.disableCheck = true;
        } else if (arg == "--list") {
            options.listOnly = true;
        } else if (arg == "--help" || arg == "-h") {
            options.showHelp = true;
        } else {
            std::printf("Unknown option: %s\n", arg.c_str());
            return false;
        }
    }

    const bool validSuite =
        options.suite == "correctness" || options.suite == "perf" ||
        options.suite == "performance" || options.suite == "stress" ||
        options.suite == "profile" || options.suite == "all";
    if (!validSuite) {
        std::printf("Invalid suite: %s\n", options.suite.c_str());
        return false;
    }
    return true;
}

bool ShouldRun(const TestCase& tc, const Options& options)
{
    if (!options.caseName.empty()) {
        return tc.name == options.caseName;
    }
    if (options.suite == "all") {
        return tc.suite != Suite::STRESS;
    }
    if (options.suite == "correctness") {
        return tc.suite == Suite::CORRECTNESS;
    }
    if (options.suite == "perf" || options.suite == "performance") {
        return tc.suite == Suite::PERFORMANCE;

    }
    if (options.suite == "profile") {
        return tc.suite == Suite::PROFILE;
    }
    return tc.suite == Suite::STRESS;
}

void PrintCaseList(const std::vector<TestCase>& cases)
{
    for (const TestCase& tc : cases) {
        const double outputMiB =
            static_cast<double>(tc.h * tc.t * kChunkSize * sizeof(float)) /
            (1024.0 * 1024.0);
        std::printf("%-38s suite=%-11s T=%-7ld H=%-3ld Hg=%-3ld K=%-3ld "
                    "chunks=%-5zu output=%.1f MiB\n",
                    tc.name.c_str(),
                    SuiteName(tc.suite),
                    tc.t,
                    tc.h,
                    tc.hg,
                    tc.k,
                    tc.chunkOffsets.size() - 1,
                    outputMiB);
    }
}

bool LaunchOperator(void* workspaceAddr,
                    uint64_t workspaceSize,
                    aclOpExecutor* executor,
                    aclrtStream stream)
{
    const int ret = aclnnChunkScaledDotKkt(
        workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        std::printf("aclnnChunkScaledDotKkt failed, ret=%d\n", ret);
        return false;
    }
    return true;
}

bool RunOneCase(TestCase tc, const Options& options, aclrtStream stream)
{
    if (options.warmupOverride >= 0) {
        tc.warmup = options.warmupOverride;
    }
    if (options.repeatOverride >= 0) {
        tc.repeat = options.repeatOverride;
    }
    if (options.sampledRowsOverride >= 0) {
        tc.sampledRows = options.sampledRowsOverride;
    }
    if (options.disableCheck) {
        tc.verifyMode = VerifyMode::NONE;
    }

    std::printf("\n[ RUN      ] %s\n", tc.name.c_str());
    if (!ValidateCase(tc)) {
        std::printf("  Invalid test configuration\n");
        return false;
    }

    const int64_t numChunks = static_cast<int64_t>(tc.chunkOffsets.size()) - 1;
    const int64_t logicalTasks = numChunks * tc.hg;
    const double outputMiB =
        static_cast<double>(tc.h * tc.t * kChunkSize * sizeof(float)) /
        (1024.0 * 1024.0);

    std::printf("  suite=%s T=%ld H=%ld Hg=%ld K=%ld chunks=%ld "
                "logicalTasks=%ld warmup=%d repeat=%d output=%.1f MiB\n",
                SuiteName(tc.suite),
                tc.t,
                tc.h,
                tc.hg,
                tc.k,
                numChunks,
                logicalTasks,
                tc.warmup,
                tc.repeat,
                outputMiB);

    HostInputs inputs;
    FillInputs(tc, inputs);

    const std::vector<int64_t> kShape = {1, tc.t, tc.hg, tc.k};
    const std::vector<int64_t> betaShape = {1, tc.h, tc.t};
    const std::vector<int64_t> gShape = {1, tc.h, tc.t};
    const std::vector<int64_t> offsetsShape = {
        static_cast<int64_t>(tc.chunkOffsets.size())};
    const std::vector<int64_t> outputShape = {1, tc.h, tc.t, kChunkSize};

    DeviceTensor kTensor;
    DeviceTensor betaTensor;
    DeviceTensor gTensor;
    DeviceTensor offsetsTensor;
    DeviceTensor outputTensor;

    if (!CreateTensorFromHost(inputs.k, kShape, ACL_BF16, kTensor) ||
        !CreateTensorFromHost(inputs.beta, betaShape, ACL_BF16, betaTensor) ||
        !CreateTensorFromHost(inputs.g, gShape, ACL_FLOAT, gTensor) ||
        !CreateTensorFromHost(tc.chunkOffsets,
                              offsetsShape,
                              ACL_INT32,
                              offsetsTensor) ||
        !CreateOutputTensor(outputShape, outputTensor)) {
        return false;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    int ret = aclnnChunkScaledDotKktGetWorkspaceSize(
        kTensor.tensor,
        betaTensor.tensor,
        gTensor.tensor,
        offsetsTensor.tensor,
        kChunkSize,
        outputTensor.tensor,
        &workspaceSize,
        &executor);
    if (ret != ACL_SUCCESS || executor == nullptr) {
        std::printf("GetWorkspaceSize failed, ret=%d executor=%p\n",
                    ret,
                    static_cast<void*>(executor));
        if (executor != nullptr) {
            aclDestroyAclOpExecutor(executor);
        }
        return false;
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr,
                          workspaceSize,
                          ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            std::printf("Workspace allocation failed, size=%lu ret=%d\n",
                        static_cast<unsigned long>(workspaceSize),
                        ret);
            aclDestroyAclOpExecutor(executor);
            return false;
        }
    }

    ret = aclSetAclOpExecutorRepeatable(executor);
    if (ret != ACL_SUCCESS) {
        std::printf("aclSetAclOpExecutorRepeatable failed, ret=%d. "
                    "This benchmark requires a repeatable executor.\n",
                    ret);
        if (workspaceAddr != nullptr) {
            aclrtFree(workspaceAddr);
        }
        aclDestroyAclOpExecutor(executor);
        return false;
    }

    bool success = true;

    for (int i = 0; i < tc.warmup; ++i) {
        if (!LaunchOperator(workspaceAddr,
                            workspaceSize,
                            executor,
                            stream)) {
            success = false;
            break;
        }
    }
    if (success && aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        std::printf("Warmup stream synchronization failed\n");
        success = false;
    }

    EventPair events;
    if (success &&
        (aclrtCreateEvent(&events.start) != ACL_SUCCESS ||
         aclrtCreateEvent(&events.end) != ACL_SUCCESS)) {
        std::printf("Failed to create timing events\n");
        success = false;
    }

    float deviceTotalMs = 0.0F;
    double wallTotalMs = 0.0;

    if (success) {
        if (aclrtRecordEvent(events.start, stream) != ACL_SUCCESS) {
            std::printf("Failed to record start event\n");
            success = false;
        }
    }

    const auto wallBegin = std::chrono::steady_clock::now();
    if (success) {
        for (int i = 0; i < tc.repeat; ++i) {
            if (!LaunchOperator(workspaceAddr,
                                workspaceSize,
                                executor,
                                stream)) {
                success = false;
                break;
            }
        }
    }

    if (success && aclrtRecordEvent(events.end, stream) != ACL_SUCCESS) {
        std::printf("Failed to record end event\n");
        success = false;
    }
    if (success && aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        std::printf("Measured stream synchronization failed\n");
        success = false;
    }
    const auto wallEnd = std::chrono::steady_clock::now();

    if (success) {
        wallTotalMs = std::chrono::duration<double, std::milli>(
            wallEnd - wallBegin).count();
        if (aclrtEventElapsedTime(&deviceTotalMs,
                                  events.start,
                                  events.end) != ACL_SUCCESS) {
            std::printf("aclrtEventElapsedTime failed\n");
            success = false;
        }
    }

    if (success) {
        const double deviceAverageUs =
            static_cast<double>(deviceTotalMs) * 1000.0 / tc.repeat;
        const double wallAverageUs = wallTotalMs * 1000.0 / tc.repeat;
        std::printf("  timing: device_total=%.3f ms device_avg=%.3f us "
                    "wall_avg=%.3f us\n",
                    deviceTotalMs,
                    deviceAverageUs,
                    wallAverageUs);
        std::printf("RESULT,%s,%s,T=%ld,H=%ld,Hg=%ld,K=%ld,chunks=%ld,"
                    "tasks=%ld,warmup=%d,repeat=%d,device_avg_us=%.3f,"
                    "wall_avg_us=%.3f\n",
                    tc.name.c_str(),
                    SuiteName(tc.suite),
                    tc.t,
                    tc.h,
                    tc.hg,
                    tc.k,
                    numChunks,
                    logicalTasks,
                    tc.warmup,
                    tc.repeat,
                    deviceAverageUs,
                    wallAverageUs);
    }

    if (success && tc.verifyMode == VerifyMode::FULL) {
        success = ValidateFullOutput(tc, inputs, outputTensor);
    } else if (success && tc.verifyMode == VerifyMode::SAMPLED) {
        success = ValidateSampledOutput(tc,
                                        inputs,
                                        outputTensor,
                                        tc.sampledRows);
    }

    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    aclDestroyAclOpExecutor(executor);

    std::printf("[ %s ] %s\n",
                success ? "       OK" : "  FAILED ",
                tc.name.c_str());
    return success;
}

}  // namespace

int main(int argc, char** argv)
{
    Options options;
    if (!ParseOptions(argc, argv, options)) {
        PrintUsage(argv[0]);
        return 2;
    }
    if (options.showHelp) {
        PrintUsage(argv[0]);
        return 0;
    }

    const std::vector<TestCase> cases = BuildTestCases();
    if (options.listOnly) {
        PrintCaseList(cases);
        return 0;
    }

    size_t selected = 0;
    for (const TestCase& tc : cases) {
        if (ShouldRun(tc, options)) {
            ++selected;
        }
    }
    if (selected == 0) {
        std::printf("No matching test case. Use --list to inspect names.\n");
        return 2;
    }

    int ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        std::printf("aclInit failed, ret=%d\n", ret);
        return ret;
    }

    ret = aclrtSetDevice(options.deviceId);
    if (ret != ACL_SUCCESS) {
        std::printf("aclrtSetDevice failed, ret=%d\n", ret);
        aclFinalize();
        return ret;
    }

    aclrtStream stream = nullptr;
    ret = aclrtCreateStream(&stream);
    if (ret != ACL_SUCCESS) {
        std::printf("aclrtCreateStream failed, ret=%d\n", ret);
        aclrtResetDevice(options.deviceId);
        aclFinalize();
        return ret;
    }

    size_t passed = 0;
    for (const TestCase& tc : cases) {
        if (!ShouldRun(tc, options)) {
            continue;
        }
        if (RunOneCase(tc, options, stream)) {
            ++passed;
        }
    }

    std::printf("\n========================================\n");
    std::printf("Passed %zu / %zu selected cases\n", passed, selected);
    std::printf("========================================\n");

    aclrtDestroyStream(stream);
    aclrtResetDevice(options.deviceId);
    aclFinalize();

    return passed == selected ? 0 : 1;
}