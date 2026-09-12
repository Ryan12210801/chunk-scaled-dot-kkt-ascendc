#include <iostream>
#include <vector>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include "acl/acl.h"
#include "aclnn_chunk_scaled_dot_kkt.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}


static uint16_t FloatToHalf(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = ((bits >> 23) & 0xff) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3ff;
    if (exp <= 0) return sign;
    if (exp >= 31) return sign | 0x7c00;
    return sign | (exp << 10) | mant;
}

static uint16_t FloatToBFloat16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    return (uint16_t)(bits >> 16);
}

template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto elemCount = GetShapeSize(shape);
    int64_t elemSize = sizeof(T);
    switch (dataType) {
        case aclDataType::ACL_FLOAT16:
        case aclDataType::ACL_BF16:
        case aclDataType::ACL_INT16:
        case aclDataType::ACL_UINT16:
            elemSize = 2;
            break;
        case aclDataType::ACL_INT8:
        case aclDataType::ACL_UINT8:
        case aclDataType::ACL_BOOL:
            elemSize = 1;
            break;
        case aclDataType::ACL_INT64:
        case aclDataType::ACL_UINT64:
        case aclDataType::ACL_DOUBLE:
            elemSize = 8;
            break;
        default:
            break;
    }
    auto size = elemCount * elemSize;
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    std::vector<uint8_t> convBuf(size);
    if (dataType == aclDataType::ACL_FLOAT16) {
        for (int64_t i = 0; i < elemCount; i++) {
            uint16_t h = FloatToHalf(static_cast<float>(hostData[i]));
            memcpy(convBuf.data() + i * 2, &h, 2);
        }
    } else if (dataType == aclDataType::ACL_BF16) {
        for (int64_t i = 0; i < elemCount; i++) {
            uint16_t b = FloatToBFloat16(static_cast<float>(hostData[i]));
            memcpy(convBuf.data() + i * 2, &b, 2);
        }
    } else if (dataType == aclDataType::ACL_DOUBLE) {
        for (int64_t i = 0; i < elemCount; i++) {
            double d = static_cast<double>(hostData[i]);
            memcpy(convBuf.data() + i * 8, &d, 8);
        }
    } else {
        auto copySize = std::min((int64_t)(elemCount * sizeof(T)), size);
        memcpy(convBuf.data(), hostData.data(), copySize);
    }
    ret = aclrtMemcpy(*deviceAddr, size, convBuf.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 构造输入 tensor
    aclTensor* k = nullptr;
    void* kDeviceAddr = nullptr;
    aclTensor* beta = nullptr;
    void* betaDeviceAddr = nullptr;
    aclTensor* g_cumsum = nullptr;
    void* g_cumsumDeviceAddr = nullptr;
    aclTensor* chunk_offsets = nullptr;
    void* chunk_offsetsDeviceAddr = nullptr;
    std::vector<int64_t> kShape = {1, 10016, 2, 128};
    std::vector<float> kHostData(2564096, 1);
    ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_BF16, &k);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::vector<int64_t> betaShape = {1, 8, 10016};
    std::vector<float> betaHostData(80128, 1);
    ret = CreateAclTensor(betaHostData, betaShape, &betaDeviceAddr, aclDataType::ACL_BF16, &beta);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::vector<int64_t> g_cumsumShape = {1, 8, 10016};
    std::vector<float> g_cumsumHostData(80128, 1);
    ret = CreateAclTensor(g_cumsumHostData, g_cumsumShape, &g_cumsumDeviceAddr, aclDataType::ACL_FLOAT, &g_cumsum);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::vector<int64_t> chunk_offsetsShape = {160};
    std::vector<int32_t> chunk_offsetsHostData(160, 1);
    ret = CreateAclTensor(chunk_offsetsHostData, chunk_offsetsShape, &chunk_offsetsDeviceAddr, aclDataType::ACL_INT32, &chunk_offsets);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 构造输出 tensor
    aclTensor* A = nullptr;
    void* ADeviceAddr = nullptr;
    std::vector<int64_t> AShape = {1, 8, 10016, 64};
    std::vector<float> AHostData(5128192, 0);
    ret = CreateAclTensor(AHostData, AShape, &ADeviceAddr, aclDataType::ACL_FLOAT, &A);
    CHECK_RET(ret == ACL_SUCCESS, return ret);


    // 调用 aclnnChunkScaledDotKktGetWorkspaceSize 第一段接口
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnChunkScaledDotKktGetWorkspaceSize(k, beta, g_cumsum, chunk_offsets, 64, A, &workspaceSize, &executor);
    CHECK_RET(ret == 0, LOG_PRINT("aclnnChunkScaledDotKktGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 申请 workspace
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 调用 aclnnChunkScaledDotKkt 第二段接口
    ret = aclnnChunkScaledDotKkt(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == 0, LOG_PRINT("aclnnChunkScaledDotKkt failed. ERROR: %d\n", ret); return ret);

    // 同步等待
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 释放资源
    aclDestroyTensor(k);
    aclrtFree(kDeviceAddr);
    aclDestroyTensor(beta);
    aclrtFree(betaDeviceAddr);
    aclDestroyTensor(g_cumsum);
    aclrtFree(g_cumsumDeviceAddr);
    aclDestroyTensor(chunk_offsets);
    aclrtFree(chunk_offsetsDeviceAddr);

    aclDestroyTensor(A);
    aclrtFree(ADeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
