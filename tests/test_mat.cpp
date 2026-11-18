#include <mycv/core/types.hpp>

// types.hpp 的编译期验收
static_assert(mycv::U8C1 == 0, "U8C1 应是深度 8U + 单通道");
static_assert(mycv::U8C3 == 16, "U8C3 = 0 + (3-1)<<3");
static_assert(mycv::F32C1 == 5, "F32C1 的 type 值等于 DEPTH_32F");
static_assert(mycv::channels(mycv::U8C3) == 3, "channels 先右移再 +1");
static_assert(mycv::channels(mycv::U8C1) == 1, "单通道不能算成 0");
static_assert(mycv::depthOf(mycv::F32C1) == mycv::DEPTH_32F, "depthOf 取低 3 位");
static_assert(mycv::elemSize1(mycv::F32C1) == 4, "float 单通道 4 字节");
static_assert(mycv::elemSize(mycv::U8C1) == 1, "U8C1 一像素一字节");
static_assert(mycv::elemSize(mycv::U8C3) == 3, "U8C3 一像素三字节");
static_assert(mycv::elemSize(mycv::F32C1) == 4, "F32C1 一像素四字节");
static_assert(mycv::elemSize(7) == 0, "未支持的深度落到 default 返回 0");

int main() { return 0; }
