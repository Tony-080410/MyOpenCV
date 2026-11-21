#pragma once

#include <cstddef>

namespace mycv {
// 深度常量
constexpr int DEPTH_8U = 0;
constexpr int DEPTH_8S = 1;
constexpr int DEPTH_16U = 2;
constexpr int DEPTH_16S = 3;
constexpr int DEPTH_32S = 4;
constexpr int DEPTH_32F = 5;
constexpr int DEPTH_64F = 6;

// 编码函数
constexpr int makeType(int depth, int cn) { return depth + ((cn - 1) << 3); }
constexpr int depthOf(int t) { return t & 7; }
constexpr int channels(int t) { return (t >> 3) + 1; }

// 类型常量
constexpr int U8C1 = makeType(DEPTH_8U, 1);
constexpr int U8C3 = makeType(DEPTH_8U, 3);
constexpr int S16C1 = makeType(DEPTH_16S, 1); // spatialGradient 的输出类型
constexpr int F32C1 = makeType(DEPTH_32F, 1);

// 元素大小
constexpr size_t elemSize1(int t) {
    switch (depthOf(t)) {
    case DEPTH_8U:
    case DEPTH_8S:
        return 1;
    case DEPTH_16U:
    case DEPTH_16S:
        return 2;
    case DEPTH_32S:
    case DEPTH_32F:
        return 4;
    case DEPTH_64F:
        return 8;
    default:
        return 0;
    }
}

// 像素字节数 = 单通道 × 通道数
constexpr size_t elemSize(int t) { return elemSize1(t) * channels(t); }

struct Point {
    int x = 0, y = 0;
};
struct Size {
    int width = 0, height = 0;
};
struct Rect {
    int x = 0, y = 0, width = 0, height = 0;
};
struct Range {
    int start = 0, end = 0;
};
struct Scalar {
    double v[4] = {0.0, 0.0, 0.0, 0.0};
};

struct Point2f {
    float x = 0, y = 0;
};
struct Size2f {
    float width = 0, height = 0;
};

// 旋转矩形：center 是中心，size.width >= size.height，
// angle 是长边与 x 轴的夹角（度），范围 [0, 90)。
// 注意和 OpenCV 的约定不同：cv::RotatedRect 的 angle 落在 (0, 90]，宽高方向相反
struct RotatedRect {
    Point2f center;
    Size2f size;
    float angle = 0;
};

} // namespace mycv
