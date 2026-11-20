#pragma once

#include <mycv/core/mat.hpp>
#include <mycv/core/types.hpp>

namespace mycv {
enum BorderType { BORDER_REPLICATE = 1, BORDER_REFLECT_101 = 4 };

// 通用二维卷积。kernel 必须是 F32C1；anchor 给 {-1,-1} 表示取核中心
Mat convolve2D(const Mat &src, const Mat &kernel, Point anchor = {-1, -1});

// 方框均值滤波，ksize 必须是正的奇数
Mat blur(const Mat &src, Size ksize);

// 一阶 Sobel，目前只支持 ksize=3 且 dx+dy==1
Mat sobel(const Mat &src, int dx, int dy, int ksize = 3);

} // namespace mycv
