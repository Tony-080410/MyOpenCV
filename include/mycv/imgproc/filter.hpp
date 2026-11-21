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

// 高斯滤波，ksize 必须是正的奇数；sigmaX/sigmaY <= 0 时按 ksize 自动取
Mat gaussianBlur(const Mat &src, Size ksize, double sigmaX = 0, double sigmaY = 0);

// 中值滤波，ksize 必须是不小于 1 的奇数
Mat medianBlur(const Mat &src, int ksize);

// 3x3 Scharr 一阶导，核比 Sobel 准（[-3,0,3;-10,0,10;-3,0,3]），不归一化
Mat scharr(const Mat &src, int dx, int dy);

// 二阶导。ksize=1 用 4 邻域核，ksize=3 用 8 邻域核；8U 输出会饱和
Mat laplacian(const Mat &src, int ksize = 1);

// Sobel 3x3 的 dx/dy 一次出，输出 16 位不饱和
void spatialGradient(const Mat &src, Mat &dx, Mat &dy);

} // namespace mycv
