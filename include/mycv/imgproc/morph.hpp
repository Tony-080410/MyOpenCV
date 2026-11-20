#pragma once

#include <mycv/core/mat.hpp>
#include <mycv/core/types.hpp>

namespace mycv {
enum MorphShape {
    MORPH_RECT = 0,
    MORPH_CROSS = 1,
    MORPH_ELLIPSE = 2, // AI实现
};

// 生成 0/1 结构元素，CV_8UC1
Mat getStructuringElement(int shape, Size ksize);

// 腐蚀 / 膨胀。kernel 必须是 CV_8UC1 的 0/1 矩阵
Mat erode(const Mat &src, const Mat &kernel);
Mat dilate(const Mat &src, const Mat &kernel);

} // namespace mycv
