#pragma once

#include <mycv/core/mat.hpp>
#include <mycv/core/types.hpp>

namespace mycv {
enum BorderType { BORDER_REPLICATE = 1, BORDER_REFLECT_101 = 4 };

// 通用二维卷积。kernel 必须是 F32C1；anchor 给 {-1,-1} 表示取核中心
Mat convolve2D(const Mat &src, const Mat &kernel, Point anchor = {-1, -1});

} // namespace mycv
