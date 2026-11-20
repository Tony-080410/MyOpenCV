#pragma once

#include <mycv/core/mat.hpp>
#include <mycv/core/types.hpp>

namespace mycv {
// 就地修改 img。thickness = 1 画轮廓，-1 填实
void line(Mat &img, Point pt1, Point pt2, const Scalar &color, int thickness = 1);
void rectangle(Mat &img, Point pt1, Point pt2, const Scalar &color, int thickness = 1);
void circle(Mat &img, Point center, int radius, const Scalar &color, int thickness = 1);

// 轴对齐椭圆（目前不支持旋转角度和圆弧）
void ellipse(Mat &img, Point center, Size axes, const Scalar &color, int thickness = 1);

} // namespace mycv
