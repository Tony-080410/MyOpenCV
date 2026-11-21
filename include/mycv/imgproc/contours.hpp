#pragma once

#include <vector>

#include <mycv/core/mat.hpp>
#include <mycv/core/types.hpp>

namespace mycv {
// 从二值图提取外轮廓，用 Moore 邻域跟踪。
// 输入必须是 U8C1，非零当前景。每条轮廓的点是闭合的（首尾不重复）。
// 只取外轮廓，内孔不单独返回。
std::vector<std::vector<Point>> findContours(const Mat &binary);

// 轮廓的外接矩形（轴对齐）
Rect boundingRect(const std::vector<Point> &contour);

// 轮廓面积（鞋带公式）
double contourArea(const std::vector<Point> &contour);

// 凸包，顺时针，去掉共线点。点数少于 3 时返回原值
std::vector<Point> convexHull(std::vector<Point> pts);

// 最小面积外接旋转矩形。内部先求凸包，再沿每条凸包边方向求包围盒取最小的
RotatedRect minAreaRect(const std::vector<Point> &contour);

} // namespace mycv
