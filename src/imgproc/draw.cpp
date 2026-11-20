#include <mycv/imgproc/draw.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#include <mycv/core/check.hpp>
#include <mycv/core/types.hpp>

namespace mycv {
namespace {

uint8_t toByte(double v) {
    if (v < 0) return 0;
    if (v > 255) return 255;
    return static_cast<uint8_t>(v);
}

// 越界直接丢弃，等于对图形做了裁剪
void setPixel(Mat &img, int x, int y, const Scalar &c, int cn) {
    if (x < 0 || x >= img.cols || y < 0 || y >= img.rows) return;
    uint8_t *p = img.ptr(y) + static_cast<std::size_t>(x) * cn;
    for (int k = 0; k < cn; ++k)
        p[k] = toByte(c.v[k]);
}

void fillSpan(Mat &img, int y, int xa, int xb, const Scalar &c, int cn) {
    if (y < 0 || y >= img.rows) return;
    if (xa < 0) xa = 0;
    if (xb >= img.cols) xb = img.cols - 1;
    for (int x = xa; x <= xb; ++x)
        setPixel(img, x, y, c, cn);
}

void drawLine(Mat &img, Point a, Point b, const Scalar &c, int cn) {
    int x0 = a.x, y0 = a.y;
    const int x1 = b.x, y1 = b.y;
    const int dx = std::abs(x1 - x0);
    const int dy = std::abs(y1 - y0);
    const int sx = (x0 < x1) ? 1 : -1;
    const int sy = (y0 < y1) ? 1 : -1;
    int err = dx - dy;

    for (;;) {
        setPixel(img, x0, y0, c, cn);
        if (x0 == x1 && y0 == y1) break;
        const int e2 = 2 * err;
        if (e2 > -dy) {
            err -= dy;
            x0 += sx;
        }
        if (e2 < dx) {
            err += dx;
            y0 += sy;
        }
    }
}

// Zingl 的整数椭圆算法。参数是外接矩形的四条边，A/B 是直径
void ellipseOutline(Mat &img, int cx, int cy, int rx, int ry, const Scalar &c, int cn) {
    int x0 = cx - rx, x1 = cx + rx;
    int y0 = cy - ry, y1 = cy + ry;

    const long A = std::abs(x1 - x0);
    const long B = std::abs(y1 - y0);
    long b1 = B % 2;
    double dx = 4.0 * (1.0 - static_cast<double>(A)) * B * B;
    double dy = 4.0 * static_cast<double>(b1 + 1) * A * A;
    double err = dx + dy + static_cast<double>(b1) * A * A;

    y0 += static_cast<int>((B + 1) / 2);
    y1 = y0 - static_cast<int>(b1);
    const long aInc = 8 * A * A;
    const long bInc = 8 * B * B;

    for (;;) {
        setPixel(img, x1, y0, c, cn);
        setPixel(img, x0, y0, c, cn);
        setPixel(img, x0, y1, c, cn);
        setPixel(img, x1, y1, c, cn);

        const double e2 = 2 * err;
        if (e2 <= dy) {
            ++y0;
            --y1;
            dy += static_cast<double>(aInc);
            err += dy;
        }
        if (e2 >= dx || 2 * err > dy) {
            ++x0;
            --x1;
            dx += static_cast<double>(bInc);
            err += dx;
        }
        if (x0 > x1) break;
    }

    // 扁椭圆（rx 很小）提前停，补齐两端尖角
    while (y0 - y1 <= B) {
        setPixel(img, x0 - 1, y0, c, cn);
        setPixel(img, x1 + 1, y0, c, cn);
        setPixel(img, x0 - 1, y1, c, cn);
        setPixel(img, x1 + 1, y1, c, cn);
        ++y0;
        --y1;
    }
}

void fillEllipse(Mat &img, int cx, int cy, int rx, int ry, const Scalar &c, int cn) {
    for (int dy = -ry; dy <= ry; ++dy) {
        const double t = 1.0 - (static_cast<double>(dy) * dy) / (static_cast<double>(ry) * ry);
        const int dx = static_cast<int>(std::lround(rx * std::sqrt(t > 0.0 ? t : 0.0)));
        fillSpan(img, cy + dy, cx - dx, cx + dx, c, cn);
    }
}

} // namespace

void line(Mat &img, Point pt1, Point pt2, const Scalar &color, int thickness) {
    MYCV_ASSERT(!img.empty());
    MYCV_ASSERT(depthOf(img.type) == DEPTH_8U);
    if (thickness != 1 && thickness != -1)
        throw std::runtime_error("line: thickness 只支持 1 和 -1");

    drawLine(img, pt1, pt2, color, channels(img.type));
}

void rectangle(Mat &img, Point pt1, Point pt2, const Scalar &color, int thickness) {
    MYCV_ASSERT(!img.empty());
    MYCV_ASSERT(depthOf(img.type) == DEPTH_8U);

    const int cn = channels(img.type);
    const int xmin = std::min(pt1.x, pt2.x), xmax = std::max(pt1.x, pt2.x);
    const int ymin = std::min(pt1.y, pt2.y), ymax = std::max(pt1.y, pt2.y);

    if (thickness == -1) {
        for (int y = ymin; y <= ymax; ++y)
            fillSpan(img, y, xmin, xmax, color, cn);
        return;
    }
    if (thickness != 1) throw std::runtime_error("rectangle: thickness 只支持 1 和 -1");

    drawLine(img, {xmin, ymin}, {xmax, ymin}, color, cn);
    drawLine(img, {xmin, ymax}, {xmax, ymax}, color, cn);
    drawLine(img, {xmin, ymin}, {xmin, ymax}, color, cn);
    drawLine(img, {xmax, ymin}, {xmax, ymax}, color, cn);
}

void ellipse(Mat &img, Point center, Size axes, const Scalar &color, int thickness) {
    MYCV_ASSERT(!img.empty());
    MYCV_ASSERT(depthOf(img.type) == DEPTH_8U);
    MYCV_ASSERT(axes.width > 0 && axes.height > 0);
    if (thickness != 1 && thickness != -1)
        throw std::runtime_error("ellipse: thickness 只支持 1 和 -1");

    const int cn = channels(img.type);
    if (thickness == -1) {
        fillEllipse(img, center.x, center.y, axes.width, axes.height, color, cn);
    } else {
        ellipseOutline(img, center.x, center.y, axes.width, axes.height, color, cn);
    }
}

void circle(Mat &img, Point center, int radius, const Scalar &color, int thickness) {
    MYCV_ASSERT(radius > 0);
    ellipse(img, center, {radius, radius}, color, thickness); // 圆是轴长相等的椭圆
}

} // namespace mycv