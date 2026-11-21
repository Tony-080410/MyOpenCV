#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <mycv/core/check.hpp>
#include <mycv/core/types.hpp>
#include <mycv/imgproc/contours.hpp>
#include <utility>

namespace mycv {
namespace {

// 8 邻域，屏幕坐标顺时针：E SE S SW W NW N NE
constexpr int DX[8] = {1, 1, 0, -1, -1, -1, 0, 1};
constexpr int DY[8] = {0, 1, 1, 1, 0, -1, -1, -1};

// 从 start 绕一个连通域的外边界走一圈
std::vector<Point> traceOne(const Mat &bin, Point start, std::vector<uint8_t> &visited) {
    std::vector<Point> contour;
    const int W = bin.cols, H = bin.rows;

    // 找 start 的第一个白邻居，确定初始方向
    int firstDir = -1;
    for (int k = 0; k < 8; ++k) {
        const int nx = start.x + DX[k], ny = start.y + DY[k];
        if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;
        if (bin.ptr(ny)[nx]) {
            firstDir = k;
            break;
        }
    }
    if (firstDir < 0) { // 孤立单点
        contour.push_back(start);
        visited[static_cast<std::size_t>(start.y) * bin.step + start.x] = 1;
        return contour;
    }

    // 起点自己也是轮廓的一部分，漏了它圆的顶点就会少一行
    contour.push_back(start);
    visited[static_cast<std::size_t>(start.y) * bin.step + start.x] = 1;

    Point p = {start.x + DX[firstDir], start.y + DY[firstDir]};
    int fromDir = firstDir; // p 是从上一步的哪个方向走过来的

    for (;;) {
        contour.push_back(p);
        visited[static_cast<std::size_t>(p.y) * bin.step + p.x] = 1;

        // 从"来路"的顺时针下一个方向开始扫，绕一圈找下一个白点。
        // 不能从来路本身开始，否则会立刻找到上一步的点，原地打转
        const int startScan = (fromDir + 4 + 1) % 8;
        int found = -1;
        for (int k = 0; k < 8; ++k) {
            const int d = (startScan + k) % 8;
            const int nx = p.x + DX[d], ny = p.y + DY[d];
            if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;
            if (bin.ptr(ny)[nx]) {
                found = d;
                break;
            }
        }
        if (found < 0) break; // 断头，正常不该出现

        const Point n = {p.x + DX[found], p.y + DY[found]};
        if (n.x == start.x && n.y == start.y) break; // 回到起点，闭合

        fromDir = found;
        p = n;
    }
    return contour;
}

// 8 连通洪泛，把整个连通域标记掉。
// 少了这一步，"左邻居是黑的白像素"会在物体内部和孔洞边上不断触发，同一块区域被跟踪多次
void floodMark(const Mat &bin, Point seed, std::vector<uint8_t> &visited) {
    const int W = bin.cols, H = bin.rows;
    std::vector<Point> stack{seed};
    visited[static_cast<std::size_t>(seed.y) * bin.step + seed.x] = 1;

    while (!stack.empty()) {
        const Point p = stack.back();
        stack.pop_back();
        for (int k = 0; k < 8; ++k) {
            const int nx = p.x + DX[k], ny = p.y + DY[k];
            if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;
            const std::size_t idx = static_cast<std::size_t>(ny) * bin.step + nx;
            if (!bin.ptr(ny)[nx] || visited[idx]) continue;
            visited[idx] = 1;
            stack.push_back({nx, ny});
        }
    }
}

} // namespace

std::vector<std::vector<Point>> findContours(const Mat &binary) {
    MYCV_ASSERT(!binary.empty());
    MYCV_ASSERT(binary.type == U8C1);

    const int W = binary.cols, H = binary.rows;
    std::vector<uint8_t> visited(static_cast<std::size_t>(H) * binary.step, 0);
    std::vector<std::vector<Point>> out;

    for (int y = 0; y < H; ++y) {
        const uint8_t *row = binary.ptr(y);
        for (int x = 0; x < W; ++x) {
            const std::size_t idx = static_cast<std::size_t>(y) * binary.step + x;
            if (!row[x] || visited[idx]) continue;
            if (x > 0 && row[x - 1]) continue; // 只从左边界起步

            out.push_back(traceOne(binary, {x, y}, visited));
            floodMark(binary, {x, y}, visited); // 整个连通域标掉
        }
    }
    return out;
}

Rect boundingRect(const std::vector<Point> &contour) {
    if (contour.empty()) return {};
    int xmin = contour[0].x, xmax = contour[0].x;
    int ymin = contour[0].y, ymax = contour[0].y;
    for (const auto &p : contour) {
        if (p.x < xmin) xmin = p.x;
        if (p.x > xmax) xmax = p.x;
        if (p.y < ymin) ymin = p.y;
        if (p.y > ymax) ymax = p.y;
    }
    return {xmin, ymin, xmax - xmin + 1, ymax - ymin + 1};
}

double contourArea(const std::vector<Point> &contour) {
    if (contour.size() < 3) return 0.0;

    // 鞋带公式：2*面积 = |Σ (x_i * y_{i+1} - x_{i+1} * y_i)|
    long long twice = 0;
    const std::size_t n = contour.size();
    for (std::size_t i = 0; i < n; ++i) {
        const Point &a = contour[i];
        const Point &b = contour[(i + 1) % n];
        twice += static_cast<long long>(a.x) * b.y - static_cast<long long>(b.x) * a.y;
    }
    return std::abs(static_cast<double>(twice)) * 0.5;
}


std::vector<Point> convexHull(std::vector<Point> pts) {
    if (pts.size() < 3) return pts;

    std::sort(pts.begin(), pts.end(), [](const Point &a, const Point &b) {
        return a.x != b.x ? a.x < b.x : a.y < b.y;
    });

    // 叉积 > 0 左转，< 0 右转，= 0 共线
    auto cross = [](const Point &o, const Point &a, const Point &b) {
        return static_cast<long long>(a.x - o.x) * (b.y - o.y) -
               static_cast<long long>(a.y - o.y) * (b.x - o.x);
    };

    std::vector<Point> hull;
    hull.reserve(pts.size() * 2);

    // 下凸包：左到右，遇到右转就把栈顶弹掉
    for (const auto &p : pts) {
        while (hull.size() >= 2 && cross(hull[hull.size() - 2], hull.back(), p) <= 0) hull.pop_back();
        hull.push_back(p);
    }

    // 上凸包：右到左，下界是 lower，别把下凸包的点弹掉
    const std::size_t lower = hull.size();
    for (int i = static_cast<int>(pts.size()) - 2; i >= 0; --i) {
        const Point &p = pts[static_cast<std::size_t>(i)];
        while (hull.size() > lower && cross(hull[hull.size() - 2], hull.back(), p) <= 0)
            hull.pop_back();
        hull.push_back(p);
    }

    hull.pop_back(); // 走完一圈回到起点，去掉重复的那个
    return hull;
}


RotatedRect minAreaRect(const std::vector<Point> &contour) {
    RotatedRect out;
    if (contour.empty()) return out;

    const std::vector<Point> hull = convexHull(contour);
    if (hull.size() < 3) { // 退化成一个点或一条线，退回轴对齐框
        const Rect r = boundingRect(contour);
        out.center = {r.x + r.width * 0.5f, r.y + r.height * 0.5f};
        out.size = {static_cast<float>(r.width), static_cast<float>(r.height)};
        out.angle = 0;
        return out;
    }

    constexpr double PI = 3.14159265358979323846;

    // 枚举凸包的每条边：最小外接矩形必然有一条边和凸包的某条边共线
    double bestArea = -1.0;
    double bestUx = 1.0, bestUy = 0.0;
    double bestMinU = 0, bestMaxU = 0, bestMinV = 0, bestMaxV = 0;

    const std::size_t n = hull.size();
    for (std::size_t i = 0; i < n; ++i) {
        const Point &a = hull[i];
        const Point &b = hull[(i + 1) % n];
        const double dx = b.x - a.x;
        const double dy = b.y - a.y;
        const double len = std::sqrt(dx * dx + dy * dy);
        if (len < 1e-12) continue;

        const double ux = dx / len, uy = dy / len; // 沿边方向
        const double vx = -uy, vy = ux;            // 法线方向

        double minU = 1e300, maxU = -1e300, minV = 1e300, maxV = -1e300;
        for (const Point &p : hull) {
            const double pu = p.x * ux + p.y * uy;
            const double pv = p.x * vx + p.y * vy;
            if (pu < minU) minU = pu;
            if (pu > maxU) maxU = pu;
            if (pv < minV) minV = pv;
            if (pv > maxV) maxV = pv;
        }

        const double area = (maxU - minU) * (maxV - minV);
        if (bestArea < 0 || area < bestArea) {
            bestArea = area;
            bestUx = ux;
            bestUy = uy;
            bestMinU = minU;
            bestMaxU = maxU;
            bestMinV = minV;
            bestMaxV = maxV;
        }
    }

    // 中心：u/v 两个方向的区间中点，转回 xy
    const double cu = (bestMinU + bestMaxU) * 0.5;
    const double cv = (bestMinV + bestMaxV) * 0.5;
    const double vx = -bestUy, vy = bestUx;
    out.center = {static_cast<float>(cu * bestUx + cv * vx),
                  static_cast<float>(cu * bestUy + cv * vy)};

    double w = bestMaxU - bestMinU;
    double h = bestMaxV - bestMinV;
    double angle = std::atan2(bestUy, bestUx) * 180.0 / PI;

    // 规范：width 一定是长边，angle 是长边与 x 轴夹角，落在 [0, 180)
    if (h > w) {
        std::swap(w, h);
        angle += 90.0;
    }
    angle = std::fmod(angle, 180.0);
    if (angle < 0) angle += 180.0;

    out.size = {static_cast<float>(w), static_cast<float>(h)};
    out.angle = static_cast<float>(angle);
    return out;
}

} // namespace mycv
