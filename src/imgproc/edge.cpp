#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#include <mycv/core/check.hpp>
#include <mycv/core/types.hpp>
#include <mycv/imgproc/edge.hpp>

namespace mycv {
namespace {

// tan(22.5°) * 2^15，OpenCV canny.cpp 里的定点常数，用整数比较代替 atan2
constexpr int TG22 = 13573;

// Canny 内部的 Sobel 必须出 16 位：8 位会饱和
// 后面 0..1020 的阈值就没意义。边界是 BORDER_REPLICATE，
// 和 convolve2D 默认的 REFLECT_101 不一样
void sobel16(const Mat &src, std::vector<int16_t> &gx, std::vector<int16_t> &gy) {
    const int H = src.rows, W = src.cols;
    gx.resize(static_cast<std::size_t>(H) * W);
    gy.resize(static_cast<std::size_t>(H) * W);

    auto cl = [](int i, int n) { return i < 0 ? 0 : (i >= n ? n - 1 : i); };

    for (int y = 0; y < H; ++y) {
        const uint8_t *rm = src.ptr(cl(y - 1, H));
        const uint8_t *r0 = src.ptr(y);
        const uint8_t *rp = src.ptr(cl(y + 1, H));
        int16_t *ox = gx.data() + static_cast<std::size_t>(y) * W;
        int16_t *oy = gy.data() + static_cast<std::size_t>(y) * W;
        for (int x = 0; x < W; ++x) {
            const int xm = cl(x - 1, W), xp = cl(x + 1, W);
            ox[x] =
                static_cast<int16_t>(-rm[xm] + rm[xp] - 2 * r0[xm] + 2 * r0[xp] - rp[xm] + rp[xp]);
            oy[x] =
                static_cast<int16_t>(-rm[xm] - 2 * rm[x] - rm[xp] + rp[xm] + 2 * rp[x] + rp[xp]);
        }
    }
}

} // namespace

Mat canny(const Mat &src, double low, double high, bool L2gradient) {
    MYCV_ASSERT(!src.empty());
    if (depthOf(src.type) != DEPTH_8U || channels(src.type) != 1)
        throw std::runtime_error("canny: 只支持 U8C1");

    const int H = src.rows, W = src.cols;
    std::vector<int16_t> gx, gy;
    sobel16(src, gx, gy);

    // （抄袭）OpenCV 先 cvFloor 成 int：L2 时阈值平方（上限 32767），L1 时直接用
    int lo = 0, hi = 0;
    if (L2gradient) {
        const double l = std::min(32767.0, low);
        const double h = std::min(32767.0, high);
        lo = static_cast<int>(std::floor(l > 0 ? l * l : l));
        hi = static_cast<int>(std::floor(h > 0 ? h * h : h));
    } else {
        lo = static_cast<int>(std::floor(low));
        hi = static_cast<int>(std::floor(high));
    }

    // 幅值和 map 都多留一圈边，外面填 0 / 1，所有邻域访问就不用判越界
    const int stride = W + 2;
    const std::size_t total = static_cast<std::size_t>(H + 2) * stride;

    std::vector<int32_t> mag(total, 0);
    for (int y = 0; y < H; ++y) {
        const int16_t *sx = gx.data() + static_cast<std::size_t>(y) * W;
        const int16_t *sy = gy.data() + static_cast<std::size_t>(y) * W;
        int32_t *m = mag.data() + static_cast<std::size_t>(y + 1) * stride + 1;
        if (L2gradient) {
            for (int x = 0; x < W; ++x)
                m[x] = int(sx[x]) * sx[x] + int(sy[x]) * sy[x];
        } else {
            for (int x = 0; x < W; ++x)
                m[x] = std::abs(int(sx[x])) + std::abs(int(sy[x]));
        }
    }

    // 0 = 弱候选（过了 NMS 没过 high），1 = 不是边，2 = 强边
    std::vector<uint8_t> map(total, 1);
    std::vector<int32_t> stack;
    stack.reserve(1024);

    for (int y = 1; y <= H; ++y) {
        const int16_t *sx = gx.data() + static_cast<std::size_t>(y - 1) * W;
        const int16_t *sy = gy.data() + static_cast<std::size_t>(y - 1) * W;
        const int32_t *ma = mag.data() + static_cast<std::size_t>(y) * stride;
        const int32_t *mp = ma - stride;
        const int32_t *mn = ma + stride;
        uint8_t *pm = map.data() + static_cast<std::size_t>(y) * stride;

        for (int x = 1; x <= W; ++x) {
            const int32_t m = ma[x];
            if (m <= lo) continue; // 严格大于，等于 low 的丢掉

            const int xs = sx[x - 1], ys = sy[x - 1];
            const int gxa = std::abs(xs);
            const int gya = std::abs(ys) << 15;
            const int tg22x = gxa * TG22;

            bool keep = false;
            if (gya < tg22x) {
                // 梯度偏水平，沿 x 找极大。左边 >、右边 >=：相等时留左边那个，
                // OpenCV 的破平方式，写成两个 > 会多出一半的边
                keep = m > ma[x - 1] && m >= ma[x + 1];
            } else if (gya > tg22x + (gxa << 16)) {
                keep = m > mp[x] && m >= mn[x];
            } else {
                // 对角：gx、gy 同号走 ↘，异号走 ↗；这两项都是严格的 >
                const int s = (xs ^ ys) < 0 ? -1 : 1;
                keep = m > mp[x - s] && m > mn[x + s];
            }
            if (!keep) continue;

            if (m > hi) {
                pm[x] = 2;
                stack.push_back(y * stride + x);
            } else {
                pm[x] = 0;
            }
        }
    }

    // 滞后阈值：从每个强边出发 8 邻域泛洪，把连着的弱候选升成边
    while (!stack.empty()) {
        const int idx = stack.back();
        stack.pop_back();
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                const int n = idx + dy * stride + dx;
                if (map[static_cast<std::size_t>(n)] == 0) {
                    map[static_cast<std::size_t>(n)] = 2;
                    stack.push_back(n);
                }
            }
        }
    }

    Mat dst;
    dst.create(H, W, U8C1);
    for (int y = 0; y < H; ++y) {
        const uint8_t *pm = map.data() + static_cast<std::size_t>(y + 1) * stride + 1;
        uint8_t *d = dst.ptr(y);
        for (int x = 0; x < W; ++x)
            d[x] = pm[x] == 2 ? 255 : 0;
    }
    return dst;
}

} // namespace mycv
