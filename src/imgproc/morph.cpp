#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <mycv/core/check.hpp>
#include <mycv/core/mat.hpp>
#include <mycv/imgproc/morph.hpp>
namespace mycv {
Mat getStructuringElement(int shape, Size ksize) {
    MYCV_ASSERT(ksize.width > 0 && ksize.height > 0);
    if (shape != MORPH_RECT && shape != MORPH_CROSS && shape != MORPH_ELLIPSE)
        throw std::runtime_error("getStructuringElement: 未知 shape");

    Mat k;
    k.create(ksize.height, ksize.width, U8C1);

    const int cx = ksize.width / 2;
    const int cy = ksize.height / 2;

    if (shape == MORPH_RECT) {
        for (int r = 0; r < k.rows; ++r)
            std::memset(k.ptr(r), 1, static_cast<std::size_t>(k.cols));
        return k;
    }

    if (shape == MORPH_CROSS) {
        for (int r = 0; r < k.rows; ++r) {
            uint8_t *p = k.ptr(r);
            for (int c = 0; c < k.cols; ++c) {
                p[c] = (r == cy || c == cx) ? 1 : 0;
            }
        }
        return k;
    }

    // MORPH_ELLIPSE：OpenCV 的整数近似，每行填中间一段 [c-dx, c+dx]
    // create 的内存未初始化，四周靠 memset 清零
    // AI写的。。。我不会。。。
    std::memset(k.data, 0, static_cast<std::size_t>(k.rows) * k.step);

    const double invR2 = cy ? 1.0 / (static_cast<double>(cy) * cy) : 0.0;
    for (int i = 0; i < k.rows; ++i) {
        const int dy = i - cy;
        if (std::abs(dy) > cy) continue;

        // half-to-even（cvRound），1.732 -> 2。用截断会得到"瘦"的数学椭圆
        const int dx =
            static_cast<int>(std::nearbyint(cx * std::sqrt((cy * cy - dy * dy) * invR2)));

        const int j1 = std::max(cx - dx, 0);
        const int j2 = std::min(cx + dx + 1, k.cols);
        uint8_t *p = k.ptr(i);
        for (int j = j1; j < j2; ++j)
            p[j] = 1;
    }
    return k;
}

namespace {

template <bool Erode> Mat morphImpl(const Mat &src, const Mat &kernel) {
    MYCV_ASSERT(!src.empty());
    MYCV_ASSERT(!kernel.empty());
    MYCV_ASSERT(depthOf(src.type) == DEPTH_8U);
    MYCV_ASSERT(depthOf(kernel.type) == DEPTH_8U);
    MYCV_ASSERT(channels(kernel.type) == 1);

    const int ax = kernel.cols / 2;
    const int ay = kernel.rows / 2;
    const int cn = channels(src.type);
    const int H = src.rows, W = src.cols;
    const int kh = kernel.rows, kw = kernel.cols;

    Mat dst;
    dst.create(H, W, src.type);

    const int yInnerLo = ay, yInnerHi = H - ay;
    const int xInnerLo = ax, xInnerHi = W - ax;

    std::vector<const uint8_t *> srows(static_cast<std::size_t>(kh));
    std::vector<const uint8_t *> krows(static_cast<std::size_t>(kh));

    // 矩形核（全 1）最常见，特化之后内层是纯 min/max 链，可以被向量化
    bool fullKernel = true;
    for (int ky = 0; ky < kh && fullKernel; ++ky) {
        const uint8_t *krow = kernel.ptr(ky);
        for (int kx = 0; kx < kw; ++kx) {
            if (krow[kx] == 0) {
                fullKernel = false;
                break;
            }
        }
    }

    for (int y = 0; y < H; ++y) {
        uint8_t *d = dst.ptr(y);
        const bool innerY = (y >= yInnerLo && y < yInnerHi);

        // 每行只取一次 kh 个行指针，不再逐像素调用 src.ptr()（2026.11.21优化）
        for (int ky = 0; ky < kh; ++ky) {
            krows[ky] = kernel.ptr(ky);
            const int sy = y + ky - ay;
            srows[ky] = (sy < 0 || sy >= H) ? nullptr : src.ptr(sy); // 越界行不反射
        }

        for (int x = 0; x < W; ++x) {
            const bool inner = innerY && (x >= xInnerLo && x < xInnerHi);
            for (int c = 0; c < cn; ++c) {
                int v = Erode ? 255 : 0; // erode 从最大开始找最小，dilate 反之
                if (inner && fullKernel) {
                    for (int ky = 0; ky < kh; ++ky) {
                        const uint8_t *srow = srows[ky];
                        for (int kx = 0; kx < kw; ++kx) {
                            const int p = srow[(x + kx - ax) * cn + c];
                            v = Erode ? std::min(v, p) : std::max(v, p);
                        }
                    }
                } else if (inner) {
                    for (int ky = 0; ky < kh; ++ky) {
                        const uint8_t *srow = srows[ky];
                        const uint8_t *krow = krows[ky];
                        for (int kx = 0; kx < kw; ++kx) {
                            if (krow[kx] == 0) continue; // 核里是 0 的位置不参与
                            const int p = srow[(x + kx - ax) * cn + c];
                            v = Erode ? std::min(v, p) : std::max(v, p);
                        }
                    }
                } else {
                    for (int ky = 0; ky < kh; ++ky) {
                        const uint8_t *srow = srows[ky];
                        if (!srow) continue;
                        const uint8_t *krow = krows[ky];
                        for (int kx = 0; kx < kw; ++kx) {
                            if (krow[kx] == 0) continue;
                            const int sx = x + kx - ax;
                            if (sx < 0 || sx >= W) continue;
                            const int p = srow[sx * cn + c];
                            v = Erode ? std::min(v, p) : std::max(v, p);
                        }
                    }
                }
                d[x * cn + c] = static_cast<uint8_t>(v);
            }
        }
    }
    return dst;
}

} // namespace

Mat erode(const Mat &src, const Mat &kernel) { return morphImpl<true>(src, kernel); }
Mat dilate(const Mat &src, const Mat &kernel) { return morphImpl<false>(src, kernel); }

} // namespace mycv