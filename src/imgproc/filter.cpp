#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <mycv/core/check.hpp>
#include <mycv/core/types.hpp>
#include <mycv/imgproc/filter.hpp>

namespace mycv {
namespace {

// BORDER_REFLECT_101: gfedcb|abcdefgh|gfedcba
int reflect101(int i, int n) {
    while (i < 0 || i >= n) {
        if (i < 0)
            i = -i;
        else
            i = 2 * n - i - 2;
    }
    return i;
}

// 2^8 定点高斯核，长度 n，和恒等于 256。
std::vector<int32_t> gaussianKernelFixed(int n, double sigma) {
    std::vector<double> k(static_cast<std::size_t>(n), 0.0);
    if (sigma <= 0) {
        static const double k3[] = {0.25, 0.5, 0.25};
        static const double k5[] = {1 / 16., 4 / 16., 6 / 16., 4 / 16., 1 / 16.};
        static const double k7[] = {2 / 64.,  7 / 64., 14 / 64., 18 / 64.,
                                    14 / 64., 7 / 64., 2 / 64.};
        static const double k9[] = {4 / 256.,  13 / 256., 30 / 256., 51 / 256., 60 / 256.,
                                    51 / 256., 30 / 256., 13 / 256., 4 / 256.};
        const double *t = nullptr;
        if (n == 3)
            t = k3;
        else if (n == 5)
            t = k5;
        else if (n == 7)
            t = k7;
        else if (n == 9)
            t = k9;
        if (t) {
            for (int i = 0; i < n; ++i)
                k[i] = t[i];
        } else {
            sigma = 0.15 * n + 0.35;
        }
    }
    if (sigma > 0) {
        const double c = (n - 1) * 0.5;
        double sum = 0.0;
        for (int i = 0; i < n; ++i) {
            const double d = i - c;
            k[i] = std::exp(-d * d / (2 * sigma * sigma));
            sum += k[i];
        }
        for (int i = 0; i < n; ++i)
            k[i] /= sum;
    }

    // 误差扩散 + 中心补齐
    std::vector<int32_t> f(static_cast<std::size_t>(n), 0);
    double err = 0.0;
    int32_t s = 0;
    for (int i = 0; i < n / 2; ++i) {
        const double adj = k[i] * 256.0 + err;
        const int32_t v = static_cast<int32_t>(std::nearbyint(adj));
        err = adj - v;
        f[i] = v;
        f[n - 1 - i] = v;
        s += v;
    }
    f[n / 2] = 256 - 2 * s;
    return f;
}

} // namespace

Mat convolve2D(const Mat &src, const Mat &kernel, Point anchor) {
    MYCV_ASSERT(!src.empty());
    MYCV_ASSERT(!kernel.empty());
    MYCV_ASSERT(depthOf(src.type) == DEPTH_8U);
    MYCV_ASSERT(depthOf(kernel.type) == DEPTH_32F);
    MYCV_ASSERT(channels(kernel.type) == 1);

    // 只有 -1 表示"取核中心"，其他负数要落到下面的断言里
    // (2026.11.20 15:48 记录）原来写的是 < 0，结果 anchor={-5,0} 也被当成默认值，断言永远不触发
    const int ax = anchor.x == -1 ? kernel.cols / 2 : anchor.x;
    const int ay = anchor.y == -1 ? kernel.rows / 2 : anchor.y;
    MYCV_ASSERT(ax >= 0 && ax < kernel.cols);
    MYCV_ASSERT(ay >= 0 && ay < kernel.rows);

    const int cn = channels(src.type);
    const int H = src.rows, W = src.cols;
    const int kh = kernel.rows, kw = kernel.cols;

    // 核预取成紧凑数组，避免内层反复 kernel.ptr<float>()
    std::vector<float> kbuf(static_cast<std::size_t>(kh) * static_cast<std::size_t>(kw));
    for (int ky = 0; ky < kh; ++ky) {
        const float *kr = kernel.ptr<float>(ky);
        std::memcpy(kbuf.data() + static_cast<std::size_t>(ky) * kw, kr,
                    static_cast<std::size_t>(kw) * sizeof(float));
    }

    Mat dst;
    dst.create(H, W, src.type);

    // 内部区：邻域完整落在图内，不需要反射。原来每个像素要调 ksize*ksize 次
    // reflect101（每次一个 while 循环），这部分现在完全省掉
    const int yLo = ay, yHi = H - kh + ay;
    const int xLo = ax, xHi = W - kw + ax;

    std::vector<const uint8_t *> srows(static_cast<std::size_t>(kh));

    for (int y = 0; y < H; ++y) {
        uint8_t *d = dst.ptr(y);
        const bool innerY = (y >= yLo && y < yHi);

        for (int ky = 0; ky < kh; ++ky) {
            const int sy = y + ky - ay;
            srows[ky] = innerY ? src.ptr(sy) : src.ptr(reflect101(sy, H));
        }

        for (int x = 0; x < W; ++x) {
            const bool inner = innerY && (x >= xLo && x < xHi);
            for (int c = 0; c < cn; ++c) {
                double sum = 0.0;
                if (inner) {
                    for (int ky = 0; ky < kh; ++ky) {
                        const uint8_t *srow = srows[ky];
                        const float *kr = kbuf.data() + static_cast<std::size_t>(ky) * kw;
                        const uint8_t *p = srow + (x - ax) * cn + c;
                        for (int kx = 0; kx < kw; ++kx)
                            sum += kr[kx] * p[kx * cn];
                    }
                } else {
                    for (int ky = 0; ky < kh; ++ky) {
                        const uint8_t *srow = srows[ky];
                        const float *kr = kbuf.data() + static_cast<std::size_t>(ky) * kw;
                        for (int kx = 0; kx < kw; ++kx) {
                            sum += kr[kx] * srow[reflect101(x + kx - ax, W) * cn + c];
                        }
                    }
                }
                long v = std::lrint(sum); // 银行家舍入，和 cvRound 一致
                if (v < 0)
                    v = 0;
                else if (v > 255)
                    v = 255;
                d[x * cn + c] = static_cast<uint8_t>(v);
            }
        }
    }
    return dst;
}

Mat blur(const Mat &src, Size ksize) {
    MYCV_ASSERT(!src.empty());
    if (ksize.width <= 0 || ksize.height <= 0 || ksize.width % 2 == 0 || ksize.height % 2 == 0)
        throw std::runtime_error("blur: ksize 必须是正的奇数");

    const int cn = channels(src.type);
    const int W = src.cols * cn; // 按"通道像素"算宽度，多通道一视同仁
    const int H = src.rows;
    const int rx = ksize.width / 2;
    const int ry = ksize.height / 2;
    const int area = ksize.width * ksize.height;

    // 第一趟：水平滑动和。原来每像素要重算 ksize 次，
    // 现在整行只做一次初始窗口，之后每个像素一加一减
    // 注意反射要按"像素"做再乘 cn，按字节反射会让三通道错位
    std::vector<int32_t> hsum(static_cast<std::size_t>(H) * W);
    for (int y = 0; y < H; ++y) {
        const uint8_t *s = src.ptr(y);
        int32_t *h = hsum.data() + static_cast<std::size_t>(y) * W;

        for (int c = 0; c < cn; ++c) {
            int sum = 0;
            for (int k = -rx; k <= rx; ++k)
                sum += s[reflect101(k, src.cols) * cn + c];
            h[c] = sum;
            for (int x = 1; x < src.cols; ++x) {
                sum -= s[reflect101(x - rx - 1, src.cols) * cn + c]; // 移出窗口
                sum += s[reflect101(x + rx, src.cols) * cn + c]; // 移进窗口
                h[x * cn + c] = sum;
            }
        }
    }

    // 第二趟：垂直滑动和 + 输出。colSum 维护"当前行上下各 ry 行的列和"
    Mat dst;
    dst.create(H, src.cols, src.type);

    std::vector<int32_t> colSum(static_cast<std::size_t>(W), 0);
    for (int k = -ry; k <= ry; ++k) {
        const int32_t *h = hsum.data() + static_cast<std::size_t>(reflect101(k, H)) * W;
        for (int x = 0; x < W; ++x)
            colSum[x] += h[x];
    }
    {
        uint8_t *d = dst.ptr(0);
        for (int x = 0; x < W; ++x)
            d[x] = static_cast<uint8_t>((colSum[x] + area / 2) / area);
    }

    for (int y = 1; y < H; ++y) {
        const int32_t *add = hsum.data() + static_cast<std::size_t>(reflect101(y + ry, H)) * W;
        const int32_t *sub = hsum.data() + static_cast<std::size_t>(reflect101(y - ry - 1, H)) * W;
        for (int x = 0; x < W; ++x)
            colSum[x] += add[x] - sub[x];

        uint8_t *d = dst.ptr(y);
        for (int x = 0; x < W; ++x)
            d[x] = static_cast<uint8_t>((colSum[x] + area / 2) / area);
    }
    return dst;
}

Mat gaussianBlur(const Mat &src, Size ksize, double sigmaX, double sigmaY) {
    MYCV_ASSERT(!src.empty());
    if (ksize.width <= 0 || ksize.height <= 0 || ksize.width % 2 == 0 || ksize.height % 2 == 0)
        throw std::runtime_error("gaussianBlur: ksize 必须是正的奇数");

    const int H = src.rows, W = src.cols, cn = channels(src.type);
    // 单行/单列图：OpenCV 把那维的核缩成 1，否则 reflect101 在 n=1 上原地打转
    const int kw = (W == 1) ? 1 : ksize.width;
    const int kh = (H == 1) ? 1 : ksize.height;
    if (kw > 2 * W - 1 || kh > 2 * H - 1)
        throw std::runtime_error("gaussianBlur: ksize 比图大太多（reflect101 不收敛）");

    // sigmaY<=0 取 sigmaX，和 OpenCV 的 sigma2=sigma1 一致
    const std::vector<int32_t> kx = gaussianKernelFixed(kw, sigmaX);
    const std::vector<int32_t> ky = gaussianKernelFixed(kh, sigmaY > 0 ? sigmaY : sigmaX);

    const int rx = kw / 2, ry = kh / 2;
    const int Wc = W * cn;

    // 反射索引预计算，按像素反射再乘 cn；直接按字节反射会让三通道错位 (2026.11.21记录)
    std::vector<int> xmap(static_cast<std::size_t>(W) * kw);
    for (int x = 0; x < W; ++x)
        for (int j = 0; j < kw; ++j)
            xmap[static_cast<std::size_t>(x) * kw + j] = reflect101(x + j - rx, W) * cn;
    std::vector<int> ymap(static_cast<std::size_t>(H) * kh);
    for (int y = 0; y < H; ++y)
        for (int j = 0; j < kh; ++j)
            ymap[static_cast<std::size_t>(y) * kh + j] = reflect101(y + j - ry, H);

    // 第一趟水平：整数累加不舍入，最大 255*256，int32 富余
    std::vector<int32_t> hsum(static_cast<std::size_t>(H) * Wc);
    for (int y = 0; y < H; ++y) {
        const uint8_t *s = src.ptr(y);
        int32_t *h = hsum.data() + static_cast<std::size_t>(y) * Wc;
        for (int x = 0; x < W; ++x) {
            const int *xm = xmap.data() + static_cast<std::size_t>(x) * kw;
            for (int c = 0; c < cn; ++c) {
                int32_t acc = 0;
                for (int j = 0; j < kw; ++j)
                    acc += kx[j] * s[xm[j] + c];
                h[x * cn + c] = acc;
            }
        }
    }

    // 第二趟垂直 + 唯一一次舍入 (acc + 2^15) >> 16，CV_DESCALE 是 round half up
    Mat dst;
    dst.create(H, W, src.type);
    for (int y = 0; y < H; ++y) {
        const int *ym = ymap.data() + static_cast<std::size_t>(y) * kh;
        uint8_t *d = dst.ptr(y);
        for (int x = 0; x < Wc; ++x) {
            int32_t acc = 0;
            for (int j = 0; j < kh; ++j)
                acc += ky[j] * hsum[static_cast<std::size_t>(ym[j]) * Wc + x];
            d[x] = static_cast<uint8_t>((acc + (1 << 15)) >> 16);
        }
    }
    return dst;
}

Mat medianBlur(const Mat &src, int ksize) {
    MYCV_ASSERT(!src.empty());
    if (ksize < 1 || ksize % 2 == 0)
        throw std::runtime_error("medianBlur: ksize 必须是不小于 1 的奇数");
    if (ksize == 1) return src.clone();

    const int H = src.rows, W = src.cols, cn = channels(src.type);
    const int r = ksize / 2;
    const int n = ksize * ksize;
    const int m = n / 2; // 中值是第 m 个（0 基）最小元素

    // 边界 REPLICATE：越界坐标夹到边缘。
    auto cl = [](int i, int dim) { return i < 0 ? 0 : (i >= dim ? dim - 1 : i); };

    std::vector<const uint8_t *> rows(static_cast<std::size_t>(H));
    for (int y = 0; y < H; ++y)
        rows[y] = src.ptr(y);

    Mat dst;
    dst.create(H, W, src.type);

    std::vector<int> hist(256);
    std::vector<const uint8_t *> win(static_cast<std::size_t>(ksize)); // 窗口的 k 行
    std::vector<int> xmap(static_cast<std::size_t>(W) * ksize);
    for (int x = 0; x < W; ++x)
        for (int j = 0; j < ksize; ++j)
            xmap[static_cast<std::size_t>(x) * ksize + j] = cl(x + j - r, W) * cn;

    for (int y = 0; y < H; ++y) {
        for (int j = 0; j < ksize; ++j)
            win[j] = rows[cl(y + j - r, H)];
        uint8_t *d = dst.ptr(y);

        for (int c = 0; c < cn; ++c) {
            std::fill(hist.begin(), hist.end(), 0);
            // x=0 的窗口
            for (int j = 0; j < ksize; ++j) {
                const uint8_t *s = win[j];
                const int *x0 = xmap.data();
                for (int i = 0; i < ksize; ++i)
                    ++hist[s[x0[i] + c]];
            }
            // below = 严格小于 med 的个数；中值满足 below <= m < below + hist[med]
            int below = 0, med = 0;
            while (below + hist[med] <= m) {
                below += hist[med];
                ++med;
            }
            d[c] = static_cast<uint8_t>(med);

            for (int x = 1; x < W; ++x) {
                // 向右滑一列：进来 cl(x+r)，出去 cl(x-r-1)
                const int addX = cl(x + r, W) * cn + c;
                const int subX = cl(x - r - 1, W) * cn + c;
                for (int j = 0; j < ksize; ++j) {
                    const uint8_t *s = win[j];
                    const uint8_t va = s[addX];
                    const uint8_t vs = s[subX];
                    ++hist[va];
                    --hist[vs];
                    if (va < med) ++below;
                    if (vs < med) --below;
                }
                // 中值指针上下修正，通常只走 0~2 步
                while (below > m) {
                    --med;
                    below -= hist[med];
                }
                while (below + hist[med] <= m) {
                    below += hist[med];
                    ++med;
                }
                d[x * cn + c] = static_cast<uint8_t>(med);
            }
        }
    }
    return dst;
}

Mat sobel(const Mat &src, int dx, int dy, int ksize) {
    MYCV_ASSERT(!src.empty());
    if (ksize != 3) throw std::runtime_error("sobel: 目前只支持 ksize=3");
    if (dx < 0 || dy < 0 || dx + dy != 1) throw std::runtime_error("sobel: 只支持 dx+dy==1");

    // OpenCV 的 Sobel 不做归一化，核和是 0；8U 输出会大量饱和到 0/255
    static const float kx[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    static const float ky[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
    const float *src9 = (dx == 1) ? kx : ky;

    Mat k;
    k.create(3, 3, F32C1);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            k.at<float>(r, c) = src9[r * 3 + c];

    return convolve2D(src, k);
}

Mat scharr(const Mat &src, int dx, int dy) {
    MYCV_ASSERT(!src.empty());
    if (dx < 0 || dy < 0 || dx + dy != 1) throw std::runtime_error("scharr: 只支持 dx+dy==1");

    // 核和是 0，不归一化；8U 输出一样会大量饱和
    static const float kx[9] = {-3, 0, 3, -10, 0, 10, -3, 0, 3};
    static const float ky[9] = {-3, -10, -3, 0, 0, 0, 3, 10, 3};
    const float *src9 = (dx == 1) ? kx : ky;

    Mat k;
    k.create(3, 3, F32C1);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            k.at<float>(r, c) = src9[r * 3 + c];

    return convolve2D(src, k);
}

Mat laplacian(const Mat &src, int ksize) {
    MYCV_ASSERT(!src.empty());
    if (ksize != 1 && ksize != 3) throw std::runtime_error("laplacian: 只支持 ksize=1 或 3");

    // ksize=1 是 4 邻域（只有上下左右），ksize=3 是 8 邻域（含对角，中心 -8）。
    static const float k1[9] = {0, 1, 0, 1, -4, 1, 0, 1, 0};
    static const float k3[9] = {2, 0, 2, 0, -8, 0, 2, 0, 2};
    const float *src9 = (ksize == 1) ? k1 : k3;

    Mat k;
    k.create(3, 3, F32C1);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            k.at<float>(r, c) = src9[r * 3 + c];

    return convolve2D(src, k);
}

void spatialGradient(const Mat &src, Mat &dx, Mat &dy) {
    MYCV_ASSERT(!src.empty());
    if (depthOf(src.type) != DEPTH_8U || channels(src.type) != 1)
        throw std::runtime_error("spatialGradient: 只支持 U8C1");

    const int H = src.rows, W = src.cols;
    dx.create(H, W, S16C1);
    dy.create(H, W, S16C1);

    // 3x3 Sobel 核直接展开，省掉 convolve2D 的核循环。
    // 输出 16 位装得下 ±1020，不饱和
    for (int y = 0; y < H; ++y) {
        const uint8_t *rm = src.ptr(reflect101(y - 1, H));
        const uint8_t *r0 = src.ptr(y);
        const uint8_t *rp = src.ptr(reflect101(y + 1, H));
        int16_t *ox = dx.ptr<int16_t>(y);
        int16_t *oy = dy.ptr<int16_t>(y);
        for (int x = 0; x < W; ++x) {
            const int xm = reflect101(x - 1, W), xp = reflect101(x + 1, W);
            ox[x] =
                static_cast<int16_t>(-rm[xm] + rm[xp] - 2 * r0[xm] + 2 * r0[xp] - rp[xm] + rp[xp]);
            oy[x] =
                static_cast<int16_t>(-rm[xm] - 2 * rm[x] - rm[xp] + rp[xm] + 2 * rp[x] + rp[xp]);
        }
    }
}

} // namespace mycv
