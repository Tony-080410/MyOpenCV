// 和 OpenCV 对比：同一份输入，逐像素比
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp> // cv::imread供测试
#include <opencv2/imgproc.hpp>

#include <mycv/core/mat.hpp>
#include <mycv/imgproc/color.hpp>
#include <mycv/imgproc/contours.hpp>
#include <mycv/imgproc/draw.hpp>
#include <mycv/imgproc/edge.hpp>
#include <mycv/imgproc/filter.hpp>
#include <mycv/imgproc/geom.hpp>
#include <mycv/imgproc/morph.hpp>
#include <mycv/imgproc/thresh.hpp>
#include <mycv/io/image.hpp>

// CMake 里定义了 MYCV_DATA_DIR；单编这个文件时退回相对路径
#ifndef MYCV_DATA_DIR
#define MYCV_DATA_DIR "data"
#endif

static std::string dataPath(const char *name) { return std::string(MYCV_DATA_DIR) + "/" + name; }

static int g_failed = 0;

#define CHECK(cond)                                                                                \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);                            \
            ++g_failed;                                                                            \
        }                                                                                          \
    } while (0)

// 两幅同尺寸图逐像素比（按通道数展开），返回最大绝对差
static int maxDiff(const mycv::Mat &a, const cv::Mat &b) {
    if (a.rows != b.rows || a.cols != b.cols) return -1;
    const int cn = mycv::channels(a.type);
    int worst = 0;
    for (int r = 0; r < a.rows; ++r) {
        const uint8_t *pa = a.ptr(r);
        const uint8_t *pb = b.ptr<uint8_t>(r);
        for (int c = 0; c < a.cols * cn; ++c) {
            const int d = pa[c] > pb[c] ? pa[c] - pb[c] : pb[c] - pa[c];
            if (d > worst) worst = d;
        }
    }
    return worst;
}

// 两幅 mycv 图之间比
static int maxDiff(const mycv::Mat &a, const mycv::Mat &b) {
    if (a.rows != b.rows || a.cols != b.cols) return -1;
    const int cn = mycv::channels(a.type);
    int worst = 0;
    for (int r = 0; r < a.rows; ++r) {
        const uint8_t *pa = a.ptr(r);
        const uint8_t *pb = b.ptr(r);
        for (int c = 0; c < a.cols * cn; ++c) {
            const int d = pa[c] > pb[c] ? pa[c] - pb[c] : pb[c] - pa[c];
            if (d > worst) worst = d;
        }
    }
    return worst;
}

// 造一张 rows×cols 的随机 BGR，mycv 和 cv::Mat 共享同一块内存，不用落盘
static mycv::Mat makeRandomBGR(int rows, int cols, unsigned seed) {
    mycv::Mat m;
    m.create(rows, cols, mycv::U8C3);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 255);
    for (int r = 0; r < rows; ++r) {
        uint8_t *p = m.ptr(r);
        for (int i = 0; i < cols * 3; ++i)
            p[i] = static_cast<uint8_t>(dist(rng));
    }
    return m;
}

// 1. 端到端：同一个文件，各自 imread + cvtColor
static void testFromFile() {
    mycv::Mat mine = mycv::imread(dataPath("test_color.ppm"));
    CHECK(!mine.empty());

    cv::Mat ref = cv::imread(dataPath("test_color.ppm"), cv::IMREAD_COLOR); // BGR
    CHECK(!ref.empty());
    CHECK(mine.rows == ref.rows && mine.cols == ref.cols);

    mycv::Mat mineGray = mycv::cvtColor(mine, mycv::COLOR_BGR2GRAY);
    cv::Mat refGray;
    cv::cvtColor(ref, refGray, cv::COLOR_BGR2GRAY);

    CHECK(maxDiff(mineGray, refGray) == 0);

    // imread 本身也对：BGR 三通道逐字节比
    int diff = 0;
    for (int r = 0; r < mine.rows; ++r) {
        const uint8_t *a = mine.ptr(r);
        const uint8_t *b = ref.ptr<uint8_t>(r);
        for (int i = 0; i < mine.cols * 3; ++i) {
            const int d = a[i] > b[i] ? a[i] - b[i] : b[i] - a[i];
            if (d > diff) diff = d;
        }
    }
    CHECK(diff == 0);
}

// 2. 大随机图：内存共享，不读文件
static void testRandom() {
    mycv::Mat mine = makeRandomBGR(480, 640, 20261118u);

    cv::Mat ref(mine.rows, mine.cols, CV_8UC3, mine.data, mine.step); // 共享 buffer
    cv::Mat refGray;
    cv::cvtColor(ref, refGray, cv::COLOR_BGR2GRAY);

    mycv::Mat mineGray = mycv::cvtColor(mine, mycv::COLOR_BGR2GRAY);
    CHECK(mineGray.rows == 480);
    CHECK(mineGray.cols == 640);
    CHECK(mineGray.type == mycv::U8C1);
    CHECK(mineGray.isContinuous());
    CHECK(maxDiff(mineGray, refGray) == 0);
}

// 3. 不连续输入：mycv 的 ROI 对比 cv::Mat 的 Rect 视图
static void testRoi() {
    mycv::Mat mine = makeRandomBGR(200, 300, 7u);
    cv::Mat ref(mine.rows, mine.cols, CV_8UC3, mine.data, mine.step);

    mycv::Mat mineSub = mine.roi({10, 20, 120, 80});
    CHECK(!mineSub.isContinuous()); // 120 宽 / 300 宽，非整宽

    mycv::Mat mineGray = mycv::cvtColor(mineSub, mycv::COLOR_BGR2GRAY);

    cv::Mat refSub = ref(cv::Rect(10, 20, 120, 80));
    cv::Mat refGray;
    cv::cvtColor(refSub, refGray, cv::COLOR_BGR2GRAY);

    CHECK(mineGray.rows == 80 && mineGray.cols == 120);
    CHECK(maxDiff(mineGray, refGray) == 0);
}

// 4. 边界值：全 0 / 全 255 / 三个纯通道
static void testEdge() {
    const int cases[][3] = {
        {0, 0, 0}, {255, 255, 255}, {255, 0, 0}, {0, 255, 0}, {0, 0, 255},
    };
    for (const auto &cs : cases) {
        cv::Mat ref(1, 1, CV_8UC3, cv::Scalar(cs[0], cs[1], cs[2])); // BGR
        cv::Mat refGray;
        cv::cvtColor(ref, refGray, cv::COLOR_BGR2GRAY);

        mycv::Mat mine;
        mine.create(1, 1, mycv::U8C3);
        mine.data[0] = static_cast<uint8_t>(cs[0]);
        mine.data[1] = static_cast<uint8_t>(cs[1]);
        mine.data[2] = static_cast<uint8_t>(cs[2]);
        mycv::Mat mineGray = mycv::cvtColor(mine, mycv::COLOR_BGR2GRAY);

        CHECK(mineGray.at<uint8_t>(0, 0) == refGray.at<uint8_t>(0, 0));
    }
}

// 5. resize：OpenCV 内部走两阶段定点 + SIMD，逐位对齐不现实，容差 1
static void testResize() {
    mycv::Mat mine = makeRandomBGR(48, 64, 11u);
    cv::Mat ref(mine.rows, mine.cols, CV_8UC3, mine.data, mine.step);

    const int sizes[][2] = {{32, 24}, {128, 96}, {96, 72}, {17, 13}, {200, 150}, {1, 1}};
    for (const auto &s : sizes) {
        mycv::Mat m = mycv::resize(mine, {s[0], s[1]}, mycv::INTER_LINEAR);
        cv::Mat r;
        cv::resize(ref, r, cv::Size(s[0], s[1]), 0, 0, cv::INTER_LINEAR);
        CHECK(m.rows == s[1] && m.cols == s[0]);
        CHECK(m.isContinuous());
        CHECK(maxDiff(m, r) <= 1);
    }

    // 恒等变换必须逐字节一致
    mycv::Mat same = mycv::resize(mine, {64, 48}, mycv::INTER_LINEAR);
    cv::Mat r;
    cv::resize(ref, r, cv::Size(64, 48), 0, 0, cv::INTER_LINEAR);
    CHECK(maxDiff(same, r) == 0);

    // 不连续输入：ROI 直接 resize
    mycv::Mat sub = mine.roi({7, 5, 40, 30});
    CHECK(!sub.isContinuous());
    mycv::Mat up = mycv::resize(sub, {80, 60}, mycv::INTER_LINEAR);
    cv::Mat upRef;
    cv::resize(ref(cv::Rect(7, 5, 40, 30)), upRef, cv::Size(80, 60), 0, 0, cv::INTER_LINEAR);
    CHECK(maxDiff(up, upRef) <= 1);
}

// 6. threshold：纯逐像素映射，零容差
static void testThreshold() {
    mycv::Mat mine = makeRandomBGR(60, 80, 13u);
    cv::Mat ref(mine.rows, mine.cols, CV_8UC3, mine.data, mine.step);

    const double threshes[] = {0.0, 100.0, 127.0, 127.9, 255.0, -1.0, 300.0};
    const int types[] = {mycv::THRESH_BINARY, mycv::THRESH_BINARY_INV, mycv::THRESH_TRUNC,
                         mycv::THRESH_TOZERO, mycv::THRESH_TOZERO_INV};

    for (double t : threshes) {
        for (int ty : types) {
            mycv::Mat m = mycv::threshold(mine, t, 255, ty);
            cv::Mat r;
            cv::threshold(ref, r, t, 255, ty); // cv 的枚举值和 mycv 相同
            CHECK(m.rows == ref.rows && m.cols == ref.cols);
            CHECK(m.isContinuous());
            CHECK(maxDiff(m, r) == 0);
        }
    }

    // maxval 的取整与 clamp
    for (double mv : {0.0, 128.4, 128.6, 255.0, 300.0, -5.0}) {
        mycv::Mat m = mycv::threshold(mine, 100, mv, mycv::THRESH_BINARY);
        cv::Mat r;
        cv::threshold(ref, r, 100, mv, cv::THRESH_BINARY);
        CHECK(maxDiff(m, r) == 0);
    }

    // 不连续输入：ROI 二值化，输出连续
    mycv::Mat sub = mine.roi({5, 5, 30, 20});
    CHECK(!sub.isContinuous());
    mycv::Mat ms = mycv::threshold(sub, 100, 255, mycv::THRESH_BINARY);
    cv::Mat rs;
    cv::threshold(ref(cv::Rect(5, 5, 30, 20)), rs, 100, 255, cv::THRESH_BINARY);
    CHECK(ms.isContinuous());
    CHECK(ms.rows == 20 && ms.cols == 30);
    CHECK(maxDiff(ms, rs) == 0);
}

// 7. convolve2D：九个核，连续输入零容差
static mycv::Mat makeKernel(int rows, int cols, const std::vector<float> &v) {
    mycv::Mat k;
    k.create(rows, cols, mycv::F32C1);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            k.at<float>(r, c) = v[static_cast<std::size_t>(r) * cols + c];
    return k;
}

static cv::Mat toCvKer(const mycv::Mat &k) {
    cv::Mat out(k.rows, k.cols, CV_32FC1);
    for (int r = 0; r < k.rows; ++r)
        for (int c = 0; c < k.cols; ++c)
            out.at<float>(r, c) = k.at<float>(r, c);
    return out;
}

static void testConvolve() {
    const float g = 1.0f / 16.0f;
    struct Kern {
        const char *name;
        int rows, cols;
        std::vector<float> v;
    };
    const std::vector<Kern> kernels = {
        {"identity", 3, 3, {0, 0, 0, 0, 1, 0, 0, 0, 0}},
        {"box 1/9",
         3,
         3,
         {1 / 9.f, 1 / 9.f, 1 / 9.f, 1 / 9.f, 1 / 9.f, 1 / 9.f, 1 / 9.f, 1 / 9.f, 1 / 9.f}},
        {"gauss 1/16", 3, 3, {g, 2 * g, g, 2 * g, 4 * g, 2 * g, g, 2 * g, g}},
        {"sobel x", 3, 3, {-1, 0, 1, -2, 0, 2, -1, 0, 1}},
        {"sobel y", 3, 3, {-1, -2, -1, 0, 0, 0, 1, 2, 1}},
        {"laplace 4", 3, 3, {0, 1, 0, 1, -4, 1, 0, 1, 0}},
        {"1x3 mean", 1, 3, {1 / 3.f, 1 / 3.f, 1 / 3.f}},
        {"2x2 mean", 2, 2, {0.25f, 0.25f, 0.25f, 0.25f}},
        {"5x5 gauss", 5, 5, {1,  4, 6, 4,  1,  4,  16, 24, 16, 4, 6, 24, 36,
                             24, 6, 4, 16, 24, 16, 4,  1,  4,  6, 4, 1}},
    };

    mycv::Mat bgr = makeRandomBGR(40, 50, 7u);
    cv::Mat bgrRef(bgr.rows, bgr.cols, CV_8UC3, bgr.data, bgr.step);

    // 单通道同一份数据
    mycv::Mat gray;
    gray.create(40, 50, mycv::U8C1);
    for (int r = 0; r < 40; ++r) {
        const uint8_t *s = bgr.ptr(r);
        uint8_t *d = gray.ptr(r);
        for (int i = 0; i < 50; ++i)
            d[i] = s[i * 3];
    }
    cv::Mat grayRef(40, 50, CV_8UC1, gray.data, gray.step);

    for (const auto &kc : kernels) {
        mycv::Mat k = makeKernel(kc.rows, kc.cols, kc.v);
        cv::Mat ck = toCvKer(k);

        mycv::Mat mg = mycv::convolve2D(gray, k);
        cv::Mat rg;
        cv::filter2D(grayRef, rg, -1, ck, cv::Point(-1, -1), 0, cv::BORDER_REFLECT_101);

        mycv::Mat mc = mycv::convolve2D(bgr, k);
        cv::Mat rc;
        cv::filter2D(bgrRef, rc, -1, ck, cv::Point(-1, -1), 0, cv::BORDER_REFLECT_101);

        CHECK(maxDiff(mg, rg) == 0);
        CHECK(maxDiff(mc, rc) == 0);
    }

    // 单位核必须逐字节等于原图
    mycv::Mat ident = makeKernel(3, 3, {0, 0, 0, 0, 1, 0, 0, 0, 0});
    CHECK(maxDiff(mycv::convolve2D(gray, ident), gray) == 0);
    CHECK(maxDiff(mycv::convolve2D(bgr, ident), bgr) == 0);

    // 核比图大：3x5 的图 + 5x5 核
    mycv::Mat tiny = makeRandomBGR(3, 5, 11u);
    cv::Mat tinyRef(3, 5, CV_8UC3, tiny.data, tiny.step);
    mycv::Mat k5 = makeKernel(5, 5, kernels.back().v);
    cv::Mat ck5 = toCvKer(k5);
    mycv::Mat mt = mycv::convolve2D(tiny, k5);
    cv::Mat rt;
    cv::filter2D(tinyRef, rt, -1, ck5, cv::Point(-1, -1), 0, cv::BORDER_REFLECT_101);
    CHECK(maxDiff(mt, rt) == 0);

    // 不连续输入：cv::filter2D 对 submatrix 有自己的边界路径，结果和 clone 后不同，
    // 所以这里只验自洽性 + 用 clone 后的连续版本对拍
    mycv::Mat sub = bgr.roi({10, 5, 20, 15});
    CHECK(!sub.isContinuous());
    mycv::Mat kbox = makeKernel(3, 3, kernels[1].v);
    cv::Mat ckbox = toCvKer(kbox);

    mycv::Mat onRoi = mycv::convolve2D(sub, kbox);
    mycv::Mat subC = sub.clone();
    mycv::Mat onClone = mycv::convolve2D(subC, kbox);
    CHECK(onRoi.isContinuous());
    CHECK(maxDiff(onRoi, onClone) == 0); // ROI 和 clone 必须等价

    cv::Mat rc;
    cv::filter2D(bgrRef(cv::Rect(10, 5, 20, 15)).clone(), rc, -1, ckbox, cv::Point(-1, -1), 0,
                 cv::BORDER_REFLECT_101);
    CHECK(maxDiff(onClone, rc) == 0);

    // 非法 anchor：-5 不是"默认值"，必须抛
    int thrown = 0;
    try {
        mycv::convolve2D(gray, ident, {-5, 0});
    } catch (const std::exception &) {
        ++thrown;
    }
    CHECK(thrown == 1);
}

// 8. blur / sobel：核生成 + convolve2D，零容差
static void testBlurSobel() {
    mycv::Mat bgr = makeRandomBGR(40, 50, 23u);
    cv::Mat bgrRef(40, 50, CV_8UC3, bgr.data, bgr.step);

    mycv::Mat gray;
    gray.create(40, 50, mycv::U8C1);
    for (int r = 0; r < 40; ++r) {
        const uint8_t *s = bgr.ptr(r);
        uint8_t *d = gray.ptr(r);
        for (int i = 0; i < 50; ++i)
            d[i] = s[i * 3 + 1];
    }
    cv::Mat grayRef(40, 50, CV_8UC1, gray.data, gray.step);

    // blur：方形 + 非方形
    const cv::Size blurSizes[] = {{3, 3}, {5, 5}, {7, 7}, {5, 3}, {3, 7}};
    for (const auto &ks : blurSizes) {
        mycv::Mat mg = mycv::blur(gray, {ks.width, ks.height});
        cv::Mat rg;
        cv::blur(grayRef, rg, ks);
        CHECK(maxDiff(mg, rg) == 0);

        mycv::Mat mc = mycv::blur(bgr, {ks.width, ks.height});
        cv::Mat rc;
        cv::blur(bgrRef, rc, ks);
        CHECK(maxDiff(mc, rc) == 0);
    }

    // sobel：dx / dy
    for (int d = 0; d <= 1; ++d) {
        const int dx = 1 - d, dy = d;

        mycv::Mat mg = mycv::sobel(gray, dx, dy);
        cv::Mat rg;
        cv::Sobel(grayRef, rg, CV_8U, dx, dy, 3);
        CHECK(maxDiff(mg, rg) == 0);

        mycv::Mat mc = mycv::sobel(bgr, dx, dy);
        cv::Mat rc;
        cv::Sobel(bgrRef, rc, CV_8U, dx, dy, 3);
        CHECK(maxDiff(mc, rc) == 0);
    }

    // 非法参数：偶数 ksize、零尺寸、dx+dy!=1、ksize!=3
    int thrown = 0;
    try {
        mycv::blur(gray, {4, 4});
    } catch (const std::exception &) {
        ++thrown;
    }
    try {
        mycv::blur(gray, {0, 3});
    } catch (const std::exception &) {
        ++thrown;
    }
    try {
        mycv::sobel(gray, 0, 0);
    } catch (const std::exception &) {
        ++thrown;
    }
    try {
        mycv::sobel(gray, 1, 1);
    } catch (const std::exception &) {
        ++thrown;
    }
    try {
        mycv::sobel(gray, 1, 0, 5);
    } catch (const std::exception &) {
        ++thrown;
    }
    CHECK(thrown == 5);
}

// 9. erode / dilate：边界跳过越界像素，零容差
static void testMorph() {
    mycv::Mat bgr = makeRandomBGR(40, 50, 31u);
    cv::Mat bgrRef(40, 50, CV_8UC3, bgr.data, bgr.step);

    mycv::Mat gray;
    gray.create(40, 50, mycv::U8C1);
    for (int r = 0; r < 40; ++r) {
        const uint8_t *s = bgr.ptr(r);
        uint8_t *d = gray.ptr(r);
        for (int i = 0; i < 50; ++i)
            d[i] = s[i * 3 + 2];
    }
    cv::Mat grayRef(40, 50, CV_8UC1, gray.data, gray.step);

    struct Spec {
        int shape, w, h;
    };
    const Spec specs[] = {
        {mycv::MORPH_RECT, 3, 3},    {mycv::MORPH_CROSS, 3, 3},   {mycv::MORPH_RECT, 5, 5},
        {mycv::MORPH_CROSS, 5, 5},   {mycv::MORPH_RECT, 5, 3},    {mycv::MORPH_RECT, 4, 4},
        {mycv::MORPH_ELLIPSE, 3, 3}, {mycv::MORPH_ELLIPSE, 5, 5}, {mycv::MORPH_ELLIPSE, 7, 7},
        {mycv::MORPH_ELLIPSE, 5, 3}, {mycv::MORPH_ELLIPSE, 4, 4}, {mycv::MORPH_ELLIPSE, 1, 1},
    };

    for (const auto &sp : specs) {
        mycv::Mat k = mycv::getStructuringElement(sp.shape, {sp.w, sp.h});
        cv::Mat ck = cv::getStructuringElement(sp.shape, cv::Size(sp.w, sp.h));

        // 结构元素本身要对
        CHECK(k.rows == ck.rows && k.cols == ck.cols);
        int kerDiff = 0;
        for (int r = 0; r < k.rows; ++r)
            for (int c = 0; c < k.cols; ++c) {
                int d =
                    static_cast<int>(k.at<uint8_t>(r, c)) - static_cast<int>(ck.at<uint8_t>(r, c));
                if (d != 0) kerDiff = 1;
            }
        CHECK(kerDiff == 0);

        // erode / dilate，单通道 + 三通道
        for (int mode = 0; mode < 2; ++mode) {
            const bool isErode = (mode == 0);

            mycv::Mat mg = isErode ? mycv::erode(gray, k) : mycv::dilate(gray, k);
            cv::Mat rg;
            if (isErode)
                cv::erode(grayRef, rg, ck);
            else
                cv::dilate(grayRef, rg, ck);
            CHECK(maxDiff(mg, rg) == 0);

            mycv::Mat mc = isErode ? mycv::erode(bgr, k) : mycv::dilate(bgr, k);
            cv::Mat rc;
            if (isErode)
                cv::erode(bgrRef, rc, ck);
            else
                cv::dilate(bgrRef, rc, ck);
            CHECK(maxDiff(mc, rc) == 0);
        }
    }

    // 灰度形态学：erode 整体不变亮，dilate 整体不变暗
    mycv::Mat k3 = mycv::getStructuringElement(mycv::MORPH_RECT, {3, 3});
    mycv::Mat e = mycv::erode(gray, k3);
    mycv::Mat d = mycv::dilate(gray, k3);
    bool eOk = true, dOk = true;
    for (int r = 0; r < 40; ++r) {
        const uint8_t *sg = gray.ptr(r);
        const uint8_t *se = e.ptr(r);
        const uint8_t *sd = d.ptr(r);
        for (int c = 0; c < 50; ++c) {
            if (se[c] > sg[c]) eOk = false;
            if (sd[c] < sg[c]) dOk = false;
        }
    }
    CHECK(eOk);
    CHECK(dOk);

    // 不连续输入
    mycv::Mat sub = bgr.roi({10, 5, 20, 15});
    CHECK(!sub.isContinuous());
    mycv::Mat subE = mycv::erode(sub, k3);
    mycv::Mat subEc = mycv::erode(sub.clone(), k3);
    CHECK(subE.isContinuous());
    CHECK(maxDiff(subE, subEc) == 0);

    // 非法：未知 shape
    int thrown = 0;
    try {
        mycv::getStructuringElement(99, {3, 3});
    } catch (const std::exception &) {
        ++thrown;
    }
    CHECK(thrown == 1);
}

// 10. draw：line/rectangle 逐像素对比；circle/ellipse 用弱断言
static mycv::Mat blackMat(int rows, int cols, int type) {
    mycv::Mat m;
    m.create(rows, cols, type);
    const int cn = mycv::channels(type);
    for (int r = 0; r < rows; ++r) {
        uint8_t *p = m.ptr(r);
        for (int i = 0; i < cols * cn; ++i)
            p[i] = 0;
    }
    return m;
}

static void testDraw() {
    // line：七条线段，和 cv::line 逐像素一致
    const int segs[][4] = {{2, 3, 25, 17}, {2, 2, 25, 2},   {5, 0, 5, 18}, {0, 0, 25, 18},
                           {3, 1, 10, 19}, {0, 15, 25, 14}, {1, 1, 24, 16}};
    for (const auto &s : segs) {
        mycv::Mat a = blackMat(20, 30, mycv::U8C1);
        cv::Mat b(20, 30, CV_8UC1, cv::Scalar(0));
        mycv::line(a, {s[0], s[1]}, {s[2], s[3]}, {255});
        cv::line(b, cv::Point(s[0], s[1]), cv::Point(s[2], s[3]), cv::Scalar(255));
        CHECK(maxDiff(a, b) == 0);
    }

    // rectangle：描边 + 填充都对
    {
        mycv::Mat a = blackMat(20, 30, mycv::U8C1);
        cv::Mat b(20, 30, CV_8UC1, cv::Scalar(0));
        mycv::rectangle(a, {3, 4}, {25, 16}, {200});
        cv::rectangle(b, cv::Point(3, 4), cv::Point(25, 16), cv::Scalar(200));
        CHECK(maxDiff(a, b) == 0);

        mycv::Mat c = blackMat(20, 30, mycv::U8C1);
        cv::Mat d(20, 30, CV_8UC1, cv::Scalar(0));
        mycv::rectangle(c, {3, 4}, {25, 16}, {200}, -1);
        cv::rectangle(d, cv::Point(3, 4), cv::Point(25, 16), cv::Scalar(200), -1);
        CHECK(maxDiff(c, d) == 0);
    }

    // circle：OpenCV 是参数化步进，像素对不上，只验对称和四个顶点
    {
        const int N = 41; // 奇数尺寸，中心落在 (20,20)
        mycv::Mat m = blackMat(N, N, mycv::U8C1);
        mycv::circle(m, {20, 20}, 10, {255});
        bool sym = true;
        for (int y = 0; y < N; ++y)
            for (int x = 0; x < N; ++x)
                if (m.at<uint8_t>(y, x) != m.at<uint8_t>(N - 1 - y, x) ||
                    m.at<uint8_t>(y, x) != m.at<uint8_t>(y, N - 1 - x))
                    sym = false;
        CHECK(sym);
        CHECK(m.at<uint8_t>(20, 30) == 255); // 右
        CHECK(m.at<uint8_t>(20, 10) == 255); // 左
        CHECK(m.at<uint8_t>(10, 20) == 255); // 上
        CHECK(m.at<uint8_t>(30, 20) == 255); // 下
        CHECK(m.at<uint8_t>(20, 20) == 0); // 圆心不画
    }

    // ellipse：四个轴顶点 + 对称 + 填充面积
    {
        const int H = 41, W = 51;
        mycv::Mat m = blackMat(H, W, mycv::U8C1);
        mycv::ellipse(m, {25, 20}, {12, 8}, {255});
        CHECK(m.at<uint8_t>(20, 25 + 12) == 255);
        CHECK(m.at<uint8_t>(20, 25 - 12) == 255);
        CHECK(m.at<uint8_t>(20 - 8, 25) == 255);
        CHECK(m.at<uint8_t>(20 + 8, 25) == 255);

        bool sym = true;
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                if (m.at<uint8_t>(y, x) != m.at<uint8_t>(H - 1 - y, x) ||
                    m.at<uint8_t>(y, x) != m.at<uint8_t>(y, W - 1 - x))
                    sym = false;
        CHECK(sym);

        mycv::Mat f = blackMat(H, W, mycv::U8C1);
        mycv::ellipse(f, {25, 20}, {12, 8}, {255}, -1);
        int cnt = 0;
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                if (f.at<uint8_t>(y, x)) ++cnt;
        // 理论面积 pi*12*8 ≈ 302，允许 10%
        CHECK(cnt > 270 && cnt < 335);
    }

    // 颜色 clamp
    {
        mycv::Mat m = blackMat(10, 10, mycv::U8C3);
        mycv::line(m, {0, 0}, {9, 9}, {300, -5, 128});
        CHECK(m.at<uint8_t>(0, 0) == 255);
        CHECK(m.at<uint8_t>(0, 1) == 0);
        CHECK(m.at<uint8_t>(0, 2) == 128);
    }

    // 越界线段：只画图内那段
    {
        mycv::Mat m = blackMat(10, 10, mycv::U8C1);
        mycv::line(m, {-20, 5}, {20, 5}, {255});
        int cnt = 0;
        for (int x = 0; x < 10; ++x)
            if (m.at<uint8_t>(5, x)) ++cnt;
        CHECK(cnt == 10);
    }
}

// 11. JPEG：解码和 cv::imread 逐字节一致，编码和 cv::imwrite 字节级相同
static void testJpeg() {
    // 彩色解码
    {
        mycv::Mat mine = mycv::imread(dataPath("test_color.jpg"));
        cv::Mat ref = cv::imread(dataPath("test_color.jpg"), cv::IMREAD_COLOR);
        CHECK(!mine.empty());
        CHECK(!ref.empty());
        CHECK(mine.rows == ref.rows && mine.cols == ref.cols);
        CHECK(mine.type == mycv::U8C3);
        CHECK(maxDiff(mine, ref) == 0);
    }

    // 灰度解码
    {
        mycv::Mat mine = mycv::imread(dataPath("test_gray.jpg"));
        cv::Mat ref = cv::imread(dataPath("test_gray.jpg"), cv::IMREAD_GRAYSCALE);
        CHECK(mine.type == mycv::U8C1);
        CHECK(maxDiff(mine, ref) == 0);
    }

    // 编码：写出去 cv 能读回，且内容接近原图
    {
        mycv::Mat src = mycv::imread(dataPath("test_color.jpg"));
        mycv::imwrite("/tmp/mycv_enc_test.jpg", src, 95);
        cv::Mat back = cv::imread("/tmp/mycv_enc_test.jpg", cv::IMREAD_COLOR);
        CHECK(!back.empty());
        CHECK(back.rows == src.rows && back.cols == src.cols);
        CHECK(maxDiff(src, back) < 40); // 有损，允许误差
    }

    // PGM 读进来到 JPEG 写出去，格式互通
    {
        mycv::Mat g = mycv::imread(dataPath("test_gray.pgm"));
        mycv::imwrite("/tmp/mycv_pgm2jpg.jpg", g, 95);
        cv::Mat back = cv::imread("/tmp/mycv_pgm2jpg.jpg", cv::IMREAD_GRAYSCALE);
        CHECK(!back.empty());
        CHECK(back.rows == g.rows && back.cols == g.cols);
    }

    // 魔数识别：扩展名写错也要按内容分派
    {
        std::FILE *in = std::fopen(dataPath("test_color.jpg").c_str(), "rb");
        CHECK(in != nullptr);
        std::vector<unsigned char> buf(4096);
        const std::size_t n = std::fread(buf.data(), 1, buf.size(), in);
        std::fclose(in);

        std::FILE *out = std::fopen("/tmp/wrong_ext.pgm", "wb");
        std::fwrite(buf.data(), 1, n, out);
        std::fclose(out);

        mycv::Mat m = mycv::imread("/tmp/wrong_ext.pgm"); // 名字是 .pgm，内容是 JPEG
        CHECK(m.type == mycv::U8C3);
        CHECK(m.rows == 24 && m.cols == 32);
    }
}

// 12. findContours：数量、外接矩形、面积和 OpenCV 对齐
static mycv::Mat fromCvU8(const cv::Mat &m) {
    mycv::Mat out;
    out.create(m.rows, m.cols, mycv::U8C1);
    for (int r = 0; r < m.rows; ++r)
        std::memcpy(out.ptr(r), m.ptr<uint8_t>(r), static_cast<std::size_t>(m.cols));
    return out;
}

static void compareContours(const char *name, const cv::Mat &cvBin) {
    mycv::Mat bin = fromCvU8(cvBin);

    const auto mine = mycv::findContours(bin);
    std::vector<std::vector<cv::Point>> theirs;
    cv::findContours(cvBin.clone(), theirs, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    CHECK(mine.size() == theirs.size());
    if (mine.size() != theirs.size()) {
        std::printf("  %-16s 数量 mycv=%zu cv=%zu\n", name, mine.size(), theirs.size());
        return;
    }

    // 两边扫描顺序不同（OpenCV 从下往上），按框左上角排序后再逐个比
    struct Entry {
        int x, y, w, h;
        double area;
    };
    std::vector<Entry> a, b;
    for (const auto &c : mine) {
        const mycv::Rect r = mycv::boundingRect(c);
        a.push_back({r.x, r.y, r.width, r.height, mycv::contourArea(c)});
    }
    for (const auto &c : theirs) {
        const cv::Rect r = cv::boundingRect(c);
        b.push_back({r.x, r.y, r.width, r.height, cv::contourArea(c)});
    }
    auto less = [](const Entry &p, const Entry &q) { return p.y != q.y ? p.y < q.y : p.x < q.x; };
    std::sort(a.begin(), a.end(), less);
    std::sort(b.begin(), b.end(), less);

    int worstArea = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        CHECK(a[i].x == b[i].x);
        CHECK(a[i].y == b[i].y);
        CHECK(a[i].w == b[i].w);
        CHECK(a[i].h == b[i].h);
        const int da = static_cast<int>(std::lround(std::abs(a[i].area - b[i].area)));
        if (da > worstArea) worstArea = da;
    }
    std::printf("  %-16s %zu 条，框全一致，面积最大差 %d 像素\n", name, mine.size(), worstArea);
}

static void testContours() {
    auto make = [](int h, int w) { return cv::Mat::zeros(h, w, CV_8UC1); };

    { // 实心矩形
        cv::Mat m = make(20, 30);
        cv::rectangle(m, {5, 4}, {20, 14}, cv::Scalar(255), -1);
        compareContours("实心矩形", m);
    }
    { // 两个分离物体
        cv::Mat m = make(30, 40);
        cv::rectangle(m, {2, 2}, {12, 12}, cv::Scalar(255), -1);
        cv::circle(m, {30, 20}, 6, cv::Scalar(255), -1);
        compareContours("两个物体", m);
    }
    { // L 形
        cv::Mat m = make(20, 20);
        cv::rectangle(m, {2, 2}, {15, 8}, cv::Scalar(255), -1);
        cv::rectangle(m, {2, 8}, {7, 16}, cv::Scalar(255), -1);
        compareContours("L 形", m);
    }
    { // 带孔
        cv::Mat m = make(24, 24);
        cv::rectangle(m, {3, 3}, {20, 20}, cv::Scalar(255), -1);
        cv::rectangle(m, {8, 8}, {14, 14}, cv::Scalar(0), -1);
        compareContours("带孔方环", m);
    }
    { // 单像素
        cv::Mat m = make(10, 10);
        m.at<uint8_t>(5, 5) = 255;
        compareContours("单像素", m);
    }
    { // 三个物体带噪声
        cv::Mat m = make(40, 50);
        cv::rectangle(m, {4, 4}, {16, 16}, cv::Scalar(255), -1);
        cv::circle(m, {35, 12}, 7, cv::Scalar(255), -1);
        cv::rectangle(m, {20, 26}, {44, 36}, cv::Scalar(255), -1);
        for (int i = 0; i < 6; ++i)
            m.at<uint8_t>(2 + i * 6, 24 + i * 3) = 255;
        compareContours("三个物体+噪声", m);
    }
}

// 13. convexHull / minAreaRect
struct NormRR {
    double cx, cy, w, h, ang;
};

// 统一成"width 是长边，angle 落在 [0,180)"，两边才好比
static NormRR normalizeRR(double cx, double cy, double w, double h, double ang) {
    if (h > w) {
        std::swap(w, h);
        ang += 90.0;
    }
    ang = std::fmod(ang, 180.0);
    if (ang < 0) ang += 180.0;
    return {cx, cy, w, h, ang};
}

static double angleGap(double a, double b) {
    double d = std::fabs(a - b);
    if (d > 90.0) d = 180.0 - d; // 角度在 0/180 附近时是同一个方向
    return d;
}

static void cmpRect(const char *name, const std::vector<mycv::Point> &pts) {
    std::vector<cv::Point> cvpts;
    cvpts.reserve(pts.size());
    for (const auto &p : pts)
        cvpts.emplace_back(p.x, p.y);

    const cv::RotatedRect cr = cv::minAreaRect(cvpts);
    const mycv::RotatedRect mr = mycv::minAreaRect(pts);

    const NormRR a = normalizeRR(mr.center.x, mr.center.y, mr.size.width, mr.size.height, mr.angle);
    const NormRR b = normalizeRR(cr.center.x, cr.center.y, cr.size.width, cr.size.height, cr.angle);

    // minAreaRect 可能存在多个等价解（三角形沿任一条边扣出的矩形面积相同），
    // 所以按定义验证：面积一致 + 所有点都在矩形内
    const double areaMine = a.w * a.h;
    const double areaCv = b.w * b.h;
    const double dArea = std::fabs(areaMine - areaCv);

    const double t = a.ang * M_PI / 180.0;
    const double ct = std::cos(t), st = std::sin(t);
    double worstOut = 0.0;
    for (const auto &p : pts) {
        const double dx = p.x - a.cx, dy = p.y - a.cy;
        const double u = dx * ct + dy * st; // 沿长边方向
        const double v = -dx * st + dy * ct; // 沿短边方向
        const double ou = std::fabs(u) - a.w * 0.5;
        const double ov = std::fabs(v) - a.h * 0.5;
        worstOut = std::max(worstOut, std::max(ou, ov));
    }

    std::printf("  %-16s 面积 %.1f / cv %.1f (差 %.2f) 最大出界 %.2f  "
                "尺寸(%.1f,%.1f) 角%.1f\n",
                name, areaMine, areaCv, dArea, worstOut, a.w, a.h, a.ang);
    CHECK(dArea < 1.0);
    CHECK(worstOut < 0.6); // 整数点取整误差

    // 凸包点数也要一致
    const auto hullMine = mycv::convexHull(pts);
    std::vector<cv::Point> hullCv;
    cv::convexHull(cvpts, hullCv);
    CHECK(hullMine.size() == hullCv.size());
}

static void testMinAreaRect() {
    // 轴对齐矩形
    cmpRect("轴对齐 50x30", {{10, 10}, {60, 10}, {60, 40}, {10, 40}});

    // 旋转 30 度的 50x20 矩形（整数点）
    cmpRect("旋转 30 度", {{83, 79}, {127, 104}, {117, 121}, {73, 96}});

    // 三角形
    cmpRect("三角形", {{0, 0}, {40, 0}, {0, 30}});

    // 正方形
    cmpRect("正方形", {{0, 0}, {20, 0}, {20, 20}, {0, 20}});

    // 圆上的点（16 边近似）
    {
        std::vector<mycv::Point> pts;
        for (int i = 0; i < 16; ++i) {
            const double t = i * M_PI / 8.0;
            pts.push_back({static_cast<int>(std::lround(100 + 40 * std::cos(t))),
                           static_cast<int>(std::lround(100 + 40 * std::sin(t)))});
        }
        cmpRect("16 边形近似圆", pts);
    }

    // 一堆带内部点的乱序点（凸包要能剔除内部点）
    {
        std::mt19937 rng(9u);
        std::uniform_int_distribution<int> dist(0, 99);
        std::vector<mycv::Point> pts;
        for (int i = 0; i < 200; ++i)
            pts.push_back({dist(rng), dist(rng)});
        cmpRect("随机 200 点", pts);
    }
}

static void testCanny() {
    // 三种输入：文件里的灰度图、随机噪声（边最多）、台阶 + 弱台阶（考滞后阈值）
    mycv::Mat pgm = mycv::imread(dataPath("test_gray.pgm"));
    CHECK(!pgm.empty());

    mycv::Mat bgr = makeRandomBGR(200, 300, 77u);
    mycv::Mat noise;
    noise.create(200, 300, mycv::U8C1);
    for (int r = 0; r < 200; ++r) {
        const uint8_t *s = bgr.ptr(r);
        uint8_t *d = noise.ptr(r);
        for (int i = 0; i < 300; ++i)
            d[i] = s[i * 3 + 1];
    }

    mycv::Mat step;
    step.create(60, 80, mycv::U8C1);
    for (int r = 0; r < 60; ++r) {
        uint8_t *d = step.ptr(r);
        for (int c = 0; c < 80; ++c)
            d[c] = (c >= 10 && c < 40) ? 200 : ((c >= 50 && c < 70) ? 3 : 0);
    }

    mycv::Mat images[] = {pgm, noise, step};
    cv::Mat refs[] = {cv::Mat(pgm.rows, pgm.cols, CV_8UC1, pgm.data, pgm.step),
                      cv::Mat(noise.rows, noise.cols, CV_8UC1, noise.data, noise.step),
                      cv::Mat(step.rows, step.cols, CV_8UC1, step.data, step.step)};

    // low 管连接、high 管种子：阈值扫一串，L1 和 L2 都要对上
    const double pairs[][2] = {{20, 60}, {60, 180}, {120, 360}, {250, 750}, {5, 500}};
    for (int k = 0; k < 3; ++k) {
        for (const auto &p : pairs) {
            mycv::Mat mine = mycv::canny(images[k], p[0], p[1]);
            cv::Mat ref;
            cv::Canny(refs[k], ref, p[0], p[1]);
            CHECK(maxDiff(mine, ref) == 0);
        }

        mycv::Mat mine = mycv::canny(images[k], 30, 90, true);
        cv::Mat ref;
        cv::Canny(refs[k], ref, 30, 90, 3, true);
        CHECK(maxDiff(mine, ref) == 0);
    }

    // 浮点阈值也是 cvFloor
    {
        mycv::Mat mine = mycv::canny(pgm, 50.5, 100.2);
        cv::Mat ref;
        cv::Canny(refs[0], ref, 50.5, 100.2);
        CHECK(maxDiff(mine, ref) == 0);
    }

    // 多通道、空图都要抛
    int thrown = 0;
    try {
        mycv::canny(bgr, 50, 150);
    } catch (const std::exception &) {
        ++thrown;
    }
    try {
        mycv::canny(mycv::Mat(), 50, 150);
    } catch (const std::exception &) {
        ++thrown;
    }
    CHECK(thrown == 2);
}

static void testGaussian() {
    mycv::Mat bgr = makeRandomBGR(120, 160, 31u);
    cv::Mat bgrRef(120, 160, CV_8UC3, bgr.data, bgr.step);

    mycv::Mat gray;
    gray.create(120, 160, mycv::U8C1);
    for (int r = 0; r < 120; ++r) {
        const uint8_t *s = bgr.ptr(r);
        uint8_t *d = gray.ptr(r);
        for (int i = 0; i < 160; ++i)
            d[i] = s[i * 3 + 1];
    }
    cv::Mat grayRef(120, 160, CV_8UC1, gray.data, gray.step);

    // ksize × sigma：sigma=0 的 3/5/7/9 走写死的整数核，其余走 exp 公式 + 误差扩散定点化
    const int sizes[] = {3, 5, 7, 9, 11, 15};
    const double sigmas[] = {0.0, 0.8, 1.2, 1.5, 2.5};
    for (int ks : sizes) {
        for (double sg : sigmas) {
            mycv::Mat mg = mycv::gaussianBlur(gray, {ks, ks}, sg);
            cv::Mat rg;
            cv::GaussianBlur(grayRef, rg, cv::Size(ks, ks), sg);
            CHECK(maxDiff(mg, rg) == 0);

            if (sg == 0.0 || sg == 1.2) { // 彩色少跑几组省时间
                mycv::Mat mc = mycv::gaussianBlur(bgr, {ks, ks}, sg);
                cv::Mat rc;
                cv::GaussianBlur(bgrRef, rc, cv::Size(ks, ks), sg);
                CHECK(maxDiff(mc, rc) == 0);
            }
        }
    }

    // 非方形核 + sigmaY 缺省取 sigmaX / 显式给 sigmaY
    {
        mycv::Mat m1 = mycv::gaussianBlur(gray, {5, 3}, 1.1);
        cv::Mat r1;
        cv::GaussianBlur(grayRef, r1, cv::Size(5, 3), 1.1);
        CHECK(maxDiff(m1, r1) == 0);

        mycv::Mat m2 = mycv::gaussianBlur(gray, {7, 5}, 0, 1.4);
        cv::Mat r2;
        cv::GaussianBlur(grayRef, r2, cv::Size(7, 5), 0, 1.4);
        CHECK(maxDiff(m2, r2) == 0);
    }

    // 单行 / 单列 / 单点：那一维的核要缩成 1，否则 reflect101 在 n=1 上原地打转
    {
        mycv::Mat row, col;
        row.create(1, 37, mycv::U8C1);
        col.create(37, 1, mycv::U8C1);
        for (int i = 0; i < 37; ++i) {
            row.ptr(0)[i] = static_cast<uint8_t>(i * 7);
            col.ptr(i)[0] = static_cast<uint8_t>(i * 11);
        }
        cv::Mat rowRef(1, 37, CV_8UC1, row.data, row.step);
        cv::Mat colRef(37, 1, CV_8UC1, col.data, col.step);

        mycv::Mat mr = mycv::gaussianBlur(row, {5, 5}, 0);
        cv::Mat rr;
        cv::GaussianBlur(rowRef, rr, cv::Size(5, 5), 0);
        CHECK(maxDiff(mr, rr) == 0);

        mycv::Mat mc = mycv::gaussianBlur(col, {5, 5}, 0);
        cv::Mat rc;
        cv::GaussianBlur(colRef, rc, cv::Size(5, 5), 0);
        CHECK(maxDiff(mc, rc) == 0);
    }

    // 偶数 ksize、比图还大的 ksize 都要抛
    int thrown = 0;
    try {
        mycv::gaussianBlur(gray, {4, 4}, 0);
    } catch (const std::exception &) {
        ++thrown;
    }
    try {
        mycv::gaussianBlur(gray, {999, 999}, 0);
    } catch (const std::exception &) {
        ++thrown;
    }
    CHECK(thrown == 2);
}

static void testMedian() {
    mycv::Mat bgr = makeRandomBGR(120, 160, 41u);
    cv::Mat bgrRef(120, 160, CV_8UC3, bgr.data, bgr.step);

    mycv::Mat gray;
    gray.create(120, 160, mycv::U8C1);
    for (int r = 0; r < 120; ++r) {
        const uint8_t *s = bgr.ptr(r);
        uint8_t *d = gray.ptr(r);
        for (int i = 0; i < 160; ++i)
            d[i] = s[i * 3 + 1];
    }
    cv::Mat grayRef(120, 160, CV_8UC1, gray.data, gray.step);

    // 椒盐噪声：中值的看家场景
    mycv::Mat salt = gray.clone();
    cv::Mat saltRef = grayRef.clone();
    std::mt19937 rng(5);
    for (int r = 0; r < 120; ++r) {
        uint8_t *d = salt.ptr(r);
        for (int c = 0; c < 160; ++c) {
            const int u = static_cast<int>(rng() & 0xffff);
            if (u < 3277)
                d[c] = 0;
            else if (u > 62258)
                d[c] = 255;
        }
    }
    saltRef = cv::Mat(120, 160, CV_8UC1, salt.data, salt.step);

    const int sizes[] = {1, 3, 5, 7, 9, 15};
    for (int ks : sizes) {
        for (int k = 0; k < 3; ++k) {
            const mycv::Mat &mi = (k == 0) ? gray : (k == 1 ? bgr : salt);
            const cv::Mat &ri = (k == 0) ? grayRef : (k == 1 ? bgrRef : saltRef);
            mycv::Mat m = mycv::medianBlur(mi, ks);
            cv::Mat r;
            cv::medianBlur(ri, r, ks);
            CHECK(maxDiff(m, r) == 0);
        }
    }

    // 退化尺寸：REPLICATE 边界下都能算（1×N、N×1、1×1、2×2）
    {
        mycv::Mat row, col, dot, two;
        row.create(1, 37, mycv::U8C1);
        col.create(37, 1, mycv::U8C1);
        dot.create(1, 1, mycv::U8C1);
        two.create(2, 2, mycv::U8C1);
        for (int i = 0; i < 37; ++i) {
            row.ptr(0)[i] = static_cast<uint8_t>(i * 7);
            col.ptr(i)[0] = static_cast<uint8_t>(i * 11);
        }
        dot.ptr(0)[0] = 123;
        two.ptr(0)[0] = 5;
        two.ptr(0)[1] = 250;
        two.ptr(1)[0] = 40;
        two.ptr(1)[1] = 90;

        const mycv::Mat *ms[] = {&row, &col, &dot, &two};
        for (const auto *m : ms) {
            cv::Mat rr(m->rows, m->cols, CV_8UC1, m->data, m->step);
            mycv::Mat mine = mycv::medianBlur(*m, 5);
            cv::Mat ref;
            cv::medianBlur(rr, ref, 5);
            CHECK(maxDiff(mine, ref) == 0);
        }
    }

    // ksize 0 / 偶数都要抛；ksize==1 是拷贝
    int thrown = 0;
    try {
        mycv::medianBlur(gray, 0);
    } catch (const std::exception &) {
        ++thrown;
    }
    try {
        mycv::medianBlur(gray, 4);
    } catch (const std::exception &) {
        ++thrown;
    }
    CHECK(thrown == 2);
    CHECK(maxDiff(mycv::medianBlur(gray, 1), gray) == 0);
}

// 两幅 CV_16S 图逐元素比，返回最大绝对差
static int maxDiff16(const mycv::Mat &a, const cv::Mat &b) {
    if (a.rows != b.rows || a.cols != b.cols) return -1;
    int worst = 0;
    for (int r = 0; r < a.rows; ++r) {
        const int16_t *pa = a.ptr<int16_t>(r);
        const int16_t *pb = b.ptr<int16_t>(r);
        for (int c = 0; c < a.cols; ++c) {
            const int d = pa[c] > pb[c] ? pa[c] - pb[c] : pb[c] - pa[c];
            if (d > worst) worst = d;
        }
    }
    return worst;
}

static void testGradientFamily() {
    mycv::Mat bgr = makeRandomBGR(40, 50, 53u);
    cv::Mat bgrRef(40, 50, CV_8UC3, bgr.data, bgr.step);

    mycv::Mat gray;
    gray.create(40, 50, mycv::U8C1);
    for (int r = 0; r < 40; ++r) {
        const uint8_t *s = bgr.ptr(r);
        uint8_t *d = gray.ptr(r);
        for (int i = 0; i < 50; ++i) d[i] = s[i * 3 + 1];
    }
    cv::Mat grayRef(40, 50, CV_8UC1, gray.data, gray.step);

    // Scharr：3x3 核 [-3,0,3;-10,0,10;-3,0,3]，核和 0、不归一化，8U 输出照样饱和
    for (int d = 0; d <= 1; ++d) {
        const int dx = 1 - d, dy = d;
        mycv::Mat mg = mycv::scharr(gray, dx, dy);
        cv::Mat rg;
        cv::Scharr(grayRef, rg, CV_8U, dx, dy);
        CHECK(maxDiff(mg, rg) == 0);

        mycv::Mat mc = mycv::scharr(bgr, dx, dy);
        cv::Mat rc;
        cv::Scharr(bgrRef, rc, CV_8U, dx, dy);
        CHECK(maxDiff(mc, rc) == 0);
    }

    // Laplacian：ksize=1 是四邻域，ksize=3 是八邻域
    for (int ks : {1, 3}) {
        mycv::Mat mg = mycv::laplacian(gray, ks);
        cv::Mat rg;
        cv::Laplacian(grayRef, rg, CV_8U, ks);
        CHECK(maxDiff(mg, rg) == 0);

        mycv::Mat mc = mycv::laplacian(bgr, ks);
        cv::Mat rc;
        cv::Laplacian(bgrRef, rc, CV_8U, ks);
        CHECK(maxDiff(mc, rc) == 0);
    }

    // spatialGradient：输出 16 位有符号，不饱和
    {
        mycv::Mat mdx, mdy;
        mycv::spatialGradient(gray, mdx, mdy);
        cv::Mat rdx, rdy;
        cv::spatialGradient(grayRef, rdx, rdy);
        CHECK(mdx.type == mycv::S16C1 && mdy.type == mycv::S16C1);
        CHECK(maxDiff16(mdx, rdx) == 0);
        CHECK(maxDiff16(mdy, rdy) == 0);

        // 随机图上梯度最大值必须超过 255，否则说明被截断成 8 位了
        int peak = 0;
        for (int r = 0; r < mdx.rows; ++r) {
            const int16_t *p = mdx.ptr<int16_t>(r);
            for (int c = 0; c < mdx.cols; ++c) {
                const int v = p[c] > 0 ? p[c] : -p[c];
                if (v > peak) peak = v;
            }
        }
        CHECK(peak > 255);
    }

    // 非法参数：scharr 的 dx+dy、laplacian 的 ksize、spatialGradient 的多通道
    int thrown = 0;
    try {
        mycv::scharr(gray, 1, 1);
    } catch (const std::exception &) {
        ++thrown;
    }
    try {
        mycv::laplacian(gray, 5);
    } catch (const std::exception &) {
        ++thrown;
    }
    {
        mycv::Mat a, b;
        try {
            mycv::spatialGradient(bgr, a, b);
        } catch (const std::exception &) {
            ++thrown;
        }
    }
    CHECK(thrown == 3);
}

int main() {
    testFromFile();
    testRandom();
    testRoi();
    testEdge();
    testResize();
    testThreshold();
    testConvolve();
    testBlurSobel();
    testCanny();
    testGaussian();
    testMedian();
    testGradientFamily();
    testMorph();
    testDraw();
    testJpeg();
    testContours();
    testMinAreaRect();

    if (g_failed == 0) {
        std::printf("test_compare_opencv: all passed\n");
        return 0;
    }
    std::printf("test_compare_opencv: %d check(s) failed\n", g_failed);
    return 1;
}
