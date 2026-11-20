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
#include <mycv/imgproc/geom.hpp>
#include <mycv/imgproc/filter.hpp>
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
        for (int c = 0; c < k.cols; ++c) out.at<float>(r, c) = k.at<float>(r, c);
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
        {"box 1/9", 3, 3,
         {1 / 9.f, 1 / 9.f, 1 / 9.f, 1 / 9.f, 1 / 9.f, 1 / 9.f, 1 / 9.f, 1 / 9.f, 1 / 9.f}},
        {"gauss 1/16", 3, 3, {g, 2 * g, g, 2 * g, 4 * g, 2 * g, g, 2 * g, g}},
        {"sobel x", 3, 3, {-1, 0, 1, -2, 0, 2, -1, 0, 1}},
        {"sobel y", 3, 3, {-1, -2, -1, 0, 0, 0, 1, 2, 1}},
        {"laplace 4", 3, 3, {0, 1, 0, 1, -4, 1, 0, 1, 0}},
        {"1x3 mean", 1, 3, {1 / 3.f, 1 / 3.f, 1 / 3.f}},
        {"2x2 mean", 2, 2, {0.25f, 0.25f, 0.25f, 0.25f}},
        {"5x5 gauss", 5, 5,
         {1, 4, 6, 4, 1, 4, 16, 24, 16, 4, 6, 24, 36, 24, 6, 4, 16, 24, 16, 4, 1, 4, 6, 4, 1}},
    };

    mycv::Mat bgr = makeRandomBGR(40, 50, 7u);
    cv::Mat bgrRef(bgr.rows, bgr.cols, CV_8UC3, bgr.data, bgr.step);

    // 单通道同一份数据
    mycv::Mat gray;
    gray.create(40, 50, mycv::U8C1);
    for (int r = 0; r < 40; ++r) {
        const uint8_t *s = bgr.ptr(r);
        uint8_t *d = gray.ptr(r);
        for (int i = 0; i < 50; ++i) d[i] = s[i * 3];
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

int main() {
    testFromFile();
    testRandom();
    testRoi();
    testEdge();
    testResize();
    testThreshold();
    testConvolve();

    if (g_failed == 0) {
        std::printf("test_compare_opencv: all passed\n");
        return 0;
    }
    std::printf("test_compare_opencv: %d check(s) failed\n", g_failed);
    return 1;
}
