#include <cstdint>
#include <cstdio>
#include <exception>

#include <mycv/core/mat.hpp>
#include <mycv/core/types.hpp>

// ---- types.hpp 的编译期验收 ----
static_assert(mycv::U8C1 == 0, "U8C1 应是深度 8U + 单通道");
static_assert(mycv::U8C3 == 16, "U8C3 = 0 + (3-1)<<3");
static_assert(mycv::F32C1 == 5, "F32C1 的 type 值等于 DEPTH_32F");
static_assert(mycv::channels(mycv::U8C3) == 3, "channels 先右移再 +1");
static_assert(mycv::channels(mycv::U8C1) == 1, "单通道不能算成 0");
static_assert(mycv::depthOf(mycv::F32C1) == mycv::DEPTH_32F, "depthOf 取低 3 位");
static_assert(mycv::elemSize1(mycv::F32C1) == 4, "float 单通道 4 字节");
static_assert(mycv::elemSize(mycv::U8C1) == 1, "U8C1 一像素一字节");
static_assert(mycv::elemSize(mycv::U8C3) == 3, "U8C3 一像素三字节");
static_assert(mycv::elemSize(mycv::F32C1) == 4, "F32C1 一像素四字节");
static_assert(mycv::elemSize(7) == 0, "未支持的深度落到 default 返回 0");

static int g_failed = 0;

#define CHECK(cond)                                                                                \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);                            \
            ++g_failed;                                                                            \
        }                                                                                          \
    } while (0)

// 造一张 rows×cols 的 U8C1
static mycv::Mat makeTestMat(int rows, int cols) {
    mycv::Mat m;
    m.create(rows, cols, mycv::U8C1);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            m.at<uint8_t>(r, c) = static_cast<uint8_t>(r * cols + c);
    return m;
}

static void test_create() {
    mycv::Mat m;
    m.create(3, 4, mycv::U8C1);
    CHECK(m.rows == 3);
    CHECK(m.cols == 4);
    CHECK(m.type == mycv::U8C1);
    CHECK(m.step == 4);
    CHECK(m.data != nullptr);
    CHECK(!m.empty());
    CHECK(m.isContinuous());
    CHECK(reinterpret_cast<uintptr_t>(m.data) % 64 == 0);

    mycv::Mat f;
    f.create(2, 2, mycv::F32C1);
    CHECK(f.step == 8);
    CHECK(f.isContinuous());

    // 重复 create 覆盖旧 buffer
    m.create(8, 8, mycv::U8C1);
    CHECK(m.rows == 8);
    CHECK(m.step == 8);

    // 非法参数测试
    int thrown = 0;
    try {
        m.create(0, 4, mycv::U8C1);
    } catch (const std::exception &) {
        ++thrown;
    }
    try {
        m.create(4, -1, mycv::U8C1);
    } catch (const std::exception &) {
        ++thrown;
    }
    try {
        m.create(2, 2, 7); // 未支持的 type
    } catch (const std::exception &) {
        ++thrown;
    }
    CHECK(thrown == 3);
    CHECK(m.rows == 8); // 失败的 create 不该把对象改坏
}

static void test_access() {
    mycv::Mat g = makeTestMat(3, 4);
    CHECK(g.at<uint8_t>(0, 0) == 0);
    CHECK(g.at<uint8_t>(1, 2) == 6);
    CHECK(g.at<uint8_t>(2, 3) == 11);
    CHECK(&g.at<uint8_t>(1, 0) == g.ptr<uint8_t>(1));
    CHECK(g.ptr<uint8_t>(1)[2] == 6);

    // const 路径
    const mycv::Mat &cg = g;
    CHECK(cg.at<uint8_t>(1, 2) == 6);
    CHECK(cg.ptr<uint8_t>(2)[3] == 11);

    // U8C3
    mycv::Mat c3;
    c3.create(2, 2, mycv::U8C3);
    for (int r = 0; r < 2; ++r)
        for (int c = 0; c < 2; ++c) {
            c3.at<uint8_t>(r, c * 3 + 0) = static_cast<uint8_t>(10 + c);
            c3.at<uint8_t>(r, c * 3 + 1) = static_cast<uint8_t>(20 + c);
            c3.at<uint8_t>(r, c * 3 + 2) = static_cast<uint8_t>(30 + c);
        }
    CHECK(c3.step == 6);
    CHECK(c3.at<uint8_t>(1, 0) == 10);
    CHECK(c3.at<uint8_t>(1, 3) == 11);
    CHECK(c3.at<uint8_t>(1, 5) == 31);

    // F32C1
    mycv::Mat f;
    f.create(2, 3, mycv::F32C1);
    f.at<float>(0, 2) = 1.25f;
    f.at<float>(1, 0) = -4.5f;
    CHECK(f.step == 12);
    CHECK(f.at<float>(0, 2) == 1.25f);
    CHECK(f.at<float>(1, 0) == -4.5f);

    // 越界、、空 Mat
    int thrown = 0;
    try {
        g.ptr(-1);
    } catch (const std::exception &) {
        ++thrown;
    }
    try {
        g.ptr(3);
    } catch (const std::exception &) {
        ++thrown;
    }
    mycv::Mat empty;
    try {
        empty.ptr(0);
    } catch (const std::exception &) {
        ++thrown;
    }
    CHECK(thrown == 3);
}

static void test_clone() {
    mycv::Mat img = makeTestMat(4, 6);
    mycv::Mat cp = img.clone();

    CHECK(cp.rows == 4);
    CHECK(cp.cols == 6);
    CHECK(cp.type == mycv::U8C1);
    CHECK(cp.step == 6);
    CHECK(cp.isContinuous());

    bool same = true;
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 6; ++c)
            if (cp.at<uint8_t>(r, c) != img.at<uint8_t>(r, c)) same = false;
    CHECK(same);

    // 深拷贝：改原图不影响副本
    img.at<uint8_t>(1, 2) = 7;
    CHECK(cp.at<uint8_t>(1, 2) == 8);
    CHECK(img.at<uint8_t>(1, 2) == 7);

    // 不连续源：必须逐行拷，第 1 行要取自原图第 2 行
    mycv::Mat sub = img.roi({2, 1, 3, 2});
    CHECK(!sub.isContinuous());
    mycv::Mat sc = sub.clone();
    CHECK(sc.isContinuous());
    CHECK(sc.step == 3);
    CHECK(sc.at<uint8_t>(0, 0) == img.at<uint8_t>(1, 2));
    CHECK(sc.at<uint8_t>(0, 2) == img.at<uint8_t>(1, 4));
    CHECK(sc.at<uint8_t>(1, 0) == img.at<uint8_t>(2, 2));
    CHECK(sc.at<uint8_t>(1, 2) == img.at<uint8_t>(2, 4));
}

static void test_roi() {
    mycv::Mat img = makeTestMat(4, 6);
    mycv::Mat sub = img.roi({2, 1, 3, 2}); // x=2 y=1 w=3 h=2

    CHECK(sub.rows == 2);
    CHECK(sub.cols == 3);
    CHECK(sub.type == mycv::U8C1);
    CHECK(sub.step == img.step); // 沿用原图步长
    CHECK(!sub.isContinuous()); // 3 列 / 6 列，非整宽
    CHECK(sub.data == img.data + img.step + 2);
    CHECK(sub.at<uint8_t>(0, 0) == 8); // 原图 (1,2)
    CHECK(sub.at<uint8_t>(1, 2) == 16); // 原图 (2,4)

    // 共享：改 ROI，原图同步变
    sub.at<uint8_t>(0, 0) = 99;
    CHECK(img.at<uint8_t>(1, 2) == 99);

    // 整宽 ROI 仍然连续
    mycv::Mat full = img.roi({0, 0, 6, 2});
    CHECK(full.step == 6);
    CHECK(full.isContinuous());

    // roi 套 roi
    mycv::Mat inner = sub.roi({1, 1, 2, 1}); // sub 内 (1,1) 起，即原图 (2,3)
    CHECK(inner.at<uint8_t>(0, 0) == img.at<uint8_t>(2, 3));

    // 越界 / 负数 / 零尺寸
    int thrown = 0;
    try {
        img.roi({0, 0, 7, 1});
    } catch (const std::exception &) {
        ++thrown;
    }
    try {
        img.roi({-1, 0, 2, 2});
    } catch (const std::exception &) {
        ++thrown;
    }
    try {
        img.roi({0, 0, 0, 2});
    } catch (const std::exception &) {
        ++thrown;
    }
    try {
        img.roi({0, 3, 2, 2}); // y+h 越界
    } catch (const std::exception &) {
        ++thrown;
    }
    CHECK(thrown == 4);
}

static void test_lifetime() {
    // ROI 比原图活得久
    mycv::Mat keeper;
    {
        mycv::Mat tmp;
        tmp.create(10, 10, mycv::U8C1);
        tmp.at<uint8_t>(3, 4) = 42;
        keeper = tmp.roi({0, 0, 5, 5});
        CHECK(keeper.at<uint8_t>(3, 4) == 42);
    }
    CHECK(keeper.at<uint8_t>(3, 4) == 42);
    keeper.at<uint8_t>(0, 0) = 5;
    CHECK(keeper.at<uint8_t>(0, 0) == 5);

    // release 原图，ROI 仍然可用
    mycv::Mat a;
    a.create(5, 5, mycv::U8C1);
    a.at<uint8_t>(2, 2) = 33;
    mycv::Mat b = a.roi({1, 1, 3, 3});
    a.release();
    CHECK(a.empty());
    CHECK(a.data == nullptr);
    CHECK(b.at<uint8_t>(1, 1) == 33);

    // 普通拷贝也只是浅拷贝：两个头共用 buffer
    mycv::Mat c = b;
    c.at<uint8_t>(0, 0) = 77;
    CHECK(b.at<uint8_t>(0, 0) == 77);
}

int main() {
    test_create();
    test_access();
    test_clone();
    test_roi();
    test_lifetime();

    if (g_failed == 0) {
        std::printf("test_mat: all passed\n");
        return 0;
    }
    std::printf("test_mat: %d check(s) failed\n", g_failed);
    return 1;
}
