# MyOpenCV --- 精简版OpenCV实现

笔者想深入探寻CV底层原理，于是心血来潮，基于OpenCV 4.10,
从零实现一套精简版 OpenCV。

## 项目结构（初步搭建）

```
MyOpenCV/
├── CMakeLists.txt
├── .clang-format               
├── include/mycv/
│   ├── mycv.hpp                门面汇总头            空
│   ├── core/
│   │   ├── types.hpp           类型常量/几何类型      完成
│   │   ├── check.hpp           MYCV_ASSERT          完成
│   │   ├── memory.hpp          对齐分配              完成
│   │   └── mat.hpp             矩阵类型              完成
│   ├── imgproc/
│   │   ├── color.hpp           cvtColor             完成（BGR2GRAY）
│   │   ├── geom.hpp            resize               完成（INTER_LINEAR）
│   │   ├── thresh.hpp          threshold            空
│   │   ├── filter.hpp          blur/sobel           空
│   │   ├── morph.hpp           erode/dilate         空
│   │   └── draw.hpp            line/rect/putText    空
│   └── io/
│       ├── image.hpp           imread/imwrite       完成（P5/P6）
│       └── video.hpp           VideoCapture         空
├── src/
│   ├── core/
|   |    ├── memory.cpp
|   |    └── mat.cpp
│   ├── imgproc/
|   |    ├── color.cpp
|   |    ├── geom.cpp
|   |    ├── thresh.cpp
|   |    ├── filter.cpp
|   |    ├── morph.cpp
|   |    └── draw.cpp
│   └── io/
|       ├── image.cpp
|       └── video.cpp
├── tests/
│   ├── test_mat.cpp            Mat 自身行为      52 条 CHECK
│   └── test_compare_opencv.cpp 和真 OpenCV 对比  5 组
├── samples/
│   ├── hello.cpp               
│   └── gray.cpp                                     空
└── data/                       测试用图
    ├── test_gray.pgm           64×48 P5
    └── test_color.ppm          32×24 P6

```

实现进度（截至2026.11.20）：

```
core/types.hpp       类型常量、几何类型           完成
core/check.hpp       MYCV_ASSERT               完成
core/memory.hpp      对齐分配器                  完成
core/mat.hpp         Mat                       完成
io/image.hpp         imread / imwrite          完成（P5/P6）
imgproc/color.hpp    cvtColor                  完成（BGR2GRAY）
imgproc/geom.hpp     resize                    完成（INTER_LINEAR）
imgproc/thresh.hpp   threshold                 未实现
imgproc/filter.hpp   blur / sobel              未实现
imgproc/morph.hpp    erode / dilate            未实现
imgproc/draw.hpp     line / rect / putText     未实现
io/video.hpp         VideoCapture              未实现
```

依赖单向向下，不反向：

```
  types.hpp      check.hpp      memory.hpp
      └──────────────┼──────────────┘
                     ▼
                 mat.hpp
        ┌────────────┼────────────┐
        ▼            ▼            ▼
   io/image.hpp  color.hpp    geom.hpp ...
        └────────────┼────────────┘
                     ▼
               samples / tests
```


## 编译与测试

```bash
cmake -B build -G Ninja
cmake --build build

cd build && ctest --output-on-failure
```

- `test_mat` 只依赖本库，测 Mat 自身行为
- `test_compare_opencv` 需要系统装的 OpenCV，同一份输入各跑一遍逐像素比

## core — Mat

一块按行排列的像素内存，六个字段描述。所有算法通过 `ptr(r)` 取行首指针，不直接接触 `data`。

```cpp
struct Mat {
    int rows, cols, type;
    size_t step;                    // 每行字节数
    uint8_t* data;                  // 指向本 Mat 第一个像素
    std::shared_ptr<uint8_t> owner; // buffer 所有权
};
```

- 浅拷贝，两个头共用一块 buffer；`clone()` 才复制像素
- `roi(Rect)` 返回新头，共享 owner，**step 沿用原图**，所以非整宽 ROI 不连续
- `at<T>(r, c)` 的 `c` 是元素序号不是列号：U8C3 上取第 c 列像素的 G 通道写 `at<uint8_t>(r, c*3+1)`
- `makeBuffer` 按 64 字节对齐分配，`F32C1` 才能安全 `reinterpret_cast<float*>`


测试：`test_mat.cpp` 64 条 CHECK，覆盖 create/step、ptr↔at 地址一致、clone 独立、ROI 共享、refcount（ROI 活过原图）、非法参数抛异常。

## core — types / check / memory

**types.hpp**：像素类型编码 `type = depth + ((cn-1) << 3)`，低 3 位是深度，高位是通道数。 \
**check.hpp**：`MYCV_ASSERT(cond)` 展开成 `if (!cond) throw std::runtime_error(...)`，消息含表达式原文和文件行号。不用 `assert`，因为 CMake 默认 Release，`NDEBUG` 会把 `assert` 整个编掉。\
**memory.hpp**：`makeBuffer(bytes)` 返回 `shared_ptr<uint8_t>`，用 `operator new(bytes, std::align_val_t{64})` 分配，deleter 必须用同样形式的带对齐的 `operator delete`。不用 `aligned_alloc`，因为它要求 size 是 alignment 的整数倍，而 `create(3,4,U8C1)` 是 12 字节。

## io — PGM/PPM 读写

支持 Netpbm 的二进制版本：`P5` 灰度 → `U8C1`，`P6` 彩色 → `U8C3`。

```cpp
Mat imread(const std::string& path);          // 失败抛 runtime_error
void imwrite(const std::string& path, const Mat& m);
```

- **P6 读进来转成 BGR，写出去转回 RGB**。OpenCV 的颜色约定是 BGR，只有转一下才能和 `cv::cvtColor` 对应上
- 头部解析必须顺序进行，没有固定偏移：字段之间任意空白分隔，`#` 到行尾是注释
- `MAXVAL` 后面**一个空白字符**，`eatSeparator` 只吃一个；CRLF 时 `\r` 后补一个 `\n`
- 读数字时多看的那个字符必须 `putback` 回去，否则 `MAXVAL` 后的分隔符被吃掉，整张图偏移一个字节
- 只支持 `MAXVAL=255`，16 位 PGM 直接抛异常

测试：往返（写出去再读回来逐像素比）、PIL 交叉验证（`imwrite` 的文件 PIL 能读且像素一致）、cv2 交叉验证、CRLF 头部、ROI 写入（不连续路径）。

## imgproc — cvtColor

```cpp
Mat cvtColor(const Mat& src, int code);   // 目前只实现 COLOR_BGR2GRAY
```

15 位定点系数，和 OpenCV 4.10 的 `color.simd_helpers.hpp` 一致：

```
gray = (b*3735 + g*19235 + r*9798 + 16384) >> 15
```

- 三个系数之和必须等于 `2^15`（32768），否则纯白 255 会算成 256 溢出
- `s[0]` 是 B，配 3735；写成 `s[0]*9798` 红蓝权重对调，出来是张合理但偏色的灰度图

测试：`test_compare_opencv` 的文件对拍、640×480 随机图、ROI 输入、五个边界色，**全部零容差**。

## imgproc — resize

```cpp
Mat resize(const Mat& src, Size dstSize, int interpolation = INTER_LINEAR);
```

双线性插值，权重 11 位定点（`COEF_SCALE = 2048`），累加后 `>> 22`。

- 坐标用半像素中心约定 `(d + 0.5) * scale - 0.5`。写成 `d * scale` 会整体偏移半个像素
- 两个权重必须互补（`w0 = 2048 - w1`）。否则各自独立四舍五入的话和可能是 2047/2049，两个权重之和不是2048了
- 边界 clamp 后要把小数部分清零，否则最后一行会拿最后两个像素重复加权
- 权重和 = `2048*2048`，最大累加 `255*2048*2048 < INT_MAX`，用 `int` 累加不溢出

测试：六种目标尺寸 + 恒等变换 + ROI 输入。恒等和 0.5 倍缩放零差异，其余比例允许差 1。

## 和 OpenCV 的差异

| 函数 | 差异 | 测试容差 |
|---|---|---|
| cvtColor(BGR2GRAY) | 逐字节一致 | 0 |
| resize(INTER_LINEAR) | 最大差 1（OpenCV 走两阶段 `ufixedpoint16` + SIMD） | 1 |
| imread / imwrite | 只有 P5/P6、8 位；通道转 BGR | 往返零差异 |

## 约束

- 像素类型只有 `U8C1` / `U8C3` / `F32C1`
- 图像 IO 只做 PGM/PPM，暂时不做 PNG/JPEG
- 检查失败抛 `std::runtime_error`，不用 `assert`
- 4K 以上图像未测 ！（没找素材呢）

## 计划（2026.11.21）

threshold / blur / sobel / erode / dilate / draw / VideoCapture
