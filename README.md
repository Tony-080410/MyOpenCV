# MyOpenCV --- 精简版OpenCV实现

笔者想深入探寻CV底层原理，于是心血来潮，基于OpenCV 4.10,
从零实现一套精简版 OpenCV。

## 项目结构

```
MyOpenCV/
├── CMakeLists.txt
├── .clang-format
├── .gitignore
├── README.md
├── LICENSE
├── include/mycv/
│   ├── mycv.hpp                 门面汇总头                       空
│   ├── core/
│   │   ├── types.hpp            类型常量/几何类型                 完成
│   │   ├── check.hpp            MYCV_ASSERT                    完成
│   │   ├── memory.hpp           对齐分配                        完成
│   │   └── mat.hpp              矩阵类型                        完成
│   ├── imgproc/
│   │   ├── color.hpp            cvtColor                       完成（BGR2GRAY）
│   │   ├── geom.hpp             resize                         完成（INTER_LINEAR）
│   │   ├── thresh.hpp           threshold                      完成（5 种 type）
│   │   ├── filter.hpp           卷积/平滑/求导 8 个函数           完成
│   │   ├── morph.hpp            erode/dilate                   完成（三种核）
│   │   ├── draw.hpp             line/rect/circle/ellipse       完成
│   │   ├── contours.hpp         findContours/凸包/minAreaRect   完成
│   │   └── edge.hpp             canny                          完成（L1/L2）
│   └── io/
│       ├── image.hpp            imread/imwrite                 完成（P5/P6 + JPG/JPEG）
│       └── video.hpp            VideoCapture                   空
├── src/
│   ├── core/
│   │   ├── mat.cpp
│   │   └── memory.cpp
│   ├── imgproc/
│   │   ├── color.cpp
│   │   ├── geom.cpp
│   │   ├── thresh.cpp
│   │   ├── filter.cpp
│   │   ├── morph.cpp
│   │   ├── draw.cpp
│   │   ├── contours.cpp
│   │   └── edge.cpp
│   └── io/
│       ├── image.cpp
│       └── video.cpp                                            空
├── tests/
│   ├── test_mat.cpp             Mat 自身行为                     64 条 CHECK
│   └── test_compare_opencv.cpp  和真 OpenCV 对比                 17 组
│ 
├── samples/
│   ├── hello.cpp
│   └── gray.cpp                                                 空
└── data/                        测试用图
    ├── test_gray.pgm            64×48 P5
    ├── test_gray.jpg            同一张图的灰度 JPG
    ├── test_color.ppm           32×24 P6
    └── test_color.jpg           同一张图的彩色 JPG
```

实现进度（截至2026.11.21）：

```
core/types.hpp        类型常量、几何类型           完成
core/check.hpp        MYCV_ASSERT               完成
core/memory.hpp       对齐分配器                  完成
core/mat.hpp          Mat                       完成
io/image.hpp          imread/imwrite            完成（P5/P6 + JPG/JPEG）
imgproc/color.hpp     cvtColor                  完成（BGR2GRAY）
imgproc/geom.hpp      resize                    完成（INTER_LINEAR）
imgproc/thresh.hpp    threshold                 完成（5 种 type）
imgproc/filter.hpp    convolve2D                完成（通用核）
imgproc/filter.hpp    blur/sobel                完成
imgproc/filter.hpp    gaussianBlur              完成（2^8 定点）
imgproc/filter.hpp    medianBlur                完成（滑动直方图）
imgproc/filter.hpp    scharr/laplacian          完成
imgproc/filter.hpp    spatialGradient           完成（16S 输出）
imgproc/morph.hpp     erode/dilate              完成（三种核）
imgproc/draw.hpp      line/rect/circle/ellipse  完成
imgproc/contours.hpp  findContours/拟合          完成
imgproc/edge.hpp      canny                     完成（L1/L2）
io/video.hpp          VideoCapture              未实现
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
- `test_compare_opencv` 需要系统装的 OpenCV，同一份输入两边各跑一遍逐像素比

性能基准单独编，不进 ctest：

```bash
cmake -DMYCV_BUILD_BENCH=ON -B build
cmake --build build --target bench
taskset -c 0 ./build/tests/bench
```

`taskset` 不是可选项：混合架构 CPU 上进程会在 P 核和 E 核之间迁移，结果差近 2 倍。优化记录见 [OPTIMIZATION.md](OPTIMIZATION.md)。

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

**types.hpp**：像素类型编码 `type = depth + ((cn-1) << 3)`，低 3 位是深度，高位是通道数；常量 `U8C1` / `U8C3` / `S16C1` / `F32C1`。 \
**check.hpp**：`MYCV_ASSERT(cond)` 展开成 `if (!cond) throw std::runtime_error(...)`，消息含表达式原文和文件行号。不用 `assert`，因为 CMake 默认 Release，`NDEBUG` 会把 `assert` 整个编掉。\
**memory.hpp**：`makeBuffer(bytes)` 返回 `shared_ptr<uint8_t>`，用 `operator new(bytes, std::align_val_t{64})` 分配，deleter 必须用同样形式的带对齐的 `operator delete`。不用 `aligned_alloc`，因为它要求 size 是 alignment 的整数倍，而 `create(3,4,U8C1)` 是 12 字节。

## io — PGM/PPM + JPEG 读写

`imread` 按**魔数**分派：`P5`/`P6` 走自己的 Netpbm 解析，`FF D8` 走 libjpeg。
`imwrite` 按**扩展名**分派：`.jpg`/`.jpeg` 走 JPEG，其余写 PGM/PPM。

```cpp
Mat imread(const std::string& path);                        // 失败抛 runtime_error
void imwrite(const std::string& path, const Mat& m, int jpegQuality = 95);
```

- **P6 读进来转成 BGR，写出去转回 RGB**。OpenCV 的颜色约定是 BGR，只有转一下才能和 `cv::cvtColor` 对应上，JPEG 同理
- 格式判断看内容不看扩展名：把 JPEG 改名成 `.pgm` 也能正确读出
- JPEG 编解码链 libjpeg（`find_package(JPEG)`，没装就只支持 PGM/PPM，配置时会打印提示）

- **P6 读进来转成 BGR，写出去转回 RGB**。OpenCV 的颜色约定是 BGR，只有转一下才能和 `cv::cvtColor` 对应上
- 头部解析必须顺序进行，没有固定偏移：字段之间任意空白分隔，`#` 到行尾是注释
- `MAXVAL` 后面**一个空白字符**，`eatSeparator` 只吃一个；CRLF 时 `\r` 后补一个 `\n`
- 读数字时多看的那个字符必须 `putback` 回去，否则 `MAXVAL` 后的分隔符被吃掉，整张图偏移一个字节
- 只支持 `MAXVAL=255`，16 位 PGM 抛异常

测试：往返（写出去再读回来逐像素比）、PIL 交叉验证（`imwrite` 的文件 PIL 能读且像素一致）、cv2 交叉验证、CRLF 头部、ROI 写入（不连续路径）；JPEG 解码和 `cv::imread` **逐字节一致**，编码和 `cv::imwrite` **字节级相同**（同一 quality 下 MD5 一致）。

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

## imgproc — threshold

```cpp
Mat threshold(const Mat& src, double thresh, double maxval, int type);
```

目前支持五种 type：`BINARY` / `BINARY_INV` / `TRUNC` / `TOZERO` / `TOZERO_INV`，枚举值和 OpenCV 相同（0~4）。

- 比较用 `>` （不用 `>=`）：等于阈值的像素归到暗的一侧（踩坑了。。。）
- `thresh` 用 `floor`，`maxval` 用 `lround`，然后 clamp 到 0..255。两个参数取整方式不同，（抄袭 OpenCV 的处理方式。。。）
- TRUNC 是 `min(v, t)` 之后必须饱和：`t = -1` 时 `min(v,-1) = -1`，直接赋给 `uint8_t` 会变成 255
- `dst = src.clone()`：每个分支都要读原值，`clone` 顺便把不连续 ROI 摊平成连续，后面可以一维扫
- 像素值先提成 `int v`，所有分支在 `int` 域算完，最后一次性转回 `uint8_t`

测试：7 个阈值 × 5 种 type + 6 个 maxval + ROI 输入，**全部零差异**。

## imgproc — filter：convolve2D

```cpp
Mat convolve2D(const Mat &src, const Mat &kernel, Point anchor = {-1, -1});
```

核用 `Mat(F32C1)` 表达，复用 `create` 的 64 字节对齐。`anchor` 只有 `-1` 表示取核中心。

- 舍入用 `std::lrint`（银行家舍入）。与OpenCV的处理保持一致 
- `sum` 用 `double`：sobel 有负系数，laplace 中心是 -4，结果可能为负，直接转 `uint8_t` 会回绕成亮白
- 边界是 `BORDER_REFLECT_101`（下标 -1 映到 1，`n` 映到 `n-2`）
- 偶数核（2×2）锚点是 `kh/2`，不是 `kh/2-1`。约定错了 2×2 的结果会整体往右下偏一格
- `anchor` 判断必须写 `== -1`。写成 `< 0` 的话 `anchor={-5,0}` 也被当成默认值，后面的范围断言永远不触发（2026.11.20 记录）

测试：九个核（单位/均值/高斯/sobel xy/laplace/1×3/2×2/5×5）在 U8C1 和 U8C3 上全部零差异，加权核比图大（3×5 图 + 5×5 核）。

## imgproc — filter：blur / sobel

```cpp
Mat blur(const Mat &src, Size ksize);
Mat sobel(const Mat &src, int dx, int dy, int ksize = 3);
```

- `blur` 拒绝偶数 ksize。偶数面积时 `sum/area` 会出现精确的 .5，`cv::blur` 利用 `boxFilter`（整数除）和 `filter2D`（浮点乘 + `cvRound`）舍入方向不同
- `sobel` 的核和是 0，**不归一化**。`/8` 是 `cv::Sobel` 的可选 `scale` 参数，默认 1.0，加了结果整体缩小 8 倍
- `sobel` 在随机噪声上约 17% 像素饱和到 0/255，这是 8 位输出的正常现象，不是 bug
- `dy=1` 的核是 `[[-1,-2,-1],[0,0,0],[1,2,1]]`（下减上）。取反后梯度角整体偏 180°，画面上看不出来，做 `atan2` 才会错

测试：blur 五种 ksize（含非方形 5×3、3×7）+ sobel 两个方向，U8C1/U8C3 全部零差异；偶数 ksize、零尺寸、`dx+dy!=1`、`ksize!=3` 五种非法输入都抛异常。

### gaussianBlur

```cpp
Mat gaussianBlur(const Mat &src, Size ksize, double sigmaX = 0, double sigmaY = 0);
```

可分离两趟 + 2^8 定点，和 `cv::GaussianBlur` **逐字节一致**。

- 核压成整数（和恒为 256）：水平趟整数累加不舍入（最大 `255*256`），垂直趟后 `(acc + 2^15) >> 16`。`CV_DESCALE` 是 round half up，不是银行家舍入，写成截断会全图偏暗 1
- **定点化不是 `round(k*256)`**，是误差扩散：左半边逐项 `cvRound(k[i]*256 + err)`、误差带到下一项，中心项用 `256 - 2*sum` 补齐。直接 round 在 5×5/7×7/11×11 上差 1~2
- `sigma <= 0` 且 ksize 是 3/5/7/9 时核是**写死的整数比**：3 → 1,2,1 /4；5 → 1,4,6,4,1 /16；7 → 2,7,14,18,14,7,2 /64；9 → 4,13,30,51,60,51,30,13,4 /256。其他尺寸才走 `sigma = 0.15n + 0.35` 的指数公式
- `sigmaY <= 0` 取 `sigmaX`；单行/单列图把那一维的核缩成 1，否则 `reflect101` 在 n=1 上原地打转
- 反射索引按像素算再乘 `cn`，直接按字节反射会让三通道错位

测试：灰度 + 彩色随机图 × ksize{3,5,7,9,11,15} × sigma{0,0.8,1.2,1.5,2.5}、非方形核 5×3 / 7×5、单行/单列、偶数 ksize 与超大 ksize 抛异常，**全部零差异**。速度：640×480 单线程 0.77~9.86 ms，比 OpenCV 慢 9~17 倍（它走 SIMD；方框核的滑动窗口在这里用不上，高斯核不均匀）。

### medianBlur

```cpp
Mat medianBlur(const Mat &src, int ksize);
```

排序取中间值，不是加权求和，所以不走 `convolve2D`。去椒盐噪声是它的主场：3×3 窗口里 8 个 10 加一个 200，`blur` 出 31、`gaussianBlur` 出 58，中值直接是 10 —— 离群值被丢掉，没参与平均。

- **边界是 `BORDER_REPLICATE`**（越界坐标夹到边缘），和 `blur` / `gaussianBlur` 的 `REFLECT_101` **相反**
- `ksize` 必须是不小于 1 的奇数；`ksize == 1` 不抛异常，直接等于拷贝
- **中值唯一**（奇数个元素），没有并列歧义，所以能做到逐字节零差异
- 实现是**滑动直方图**：256 桶 + 中值指针，每像素只做 2k 次桶增减，耗时基本不随 k 变（k=3/5/9 都在 6 ms 上下）。排序版（每像素 `nth_element`）k=3 要 21 ms、k=9 要 199 ms
- 中值指针的维护：`below` = 严格小于 `med` 的个数，中值满足 `below <= m < below + hist[med]`，滑一格后上下各修正一次，通常 0~2 步

测试：灰度 / 彩色 / 椒盐 × ksize{1,3,5,7,9,15}、单行 / 单列 / 1×1 / 2×2 四种退化尺寸、ksize 0 与偶数抛异常，**全部零差异**。速度：640×480 单线程 k=3 6.4 ms / k=5 5.8 ms / k=9 6.2 ms（k=9 和 `cv::medianBlur` 持平，k=3 慢 35 倍，它走 SIMD 排序网络）。

### Scharr / Laplacian / spatialGradient

```cpp
Mat scharr(const Mat &src, int dx, int dy);             // 3x3
Mat laplacian(const Mat &src, int ksize = 1);           // ksize 只支持 1 或 3
void spatialGradient(const Mat &src, Mat &dx, Mat &dy); // 输出 S16C1
```

三个小核导数。前两个走 `convolve2D`，`spatialGradient` 单独展开。

- **Scharr 核** `[-3,0,3;-10,0,10;-3,0,3]`（一维 `[3,10,3]` 和 `[-1,0,1]` 的外积），方向精度比 Sobel 高。核和仍是 0、不归一化，8U 输出照样饱和
- **Laplacian 的两个核**：`ksize=1` 是四邻域 `[0,1,0;1,-4,1;0,1,0]`，`ksize=3` 是八邻域 `[2,0,2;0,-8,0;2,0,2]`（= `[1,2,1]^T ⊗ [1,-2,1]`）。二阶核不能拆成两趟一维
- **`spatialGradient` 输出 `S16C1`**：16 位装得下 ±1020，**不饱和**，这就是它和 `sobel()` 的区别。输出前先 `create(H, W, S16C1)`，不能复用一个 U8 的 dst
- `spatialGradient` 只支持 U8C1（OpenCV 同样如此），多通道抛异常

测试：灰度 + 彩色 × scharr dx/dy、laplacian ksize 1/3 全部零差异；spatialGradient 和 `cv::spatialGradient` 逐元素零差异，另加一条「峰值必须 > 255」的饱和检查；`scharr(1,1)`、`laplacian(5)`、三通道 spatialGradient 都抛异常。速度：640×480 单线程 scharr 2.02 ms、laplacian 2.02 ms、spatialGradient 0.42 ms（核展开省掉了核循环，比走 `convolve2D` 快 5 倍）。

## imgproc — morph：erode / dilate

```cpp
Mat getStructuringElement(int shape, Size ksize);   // MORPH_RECT / MORPH_CROSS
Mat erode(const Mat &src, const Mat &kernel);
Mat dilate(const Mat &src, const Mat &kernel);
```

邻域取最小 / 最大，结果一定是邻域里已经存在的像素值。

- **边界策略**：越界像素直接跳过（`if (sy < 0 || sy >= rows) continue;`），不反射也不填充。等价于 erode 外面填 255、dilate 填 0。沿用 `reflect101` 会让轮廓整体缩一圈
- `erode` 初值 255，`dilate` 初值 0（8 位里的 ±∞）。
- 核里是 0 的位置要跳过，否则十字核退化成矩形核
- `getStructuringElement` 的 `create(ksize.height, ksize.width, U8C1)`：先行后列。
- `MORPH_ELLIPSE` 不是数学椭圆，是整数近似：每行算半宽 `dx = round(c·√((r²-dy²)/r²))`，填中间 `[c-dx, c+dx]`。`round` 要 half-to-even（`nearbyint`），1.732 → 2；用截断会得到瘦一圈的错误结果。`create` 的内存未初始化，椭圆分支要先 `memset` 清零

测试：RECT/CROSS × 3×3/5×5/5×3/4×4 六种核 + ELLIPSE × 3×3/5×5/7×7/5×3/4×4/1×1 六种，erode/dilate 在 U8C1/U8C3 上全部零差异；灰度图验证 erode 不变亮、dilate 不变暗；ROI 输入自洽；未知 shape 抛异常。

## imgproc — draw：line / rectangle / circle / ellipse

```cpp
void line(Mat &img, Point pt1, Point pt2, const Scalar &color, int thickness = 1);
void rectangle(Mat &img, Point pt1, Point pt2, const Scalar &color, int thickness = 1);
void circle(Mat &img, Point center, int radius, const Scalar &color, int thickness = 1);
void ellipse(Mat &img, Point center, Size axes, const Scalar &color, int thickness = 1);
```

就地往图上画点，不改变其他像素。`thickness = 1` 描边、`-1` 填充。

- `line` 用 Bresenham
- `rectangle` ：四条 `line`，四个角画两次无害
- `circle` 内部调 `ellipse`，圆就是 `axes = {r, r}` 的特例，一套光栅化算法
- `ellipse` 用 Zingl 的整数算法。**参数 `A`/`B` 是直径**（`A = |x1-x0| = 2·rx`），传半轴会画出只有一半大的椭圆
- 填充走的是另一条路：浮点扫描线，`dx = rx·√(1-(dy/ry)²)`
- 越界点直接丢弃 = 裁剪，部分在图外的线段会自动只画里面那段
- `Scalar` 是 `double`，写进 `U8` 前必须 clamp（`{300,-5,128}` → `255/0/128`）

**circle / ellipse 和 OpenCV 对不上**：OpenCV 用参数化步进，`cv::circle` 在 r=5 时第 2 行是 `###.###`（中间缺一格）。复刻它要读 `drawing.cpp` 的定点算法，投入产出比太低。测试里用弱断言：轴对称、四个轴顶点落在半径上、填充面积在 `π·rx·ry` 的 10% 内。

测试：七条线段 + 矩形描边/填充逐像素对比；圆和椭圆的对称性、顶点、面积、颜色 clamp、越界裁剪。

## imgproc — contours：findContours

```cpp
std::vector<std::vector<Point>> findContours(const Mat &binary);   // 只取外轮廓
Rect boundingRect(const std::vector<Point> &contour);              // 轴对齐外接矩形
double contourArea(const std::vector<Point> &contour);             // 鞋带公式
std::vector<Point> convexHull(std::vector<Point> pts);             // 单调链
RotatedRect minAreaRect(const std::vector<Point> &contour);        // 最小旋转外接矩形
```

- 使用 **Moore 邻域跟踪**：从每个连通域最左上的白像素起步，沿 8 邻域绕一圈
- **起点手动加进轮廓**。`contour` 从第二个点开始推的话，圆的顶点会漏掉一行，外接矩形少 1 像素
- 扫描从"来路"的顺时针下一位开始，不能从来路本身开始，否则原地打转（2026.11.21）
- `boundingRect` 的宽高是 `max - min + 1`，`+1` 是端点本身
- `contourArea` 用鞋带公式，`long long` 累加；结果可能为负，取绝对值再乘 0.5

测试：实心矩形、两个分离物体、L 形、带孔方环、单像素、三个物体+噪声，共六组。轮廓数量、每个外接矩形、面积**全部和 OpenCV 一致**（面积差 0 像素）。

### convexHull / minAreaRect

- `convexHull`  **Andrew 单调链** 算法：按 (x,y) 排序，从左到右扫出下凸包、从右到左扫出上凸包，遇到右转就弹栈。`cross <= 0` 去掉共线点，只用整数叉积、没有浮点误差
- `minAreaRect` 枚举**凸包的每一条边**：最小外接矩形必然有一条边和凸包的某条边共线。每条边方向投影求包围盒，取面积最小的。
- `RotatedRect` 的约定是**自己的**：`width >= height`，`angle` 是长边与 x 轴夹角、落在 `[0,180)`。OpenCV 的 `cv::RotatedRect` 是 `angle ∈ (0,90]` 且宽高方向相反，对比前要先归一化
- **minAreaRect 可能有多解**：三角形沿任一条边扣出的矩形面积相同，正方形和近似圆也一样。所以测试按定义验证（面积一致 + 所有点在框内），不比中心点和角度

测试：轴对齐矩形、旋转 30°、三角形、正方形、16 边形近似圆、随机 200 点。面积和 OpenCV **差 0.00**，所有点都在框内

## imgproc — edge：canny

```cpp
Mat canny(const Mat &src, double low, double high, bool L2gradient = false);
```

Sobel → 幅值 → 非极大值抑制 → 双阈值滞后，四步。src 必须是 `U8C1`。
- **边界是 `BORDER_REPLICATE`**，和 `convolve2D` 默认的 `REFLECT_101` 相反。噪声图上两种边界有 994/60000 个像素幅值不同
- 方向分档用定点：`gya = |gy| << 15`、`TG22 = 13573`（tan22.5°·2¹⁵）。`gya < |gx|·TG22` 走水平、`gya > |gx|·(TG22+65536)` 走垂直、其余走对角
- **破平方式不对称**：水平/垂直是 `m > 前一个 && m >= 后一个`，对角是两个严格 `>`。全写成 `>` 边会被劈成两半（200×300 噪声图上错 1913 像素）
- `L2gradient = true` 时幅值用 `gx²+gy²`，阈值先 `min(32767, t)` 再平方
- 多通道抛异常（OpenCV 取各通道最大梯度），`apertureSize` 只支持 3

测试：pgm 文件图 + 200×300 随机噪声 + 台阶/弱台阶图，5 组阈值 × 3 张图 + L2 + 浮点阈值，**全部零差异**；多通道和空图抛异常。速度：640×480 单线程 3.63 ms，`cv::Canny` 3.94 ms。


## 和 OpenCV 的差异

| 函数 | 差异 | 测试容差 |
|---|---|---|
| cvtColor(BGR2GRAY) | 逐字节一致 | 0 |
| resize(INTER_LINEAR) | 最大差 1（OpenCV 走两阶段 `ufixedpoint16` + SIMD） | 1 |
| threshold | 逐字节一致 | 0 |
| convolve2D | 逐字节一致（连续输入） | 0 |
| blur / sobel | 逐字节一致 | 0 |
| gaussianBlur | 逐字节一致（ksize 3~15、sigma 0~5、含 1×N） | 0 |
| medianBlur | 逐字节一致（ksize 1~15、含 1×N/1×1/2×2） | 0 |
| scharr / laplacian | 逐字节一致 | 0 |
| spatialGradient | 逐元素一致，输出 16S 不饱和 | 0 |
| canny | 逐字节一致（L1/L2、浮点阈值、小图） | 0 |
| erode / dilate | 逐字节一致 | 0 |
| line / rectangle | 逐字节一致 | 0 |
| findContours | 轮廓数量、外接矩形、面积都和 OpenCV 一致（点顺序不同） | 0 |
| convexHull | 凸包点数一致 | 0 |
| minAreaRect | 面积一致（多解时角度可能不同，属正常） | 0 |
| circle / ellipse | 形状正确，光栅化路径不同（OpenCV 用参数化步进，圆有缺格） | 不对照 |
| imread / imwrite | PGM/PPM 只支持 P5/P6、8 位；通道转 BGR | 往返零差异 |
| JPG 解码 | 同一个 libjpeg，逐字节一致 | 0 |
| JPEG 编码 | 同一 quality 下和 cv::imwrite 字节级相同 | MD5 相同 |

## 约束

- 像素类型只有 `U8C1` / `U8C3` / `F32C1`
- 图像 IO：PGM/PPM（自己解析）+ JPEG（libjpeg），不做 PNG
- 检查失败抛 `std::runtime_error`，不用 `assert`
- 4K 以上图像未测 ！（没找素材呢）

## 计划（2027）

VideoCapture
