#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <istream>
#include <mycv/core/check.hpp>
#include <mycv/core/types.hpp>
#include <mycv/io/image.hpp>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef MYCV_HAVE_JPEG
#include <csetjmp>
#include <jpeglib.h>
#endif

namespace mycv {
namespace {

// ---------- PGM / PPM ----------

// 跳过空白和 # 注释，读一个十进制整数。数字后的分隔符还回流出
int readInt(std::istream &in) {
    int ch;
    do {
        ch = in.get();
        if (ch == EOF) throw std::runtime_error("P5/P6: 头部意外结束");
        if (ch == '#') {
            while (ch != EOF && ch != '\n')
                ch = in.get();
        }
    } while (std::isspace(ch));

    int v = 0;
    bool any = false;
    while (ch >= '0' && ch <= '9') {
        v = v * 10 + (ch - '0');
        any = true;
        ch = in.get();
    }
    if (!any) throw std::runtime_error("P5/P6: 头部数字非法");
    if (ch != EOF) in.putback(static_cast<char>(ch));
    return v;
}

// maxval 后面恰好一个空白。CRLF 是两个，'\r' 之后补一个 '\n'
void eatSeparator(std::istream &in) {
    const int sep = in.get();
    if (sep == '\r' && in.peek() == '\n') in.get();
}

Mat readPnm(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("imread: 打不开 " + path);

    if (in.get() != 'P') throw std::runtime_error("imread: 不是 PGM/PPM " + path);
    const int kind = in.get();
    if (kind != '5' && kind != '6') throw std::runtime_error("imread: 魔数不是 P5/P6 " + path);

    const int w = readInt(in);
    const int h = readInt(in);
    const int maxv = readInt(in);
    if (w <= 0 || h <= 0) throw std::runtime_error("imread: 尺寸非法 " + path);
    if (maxv != 255) throw std::runtime_error("imread: 只支持 MAXVAL=255 " + path);

    eatSeparator(in);

    const int chans = (kind == '6') ? 3 : 1;
    Mat m;
    m.create(h, w, chans == 3 ? U8C3 : U8C1);

    const std::size_t want = static_cast<std::size_t>(m.rows) * m.step;
    in.read(reinterpret_cast<char *>(m.data), static_cast<std::streamsize>(want));
    if (!in || static_cast<std::size_t>(in.gcount()) != want)
        throw std::runtime_error("imread: 像素数据不足 " + path);

    // PPM 存 RGB，转成 BGR 和 OpenCV 一致
    if (chans == 3) {
        uint8_t *p = m.data;
        const std::size_t n = static_cast<std::size_t>(h) * static_cast<std::size_t>(w);
        for (std::size_t i = 0; i < n; ++i, p += 3)
            std::swap(p[0], p[2]);
    }
    return m;
}

void writePnm(const std::string &path, const Mat &m) {
    const int chans = channels(m.type);

    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("imwrite: 打不开 " + path);

    out << (chans == 1 ? "P5\n" : "P6\n") << m.cols << " " << m.rows << "\n255\n";

    const std::size_t rowBytes = static_cast<std::size_t>(m.cols) * static_cast<std::size_t>(chans);
    if (chans == 1 && m.isContinuous()) {
        out.write(reinterpret_cast<const char *>(m.data),
                  static_cast<std::streamsize>(static_cast<std::size_t>(m.rows) * rowBytes));
    } else {
        std::vector<uint8_t> row(rowBytes);
        for (int r = 0; r < m.rows; ++r) {
            const uint8_t *src = m.ptr(r);
            if (chans == 1) {
                out.write(reinterpret_cast<const char *>(src),
                          static_cast<std::streamsize>(rowBytes));
            } else {
                for (std::size_t i = 0; i < rowBytes; i += 3) { // BGR -> RGB
                    row[i] = src[i + 2];
                    row[i + 1] = src[i + 1];
                    row[i + 2] = src[i];
                }
                out.write(reinterpret_cast<const char *>(row.data()),
                          static_cast<std::streamsize>(rowBytes));
            }
        }
    }
    if (!out) throw std::runtime_error("imwrite: 写入失败 " + path);
}

// ---------- JPEG ----------

#ifdef MYCV_HAVE_JPEG

// libjpeg 的错误回调必须 longjmp，因为它没有异常接口
struct JpegErrorMgr {
    jpeg_error_mgr pub;
    std::jmp_buf jump;
};

void jpegErrorExit(j_common_ptr cinfo) {
    auto *err = reinterpret_cast<JpegErrorMgr *>(cinfo->err);
    std::longjmp(err->jump, 1);
}

Mat readJpeg(const std::string &path) {
    std::FILE *fp = std::fopen(path.c_str(), "rb");
    if (!fp) throw std::runtime_error("imread: 打不开 " + path);

    jpeg_decompress_struct cinfo{};
    JpegErrorMgr jerr{};
    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = jpegErrorExit;

    if (setjmp(jerr.jump)) {
        jpeg_destroy_decompress(&cinfo);
        std::fclose(fp);
        throw std::runtime_error("imread: JPEG 解码失败 " + path);
    }

    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, fp);
    jpeg_read_header(&cinfo, TRUE);

    // 灰度 JPEG 保持单通道，彩色输出 RGB（后面转 BGR）
    const bool gray = (cinfo.num_components == 1);
    cinfo.out_color_space = gray ? JCS_GRAYSCALE : JCS_RGB;
    jpeg_start_decompress(&cinfo);

    const int w = static_cast<int>(cinfo.output_width);
    const int h = static_cast<int>(cinfo.output_height);
    const int cn = gray ? 1 : 3;

    Mat m;
    m.create(h, w, gray ? U8C1 : U8C3);

    std::vector<uint8_t> row(static_cast<std::size_t>(w) * cn);
    while (cinfo.output_scanline < cinfo.output_height) {
        const JDIMENSION y = cinfo.output_scanline;
        JSAMPROW rp = row.data();
        jpeg_read_scanlines(&cinfo, &rp, 1);
        uint8_t *d = m.ptr(static_cast<int>(y));
        if (gray) {
            std::memcpy(d, row.data(), static_cast<std::size_t>(w));
        } else {
            for (int x = 0; x < w; ++x) { // RGB -> BGR
                d[x * 3 + 0] = row[x * 3 + 2];
                d[x * 3 + 1] = row[x * 3 + 1];
                d[x * 3 + 2] = row[x * 3 + 0];
            }
        }
    }

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    std::fclose(fp);
    return m;
}

void writeJpeg(const std::string &path, const Mat &m, int quality) {
    if (quality < 1) quality = 1;
    if (quality > 100) quality = 100;

    std::FILE *fp = std::fopen(path.c_str(), "wb");
    if (!fp) throw std::runtime_error("imwrite: 打不开 " + path);

    jpeg_compress_struct cinfo{};
    JpegErrorMgr jerr{};
    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = jpegErrorExit;

    if (setjmp(jerr.jump)) {
        jpeg_destroy_compress(&cinfo);
        std::fclose(fp);
        throw std::runtime_error("imwrite: JPEG 编码失败 " + path);
    }

    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, fp);

    const int cn = channels(m.type);
    cinfo.image_width = static_cast<JDIMENSION>(m.cols);
    cinfo.image_height = static_cast<JDIMENSION>(m.rows);
    cinfo.input_components = cn;
    cinfo.in_color_space = (cn == 1) ? JCS_GRAYSCALE : JCS_RGB;

    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, quality, TRUE); // TRUE = 强制基线量化表，和 imwrite 一致
    jpeg_start_compress(&cinfo, TRUE);

    const std::size_t rowBytes = static_cast<std::size_t>(m.cols) * cn;
    std::vector<uint8_t> row(rowBytes);
    while (cinfo.next_scanline < cinfo.image_height) {
        const uint8_t *s = m.ptr(static_cast<int>(cinfo.next_scanline));
        if (cn == 1) {
            std::memcpy(row.data(), s, rowBytes);
        } else {
            for (int x = 0; x < m.cols; ++x) { // BGR -> RGB
                row[x * 3 + 0] = s[x * 3 + 2];
                row[x * 3 + 1] = s[x * 3 + 1];
                row[x * 3 + 2] = s[x * 3 + 0];
            }
        }
        JSAMPROW rp = row.data();
        jpeg_write_scanlines(&cinfo, &rp, 1);
    }

    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    std::fclose(fp);
}

#endif // MYCV_HAVE_JPEG

// 扩展名比较
bool hasExt(const std::string &path, const char *ext) {
    const std::size_t n = std::strlen(ext);
    if (path.size() <= n) return false;
    for (std::size_t i = 0; i < n; ++i) {
        const char a =
            static_cast<char>(std::tolower(static_cast<unsigned char>(path[path.size() - n + i])));
        if (a != ext[i]) return false;
    }
    return true;
}

} // namespace

Mat imread(const std::string &path) {
    // 先看前两个字节定格式
    std::ifstream probe(path, std::ios::binary);
    if (!probe) throw std::runtime_error("imread: 打不开 " + path);

    unsigned char magic[2] = {0, 0};
    probe.read(reinterpret_cast<char *>(magic), 2);
    const std::streamsize got = probe.gcount();
    probe.close();

    if (got == 2 && magic[0] == 0xFF && magic[1] == 0xD8) {
#ifdef MYCV_HAVE_JPEG
        return readJpeg(path);
#else
        throw std::runtime_error("imread: 编译时没开 JPEG" + path);
#endif
    }
    return readPnm(path);
}

void imwrite(const std::string &path, const Mat &m, int jpegQuality) {
    MYCV_ASSERT(!m.empty());
    MYCV_ASSERT(depthOf(m.type) == DEPTH_8U); // 只写 8 位

    const int chans = channels(m.type);
    MYCV_ASSERT(chans == 1 || chans == 3);

    if (hasExt(path, ".jpg") || hasExt(path, ".jpeg")) {
#ifdef MYCV_HAVE_JPEG
        writeJpeg(path, m, jpegQuality);
        return;
#else
        throw std::runtime_error("imwrite: 编译时没开 JPEG" + path);
#endif
    }
    writePnm(path, m);
}

} // namespace mycv
