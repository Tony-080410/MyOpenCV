#include "mycv/core/check.hpp"
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <ios>
#include <istream>
#include <mycv/core/mat.hpp>
#include <mycv/core/types.hpp>
#include <mycv/io/image.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mycv {
namespace {
// 跳过空白和#注释，读进一个 十进制数
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

// maxval后面一个空白字符，CRLF两个空白字符，'\r'之后补一个'\n'
void eatSeprator(std::istream &in) {
    const int sep = in.get();
    if (sep == '\r' && in.peek() == '\n') {
        in.get();
    }
}

} // namespace

Mat imread(const std::string &path) {
    std::ifstream in(path, std::ios::binary);

    if (!in) throw std::runtime_error("imreed: 打不开" + path);
    if (in.get() != 'P') throw std::runtime_error("imread: 不是PGM/PPM " + path);

    const int kind = in.get();

    if (kind != '5' && kind != '6') throw std::runtime_error("imread: 不是P5/P6 " + path);

    const int w = readInt(in);
    const int h = readInt(in);
    const int maxv = readInt(in);

    if (w <= 0 || h <= 0) throw std::runtime_error("imread: 非法尺寸 " + path);

    if (maxv != 255) throw std::runtime_error("imread: 只支持MAXVAL=255 " + path);

    eatSeprator(in);

    const int chans = (kind == '6') ? 3 : 1;

    Mat m;
    m.create(h, w, chans == 3 ? U8C3 : U8C1);
    const size_t want = static_cast<size_t>(m.rows) * m.step;
    in.read(reinterpret_cast<char *>(m.data), static_cast<std::streamsize>(want));

    if (!in || static_cast<std::size_t>(in.gcount()) != want)
        throw std::runtime_error("imread: 像素数据不足 " + path);

    // PPM是RGB，转换成BGR
    if (chans == 3) {
        uint8_t *p = m.data;
        const size_t n = static_cast<size_t>(h) * static_cast<size_t>(w);
        for (size_t i = 0; i < n; ++i, p += 3) {
            std::swap(p[0], p[2]);
        }
    }

    return m;
}

void imwrite(const std::string &path, const Mat &m) {
    MYCV_ASSERT(!m.empty());
    MYCV_ASSERT(depthOf(m.type) == DEPTH_8U); // 只写8位

    const int chans = channels(m.type);

    MYCV_ASSERT(chans == 1 || chans == 3);

    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("imwrite: 打不开 " + path);

    out << (chans == 1 ? "P5\n" : "P6\n") << m.cols << " " << m.rows << "\n255\n";

    const size_t rowBytes = static_cast<size_t>(m.cols) * static_cast<size_t>(chans);
    if (chans == 1 && m.isContinuous()) {
        // 如果是单通道且连续，那么直接整块读取
        out.write(reinterpret_cast<const char *>(m.data),
                  static_cast<std::streamsize>(static_cast<size_t>(m.rows) * rowBytes));
    } else {
        std::vector<uint8_t> row(rowBytes);
        for (int r = 0; r < m.rows; ++r) {
            const uint8_t *src = m.ptr(r);
            if (chans == 1) {
                out.write(reinterpret_cast<const char *>(src),
                          static_cast<std::streamsize>(rowBytes));
            } else {
                for (size_t i = 0; i < rowBytes; i += 3) {
                    // BGR转换成RGB
                    row[i] = src[i + 2];
                    row[i + 1] = src[i + 1];
                    row[i + 2] = src[i];
                }
                out.write(reinterpret_cast<const char *>(row.data()),
                          static_cast<std::streamsize>(rowBytes));
            }
        }
    }
    if (!out) {
        throw std::runtime_error("imwrite: 写入失败" + path);
    }
}

} // namespace mycv