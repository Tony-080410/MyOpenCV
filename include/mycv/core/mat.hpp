#pragma once

#include "mycv/core/types.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
namespace mycv {
struct Mat {
    int rows = 0;
    int cols = 0;
    int type = 0; // U8C1 / U8C3 / F32C1

    size_t step = 0; // 每一行字节数
    uint8_t *data = nullptr; // 指向Mat的第一个像素
    std::shared_ptr<uint8_t> owner; // buffer所有权

    bool empty() const { return data == nullptr || rows <= 0 || cols <= 0; }
    bool isContinuous() const {
        return step == static_cast<size_t>(cols) * elemSize(type);
    } // 判断是否连续：连续内存能够当作一维数组使用，卷积时可以直接跑指针，不用操心行尾

    // 模板访问器
    uint8_t *ptr(int r);
    const uint8_t *ptr(int r) const;

    template <class T> T *ptr(int r) { return reinterpret_cast<T *>(ptr(r)); }

    template <class T> const T *ptr(int r) const { return reinterpret_cast<const T *>(ptr(r)); }

    template <class T> T &at(int r, int c) { return ptr<T>(r)[c]; }

    template <class T> const T &at(int r, int c) const { return ptr<T>(r)[c]; }

    void create(int rows, int cols, int type); // 分配 rows×cols，step = cols*elemSize
    Mat clone() const; // 深拷贝，新 buffer，连续
    Mat roi(const Rect &r) const; // 共享 owner，data 偏移，同 step
    void release(); // 清空，owner 计数减一
};

} // namespace mycv