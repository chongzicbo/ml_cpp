#ifndef POINT_H
#define POINT_H

struct Point
{
    double x; // 特征值
    double y; // 标签值

    // 添加 swap 方法
    void swap(Point &other) noexcept
    {
        std::swap(x, other.x);
        std::swap(y, other.y);
    }
};

// 全局 swap 函数，用于支持 std::swap
inline void swap(Point &a, Point &b) noexcept
{
    a.swap(b);
}

#endif // POINT_H