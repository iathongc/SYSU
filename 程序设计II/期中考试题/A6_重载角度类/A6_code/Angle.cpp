#include "Angle.hpp"

Angle operator+(const Angle& a, const Angle& b) {
    int n_degree = a.degree+b.degree;
    int n_minute = a.minute+b.minute;
    int n_second = a.second+b.second;

    if (n_second > 60) {
        n_minute ++;
        n_second -= 60;
    }
    if (n_minute > 60) {
        n_degree ++;
        n_minute -= 60;
    }

    return Angle(n_degree, n_minute, n_second);
}

Angle operator-(const Angle& a, const Angle& b) {
    // 先将两个角度转换为总秒数进行计算
    int total_a = a.degree * 3600 + a.minute * 60 + a.second;
    int total_b = b.degree * 3600 + b.minute * 60 + b.second;
    int total_diff = total_a - total_b;

    // 处理结果为负的情况
    bool is_negative = false;
    if (total_diff < 0) {
        is_negative = true;
        total_diff = -total_diff;
    }

    // 将总秒数转换回度分秒
    int n_degree = total_diff / 3600;
    int remaining = total_diff % 3600;
    int n_minute = remaining / 60;
    int n_second = remaining % 60;

    // 如果结果是负的，所有分量都设为负
    if (is_negative) {
        n_degree = -n_degree;
        n_minute = -n_minute;
        n_second = -n_second;
    }

    return Angle(n_degree, n_minute, n_second);
}

istream& operator>>(istream& is, Angle& ang) {
    is >> ang.degree >> ang.minute >> ang.second;
    return is;
}

ostream& operator<<(ostream& os, const Angle& ang) {
    os << ang.degree << "°" << ang.minute << "′" << ang.second << "′′";
    return os;
}

bool Angle::operator>(const Angle& ang) {
    if (degree > ang.degree)
        return 1;
    else if (degree < ang.degree)
        return 0;
    else {
        if (minute > ang.minute)
            return 1;
        else if (minute < ang.minute)
            return 0;
        else {
            if (second > ang.second)
                return 1;
            else if (second < ang.second)
                return 0;
        }
    }
}

bool Angle::operator<(const Angle& ang) {
    if (degree < ang.degree)
        return 1;
    else if (degree > ang.degree)
        return 0;
    else {
        if (minute < ang.minute)
            return 1;
        else if (minute > ang.minute)
            return 0;
        else {
            if (second < ang.second)
                return 1;
            else if (second > ang.second)
                return 0;
        }
    }
}    

bool Angle::operator==(const Angle& ang) {
    return (degree == ang.degree && minute == ang.minute && second == ang.second);
}