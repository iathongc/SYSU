#include "Rectangle.h"

Rectangle::Rectangle(float length, float width, int len) {
    size = len;
    attribution_list = new float[size];
    for (int i=0; i<size; i++) {
        attribution_list[0] = length;
        attribution_list[1] = width;
        attribution_list[2] = length*width;
        attribution_list[3] = (length+width)*2;
    }
}

Rectangle::~Rectangle() {
    delete[] attribution_list;
}