#include "cylinder.hpp"

double Cylinder::get_radius() {
    return radius;
}

double Cylinder::get_height() {
    return height;
}

void Cylinder::set_radius(double r) {
    radius = r;
}

void Cylinder::set_height(double h){
    height = h;
}

double Cylinder::get_area() {
    return PI*radius*radius;
}

double Cylinder::get_volume() {
    return get_area()*height;
}