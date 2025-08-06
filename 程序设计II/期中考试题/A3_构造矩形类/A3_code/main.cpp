#include "Rectangle.h"

int main()
{
    float length, width;
    cin >> length >> width;

    Rectangle rectangle(length,width,4);
    rectangle.RectanglePrint();
    return 0;
}