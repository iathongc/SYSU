#include "Employee.h"

Employee::Employee(int* id, string* name, int len):size(len) {
    id_list = new int[size];
    name_list = new string[size];

    for (int i=0; i<size; i++) {
        id_list[i] = id[i];
        name_list[i] = name[i];
    }
}

Employee::Employee(const Employee& other) {
    size = other.size;
    id_list = new int[size];
    name_list = new string[size];

    for (int i=0; i<size; i++) {
        id_list[i] = other.id_list[i];
        name_list[i] = other.name_list[i];
    }
}

Employee::~Employee() {
    delete[] id_list;
    delete[] name_list;
}