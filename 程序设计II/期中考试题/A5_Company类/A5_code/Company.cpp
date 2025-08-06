#include "Company.hpp"
#include <iostream>
using namespace std;

int Company::numberOfCompany = 0;

Company::Company(int r, int v):realId(r), virtualId(v) {
    numberOfCompany++;
    cout << "Congratulations on the registration of Company " << realId << endl;
}

Company::~Company() {
    numberOfCompany--;
}

int Company::getRealId() const {
    return realId;
}

int Company::getId() {
    return virtualId;
}

void Company::modifyId(int newVirtualId) {
    virtualId = newVirtualId;
}

int Company::getNumberOfCompany() {
    return numberOfCompany;
}