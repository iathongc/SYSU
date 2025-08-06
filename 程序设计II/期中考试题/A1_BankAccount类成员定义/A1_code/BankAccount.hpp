#include <iostream>
#include <string>
using namespace std;

class BankAccount {
public:
    string get_accountNumber() {
        return accountNumber;
    }

    string get_ownerName() {
        return ownerName;
    }
    
    int get_balance() {
        return balance;
    }

    void set_accountNumber(string& n) {
        accountNumber = n;
    }

    void set_ownerName(string& n) {
        ownerName = n;
    }

    void set_balance(int b) {
        balance = b;
    }

private:
    string accountNumber;
    string ownerName;
    int balance;
};