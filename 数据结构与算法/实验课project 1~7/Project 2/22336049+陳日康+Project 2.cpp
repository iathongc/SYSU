#include <iostream>
#include <cstring>
#include <algorithm>
#include <stack>
#include <unordered_map>
using namespace std;
stack<int> num;
stack<char> op;

void eval () {
    auto b = num.top();
    num.pop();
    auto a = num.top();
    num.pop();
    auto c = op.top();
    op.pop();
    int x;
    if (c == '+') {
		x = a+b;
	}
    else if (c == '-') {
		x = a-b;
	}
    else if (c == '*') {
		x = a*b;
	}
	else if (c == '^') {
		x = a*a;
	}
	else if (c == '#') {
		x = b;																				//右結合，取b的值
	}
    else {
		x = a/b;
	}
    num.push(x);
}

int main () {
    unordered_map<char, int> pr{ {'+', 1}, {'-', 1}, {'*', 2}, {'/', 2}, {'#', 0} };
    int choice;
    do {
        cout << "\t       算法運算式求值演算" << endl;
        cout << "\t           多項式計算" << endl;
        cout << "-------------------------------------------------------------" <<endl;
        cout << "\t          請選擇操作:" << endl;
        cout << "\t        1. 計算運算式" << endl;
        cout << "\t           0. 退出" << endl;
        cout << "-------------------------------------------------------------" << endl;
        cout << "請輸入你的選擇: ";
        cin >> choice;
        cout << endl;
        cin.ignore();  																		// 清除輸入緩衝區
        switch (choice) {
            case 1: {
                string str;
                cout << "請輸入運算式（不含空格）: ";
                cin >> str;
                // 計算運算式的代碼
 				for (int i=0; i<str.size(); i++) {
        			auto c = str[i];
        			if (isdigit(c)) {
            			int x=0, j=i;
            			while (j<str.size() && isdigit(str[j]))
                			x = x*10 + str[j++]-'0';
            				i = j-1;
            				num.push(x);													//將解析的運算元壓入 num 棧
        				}
        			else if (c == '(')														//左括弧直接入棧
						op.push(c);													
       				else if (c == ')') {													//處理右括弧，彈出操作符並計算
            			while (op.top() != '(')	{
							eval ();
						}														
            			op.pop();															//彈出左括弧
        			}
        			else {
            			while (op.size() && op.top() != '(' && pr[op.top()] >= pr[c]) {
							eval ();														//處理操作符，根據優先順序彈出運算元和操作符並計算
						}														
            			op.push(c);															//將當前操作符入棧
        			}
    			}
    			while (op.size()) {
					eval ();																//處理剩餘的操作符和運算元																
				}
                cout << "計算結果為: " << num.top() << endl;			
				cout << endl << endl << endl;
                break;
            }
            case 0:
                cout << "已退出" << endl;
                cout << "--------------------------------" << endl;
                break;
            default:
                cout << "無效選擇，請重新選擇" << endl;
                break;
        }
    }
	while (choice != 0);
    return 0;
}
