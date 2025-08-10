#include <iostream>
#include <cctype>
#include <map>
#include <string>
#include <algorithm>
#include <cmath>
using namespace std;

struct BinaryNode {
    char data;
    BinaryNode* left;
    BinaryNode* right;
    BinaryNode (char d) {
    	data = d;
		left = nullptr;
		right = nullptr; 
	}
};

typedef BinaryNode* BinaryTree;
map<char, int> VarValue; 																// 變數賦值

BinaryTree ReadExpr (const string& expression, int& index) {
    if (index >= expression.length ()) {
        return nullptr;
    }
    char token = expression[index++];
    if (islower (token)) {
        return new BinaryNode(token);
    }
    else if (isdigit (token) || (token == '-' && isdigit(expression[index]))) {
        return new BinaryNode(token);
    }
    else if (token == '+' || token == '-' || token == '*' || token == '/' || token == '^') {
        BinaryTree left = ReadExpr (expression, index);
        BinaryTree right = ReadExpr (expression, index);
        BinaryNode* root = new BinaryNode (token);
        root->left = left;
        root->right = right;
        return root;
    }
    else if (token == 'P') { 															// 新的操作符 P
        BinaryTree E1 = ReadExpr (expression, index);
        char P = expression [index++];
        BinaryTree E2 = ReadExpr (expression, index);
        BinaryNode* root = new BinaryNode (P);
        root->left = E1;
        root->right = E2;
        return root;
    }
    return nullptr;
}

void Assign (char V, int c) {
    VarValue[V] = c;
}

int Power (int base, int exponent) {
    if (exponent == 0) {
        return 1;
    }
    return base * Power (base, exponent - 1);
}

int Value (BinaryTree root) {
    if (root) {
        if (islower (root->data)) {
            return VarValue[root->data];
        }
        else if (isdigit (root->data)) {
            return root->data - '0';
        }
        else {
            int leftVal = Value (root->left);
            int rightVal = Value (root->right);
            char op = root->data;

            int result = 0;

            switch (op) {
                case '+':
                    result = leftVal + rightVal;
                    break;
                case '-':
                    result = leftVal - rightVal;
                    break;
                case '*':
                    result = leftVal * rightVal;
                    break;
                case '/':
                    if (rightVal != 0) {
                        result = leftVal / rightVal;
                    }
                    else {
                        cout << "除數不能為0" << endl;
                        result = -1;
                    }
                    break;
                case '^':
                    result = Power(leftVal, rightVal);
                    break;
                default:
                    cout << "未知操作符: " << op << endl;
                    result = -1;
            }
            return result;
        }
    }
    return 0;
}

// 函數用於創建一個二叉樹節點
BinaryTree CreateNode (char data) {
    return new BinaryNode(data);
}

// 函數用於輸出二叉樹的算術運算式
void PrintExpression (BinaryTree root) {
    if (root) {
        if (root->left || root->right) {
            cout << "(";
        }

        PrintExpression (root->left);
        cout << root->data;

        PrintExpression (root->right);

        if (root->left || root->right) {
            cout << ")";
        }
    }
}

BinaryTree MergeExpressions (BinaryTree expr1, BinaryTree expr2, char op) {
    BinaryNode* root = new BinaryNode (op);
    root->left = expr1;
    root->right = expr2;
    return root;
}

int main () {
	cout << "運算式類型的實現" << endl;
	cout << endl; 
    while (true) {
        cout << "選擇一個操作:" << endl;
        cout << "1. 為運算式賦值" << endl;
        cout << "2. 合併運算式" << endl;
        cout << "3. 退出程式" << endl; 
        int choice;
        cout << "你的選擇是：";
        cin >> choice;
        cout << endl;
        int index = 0;

        if (choice == 1) {
            cout << "為運算式賦值：" << endl;
            string expression;
            cout << "請輸入一個正確的前綴算術運算式: ";
            cin >> expression;

            index = 0;
            BinaryTree root = ReadExpr (expression, index);

            if (!root) {
                cout << "運算式不正確" << endl;
                return 1;
            }

            cout << "初始運算式為: ";
            PrintExpression (root);
            cout << endl;

            char var;
            int value;
            cout << endl;
            cout << "是否為變數賦值 (Y/N)? ";
            char assignOption;
            cin >> assignOption;

            if (toupper (assignOption) == 'Y') {
                while (true) {
                    cout << "請輸入變數和賦值，用空格分隔（例如：x 1）: ";
                    cin >> var >> value;
                    Assign(var, value);
                    cout << "變數 " << var << " 賦值為 " << value << endl;
                    cout << endl;
                    cout << "是否繼續為變數賦值 (Y/N)? ";
                    char continueAssign;
                    cin >> continueAssign;
                    if (toupper (continueAssign) != 'Y') {
                        break;
                    }
                }
            }

            int result = Value (root);
            cout << endl;
            cout << "運算式的值為: " << result << endl;
            cout << endl;
            cout << "--------------------------------------------------------" << endl;
            cout << endl;
        }

        else if (choice == 2) {
            cout << "合併運算式：" << endl;
            string expression1, expression2;
            cout << "輸入第一個前綴運算式: ";
            cin >> expression1;
            index = 0;
            BinaryTree root1 = ReadExpr (expression1, index);

            if (!root1) {
                cout << "無效運算式" << endl;
                return 1;
            }

            cout << "第一個前綴運算式中綴運算式: ";
            PrintExpression (root1);
            cout << endl;
            cout << endl;
            cout << "輸入第二個前綴運算式: ";
            cin >> expression2;
            index = 0;
            BinaryTree root2 = ReadExpr (expression2, index);

            if (!root2) {
                cout << "無效運算式" << endl;
                return 1;
            }

            cout << "第二個前綴運算式中綴運算式: ";
            PrintExpression (root2);
            cout << endl;
            cout << endl;
            char mergeOp;
            cout << "輸入合併操作符 (+, -, *, /, ^): ";
            cin >> mergeOp;

            BinaryTree mergedExpr = MergeExpressions (root1, root2, mergeOp);
            cout << endl;
            cout << "合併後的中綴運算式: ";
            PrintExpression (mergedExpr);
            cout << endl;
            cout << endl;
            cout << "--------------------------------------------------------" << endl;
            cout << endl;
        }
        
		else if (choice == 3) {
			cout << "已退出程序" << endl;
			break;
		}
		
		else {
			cout << "無效的選擇" << endl;
			cout << endl;
			cout << "--------------------------------------------------------" << endl;
			cout << endl;
		}
    }
    return 0;
}
