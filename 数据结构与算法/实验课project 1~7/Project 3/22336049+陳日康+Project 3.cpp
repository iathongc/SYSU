#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <iomanip>
using namespace std;

vector<int> computeLPS(const string& pattern) {
    int len = 0;
    vector<int> lps(pattern.size(), 0);
    int i = 1;
    while (i < pattern.size()) {
        if (pattern[i] == pattern[len]) {
            len++;
            lps[i] = len;
            i++;
        }
		else {
            if (len != 0) {
                len = lps[len - 1];
            }
			else {
                lps[i] = 0;
                i++;
            }
        }
    }
    return lps;
}

bool isWordCharacter(char c) {
    return isalnum(c) || c == '_';
}

void kmpSearch(const string& text, const string& pattern, map<int, int>& occurrences, int& totalOccurrences, int lineNumber) {
    vector<int> lps = computeLPS(pattern);
    int i = 0;
    int j = 0;

    while (i < text.size()) {
        if (isWordCharacter(text[i]) && !isWordCharacter(text[i - 1])) {
            j = 0;
        }
        
        if (pattern[j] == text[i]) {
            i++;
            j++;
        }

        if (j == pattern.size()) {
            if ((i == text.size() || !isWordCharacter(text[i])) && (i - j == 0 || !isWordCharacter(text[i - j - 1]))) {
                occurrences[lineNumber]++;
                totalOccurrences++;
            }
            j = lps[j - 1];
        }
		else if (i < text.size() && pattern[j] != text[i]) {
            if (j != 0) {
                j = lps[j - 1];
            }
			else {
                i++;
            }
        }
    }
}

int main() {
    string filename;
    cout << "請輸入檔案名稱：";
    cin >> filename;

    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "無法讀入檔案" << endl;
        return 1;
    }
    cout << "成功讀入檔案！" << endl;
    cout << endl;
    string pattern;
    cout << "請輸入你需要查找的字符：";
    cin.ignore();  // 清除輸入緩衝區
    getline(cin, pattern);
    cout << endl;

    map<int, int> occurrences;
    int totalOccurrences = 0;
    string line;
    int lineNumber = 1;
    while (getline(file, line)) {
        kmpSearch(line, pattern, occurrences, totalOccurrences, lineNumber);
        lineNumber++;
    }

    file.close();

    cout << "搜尋結果如下：";
    cout << endl;
    for (const auto& entry : occurrences) {
        cout << "行號：" << setw(2) << entry.first << "	" << "出現次數：" << entry.second << endl;
    }
	cout << endl;
    cout << "總出現次數：" << totalOccurrences << endl;
    return 0;
}
