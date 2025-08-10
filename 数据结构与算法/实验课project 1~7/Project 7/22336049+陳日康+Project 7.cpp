#include <iostream>
#include <time.h>   			//time用到的頭文件
#include <cstdlib>				//隨機數srand用到的頭文件
#include <ctype.h>  			//toascii()用到的頭文件
#include <cstring> 				//查找姓名時比較字串用的頭文件
#include <iomanip>
#define HASH_LEN 50 			//哈希表的長度
#define P 47        			//小於哈希表長度的P
#define NAME_LEN 30 			//姓名表的長度
using namespace std;

typedef struct {				//姓名表
    char *name;   				//名字的拼音
    int hashCode; 				//拼音所對應的ASCII和
} NAME;

typedef struct { 				//哈希表
    char *name; 				//名字的拼音
    int key;    				//拼音所對應的ASCII總和，即關鍵字
    int si;     				//查找長度
} HASH;

NAME Name[HASH_LEN]; 			//全域定義姓名表，最大長度為50
HASH Hash[HASH_LEN]; 			//全域定義哈希表，最大長度為50
int i, j;            			//全域定義隨機數，循環用的i、j

/**
 * 獲取姓名的 ASCII 之和
 */
int gethashCode(char *name) {
    int s = 0;
    for (j=0; *(name + j)!='\0'; j++) {
        s += toascii(*(name + j));
    }
    return s;
}

void InitName() {							//姓名表的初始化
    Name[0].name = "Cheniathong";
    Name[1].name = "Chenchuan";
    Name[2].name = "Hangbohao";
    Name[3].name = "Chenyonghui";
    Name[4].name = "Chengang";
    Name[5].name = "Zhoujielun";
    Name[6].name = "Ngcheokian";
    Name[7].name = "Huangjianqiao";
    Name[8].name = "Guizixuan";
    Name[9].name = "Chenrikang";
    Name[10].name = "Huangyushuo";
    Name[11].name = "Wangshaoquan";
    Name[12].name = "Wuyangtianlang";
    Name[13].name = "Wuzhuoxin";
    Name[14].name = "Liuxionghao";
    Name[15].name = "Zhuxun";
    Name[16].name = "Luyao";
    Name[17].name = "Eriktenhag";
    Name[18].name = "Maddison";
    Name[19].name = "Huangyongming";
    Name[20].name = "Sonheungmin";
    Name[21].name = "Fandewen";
    Name[22].name = "Yangyue";
    Name[23].name = "Zhanghongkai";
    Name[24].name = "Romero";
    Name[25].name = "Richa";
    Name[26].name = "Liangjingqian";
    Name[27].name = "Wulei";
    Name[28].name = "Lihaorong";
    Name[29].name = "Bentancur";

    for (i=0; i<NAME_LEN; i++) {					//將字串的各個字元所對應的ASCII碼相加，所得的整數做為哈希表的關鍵字
        Name[i].hashCode = gethashCode(Name[i].name);
    }
}

void CreateHash() {                                 //建立哈希表
    for (i=0; i<HASH_LEN; i++) { 					//清空哈希表，未經此操作將儲存空資料
        Hash[i].name = "\0";
        Hash[i].key = 0;
        Hash[i].si = 0;
    }
    for (i=0; i<NAME_LEN; i++) {
        int si = 1;                       			//查找長度默認為1
        int adr = (Name[i].hashCode) % P; 			//除留餘數法H(name)=name%P，除數為P=47

        if (Hash[adr].si != 0) { 					//如果衝突，使用線性探測法處理衝突
            int currAddr = adr;
            //從衝突下一個位置開始探測,到達最後一個再從第一個開始
            do {
                si++; 								// 查找長度+1
                adr = (adr + 1) % HASH_LEN;
            } while (Hash[adr].si != 0 && adr != currAddr);

            // 直到回到當前位置時還沒找到，說明哈希表已經滿了
            if (adr == currAddr) {
            	cout << "哈希表已滿" << endl;
                return;
            }
        }
        Hash[adr].key = Name[i].hashCode;
        Hash[adr].name = Name[i].name;
        Hash[adr].si = si;
    }
}

void DisplayName() {                                									//顯示姓名表
	cout << endl;
	cout << "姓名表如下：" << endl;
    for (i = 0; i < NAME_LEN; i++) {
        cout << "地址：" << left << setw(10) << i << "姓名：" << left << setw(20) << Name[i].name << "關鍵字：" << left << setw(10) << Name[i].hashCode << endl;
    }
    cout << endl;
    cout << "=======================================================" << endl;
}

void DisplayHash() { 																	// 顯示哈希表
    float asl = 0.0;
    cout << endl << endl << " 地址 \t\t 姓名 \t\t 關鍵字 \t 搜索長度" << endl; 			//顯示的格式
    for (i=0; i<HASH_LEN; i++) {
        printf("%2d %18s \t\t  %d \t\t  %d\n", i, Hash[i].name, Hash[i].key, Hash[i].si);
        asl += Hash[i].si;
    }
    asl /= NAME_LEN; 																	//求得ASL
    cout << endl << endl;
    cout << "平均查找長度：ASL(" << NAME_LEN << ")" << " = " << asl << endl;
    cout << endl;
    cout << "=======================================================" << endl;
}

// 查詢
void FindName () {
    char name[20] = {0};
    int hashCode = 0, si = 1;
	cout << endl;
    cout << "請輸入想要查找的名字: ";
    cin >> name;
    getchar();

    hashCode = gethashCode(name); 										//求出姓名的拼音所對應的ASCII作為關鍵字
    int adr = hashCode % P;       										//除留餘數法去地址
    int j = 0;

    // 如果hash位址為空，則直接認為不存在
    if (Hash[adr].key == 0) {
    	cout << endl;
        cout << "你輸入的名字不在姓名表中！" << endl;
        cout << endl;
    	cout << "=======================================================" << endl;
        return;
    }
    
    // 如果hashCode和name都相等
    if (Hash[adr].key == hashCode && 0 == strcmp(Hash[adr].name, name)) {
    	cout << endl;
    	cout << "姓名: " << left << setw(17) << Hash[adr].name << "關鍵字: " << left << setw(12) << hashCode << "地址: " << left << setw(15) << adr;
    	cout << endl << endl;
    	cout << "=======================================================" << endl;
    }

    // 如果不相等，則進行線性探測搜索
    else {
        int currAddr = adr;
        //從衝突下一個位置開始探測,到達最後一個再從第一個開始
        do {
            si++; 																			// 查找長度+1
            adr = (adr+1) % HASH_LEN;
            // 如果找到，直接break跳出循環
            if (Hash[adr].key == hashCode && 0 == strcmp(Hash[adr].name, name)) {
            	cout << endl;
            	cout << "姓名: " << left << setw(17) << Hash[adr].name << "關鍵字: " << left << setw(12) << hashCode << "地址: " << left << setw(12) << adr << "查找長度為: " << left << setw(15) << si;
            	cout << endl << endl;
        		cout << "=======================================================" << endl;
        		break;
			}
        } while (adr != currAddr);

        // 直到回到當前位置時還沒找到，說明不存在
        if (adr == currAddr) {
        	cout << endl;
            cout << "你輸入的名字不在姓名表中！" << endl;
            cout << endl;
        	cout << "=======================================================" << endl;
        }
	}
}

void view () {																		//交互介面
    cout << "=======================================================";
    cout << endl;
    cout << "              Project 7 人名哈希表                   " << endl;
	cout << "A: 列印姓名表" << endl;
	cout << "B: 列印哈希表" << endl;
	cout << "C: 查找人名" << endl;
	cout << "D: 退出程序" << endl;
    cout << "=======================================================" << endl;
}

int main() {				//主函數
    char c;
    int a = 1;
    InitName();   			//調用初始化姓名表函數
    CreateHash(); 			//調用創建哈希表函數
    view();       			//調用交互介面函數
    while (a) {
    	cout << endl;
        cout << "請輸入操作選項: ";
        cin >> c;
        getchar();
        switch (c) {		//根據選擇進行判斷，直到選擇退出時才可以退出
        case 'A':
        case 'a':
            DisplayName();
            break; 			//列印姓名表
        case 'B':
        case 'b':
            DisplayHash();
            break; 			//列印哈希表
        case 'C':
        case 'c':
            FindName();
            break; 			//調用查找函數
        case 'D':
        case 'd':
            a = 0;
            break; 			//退出循環，終止程式
        default:
        	cout << endl;
            cout << "錯誤的選項！請重新輸入正確的選項!" << endl;
            break;
        }
    }
    return 0;
}
