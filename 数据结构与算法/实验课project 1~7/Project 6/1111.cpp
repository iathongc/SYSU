#include <iostream>
#include <stack>
#include <queue>
#include <cstring>
using namespace std;

struct Edge {
    int chu_node;
    Edge* ru_edge;
    Edge* chu_edge;
};

struct Node {
    char location[100];
    Edge* first_edge;
};

// 添加 DFS 函數
void performDFS(Node* site, int n, int origin) {
    stack<int> location;
    queue<int> vex1;
    queue<int> vex2;
    int vist[1000] = {0};
    int count = 0;

    cout << "DFS Traversal Order:" << endl;
    cout << site[origin].location << endl;
    location.push(origin);
    count++;
    vist[origin] = 1;
    Edge* temp1;
    int temp2 = origin;

    while (!location.empty()) {
        int flag = 0;
        temp1 = site[temp2].first_edge;

        while (temp1) {
            if (!vist[temp1->chu_node] || !vist[temp1->ru_edge->chu_node]) {
                location.push(temp2);
                vex1.push(temp2);

                if (temp2 == temp1->ru_edge->chu_node)
                    temp2 = temp1->chu_node;
                else
                    temp2 = temp1->ru_edge->chu_node;

                vex2.push(temp2);
                flag = 1;
                vist[temp2] = 1;
                cout << site[temp2].location << endl;
                count++;
                break;
            }

            if (temp1->ru_edge->chu_node == temp2)
                temp1 = temp1->ru_edge;
            else if (temp1->chu_node == temp2)
                temp1 = temp1->chu_edge;
        }

        if (!flag) {
            temp2 = location.top();
            location.pop();
        }
    }

    cout << "DFS Edge Set:" << endl;
    while (!vex1.empty()) {
        cout << site[vex1.front()].location << ' ' << site[vex2.front()].location << endl;
        vex1.pop();
        vex2.pop();
    }
}

// 添加 BFS 函數
void performBFS(Node* site, int n, int origin) {
    queue<int> bfs;
    int vist1[1000] = {0};
    vist1[origin] = 1;

    cout << "BFS Traversal Order:" << endl;
    bfs.push(origin);

    while (!bfs.empty()) {
        int temp2 = bfs.front();
        bfs.pop();
        cout << site[temp2].location << endl;
        Edge* temp1 = site[temp2].first_edge;

        while (temp1) {
            if (!vist1[temp1->chu_node] || !vist1[temp1->ru_edge->chu_node]) {
                if (temp1->chu_edge->chu_node == temp2) {
                    bfs.push(temp1->ru_edge->chu_node);
                    vist1[temp1->ru_edge->chu_node] = 1;
                } else {
                    bfs.push(temp1->chu_edge->chu_node);
                    vist1[temp1->chu_edge->chu_node] = 1;
                }
            }

            temp1 = temp1->chu_edge;
        }
    }

    cout << "BFS Edge Set:" << endl;
    // 輸出 BFS 生成樹的邊集
    for (int i = 0; i < n; i++) {
        Edge* temp1 = site[i].first_edge;
        while (temp1) {
            if (vist1[temp1->chu_node] && vist1[temp1->ru_edge->chu_node]) {
                cout << site[temp1->chu_node].location << ' ' << site[temp1->ru_edge->chu_node].location << endl;
            }
            temp1 = temp1->chu_edge;
        }
    }
}

// 非遞迴 DFS
void iterativeDFS(Node* site, int n, int origin) {
    stack<int> dfs_stack;
    int vist[1000] = {0};

    cout << "Non-recursive DFS Traversal Order:" << endl;
    dfs_stack.push(origin);
    vist[origin] = 1;

    while (!dfs_stack.empty()) {
        int temp2 = dfs_stack.top();
        dfs_stack.pop();
        cout << site[temp2].location << endl;

        Edge* temp1 = site[temp2].first_edge;
        while (temp1) {
            if (!vist[temp1->chu_node] || !vist[temp1->ru_edge->chu_node]) {
                int next_node = (temp1->chu_edge->chu_node == temp2) ? temp1->ru_edge->chu_node : temp1->chu_edge->chu_node;
                dfs_stack.push(next_node);
                vist[next_node] = 1;
            }
            temp1 = temp1->chu_edge;
        }
    }
}

int main() {
    Node site[1000];
    int n;
    cout << "Please input the number of locations:" << endl;
    cin >> n;
    cout << "Please input the locations:" << endl;
    for (int i = 0; i < n; i++) {
        cin >> site[i].location;
        site[i].first_edge = nullptr;
    }

    int m;
    cout << "Please input the edges (enter 0 0 to end):" << endl;
    Edge* s;

    while (true) {
        s = new Edge;
        cin >> s->chu_node;
        cin >> s->ru_edge->chu_node;

        if (s->chu_node == 0 && s->ru_edge->chu_node == 0) {
            break;
        }

        s->chu_edge = site[s->chu_node].first_edge;
        s->ru_edge = site[s->ru_edge->chu_node].first_edge;
        site[s->chu_node].first_edge = s;
        site[s->ru_edge->chu_node].first_edge = s;
    }

    while (true) {
        cout << "Choose traversal type (1 for DFS, 2 for BFS, 3 for non-recursive DFS, 0 to exit):" << endl;
        int traversal_type;
        cin >> traversal_type;

        if (traversal_type == 1) {
            cout << "Please input the origin:" << endl;
            char a[20];
            cin >> a;

            int origin;
            for (int i = 0; i < n; i++) {
                if (!strcmp(a, site[i].location)) {
                    origin = i;
                    break;
                }
            }

            performDFS(site, n, origin);
        } else if (traversal_type == 2) {
            cout << "Please input the origin:" << endl;
            char a[20];
            cin >> a;

            int origin;
            for (int i = 0; i < n; i++) {
                if (!strcmp(a, site[i].location)) {
                    origin = i;
                    break;
                }
            }

            performBFS(site, n, origin);
        } else if (traversal_type == 3) {
            cout << "Please input the origin:" << endl;
            char a[20];
            cin >> a;

            int origin;
            for (int i = 0; i < n; i++) {
                if (!strcmp(a, site[i].location)) {
                    origin = i;
                    break;
                }
            }

            iterativeDFS(site, n, origin);
        } else if (traversal_type == 0) {
            break;
        } else {
            cout << "Invalid traversal type." << endl;
        }
    }

    return 0;
}
