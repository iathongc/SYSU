import copy
import time
import heapq
import os
import psutil

TARGET_STATE = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0]
MOVEMENT_OFFSETS = [[0, 1], [0, -1], [1, 0], [-1, 0]]
VISITED_STATES = set()
flag = -1

class PuzzleNode:
    def __init__(self, f_score, state, path, g_score):
        self.f_score = f_score    #A*算法中的估价函数，表示从初始状态经过n到达目标状态的估计最小成本
        self.state = state        #表示拼图的当前状态，一个长度为 16 的列表
        self.path = path
        self.g_score = g_score    #表示从初始状态到当前状态的实际成本

    def __lt__(self, other_node):
        # 定义节点之间的比较，优先队列需要
        # 如果 f_score 相等，则比较 g_score；f_score小的或 g_score大的节点被视为较小
        if self.f_score == other_node.f_score:
            return self.g_score > other_node.g_score
        return self.f_score < other_node.f_score

def manhattan_distance(state):         #计算曼哈顿距离
    distance = 0
    for i in range(0, 16):
        if state[i] != 0 and state[i] != TARGET_STATE[i]:
            x = (state[i] - 1) // 4
            y = state[i] - 4 * x - 1
            distance += abs(x - (i // 4)) + abs(y - (i % 4))
    return distance

def linear_conflict(state):
    board = [state[0:4], state[4:8], state[8:12], state[12:16]]
    n = len(board)
    conflicts = 0
    #计算每一行的线性冲突
    for i in range(n):
        for j in range(n):
            tile1 = board[i][j]
            if tile1 == 0:
                continue
            goal_row1, goal_col1 = (tile1 - 1) // n, (tile1 - 1) % n
            if goal_row1 == i:             #需要在同一行内进一步检查
                for k in range(j + 1, n):
                    tile2 = board[i][k]
                    if tile2 == 0:
                        continue
                    goal_row2, goal_col2 = (tile2 - 1) // n, (tile2 - 1) % n
                    if goal_row2 == i and goal_col2 < goal_col1:
                        conflicts += 1
    #计算每一列的线性冲突
    for j in range(n):
        for i in range(n):
            tile1 = board[i][j]
            if tile1 == 0:
                continue
            goal_row1, goal_col1 = (tile1 - 1) // n, (tile1 - 1) % n
            if goal_col1 == j:             #需要在同一行内进一步检查
                for k in range(i + 1, n):
                    tile2 = board[k][j]
                    if tile2 == 0:
                        continue
                    goal_row2, goal_col2 = (tile2 - 1) // n, (tile2 - 1) % n
                    if goal_col2 == j and goal_row2 < goal_row1:
                        conflicts += 1
    return manhattan_distance(state) + conflicts

def heuristic_cost(state):
    return linear_conflict(state)

#生成子节点
def generate_moves(state, path, history,initial_state):
    for i in range(0, 16):
        if state[i] == 0:
            index = i
            break
    possible_moves = []
    x = index // 4
    y = index % 4
    for i in range(0, 4):
        fx = MOVEMENT_OFFSETS[i][0]
        fy = MOVEMENT_OFFSETS[i][1]
        new_index = index + fx * 4 + fy
        if x + fx > -1 and x + fx < 4 and y + fy > -1 and y + fy < 4:
            new_state = copy.deepcopy(state)
            new_state[index] = new_state[new_index]
            new_state[new_index] = 0
            possible_moves.append([new_state, str(new_state[index])])
            if new_state == TARGET_STATE:
                global flag
                flag = 1
                lt = path.split()
                path_count = len(lt) + 1
                #print(type(initial_state[0]))
                print()
                print("Path:", path + str(new_state[index]) + " ")
                print()
                print("Path count:", path_count)
                print()
                print("Total number of visited nodes:", history)
                print_format(initial_state,lt)
                return possible_moves  # 在找到目标状态后立即返回
    return possible_moves

def print_format(state,lt_str):
    zero_pos = 0
    move_pos = 1
    lt = [int(item) for item in lt_str]
    #print(type(lt[6]))
    #print(type(state[6]))
    for i in range(len(lt)):
        print()
        print(f"Step {i+1}:")
        #print(f"lt[{i+1}]为:{lt[i]}")
        for j in range(16):
            #print(lt[i],state[j],sep="=?",end=" ")
            #print(type(lt[i]),type(state[j]),end = " ")
            if state[j] == 0:
                zero_pos = j
        for ij in range(16):
            #print(lt[i],state[j],sep="=?",end=" ")
            #print(type(lt[i]),type(state[j]),end = " ")
            if state[ij] == lt[i]:
                move_pos = ij
        #print(f"zero_pos为:{zero_pos}")
        #print(f"move_pos为：{move_pos}")
        state[zero_pos] = state[move_pos]
        state[move_pos] = 0
        #print(f"第{i+1}步为：")
        for k in range(16):
            print(state[k],end=' ')
            if(k+1)%4 == 0:
                print()
    state=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0]
    print()
    print(f"Step {len(lt)+1}:")
    for k in range(16):
        print(state[k], end=' ')
        if (k + 1) % 4 == 0:
            print()

        #print(state)
def a_star_search(initial_state):       #遍历
    initial_node = PuzzleNode(heuristic_cost(initial_state), initial_state, "", 0)
    open_nodes = [initial_node]
    heapq.heapify(open_nodes)           #使用堆可以省时
    history_count = 0
    start_time = time.time()
    while len(open_nodes) != 0:
        top_node = heapq.heappop(open_nodes)
        history_count += 1
        if flag != -1:
            break
        for move in generate_moves(top_node.state, top_node.path, history_count,initial_state):
            if tuple(move[0]) in VISITED_STATES:
                continue
            VISITED_STATES.add(tuple(move[0]))
            child_node = PuzzleNode(top_node.g_score + heuristic_cost(move[0]), move[0], str(top_node.path + move[1] + " "), top_node.g_score + 1)
            heapq.heappush(open_nodes, child_node)
    if len(open_nodes) == 0:
        print("No solution.")
    end_time = time.time()
    print()
    print("Runtime:", end_time - start_time)
    print()
    print("Memory usage: %.4f GB" % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))

# Input reading
user_input = input().split()
initial_state = []
for val in user_input:
    initial_state.append(int(val))
a_star_search(initial_state)
