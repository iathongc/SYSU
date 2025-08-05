import copy
import time
import os
import psutil

goal = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0]
next = [[-1, 0], [1, 0], [0, -1], [0, 1]]
close = set()
history = 0

def man_dist(arr):  # 计算曼哈顿距离
    ans = 0
    for i in range(0, 16):
        if arr[i] != 0 and arr[i] != goal[i]:
            x = int((arr[i] - 1) / 4)
            y = arr[i] - 4 * x - 1
            ans += abs(x - int(i / 4)) + abs(y - (i % 4))
    return ans

def linear_conflict(arr):
    board = [arr[0:4], arr[4:8], arr[8:12], arr[12:16]]
    n = len(board)
    linear_conflicts = 0
    # 计算每一行的线性冲突
    for i in range(n):
        for j in range(n):
            tile1 = board[i][j]
            if tile1 == 0:
                continue
            goal_row1, goal_col1 = (tile1 - 1) // n, (tile1 - 1) % n
            if goal_row1 == i:  # 需要在同一行内进一步检查
                for k in range(j + 1, n):
                    tile2 = board[i][k]
                    if tile2 == 0:
                        continue
                    goal_row2, goal_col2 = (tile2 - 1) // n, (tile2 - 1) % n
                    if goal_row2 == i and goal_col2 < goal_col1:
                        linear_conflicts += 1
    # 计算每一列的线性冲突
    for j in range(n):
        for i in range(n):
            tile1 = board[i][j]
            if tile1 == 0:
                continue
            goal_row1, goal_col1 = (tile1 - 1) // n, (tile1 - 1) % n
            if goal_col1 == j:  # 需要在同一列内进一步检查
                for k in range(i + 1, n):
                    tile2 = board[k][j]
                    if tile2 == 0:
                        continue
                    goal_row2, goal_col2 = (tile2 - 1) // n, (tile2 - 1) % n
                    if goal_col2 == j and goal_row2 < goal_row1:
                        linear_conflicts += 1
    return man_dist(arr) + linear_conflicts

def h(arr):
    return linear_conflict(arr)

def children(arr_):
    index = 0
    for i in range(0, 16):
        if arr_[i] == 0:
            index = i  # 找到0的坐标
    ans = []
    x = index // 4
    y = index % 4
    for i in range(0, 4):
        fx = next[i][0]
        fy = next[i][1]
        new_index = index + fx * 4 + fy
        if x + fx > -1 and x + fx < 4 and y + fy > -1 and y + fy < 4:
            arr = copy.deepcopy(arr_)
            arr[index] = arr[new_index]
            arr[new_index] = 0
            ans.append(arr)
    return sorted(ans, key=lambda x: h(x))

def dfs(way, g, bound):
    global history
    node = way[-1]
    f = g + h(node)
    history += 1
    if f > bound:
        return f
    if node == goal:
        return -1  # 找到目标

    min = 6666
    for i in children(node):
        k = tuple(i)
        if k in close: continue
        way.append(i)
        close.add(k)  # 路经检测

        t = dfs(way, g + 1, bound)
        if t == -1:
            return -1
        if t < min:
            min = t
        way.pop()
        close.remove(k)
    return min

def IDA(start):
    bound = h(start)
    way = [start]  # 更新的路径合集
    close.add(tuple(start))
    while (1):
        t = dfs(way, 0, bound)
        if t == -1:
            return way
        if t > 60:
            return None
        bound = t  # 更新bound

def print_ans(ans):
    ans_list = []
    for k in range(1, len(ans)):
        index = 0
        for i in range(0, 16):
            if ans[k - 1][i] == 0:
                index = i
                break
        ans_list.append(str(ans[k][index]))  # 将数字转换为字符串
    l = len(ans_list) + 1
    print()
    print("Path:", ' '.join(ans_list))  # 使用空格连接列表中的元素
    print()
    print(f"Path count: {l}",end=" ")
    print()
    print()
    return ans_list

def print_format(state_str,lt_str):
    zero_pos = 0
    move_pos = 1
    lt = [int(item) for item in lt_str]
    state = [int(item) for item in state_str]
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

# 数据读入
temp = input().split()
start = time.time()
arr = []
sum = 0
for i in range(0, 16):
    arr.append(int(temp[sum]))
    sum += 1

ans = IDA(arr)
lt_str = print_ans(ans)
print("Total number of visited nodes:", history)
print_format(temp,lt_str)
end = time.time()
print()
print("Runtime:", end - start)
print()
print(u'Memory usage:%.4f GB'
      % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
