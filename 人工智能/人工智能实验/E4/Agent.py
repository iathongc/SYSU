import math
import copy

dir = [[1, 0], [0, 1], [1, 1], [-1, 1]]  # 方向
# 各种棋形在评分数组中的下标
sum = 0
sum1 = 0
NONE = 0,
SLEEP_TWO = 1
LIVE_TWO = 2
SLEEP_THREE = 3
LIVE_THREE = 4
CHONG_FOUR = 5
LIVE_FOUR = 6
LIVE_FIVE = 7
CHESS_TYPE_NUM = 8
FIVE = 7  # 连五
FOUR, THREE, TWO = 6, 4, 2  # 活四、活三、活二
SFOUR = 5  # 冲四
STHREE, STWO = 3, 1  # 冲三、冲二

class Chess_AI():
    def __init__(self, chess_len):
        self.len = chess_len  # 棋盘的宽度
        self.record = [[[0, 0, 0, 0] for x in range(chess_len)] for y in
                       range(chess_len)]  # 记录每一个位置的竖直 横向 左斜 右斜方向是否被访问评分过
        self.count = [[0 for x in range(CHESS_TYPE_NUM)] for i in range(2)]  # 记录黑白棋子各种棋形的数目
        self.pos_score = [[(7 - max(abs(x - 7), abs(y - 7))) for x in range(chess_len)] for y in
                          range(chess_len)]  # 给棋盘上每个位置的初始评分，由中间向四周递减

    def reset(self):  # 棋盘评分重置函数
        for y in range(self.len):
            for x in range(self.len):
                for i in range(4):
                    self.record[y][x][i] = 0  # 将各个方向访问评分标志置为0，代表未评分

        for i in range(len(self.count)):
            for j in range(len(self.count[0])):
                self.count[i][j] = 0  # 将黑白各种棋形数目置为0

    def getline(self, x, y):  # 获取x、y这个位置的竖直 横向 左斜 右斜的直线棋子列表
        line = []
        value = self.board[x][y]
        for dirs in dir:
            line.append([])
            line[-1].append(value)
            for i in range(1, 5):
                tmp_x = x + dir[i]
                tmp_y = y + dir[i]
                if tmp_x >= len or tmp_y >= len or tmp_x < 0 or tmp_y < 0:
                    line[-1].append(abs(1 - value))
                else:
                    line[-1].append(self.board[tmp_x][tmp_y])
            for i in range(1, 5):
                tmp_x = x - dir[i]
                tmp_y = y - dir[i]
                if tmp_x >= len or tmp_y >= len or tmp_x < 0 or tmp_y < 0:
                    line[-1].append(abs(1 - value))
                else:
                    line[-1].append(self.board[tmp_x][tmp_y])
        return line

    def has_neighbor(self, board, x, y, radius):  # 判断x、y周围是否有棋子
        start_x, end_x = (x - radius), (x + radius)
        start_y, end_y = (y - radius), (y + radius)
        global sum
        if sum == 0: return True

        sum += 1
        for i in range(start_y, end_y + 1):
            for j in range(start_x, end_x + 1):
                if i >= 0 and i < self.len and j >= 0 and j < self.len:
                    if board[i][j] != -1:
                        return True
        return False

    # get all positions near chess
    def get_move(self, board):  # 获取可下棋子位置
        moves = []
        radius = 2
        for y in range(self.len):
            for x in range(self.len):
                if board[y][x] == -1 and self.has_neighbor(board, x, y, radius):
                    score = self.pos_score[y][x]
                    moves.append((score, y, x))
        moves.sort(reverse=True)
        return moves

    def get_line(self, board, x, y, dir_offset, mine, opponent):  # 获取x、y这个位置的dir_offset方向的直线棋子列表
        line = [0 for i in range(9)]
        tmp_x = x + (-5 * dir_offset[0])
        tmp_y = y + (-5 * dir_offset[1])
        for i in range(9):
            tmp_x += dir_offset[0]
            tmp_y += dir_offset[1]
            if (tmp_x < 0 or tmp_x >= self.len or
                    tmp_y < 0 or tmp_y >= self.len):
                line[i] = opponent  # 将超出范围的位置当作敌方棋子
            else:
                line[i] = board[tmp_y][tmp_x]

        return line

    def evaluate_point(self, board, x, y, mine, opponent):  # 对x、y四个方向评分
        dir_offset = [(1, 0), (0, 1), (1, 1), (1, -1)]  # direction from left to right
        for i in range(4):
            if self.record[y][x][i] == 0:
                self.analysisLine(board, x, y, i, dir_offset[i], mine, opponent, self.count[mine])

    def evaluate(self, board, turn, checkWin=False):  # 估价函数
        self.reset()
        mine = turn  # turn 1代表黑子下，0代表白子下
        opponent = abs(1 - turn)
        for y in range(self.len):
            for x in range(self.len):
                if board[y][x] == mine:
                    self.evaluate_point(board, x, y, mine, opponent)
                elif board[y][x] == opponent:
                    self.evaluate_point(board, x, y, opponent, mine)
        mine_count = self.count[turn]
        opponent_count = self.count[abs(turn - 1)]
        if checkWin:
            return mine_count[FIVE] > 0
        else:
            mscore, oscore = self.getScore(mine_count, opponent_count)
            return (mscore - oscore)

    def analysisLine(self, board, x, y, dir_index, dir_offset, mine, opponent, count):  # 对x、y某个方向的棋形进行统计
        # record line range[left, right] as analysized
        def setRecord(self, x, y, left, right, dir_index, dir_offset):  # 标志某个方向的某些棋子已经被评分了，避免重复计算
            tmp_x = x + (-5 + left) * dir_offset[0]
            tmp_y = y + (-5 + left) * dir_offset[1]
            for i in range(left, right + 1):
                tmp_x += dir_offset[0]
                tmp_y += dir_offset[1]
                self.record[tmp_y][tmp_x][dir_index] = 1

        empty = -1
        left_idx, right_idx = 4, 4

        line = self.get_line(board, x, y, dir_offset, mine, opponent)  # 获取对应方向的棋子列表

        while right_idx < 8:
            if line[right_idx + 1] != mine:
                break
            right_idx += 1
        while left_idx > 0:
            if line[left_idx - 1] != mine:
                break
            left_idx -= 1

        left_range, right_range = left_idx, right_idx
        while right_range < 8:
            if line[right_range + 1] == opponent:
                break
            right_range += 1
        while left_range > 0:
            if line[left_range - 1] == opponent:
                break
            left_range -= 1

        chess_range = right_range - left_range + 1
        if chess_range < 5:
            setRecord(self, x, y, left_range, right_range, dir_index, dir_offset)
            return 0

        setRecord(self, x, y, left_idx, right_idx, dir_index, dir_offset)

        m_range = right_idx - left_idx + 1

        # M:mine chess, P:opponent chess or out of range, X: empty
        if m_range == 5:
            count[FIVE] += 1

        # Live Four : XMMMMX
        # Chong Four : XMMMMP, PMMMMX
        if m_range == 4:
            left_empty = right_empty = False
            if line[left_idx - 1] == empty:
                left_empty = True
            if line[right_idx + 1] == empty:
                right_empty = True
            if left_empty and right_empty:
                count[FOUR] += 1
            elif left_empty or right_empty:
                count[SFOUR] += 1

        # Chong Four : MXMMM, MMMXM, the two types can both exist
        # Live Three : XMMMXX, XXMMMX
        # Sleep Three : PMMMX, XMMMP, PXMMMXP
        if m_range == 3:
            left_empty = right_empty = False
            left_four = right_four = False
            if line[left_idx - 1] == empty:
                if line[left_idx - 2] == mine:  # MXMMM
                    setRecord(self, x, y, left_idx - 2, left_idx - 1, dir_index, dir_offset)
                    count[SFOUR] += 1
                    left_four = True
                left_empty = True

            if line[right_idx + 1] == empty:
                if line[right_idx + 2] == mine:  # MMMXM
                    setRecord(self, x, y, right_idx + 1, right_idx + 2, dir_index, dir_offset)
                    count[SFOUR] += 1
                    right_four = True
                right_empty = True

            if left_four or right_four:
                pass
            elif left_empty and right_empty:
                if chess_range > 5:  # XMMMXX, XXMMMX
                    count[THREE] += 1
                else:  # PXMMMXP
                    count[STHREE] += 1
            elif left_empty or right_empty:  # PMMMX, XMMMP
                count[STHREE] += 1

        # Chong Four: MMXMM, only check right direction
        # Live Three: XMXMMX, XMMXMX the two types can both exist
        # Sleep Three: PMXMMX, XMXMMP, PMMXMX, XMMXMP
        # Live Two: XMMX
        # Sleep Two: PMMX, XMMP
        if m_range == 2:
            left_empty = right_empty = False
            left_three = right_three = False
            if line[left_idx - 1] == empty:
                if line[left_idx - 2] == mine:
                    setRecord(self, x, y, left_idx - 2, left_idx - 1, dir_index, dir_offset)
                    if line[left_idx - 3] == empty:
                        if line[right_idx + 1] == empty:  # XMXMMX
                            count[THREE] += 1
                        else:  # XMXMMP
                            count[STHREE] += 1
                        left_three = True
                    elif line[left_idx - 3] == opponent:  # PMXMMX
                        if line[right_idx + 1] == empty:
                            count[STHREE] += 1
                            left_three = True

                    elif line[left_idx - 3] == mine:  # MMXMM
                        setRecord(self, x, y, left_idx - 1, left_idx - 2, dir_index, dir_offset)
                        count[SFOUR] += 1
                        right_three = True

                left_empty = True

            if line[right_idx + 1] == empty:
                if line[right_idx + 2] == mine:
                    if line[right_idx + 3] == mine:  # MMXMM
                        setRecord(self, x, y, right_idx + 1, right_idx + 2, dir_index, dir_offset)
                        count[SFOUR] += 1
                        right_three = True
                    elif line[right_idx + 3] == empty:
                        # ?????setRecord(self, x, y, right_idx+1, right_idx+2, dir_index, dir)
                        if left_empty:  # XMMXMX
                            count[THREE] += 1
                        else:  # PMMXMX
                            count[STHREE] += 1
                        right_three = True
                    elif left_empty:  # XMMXMP
                        count[STHREE] += 1
                        right_three = True

                right_empty = True

            if left_three or right_three:
                pass
            elif left_empty and right_empty:  # XMMX
                count[TWO] += 1
            elif left_empty or right_empty:  # PMMX, XMMP
                count[STWO] += 1

        # Live Two: XMXMX, XMXXMX only check right direction
        # Sleep Two: PMXMX, XMXMP
        if m_range == 1:
            left_empty = right_empty = False
            if line[left_idx - 1] == empty:
                if line[left_idx - 2] == mine:
                    if line[left_idx - 3] == empty:
                        if line[right_idx + 1] == opponent:  # XMXMP
                            count[STWO] += 1
                left_empty = True

            if line[right_idx + 1] == empty:
                if line[right_idx + 2] == mine:
                    if line[right_idx + 3] == empty:
                        if left_empty:  # XMXMX
                            # setRecord(self, x, y, left_idx, right_idx+2, dir_index, dir)
                            count[TWO] += 1
                        else:  # PMXMX
                            count[STWO] += 1
                elif line[right_idx + 2] == empty:
                    if line[right_idx + 3] == mine and line[right_idx + 4] == empty:  # XMXXMX
                        count[TWO] += 1

        return 0

    def getScore(self, mine_count, opponent_count):  # 根据棋形评分
        mscore, oscore = 0, 0
        if mine_count[FIVE] > 0:  # 我方连五，返回获胜
            return (1000000, 0)
        if opponent_count[FIVE] > 0:  # 敌方连五，返回失败
            return (0, 1000000)

        if mine_count[SFOUR] >= 2:
            mine_count[FOUR] += 1

        if opponent_count[FOUR] > 0:
            return (0, 9050)
        if opponent_count[SFOUR] > 0:
            return (0, 9040)

        if mine_count[FOUR] > 0:
            return (9030, 0)
        if mine_count[SFOUR] > 0 and mine_count[THREE] > 0:
            return (9020, 0)

        if opponent_count[THREE] > 0 and mine_count[SFOUR] == 0:
            return (0, 9010)

        if (mine_count[THREE] > 1 and opponent_count[THREE] == 0 and opponent_count[STHREE] == 0):
            return (9000, 0)

        if mine_count[SFOUR] > 0:
            mscore += 2000

        if mine_count[THREE] > 1:
            mscore += 500
        elif mine_count[THREE] > 0:
            mscore += 100

        if opponent_count[THREE] > 1:
            oscore += 2000
        elif opponent_count[THREE] > 0:
            oscore += 400

        if mine_count[STHREE] > 0:
            mscore += mine_count[STHREE] * 10
        if opponent_count[STHREE] > 0:
            oscore += opponent_count[STHREE] * 10

        if mine_count[TWO] > 0:
            mscore += mine_count[TWO] * 4
        if opponent_count[TWO] > 0:
            oscore += opponent_count[TWO] * 4

        if mine_count[STWO] > 0:
            mscore += mine_count[STWO] * 4
        if opponent_count[STWO] > 0:
            oscore += opponent_count[STWO] * 4

        return (mscore, oscore)

    def check_win(self, board, i, j):  # 检测下在i、j位置是否能赢
        if board[i][j] == -1:
            return False
        color = board[i][j]
        for dire in dir:
            x, y = i, j
            x1, y1 = i, j
            chess = []
            while board[x1][y1] == color:
                chess.append((x1, y1))
                x1, y1 = x1 + dire[0], y1 + dire[1]
                if x1 < 0 or y1 < 0 or x1 >= self.len or y1 >= self.len:
                    break
            x1, y1 = x - dire[0], y - dire[1]
            if x1 < 0 or y1 < 0 or x1 >= self.len or y1 >= self.len:
                continue
            while board[x1][y1] == color:
                chess.append((x1, y1))
                x1, y1 = x1 - dire[0], y1 - dire[1]
                if x1 < 0 or y1 < 0 or x1 >= self.len or y1 >= self.len:
                    break
            if len(chess) >= 5:
                return True
        return False

def renew(list):  # 对棋盘列表重构
    tmp = copy.deepcopy(list)
    for i in range(len(list)):
        for j in range(len(list[i])):
            if list[i][j] == (0, 0, 0):
                tmp[i][j] = 1
            if list[i][j] == (255, 255, 255):
                tmp[i][j] = 0
    return tmp

def search(board, Ai, alpha, beta, if_max, turn, depth, limit):  # 深度优先搜索
    global sum1
    sum1 += 1
    moves = Ai.get_move(board)
    max_score = -1000000  #
    min_score = 1000000  #
    alpha_tmp = alpha
    beta_tmp = beta
    x_ans = -1
    y_ans = -1
    for move in moves:
        x, y = move[1], move[2]
        board[x][y] = turn
        if turn == 0:
            if Ai.check_win(board, x, y):
                x_ans = x
                y_ans = y
                min_score = -1000000  #
        else:
            if Ai.check_win(board, x, y):
                x_ans = x
                y_ans = y
                max_score = 1000000  #
        if depth == limit:
            sum1 += 1
            score = Ai.evaluate(board, turn)
            if turn % 2 == 0:
                score = -score
        else:
            x_tmp, y_tmp, score = search(board, Ai, alpha_tmp, beta_tmp, 1 - if_max, 1 - turn, depth + 1, limit)
        board[x][y] = -1
        if if_max == 1:
            if score > max_score:
                x_ans = x
                y_ans = y
                max_score = score
            if max_score > beta_tmp or max_score == beta_tmp:
                break
            if max_score > alpha_tmp:
                alpha_tmp = max_score
        else:
            if score < min_score:
                x_ans = x
                y_ans = y
                min_score = score
            if min_score < alpha_tmp or min_score == alpha_tmp:
                break
            if min_score < beta_tmp:
                beta_tmp = min_score
    if if_max == 1:
        return (x_ans, y_ans, max_score)
    else:
        return (x_ans, y_ans, min_score)

def AlphaBetaSearch(board1, EMPTY, BLACK, WHITE, black):  # alpha-beta剪枝搜索
    board = renew(board1)
    Ai = Chess_AI(len(board))
    if_max = 1
    turn = 0
    alpha = -10000000000
    beta = 100000000000
    if black:
        turn = 1
    limit = 2  # 搜索深度
    x, y, score = search(board, Ai, alpha, beta, if_max, turn, 1, limit)
    global sum1
    print("------------------------")
    print(sum1)
    return (x, y, score)