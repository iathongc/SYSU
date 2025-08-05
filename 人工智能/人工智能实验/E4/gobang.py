# coding=utf-8
# 转载自知乎“一行玩python”
import pygame
import game
from argparse import ArgumentParser
from Agent import AlphaBetaSearch

ROWS = 15
SIDE = 30

SCREEN_WIDTH = ROWS * SIDE
SCREEN_HEIGHT = ROWS * SIDE

EMPTY = -1
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
DIRE = [(1, 0), (0, 1), (1, 1), (1, -1)]

class Gobang(game.game):
    def __init__(self, title, size, fps=15, chess_file=None):
        super(Gobang, self).__init__(title, size, fps)
        self.board = [[EMPTY for i in range(ROWS)] for j in range(ROWS)]

        # chess_file输入初始局面，#为空，1为黑，0为白
        if "D:\Python\python project\11.txt" != None:
            with open(r'D:\Python\python project\11.txt', "r") as f:
                for i, line in enumerate(f.readlines()):
                    for j, x in enumerate(line.strip().split(" ")):
                        if x == "1":
                            self.board[i][j] = BLACK
                        if x == "0":
                            self.board[i][j] = WHITE

        self.select = (-1, -1)
        self.black = True
        # self.black = False
        self.draw_board()
        self.bind_click(1, self.click)
        # 修改使用AI还是人
        self.player1 = self.AI_player
        self.player2 = self.Human_player

    def click(self, x, y):  # 落子，成功落子return True，否则return False
        if self.end:
            return False
        i, j = y // SIDE, x // SIDE  # 注意坐标关系
        if self.board[i][j] != EMPTY:
            pygame.display.set_caption("五子棋 ---- %s棋错误落子" % ("黑" if self.black else "白"))
            return False
        self.board[i][j] = BLACK if self.black else WHITE
        self.draw_chess(self.board[i][j], i, j)
        self.black = not self.black

        chess = self.check_win()
        if chess:
            self.end = True
            i, j = chess[0]
            winer = "Black"
            if self.board[i][j] == WHITE:
                winer = "White"
            pygame.display.set_caption("五子棋 ---- %s win!" % (winer))
            for c in chess:
                i, j = c
                self.draw_chess((100, 255, 255), i, j)
                self.timer.tick(5)

        return True

    def check_win(self):
        for i in range(ROWS):
            for j in range(ROWS):
                win = self.check_chess(i, j)
                if win:
                    return win
        return None

    def check_chess(self, i, j):
        if self.board[i][j] == EMPTY:
            return None
        color = self.board[i][j]
        for dire in DIRE:
            x, y = i, j
            chess = []
            while self.board[x][y] == color:
                chess.append((x, y))
                x, y = x + dire[0], y + dire[1]
                if x < 0 or y < 0 or x >= ROWS or y >= ROWS:
                    break
            if len(chess) >= 5:
                return chess
        return None

    def draw_chess(self, color, i, j):
        center = (j * SIDE + SIDE // 2, i * SIDE + SIDE // 2)
        pygame.draw.circle(self.screen, color, center, SIDE // 2 - 2)
        pygame.display.update(pygame.Rect(j * SIDE, i * SIDE, SIDE, SIDE))

    def draw_board(self):
        self.screen.fill((139, 87, 66))
        for i in range(ROWS):
            start = (i * SIDE + SIDE // 2, SIDE // 2)
            end = (i * SIDE + SIDE // 2, ROWS * SIDE - SIDE // 2)
            pygame.draw.line(self.screen, 0x000000, start, end)
            start = (SIDE // 2, i * SIDE + SIDE // 2)
            end = (ROWS * SIDE - SIDE // 2, i * SIDE + SIDE // 2)
            pygame.draw.line(self.screen, 0x000000, start, end)
        center = ((ROWS // 2) * SIDE + SIDE // 2, (ROWS // 2) * SIDE + SIDE // 2)
        pygame.draw.circle(self.screen, (0, 0, 0), center, 4)
        pygame.display.update()

        for i in range(ROWS):
            for j in range(ROWS):
                if self.board[i][j] is not EMPTY:
                    self.draw_chess(self.board[i][j], i, j)

    def AI_player(self):
        # 修改该函数
        if not self.black: return
        x, y, alpha = AlphaBetaSearch(self.board, EMPTY, BLACK, WHITE, self.black)
        self.click(y * SIDE, x * SIDE)
        print(x, y, alpha)

if __name__ == '__main__':
    print('''
    欢迎挑战五子棋残局!
    click LEFT MOUSE BUTTON to play game.
    ''')
    parser = ArgumentParser(description='gobang')
    parser.add_argument('--chess_file', type=str, default=None)
    args = parser.parse_args()
    gobang = Gobang("五子棋", (SCREEN_WIDTH, SCREEN_HEIGHT), chess_file=args.chess_file)
    gobang.run()