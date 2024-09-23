import os
import numpy as np
from copy import deepcopy
import random
import time
import pygame


class GameState:
    def __init__(self, board=None):
        self.board = board if board is not None else np.zeros((4, 4), dtype=int)
        self.player_turn = True  # 对于玩家为真，对于 AI 为假
        self.score = 0
        self.dir = random.choice([0, 1, 2, 3])

    def find_max_index(self):
        """寻找最大数的坐标值"""
        max_value = self.board[0][0]
        max_index = (0, 0)
        for i in range(len(self.board)):
            for j in range(len(self.board[i])):
                if self.board[i][j] > max_value:
                    max_value = self.board[i][j]
                    max_index = (i, j)
        return max_index

    def Is_game_succeed(self):
        """寻找期盼的最大数"""
        max_value = np.max(np.max(self.board))
        if max_value >= 2048:
            return True
        else:
            return False

    def move(self, direction):
        # 方向：0 - 左，1 - 上，2 - 右，3 - 下
        moved = False
        new_board = np.copy(self.board)
        total_score = 0
        if direction == 0:  # Left
            self.dir = direction
            for row in new_board:
                non_zero = row[row != 0]
                merged, score = self._merge_left(non_zero)
                total_score += score
                row[:len(merged)] = merged
                row[len(merged):] = 0
                if not np.all(row == self.board[row != 0]):
                    moved = True
        elif direction == 1:  # Up
            self.dir = direction
            transposed = np.transpose(new_board)
            for row in transposed:
                non_zero = row[row != 0]
                merged, score = self._merge_left(non_zero)
                total_score += score
                row[:len(merged)] = merged
                row[len(merged):] = 0
                if not np.all(row == self.board[row != 0]):
                    moved = True
            new_board = np.transpose(transposed)
        elif direction == 2:  # Right
            self.dir = direction
            for row in new_board:
                non_zero = row[row != 0][::-1]
                merged, score = self._merge_left(non_zero)
                total_score += score
                row[-len(merged):] = merged[::-1]
                row[:-len(merged)] = 0
                if not np.all(row == self.board[row != 0]):
                    moved = True
        elif direction == 3:  # Down
            self.dir = direction
            transposed = np.transpose(new_board)
            for row in transposed:
                non_zero = row[row != 0][::-1]
                merged, score = self._merge_left(non_zero)
                total_score += score
                row[-len(merged):] = merged[::-1]
                row[:-len(merged)] = 0
                if not np.all(row == self.board[row != 0]):
                    moved = True
            new_board = np.transpose(transposed)

        if moved:
            self.board = new_board
            self.score += total_score
        return moved

    def _merge_left(self, non_zero):
        if len(non_zero) == 0:
            return [0], 0  # 返回一个包含单个零和合并之和的列表
        merged = []
        score = 0  # 初始化本次合并的分数
        skip = False
        for i in range(len(non_zero)):
            if skip:
                skip = False
                continue
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                merged.append(non_zero[i] * 2)
                score += non_zero[i] * 2  # 添加此次合并的得分
                skip = True
            else:
                merged.append(non_zero[i])
        return merged, score

    def insert_tile(self, x, y, value):
        if self.board[x, y] == 0:
            self.board[x, y] = value
            return True
        return False

    def remove_tile(self, x, y):
        self.board[x, y] = 0

    def get_available_cells(self):
        """返回一个由元组组成的列表，这些元组表示空单元格的坐标"""
        available_cells = []
        for row in range(4):
            for col in range(4):
                if self.board[row][col] == 0:
                    available_cells.append((row, col))
        return available_cells

    def can_merge(self):
        """检查是否有任何可以合并的相邻方块"""
        for row in range(4):
            for col in range(3):
                if self.board[row][col] == self.board[row][col + 1]:
                    return True
        for row in range(3):
            for col in range(4):
                if self.board[row][col] == self.board[row + 1][col]:
                    return True
        return False

    def is_game_over(self):
        """通过查看可能的走法来检查游戏是否结束"""
        # 检查是否存在任何空单元格
        if self.get_available_cells():
            return False
        # 检查是否有任何可以合并的相邻方块
        if self.can_merge():
            return False
        return True

    def undo_move(self, direction):
        # 为了检查游戏结束的情况，我们需要在测试后撤销这一动作.
        self.board = np.copy(self.board)

    def __str__(self):
        return '\n'.join(['\t\t'.join([str(cell) for cell in row]) for row in self.board])


class AI:
    def __init__(self, game_state):
        self.game_state = game_state

    def evaluate(self):
        smoothness = self.smoothness()
        monotonicity = self.monotonicity()
        empty_cells = len(self.game_state.get_available_cells())
        # Avoid division by zero
        if empty_cells == 0:
            empty_cells = 1
        max_value = np.max(self.game_state.board)
        return smoothness * 2.7 + monotonicity * 1.3 + np.log(empty_cells) * 0.1 + np.log(max_value) * 1.5

    def smoothness(self):
        smoothness = 0
        for x in range(4):
            for y in range(4):
                if self.game_state.board[x, y] != 0:
                    value = np.log2(self.game_state.board[x, y])
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < 4 and 0 <= ny < 4:
                            if self.game_state.board[nx, ny] != 0:
                                target_value = np.log2(self.game_state.board[nx, ny])
                                smoothness -= abs(value - target_value)
        return smoothness

    def monotonicity(self):
        totals = [0, 0, 0, 0]
        for x in range(4):
            current, next = 0, 1
            while next < 4:
                while next < 4 and self.game_state.board[x, next] == 0:
                    next += 1
                if next >= 4:
                    next -= 1
                currentValue = np.log2(self.game_state.board[x, current]) if self.game_state.board[
                                                                                 x, current] != 0 else 0
                nextValue = np.log2(self.game_state.board[x, next]) if self.game_state.board[x, next] != 0 else 0
                if currentValue > nextValue:
                    totals[0] += nextValue - currentValue
                elif nextValue > currentValue:
                    totals[1] += currentValue - nextValue
                current = next
                next += 1
        for y in range(4):
            current, next = 0, 1
            while next < 4:
                while next < 4 and self.game_state.board[next, y] == 0:
                    next += 1
                if next >= 4:
                    next -= 1
                currentValue = np.log2(self.game_state.board[current, y]) if self.game_state.board[
                                                                                 current, y] != 0 else 0
                nextValue = np.log2(self.game_state.board[next, y]) if self.game_state.board[next, y] != 0 else 0
                if currentValue > nextValue:
                    totals[2] += nextValue - currentValue
                elif nextValue > currentValue:
                    totals[3] += currentValue - nextValue
                current = next
                next += 1
        return max(totals[0], totals[1]) + max(totals[2], totals[3])

    def search(self, depth, alpha, beta, positions, cutoffs):
        best_score = alpha if self.game_state.player_turn else beta
        best_move = -1
        directions = [0, 1, 2, 3]

        if self.game_state.player_turn:  # 最大层
            for direction in directions:
                new_grid = GameState(deepcopy(self.game_state.board))
                if new_grid.move(direction):
                    positions += 1
                    if depth == 0:
                        score = self.evaluate()
                    else:
                        new_ai = AI(GameState(new_grid.board))
                        new_ai.game_state.player_turn = False
                        result = new_ai.search(depth - 1, best_score, beta, positions, cutoffs)
                        score = result['score']
                        positions = result['positions']
                        cutoffs = result['cutoffs']
                    if score > best_score:
                        best_score = score
                        best_move = direction
                    if best_score > beta:
                        cutoffs += 1
                        return {'move': best_move, 'score': best_score, 'positions': positions, 'cutoffs': cutoffs}
            return {'move': best_move, 'score': best_score, 'positions': positions, 'cutoffs': cutoffs}
        else:  # Min layer
            available_cells = self.game_state.get_available_cells()
            if not available_cells:  # 检查可用的单元格
                return {'move': -1, 'score': 0, 'positions': positions, 'cutoffs': cutoffs}

            candidates = []
            for value in [2, 4]:
                for cell in available_cells:
                    self.game_state.insert_tile(*cell, value)
                    score = -self.smoothness() + self.islands()
                    candidates.append({'cell': cell, 'value': value, 'score': score})
                    self.game_state.remove_tile(*cell)

            # 选择能导致最佳得分的走法
            best_candidate = max(candidates, key=lambda c: c['score'])
            best_cell, best_value = best_candidate['cell'], best_candidate['value']
            self.game_state.insert_tile(*best_cell, best_value)

            # 现在检查一下棋盘是否已满但仍可能有合并。
            if self.game_state.is_game_over():
                # 尝试所有四个方向，并选择导致合并的那个方向
                for direction in directions:
                    new_grid = GameState(deepcopy(self.game_state.board))
                    if new_grid.move(direction):
                        # 出现了一次合并，所以这是一个有效的举措
                        return {'move': direction, 'score': -100, 'positions': positions, 'cutoffs': cutoffs}

            return {'move': -1, 'score': 0, 'positions': positions, 'cutoffs': cutoffs}

    def iterative_deepening_search(self, timeout_ms):
        best_move = -1
        start_time = time.time()
        depth = 0
        while (time.time() - start_time) * 1000 < timeout_ms:
            result = self.search(depth, -np.inf, np.inf, 0, 0)
            if result['move'] != -1:
                best_move = result['move']
            depth += 1
        return best_move

    def islands(self):
        islands = 0
        marked = np.zeros((4, 4), dtype=bool)
        for x in range(4):
            for y in range(4):
                if self.game_state.board[x, y] != 0:
                    marked[x, y] = False
        for x in range(4):
            for y in range(4):
                if self.game_state.board[x, y] != 0 and not marked[x, y]:
                    islands += 1
                    self._mark(x, y, self.game_state.board[x, y], marked)
        return islands

    def _mark(self, x, y, value, marked):
        if 0 <= x < 4 and 0 <= y < 4 and self.game_state.board[x, y] == value and not marked[x, y]:
            marked[x, y] = True
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                self._mark(nx, ny, value, marked)


class Show:
    def __init__(self, game_state):
        self.game_state = game_state
        self.color_bar = {0: (50, 205, 50), 2: (255, 228, 225), 4: (255, 218, 185), 8: (188, 143, 143),
                          16: (210, 180, 140), 32: (244, 164, 96), 64: (210, 105, 30), 128: (160, 82, 45),
                          256: (34, 139, 34), 512: (50, 205, 50), 1024: (0, 100, 0), 2048: (250, 128, 114)}
        self.screen = pygame.display.set_mode((600, 700))
        self.font = pygame.font.Font(r"font/联想小新潮酷体.ttf", 30)
        self.direction = {0: '上', 1: '左', 2: '下', 3: '右'}
        self.music_dir = 'music'
        self.music_control = '关'
        self.ai_control = '关'
        tmp_score = []  # 获取最大历史分数
        with open('history_score.txt', 'r') as f:
            for line in f:
                line = line.strip()
                tmp_score.append(int(line))
        self.highest_score = np.max(tmp_score)

    def game_board(self):
        """设置棋盘相关显示"""
        pygame.display.set_caption('Redal-Python-2048')
        self.screen.fill((150, 200, 150))
        for row in range(4):
            for col in range(4):
                # 绘制棋盘
                pygame.draw.rect(self.screen, self.color_bar[self.game_state.board[row][col]],
                                 [100 + 100 * row, 150 + 100 * col, 95, 95], 0, 10)
                # 绘制数字字体
                if self.game_state.board[row][col] != 0:
                    text_number = self.font.render(str(self.game_state.board[row][col]), True, (255, 255, 255),
                                                   self.color_bar[self.game_state.board[row][col]])
                else:
                    text_number = self.font.render('', True, (255, 255, 255),
                                                   self.color_bar[self.game_state.board[row][col]])
                text_rect = text_number.get_rect(center=(147.5 + row * 100, 197.5 + col * 100))
                self.screen.blit(text_number, text_rect)

        context_list = ['得分', str(self.game_state.score), '帮助', self.direction[self.game_state.dir]]
        for row, x in enumerate(context_list):  # 绘制得分显示和帮助显示
            pygame.draw.rect(self.screen, (50, 180, 50), [100 + 100 * row, 97.5, 95, 47.5], 0, 10)
            text_score_help = self.font.render(x, True, (255, 255, 255), (50, 180, 50))
            score_help_rect = text_score_help.get_rect(center=(147.5 + 100 * row, 121.25))
            self.screen.blit(text_score_help, score_help_rect)

        music_ai_list = ['音乐', self.music_control, '智能', self.ai_control]
        for row, x in enumerate(music_ai_list):
            pygame.draw.rect(self.screen, (50, 240, 50), [100 + 100 * row, 45, 95, 47.5], 0, 10)
            text_music_ai = self.font.render(x, True, (255, 255, 255), (50, 240, 50))
            music_ai_rect = text_music_ai.get_rect(center=(147.5 + 100 * row, 68.75))
            self.screen.blit(text_music_ai, music_ai_rect)

        score_restart_list = ['最高分', str(self.highest_score), '重开', '点击']
        for row, x in enumerate(score_restart_list):
            pygame.draw.rect(self.screen, (100, 220, 50), [100 + 100 * row, 550, 95, 47.5], 0, 10)
            text_restart_score = self.font.render(x, True, (255, 255, 255), (100, 220, 50))
            restart_score_rect = text_restart_score.get_rect(center=(147.5 + 100 * row, 573.75))
            self.screen.blit(text_restart_score, restart_score_rect)
        pygame.display.flip()  # 更新屏幕

    def music_on_off(self):
        if self.music_control == '开':
            music_paths = [os.path.join(self.music_dir, music_file) for music_file in os.listdir(self.music_dir)]
            pygame.mixer.music.load(random.choice(music_paths))
            pygame.mixer.music.play(loops=-1)  # 循环播放
        elif self.music_control == '关':
            pygame.mixer.music.stop()

    def show_game_lost(self):
        """绘制失败游戏界面"""
        pygame.init()
        lost_font = pygame.font.Font(r'font/联想小新潮酷体.ttf', 80)
        text_pos = (150, 300)

        # 绘制失败文字
        lost_rect_surface = lost_font.render('游戏失败', True, (50, 255, 50), None)
        lost_rect_surface.set_alpha(50)  # 设置为半透明状态
        self.screen.blit(lost_rect_surface, text_pos)

    def show_game_succeed(self):
        """绘制胜利游戏界面"""
        succeed_font = pygame.font.Font(r'font/联想小新潮酷体.ttf', 80)
        text_pos = (150, 300)

        # 绘制失败文字
        succeed_rect_surface = succeed_font.render('游戏胜利', True, (50, 255, 50), None)
        succeed_rect_surface.set_alpha(50)  # 设置为半透明状态
        self.screen.blit(succeed_rect_surface, text_pos)


def main():
    # 初始化游戏
    pygame.init()
    initial_board = np.zeros((4, 4), dtype=int)
    initial_board[random.randint(0, 3), random.randint(0, 3)] = 0
    game_state = GameState(initial_board)
    game_show = Show(game_state)

    # 设置初始状态
    game_state.insert_tile(*random.choice(game_state.get_available_cells()), 2)
    game_show.music_on_off()
    game_show.game_board()

    # 设置鼠标点击区域
    music_pos = pygame.Rect(200, 45, 95, 47.5)
    ai_pos = pygame.Rect(400, 45, 95, 47.5)
    restart_pos = pygame.Rect(400, 550, 95, 47.5)

    running = True
    while running:
        pygame.init()
        pygame.time.Clock().tick(60)
        for event in pygame.event.get():
            # 判断是否离开
            if event.type == pygame.QUIT:
                running = False

            # 判断游戏是否满足结束条件
            if game_state.is_game_over():
                running = False
            if game_state.Is_game_succeed():
                running = False

            # 检测鼠标点击状态
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                # 检测鼠标点击区域
                if music_pos.collidepoint(mouse_pos):
                    if game_show.music_control == '开':
                        game_show.music_control = '关'
                        game_show.music_on_off()  # 控制背景音乐
                    elif game_show.music_control == '关':
                        game_show.music_control = '开'
                        game_show.music_on_off()  # 控制背景音乐
                elif ai_pos.collidepoint(mouse_pos):
                    if game_show.ai_control == '开':
                        game_show.ai_control = '关'
                    elif game_show.ai_control == '关':
                        game_show.ai_control = '开'
                elif restart_pos.collidepoint(mouse_pos):
                    initial_board = np.zeros((4, 4), dtype=int)
                    initial_board[random.randint(0, 3), random.randint(0, 3)] = 2
                    game_state.board = initial_board

        if game_show.ai_control == '开':
            """AI模式运行"""
            ai = AI(game_state)
            move = ai.iterative_deepening_search(200)
            if move != -1:
                moved = game_state.move(move)  # 假设 move 方法返回一个布尔值
                if moved:  # 只有当移动有效时才插入新数字
                    available_cells = game_state.get_available_cells()
                    if available_cells:
                        game_state.insert_tile(*random.choice(available_cells), 2)
                    else:
                        x, y = game_state.find_max_index()
                        if x <= 1 and y <= 1:
                            game_state.move(random.choice([0, 1]))
                        elif x <= 1 and y >= 2:
                            game_state.move(random.choice([0, 3]))
                        elif x >= 2 and y <= 1:
                            game_state.move(random.choice([1, 2]))
                        elif x >= 2 and y >= 2:
                            game_state.move(random.choice([2, 3]))
                        else:
                            # 如果没有合法的移动，则游戏结束
                            break
            print(f'Get Scores: {game_state.score}')
            print(game_state.__str__())
            game_show.game_board()
            pygame.display.flip()

        if game_show.ai_control == '关':
            """玩家模式运行"""
            # 获取按键的状态
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT] or keys[ord('a')]:
                moved = game_state.move(1)  # 假设 move 方法返回一个布尔值
                if moved:  # 只有当移动有效时才插入新数字
                    available_cells = game_state.get_available_cells()
                    if available_cells:
                        game_state.insert_tile(*random.choice(available_cells), 2)
            elif keys[pygame.K_RIGHT] or keys[ord('d')]:
                moved = game_state.move(3)  # 假设 move 方法返回一个布尔值
                if moved:  # 只有当移动有效时才插入新数字
                    available_cells = game_state.get_available_cells()
                    if available_cells:
                        game_state.insert_tile(*random.choice(available_cells), 2)
            elif keys[pygame.K_UP] or keys[ord('w')]:
                moved = game_state.move(0)  # 假设 move 方法返回一个布尔值
                if moved:  # 只有当移动有效时才插入新数字
                    available_cells = game_state.get_available_cells()
                    if available_cells:
                        game_state.insert_tile(*random.choice(available_cells), 2)
            elif keys[pygame.K_DOWN] or keys[ord('s')]:
                moved = game_state.move(2)  # 假设 move 方法返回一个布尔值
                if moved:  # 只有当移动有效时才插入新数字
                    available_cells = game_state.get_available_cells()
                    if available_cells:
                        game_state.insert_tile(*random.choice(available_cells), 2)

        game_show.game_board()  # 游戏结束,显示结束棋盘
        pygame.display.flip()
        pygame.display.update()
    # 保存分数
    with open('history_score.txt', 'a') as f:
        if game_state.score != 0:
            # 若分数为0，则不保存
            f.write(str(game_state.score) + '\n')

    # 游戏失败画面
    if game_state.is_game_over():
        pygame.init()
        # 进入游戏结束循环
        running = True
        while running:
            game_show.show_game_lost()
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break  # 退出游戏结束循环
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    # 检测鼠标点击区域
                    if restart_pos.collidepoint(mouse_pos):
                        main()

    # 游戏胜利画面
    elif game_state.Is_game_succeed():
        pygame.init()
        # 进入游戏胜利循环
        running = True
        while running:
            game_show.show_game_succeed()
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break  # 退出游戏胜利循环
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    # 检测鼠标点击区域
                    if restart_pos.collidepoint(mouse_pos):
                        main()
    print("Game Over")
    print(game_state)
    pygame.quit()


if __name__ == '__main__':
    main()
