import numpy as np
from copy import deepcopy
import random
import time


class GameState:
    def __init__(self, board=None):
        self.board = board if board is not None else np.zeros((4, 4), dtype=int)
        self.player_turn = True  # True for player, False for AI
        self.score = 0

    def find_max(self):
        max_value = self.board[0][0]
        max_index = (0, 0)
        for i in range(len(self.board)):
            for j in range(len(self.board[i])):
                if self.board[i][j] > max_value:
                    max_value = self.board[i][j]
                    max_index = (i, j)
        return max_index

    def move(self, direction):
        # Directions: 0- Left, 1- Up, 2- Right, 3- Down
        moved = False
        new_board = np.copy(self.board)
        total_score = 0
        if direction == 0:  # Left
            for row in new_board:
                non_zero = row[row != 0]
                merged, score = self._merge_left(non_zero)
                total_score += score
                row[:len(merged)] = merged
                row[len(merged):] = 0
                if not np.all(row == self.board[row != 0]):
                    moved = True
        elif direction == 1:  # Up
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
            for row in new_board:
                non_zero = row[row != 0][::-1]
                merged, score = self._merge_left(non_zero)
                total_score += score
                row[-len(merged):] = merged[::-1]
                row[:-len(merged)] = 0
                if not np.all(row == self.board[row != 0]):
                    moved = True
        elif direction == 3:  # Down
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
            return [0], 0  # Return a list with a single zero and the sum of merges
        merged = []
        score = 0  # Initialize the score for this merge
        skip = False
        for i in range(len(non_zero)):
            if skip:
                skip = False
                continue
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                merged.append(non_zero[i] * 2)
                score += non_zero[i] * 2  # Add the score of this merge
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
        """Returns a list of tuples representing the coordinates of empty cells."""
        available_cells = []
        for row in range(4):
            for col in range(4):
                if self.board[row][col] == 0:
                    available_cells.append((row, col))
        return available_cells

    def can_merge(self):
        """Checks if there are any adjacent tiles that can be merged."""
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
        """Checks if the game is over by looking for possible moves."""
        # Check if there are any empty cells
        if self.get_available_cells():
            return False
        # Check for any adjacent tiles that can be merged
        if self.can_merge():
            return False
        return True

    def undo_move(self, direction):
        # For the purpose of checking game over, we need to undo the move after testing it.
        self.board = np.copy(self.board)

    # def __str__(self):
    #     return '\n'.join([''.join([str(cell) for cell in row]) for row in self.board])

    def __str__(self):
        return '\n'.join([' '.join([str(cell) for cell in row]) for row in self.board])


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
        return smoothness * 0.1 + monotonicity * 1.3 + np.log(empty_cells) * 2.7 + np.log(max_value) * 1.0

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

        if self.game_state.player_turn:  # Max layer
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
            if not available_cells:  # Check for available cells
                return {'move': -1, 'score': 0, 'positions': positions, 'cutoffs': cutoffs}

            candidates = []
            for value in [2, 4]:
                for cell in available_cells:
                    self.game_state.insert_tile(*cell, value)
                    score = -self.smoothness() + self.islands()
                    candidates.append({'cell': cell, 'value': value, 'score': score})
                    self.game_state.remove_tile(*cell)

            # Choose the move that leads to the best score
            best_candidate = max(candidates, key=lambda c: c['score'])
            best_cell, best_value = best_candidate['cell'], best_candidate['value']
            self.game_state.insert_tile(*best_cell, best_value)

            # Now check if the board is full but there can still be merges
            if self.game_state.is_game_over():
                # Try all four directions and choose the one that results in a merge
                for direction in directions:
                    new_grid = GameState(deepcopy(self.game_state.board))
                    if new_grid.move(direction):
                        # There was a merge, so this is a valid move
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


def main():
    # 初始化游戏
    initial_board = np.zeros((4, 4), dtype=int)
    initial_board[random.randint(0, 3), random.randint(0, 3)] = 2
    game_state = GameState(initial_board)

    # 设置初始状态
    game_state.insert_tile(*random.choice(game_state.get_available_cells()), 2)

    # 主循环
    while not game_state.is_game_over():
        print(game_state)
        ai = AI(game_state)
        move = ai.iterative_deepening_search(200)
        if move != -1:
            game_state.move(move)

        # 检查是否有可用的空格
        available_cells = game_state.get_available_cells()
        if available_cells:
            game_state.insert_tile(*random.choice(available_cells), 2)
        else:
            x, y = game_state.find_max()
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

    print("Game Over")
    print(game_state)


if __name__ == '__main__':
    main()
