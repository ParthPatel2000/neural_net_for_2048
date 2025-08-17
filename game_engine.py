#game_engine.py

import math
from random import random, choice

class GameEngine:
    
    boardSize = (5, 4)  # height, width
    gameboard = []
    score = 0  # current score of the player
    moves = 0  # number of moves made
    is_game_over = False  # whether the game is over

    def __init__(self):
        self.gameboard = [[0 for _ in range(self.boardSize[1])] for _ in range(self.boardSize[0])]  # 2D list representing the game board
        self.score = 0  # current score of the player
        self.moves = 0  # number of moves made
        self.is_game_over = False  # whether the game is over

    def __init__(self, rows=5, cols=4):
        self.boardSize = (rows, cols)
        self.gameboard = [[0 for _ in range(cols)] for _ in range(rows)]  # 2D list representing the game board
        self.score = 0  # current score of the player
        self.moves = 0  # number of moves made
        self.is_game_over = False  # whether the game is over

    def __str__(self):
        board_str = "\n".join(["\t".join(map(str, row)) for row in self.gameboard])
        return f"{board_str}\nScore: {self.score}, Moves: {self.moves}"

    def reset_game(self):
        self.gameboard = [[0 for _ in range(self.boardSize[1])] for _ in range(self.boardSize[0])]  # 2D list representing the game board
        self.score = 0
        self.moves = 0
        self.is_game_over = False

    def gameState(self):
        return {
            "board": self.gameboard,
            "score": self.score,
            "moves": self.moves,
            "is_game_over": self.is_game_over
        }
        
    def addTile(self):
        zeros = [(i,j) for i in range(len(self.gameboard)) for j in range(len(self.gameboard[i])) if self.gameboard[i][j] == 0]
        if zeros:
            i, j = choice(zeros)
            self.gameboard[i][j] = 2 if random() < 0.9 else 4
        else:
            self.is_game_over = True
    
    def move(self, direction):
        if self.is_game_over:
            return

        match(direction):
            case "up":
                self.up()
            case "down":
                self.down()
            case "left":
                self.left()
            case "right":
                self.right()

        self.moves += 1

        # Implement the logic for moving tiles in the specified direction
        # Update self.gameboard, self.score, and self.moves accordingly
        
    def up(self):
        added_score = 0
        cols = len(self.gameboard[0])
        for col in range(cols):
            rows = len(self.gameboard)
            trailPtr = -1
            for row in range(rows):
                if self.gameboard[row][col] != 0:
                    if trailPtr == -1:
                        if row != 0:
                            self.gameboard[0][col] = self.gameboard[row][col]
                            self.gameboard[row][col] = 0
                        trailPtr = 0
                    else:
                        if self.gameboard[trailPtr][col] == 0:
                            self.gameboard[trailPtr][col] = self.gameboard[row][col]
                            self.gameboard[row][col] = 0
                        elif self.gameboard[trailPtr][col] == self.gameboard[row][col]:
                            self.gameboard[trailPtr][col] *= 2
                            added_score += self.gameboard[trailPtr][col]
                            self.gameboard[row][col] = 0
                            trailPtr += 1
                        else:
                            trailPtr += 1
                            if trailPtr != row:
                                self.gameboard[trailPtr][col] = self.gameboard[row][col]
                                self.gameboard[row][col] = 0
        self.addTile()
        self.score += added_score + math.floor(self.moves / 10) * 10
        
    def down(self):
        self.gameboard.reverse()
        self.up()
        self.gameboard.reverse()
    
    def transpose(self):
        self.gameboard = [list(row) for row in zip(*self.gameboard)]

    def left(self):
        self.transpose()
        self.up()
        self.transpose()
    
    def right(self):
        self.transpose()
        self.down()
        self.transpose()
        
if __name__ == "__main__":
    game = GameEngine()
    game.addTile()
    game.addTile()
    print(game)
    print("************************")

