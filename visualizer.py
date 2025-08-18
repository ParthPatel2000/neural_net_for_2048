import pygame
import sys, signal
from game_engine import GameEngine
from heuristics_ai import get_move
from tabular_qlearning import QLearningAgent
import deep_Q_learning

pygame.init()

tile_size = 100
padding = 10

# Initialize game
game = GameEngine(5, 4)
game.addTile()
game.addTile()

rows = len(game.gameboard)
cols = len(game.gameboard[0])

width = cols * (tile_size + padding) + padding
height = rows * (tile_size + padding) + padding

screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("2048 Visualizer")
font = pygame.font.Font(None, 40)

colors = {
    0: (205, 193, 180),
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46),
}


def draw_board(board):
    screen.fill((187, 173, 160))
    rows = len(board)
    cols = len(board[0])
    for y in range(rows):
        for x in range(cols):
            value = board[y][x]
            rect_x = padding + x * (tile_size + padding)
            rect_y = padding + y * (tile_size + padding)
            pygame.draw.rect(
                screen,
                colors.get(value, (60, 58, 50)),
                (rect_x, rect_y, tile_size, tile_size),
                border_radius=8,
            )
            if value != 0:
                text_surface = font.render(str(value), True, (0, 0, 0))
                text_rect = text_surface.get_rect(
                    center=(rect_x + tile_size // 2, rect_y + tile_size // 2)
                )
                screen.blit(text_surface, text_rect)


# flag for graceful shutdown
interrupted = False

def handle_interrupt(sig, frame):
    global interrupted
    print("Ctrl+C detected. Will exit after this game finishes.")
    interrupted = True

signal.signal(signal.SIGINT, handle_interrupt)
signal.signal(signal.SIGTERM, handle_interrupt)

agent = QLearningAgent(actions=["UP", "DOWN", "LEFT", "RIGHT"])

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            agent.save()
            pygame.quit()
            sys.exit()

    state = game.gameboard

    if not game.gameState()["is_game_over"]:
        # Q-learning chooses action
        action = agent.choose_action(state)
        prev_score = game.score
        game.move(action.lower())  # engine uses lowercase
        reward = game.score - prev_score
        agent.learn(state, action, reward, game.gameboard)
        print("AI Move:", action, "score:", game.score)
    else:
        print("Game Over!")
        agent.save()
        if interrupted:  # only quit if Ctrl+C was pressed
            running = False
        else:
            # restart new game
            game = GameEngine(5, 4)
            game.addTile()
            game.addTile()

    draw_board(game.gameboard)
    pygame.display.flip()
    pygame.time.delay(100)

pygame.quit()
sys.exit()



# while running:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             pygame.quit()
#             sys.exit()

#     # --- AI move step ---
#     if not game.gameState()["is_game_over"]:
#         best_move = get_move(game.gameboard, "clusterTiles")  # or any model
#         game.move(best_move["direction"].lower())  # execute AI move
#         print("AI Move:", best_move["direction"])
#     else:
#         print("Game Over!")
#         draw_board(game.gameboard)  # draw final board first
#         text = font.render("GAME OVER", True, (0, 0, 0))  # black text
#         text_rect = text.get_rect(
#             center=(screen.get_width() // 2, screen.get_height() // 2)
#         )  # adjust to window center
#         screen.blit(text, text_rect)
#         pygame.display.flip()
#         pygame.time.delay(2000)  # pause 2 seconds
#         running = False

#     draw_board(game.gameboard)
#     pygame.display.flip()
#     pygame.time.delay(100)  # slow down for visualization
