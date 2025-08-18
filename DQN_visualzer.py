import numpy as np
import pygame, sys, signal, os
import torch
import torch.optim as optim
from game_engine import GameEngine, board_to_tensor
from DQN import (
    DQN2048,
    ReplayBuffer,
    select_action,
    train_step,
    load_buffer,
    load_model,
    save_buffer,
    save_model,
)
import pickle
from collections import deque

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


# -------------Global setups vars-------------
model_path = "DQN_model/dqn2048.pth"
buffer_path = "DQN_model/replay_buffer.pkl"
# ----------- Hyperparameters and Config -----------
BOARD_ROWS = 5
BOARD_COLS = 4
MODEL_PATH = "DQN_model/dqn2048.pth"
BUFFER_PATH = "DQN_model/replay_buffer.pkl"
REPLAY_BUFFER_SIZE = 10000
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.3
EPSILON_DECAY = 0.995
TARGET_UPDATE_FREQ = 50
SAVE_FREQ = 500
TILE_SIZE = 100
PADDING = 10

# Use these variables below in the rest of the code
tile_size = TILE_SIZE
padding = PADDING
model_path = MODEL_PATH
buffer_path = BUFFER_PATH

# ------------------ Graceful shutdown ------------------
interrupted = False


def handle_interrupt(sig, frame):
    global interrupted
    print("Ctrl+C detected. Saving model and buffer before exit...")
    print("Saved successfully. Exiting.")
    interrupted = True


signal.signal(signal.SIGINT, handle_interrupt)
signal.signal(signal.SIGTERM, handle_interrupt)

# ------------------ DQN setup ------------------
num_actions = 4
net = DQN2048(num_actions=num_actions)
target_net = DQN2048(num_actions=num_actions)

# --- Load saved model if exists ---
if os.path.exists(model_path):
    net.load_state_dict(torch.load(model_path))
    net.eval()
    target_net.load_state_dict(net.state_dict())
    target_net.eval()
    print("Loaded saved DQN model.")
else:
    target_net.load_state_dict(net.state_dict())
    target_net.eval()

optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
buffer = ReplayBuffer(10000)

# --- Load replay buffer if exists ---
if os.path.exists(buffer_path):
    load_buffer(buffer, buffer_path)
    print("Loaded replay buffer.")

epsilon = EPSILON_START
epsilon_min = EPSILON_MIN
epsilon_decay = EPSILON_DECAY
target_update_freq = TARGET_UPDATE_FREQ  # frames
action_map = {0: "up", 1: "down", 2: "left", 3: "right"}

# ------------------- Print Helper -------------------
import sys


def print_model_stats(
    last_game_score, last_100_median, last_100_avg, current_game, epsilon_decay
):
    sys.stdout.write("\033[F\033[K")  # move up, clear line
    sys.stdout.write("\033[F\033[K")  # move up again, clear
    sys.stdout.write("\033[F\033[K")  # move up again, clear
    sys.stdout.write("\033[F\033[K")  # move up again, clear
    sys.stdout.write(f"Epsilon Decay: {epsilon_decay}\n")
    sys.stdout.write(f"Game Finished with Score: {last_game_score}\n")
    sys.stdout.write(
        f"median Score (last 100 games): {last_100_median} and Average Score: {last_100_avg}\n"
    )
    sys.stdout.write(f"Current Game: {current_game}\n")
    sys.stdout.flush()


# ------------------ Initialize game ------------------
game = GameEngine(5, 4)
game.addTile()
game.addTile()

frame_count = 0
games_count = 0
games_score = deque(maxlen=100)
median_score = 0
avg_score = 0

# Flag for conditional rendering
render_next_game = False

print("\n\n\n")
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_v:  # Press "v" to visualize the next game
                if render_next_game:
                    render_next_game = False
                else:
                    render_next_game = True

    if not game.is_game_over:
        # --- DQN selects action ---
        state_tensor = board_to_tensor(game.gameboard)
        action_idx = select_action(state_tensor, net, epsilon)
        direction = action_map[action_idx]

        # --- Take action ---
        prev_score = game.score
        game.move(direction)
        reward = game.score - prev_score

        next_state_tensor = board_to_tensor(game.gameboard)
        done = game.is_game_over

        # --- Store experience ---
        buffer.push(state_tensor, action_idx, reward, next_state_tensor, done)

        # --- Train step ---
        train_step(net, target_net, buffer, optimizer, batch_size=32, gamma=0.99)

        # # --- Epsilon decay ---
        # epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # --- Target network update ---
        frame_count += 1
        if frame_count % target_update_freq == 0:
            target_net.load_state_dict(net.state_dict())

    else:
        games_count += 1
        games_score.append(game.score)
        median_score = np.median(games_score)
        avg_score = np.mean(games_score)
        print_model_stats(game.score, median_score, avg_score, games_count, epsilon)

        # Decay epsilon at the end of each game
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if interrupted:
            running = False
        else:
            # Restart new game
            game = GameEngine(5, 4)
            game.addTile()
            game.addTile()

            # Optionally force rendering for new game
            if render_next_game:
                draw_board(game.gameboard)
                pygame.display.flip()
                pygame.time.delay(500)  # small pause to see the start

    # --- Conditional rendering ---
    if render_next_game:
        draw_board(game.gameboard)
        pygame.display.flip()
        pygame.time.delay(50)

    if frame_count % 500 == 0:
        save_model(net, model_path, False)  # removing logging for less clutter
        save_buffer(buffer, buffer_path, False)  # removing logging for less clutter

save_model(net, model_path)  # Saving and logging, so that i know everything was saved
save_buffer(
    buffer, buffer_path
)  # Saving and logging, so that i know everything was saved
pygame.quit()
sys.exit()
