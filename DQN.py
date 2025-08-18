import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from game_engine import GameEngine
from collections import deque
import pickle


class DQN2048(nn.Module):
    def __init__(self, num_actions=4):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=2)  # 5x4 → 4x3
        self.conv2 = nn.Conv2d(16, 64, kernel_size=2)  # 4x3 → 3x2

        self.fc1 = nn.Linear(64 * 3 * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Store a transition in the buffer
        state, next_state: torch tensors
        action: int
        reward: float
        done: bool
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Randomly sample a batch of transitions
        Returns: tuple of tensors
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.cat(states)  # (batch, 1, 5, 4)
        next_states = torch.cat(next_states)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


def select_action(state, net, epsilon=0.1):
    """
    Choose an action using epsilon-greedy policy.
    state: torch tensor (1,1,5,4)
    net: DQN network
    epsilon: float, probability of choosing random action
    """
    if random.random() < epsilon:
        # Exploration: random action
        return random.randint(0, 3)  # 4 possible moves: 0,1,2,3
    else:
        # Exploitation: choose action with max Q-value
        with torch.no_grad():
            q_values = net(state)  # shape: (1,4)
            return torch.argmax(q_values).item()


def board_to_tensor(board):
    """
    Convert 5x4 2048 board to a PyTorch tensor.
    board: 2D list or numpy array
    returns: torch tensor of shape (1, 1, 5, 4)
    """
    board = np.array(board, dtype=np.float32)
    board = np.log2(
        np.clip(board, a_min=1, a_max=None)
    )  # replaces 0 with 1 before log2
    tensor = torch.tensor(board).unsqueeze(0).unsqueeze(0)  # shape: (1,1,5,4)
    return tensor


def train_step(net, target_net, replay_buffer, optimizer, batch_size=32, gamma=0.99):
    if len(replay_buffer) < batch_size:
        return  # not enough samples yet

    # Sample a batch
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    # Predicted Q-values for taken actions
    q_pred = net(states).gather(1, actions)  # shape: (batch_size,1)

    # Compute target Q-values
    with torch.no_grad():
        q_next = target_net(next_states).max(1)[0].unsqueeze(1)  # shape: (batch_size,1)
        q_target = rewards + gamma * q_next * (1 - dones)

    # Loss
    loss = F.mse_loss(q_pred, q_target)

    # Backpropagation
    optimizer.zero_grad()  # clear old gradients
    loss.backward()  # compute new gradients for current batch
    optimizer.step()  # update network weights

    return loss.item()


def save_model(net, path="models/dqn2048.pth", log=True):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(net.state_dict(), path)
    if log:
        print(f"Model saved to {path}")


def load_model(net, path="models/dqn2048.pth"):
    net.load_state_dict(torch.load(path))
    net.eval()
    print(f"Model loaded from {path}")


def save_buffer(buffer, path="buffers/replay_buffer.pkl", log=True):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Convert tensors to CPU numpy for serialization
    buffer_data = [(s.cpu(), a, r, ns.cpu(), d) for s, a, r, ns, d in buffer.buffer]
    with open(path, "wb") as f:
        pickle.dump(buffer_data, f)
    if log:
        print(f"Replay buffer saved to {path}")


def load_buffer(buffer, path="buffers/replay_buffer.pkl"):
    with open(path, "rb") as f:
        buffer_data = pickle.load(f)
    # Rebuild deque
    buffer.buffer = deque(buffer_data, maxlen=buffer.buffer.maxlen)
    print(f"Replay buffer loaded from {path}")


if __name__ == "__main__":
    game = GameEngine()
    game.addTile()
    game.addTile()
    board_tensor = board_to_tensor(game.gameboard)
    print("Game board tensor shape:", board_tensor.shape)
    print("Game board tensor:", board_tensor)

    ### NEURAL NETWORK Testing
    net = DQN2048()
    q_values = net(board_tensor)
    # print(q_values)  # shape: (1,4)
    # print(q_values.argmax(dim=1))  # best action

    # for i in range(5):
    #     action = select_action(board_tensor, net, epsilon=0.2)
    #     print(f"Selected action {i+1}: {action}")

    #### Replay Buffer Testing
    # buffer = ReplayBuffer(100)

    # # Example experience
    # state = board_tensor
    # next_state = board_tensor  # normally this would be the new board
    # buffer.push(state, 3, 1.0, next_state, False)
    # buffer.push(state, 3, 1.0, next_state, False)
    # buffer.push(state, 3, 1.0, next_state, False)
    # buffer.push(state, 3, 1.0, next_state, False)

    # print(len(buffer))  # 4
    # s, a, r, ns, d = buffer.sample(4)
    # print("state:", s.shape, "\naction:", a, "\nreward:", r, "\nnext_state:", ns.shape, "\ndone:", d)

    #### DQN TESTING
    # Hyperparameters
    # num_episodes = 1000
    # batch_size = 32
    # gamma = 0.99
    # epsilon = 1.0
    # epsilon_min = 0.1
    # epsilon_decay = 0.995
    # target_update_freq = 10  # episodes

    # # Initialize networks
    # net = DQN2048(num_actions=4)
    # target_net = DQN2048(num_actions=4)
    # target_net.load_state_dict(net.state_dict())
    # target_net.eval()

    # # Optimizer
    # optimizer = optim.Adam(net.parameters(), lr=1e-3)

    # # Replay buffer
    # buffer = ReplayBuffer(10000)

    # # Map integers to directions
    # action_map = {0: "up", 1: "down", 2: "left", 3: "right"}

    # # Training loop
    # for episode in range(num_episodes):
    #     game = GameEngine()
    #     game.addTile()
    #     game.addTile()
    #     state = board_to_tensor(game.gameboard)

    #     done = False
    #     total_reward = 0

    #     while not game.is_game_over:
    #         # 1. Epsilon-greedy action
    #         if torch.rand(1).item() < epsilon:
    #             action_idx = torch.randint(0, 4, (1,)).item()
    #         else:
    #             with torch.no_grad():
    #                 q_values = net(state)
    #                 action_idx = torch.argmax(q_values).item()

    #         direction = action_map[action_idx]

    #         # 2. Take action
    #         prev_score = game.score
    #         game.move(direction)
    #         reward = game.score - prev_score  # reward = score difference
    #         total_reward += reward

    #         next_state = board_to_tensor(game.gameboard)
    #         done = game.is_game_over

    #         # 3. Store transition
    #         buffer.push(state, action_idx, reward, next_state, done)

    #         # 4. Train step
    #         if len(buffer) >= batch_size:
    #             # Sample minibatch
    #             states, actions, rewards, next_states, dones = buffer.sample(batch_size)

    #             # Predicted Q
    #             q_pred = net(states).gather(1, actions)

    #             # Target Q
    #             with torch.no_grad():
    #                 q_next = target_net(next_states).max(1)[0].unsqueeze(1)
    #                 q_target = rewards + gamma * q_next * (1 - dones)

    #             # Loss and backprop
    #             loss = F.mse_loss(q_pred, q_target)
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()

    #         state = next_state

    #     # 5. Epsilon decay
    #     epsilon = max(epsilon_min, epsilon * epsilon_decay)

    #     # 6. Target network update
    #     if episode % target_update_freq == 0:
    #         target_net.load_state_dict(net.state_dict())

    #     print(f"Episode {episode}, Total reward: {total_reward}, Epsilon: {epsilon:.3f}")
