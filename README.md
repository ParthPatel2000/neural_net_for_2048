
# 2048 Game AI: Neural Network Project

This project demonstrates the step-by-step progression in building an AI for the game 2048, starting from simple neural network concepts and evolving to more complex solutions. Each stage highlights why increased complexity was necessary for better game performance.

## Project Progression

### 1. Basic Neural Network Training
- **Goal:** Understand how neural networks learn from data.
- **Implementation:** Simple regression/classification using PyTorch (`train.py`).
- **Why:** Foundation for all AI—teaches the basics of model training, loss functions, and optimization.

### 2. Data Preparation & Cleaning
- **Goal:** Prepare game data for training.
- **Implementation:** Scripts like `clean_data.py` convert raw game logs to usable formats.
- **Why:** Clean, structured data is essential for effective learning and generalization.

### 3. Increasing Model Complexity
- **Goal:** Improve AI's ability to play 2048.
- **Implementation:** Experiment with deeper networks, more layers, and advanced architectures.
- **Why:** Simple models can't capture the game's strategic depth. More complex models learn better policies and achieve higher scores.

### 4. Exporting & Integration
- **Goal:** Use the trained model in different environments (e.g., JavaScript for web-based 2048).
- **Implementation:** Export models to ONNX or TensorFlow.js formats for browser/JS use.
- **Why:** Enables real-time AI gameplay in web apps and cross-platform deployment.

### 5. Visualization & Analysis
- **Goal:** Understand and debug AI decisions.
- **Implementation:** Tools like `visualizer.py` to inspect moves and model predictions.
- **Why:** Visualization helps refine strategies and spot weaknesses in the AI.

## Project Structure
- `requirements.txt`: Python dependencies
- `train.py`: Main training script
- `clean_data.py`: Data cleaning utility
- `convert_existing_model_to_onnx.py`: Model export script
- `visualizer.py`: Visualization tool
- `raw_data/`, `processed_data/`: Data folders

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Prepare data:
   ```bash
   python clean_data.py
   ```
3. Train the model:
   ```bash
   python train.py
   ```
4. Export for web/JS (optional):
   ```bash
   python convert_existing_model_to_onnx.py
   ```
5. Visualize results:
   ```bash
   python visualizer.py
   ```


## Evolution of AI Architectures for 2048

### 1. Fully Connected (FC) Networks
- **Why Start Here:** FC networks are simple and easy to implement. They treat the board as a flat vector, allowing basic learning.
- **Limitation:** They ignore spatial relationships between tiles, so they struggle to learn advanced strategies.

### 2. Convolutional Neural Networks (CNN)
- **Why Move to CNN:** CNNs can capture spatial patterns and local interactions on the board, which are crucial for 2048.
- **Limitation:** While better than FC, CNNs still rely on supervised learning and may not learn optimal long-term strategies. (Reason- my heuristics were bad and i didnt want to spend too much time tweaking them when i can let neural nets build their own strats.)

### 3. Q-Learning with Tables
- **Why Try Q-Learning:** Q-learning introduces reinforcement learning, allowing the AI to learn from rewards and explore strategies.
- **Limitation:** Table-based Q-learning is only feasible for small state spaces. 2048 has too many possible board states for this approach to scale.

### 4. Deep Q-Networks (DQN)
- **Why Move to DQN:** DQN combines deep learning with Q-learning, using neural networks to approximate Q-values for large state spaces. This enables the AI to learn complex strategies and generalize across many board configurations.
- **Result:** DQN provides the best performance, balancing learning efficiency and strategic depth, making it ideal for mastering 2048.

Each transition was necessary to overcome the limitations of the previous approach and achieve higher scores and smarter gameplay in 2048.
