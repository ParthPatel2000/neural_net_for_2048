import signal
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
import random

app = Flask(__name__)
CORS(app)

directions = ["UP", "DOWN", "LEFT", "RIGHT"]

@app.route("/get_move", methods=["POST"])
def get_move():
    move = random.choice(directions)
    return jsonify({"direction": move})

def save_and_exit(signal_received, frame):
    print("SIGINT received. Saving checkpoint...")
    # Save your model and replay buffer here
    sys.exit(0)

# Register signal handler BEFORE running Flask
signal.signal(signal.SIGINT, save_and_exit)

if __name__ == "__main__":
    app.run(port=5000)


# from flask import Flask, request, jsonify
# import numpy as np
# import pickle

# app = Flask(__name__)

# # Load your Q-table or trained model
# with open("q_table.pkl", "rb") as f:
#     q_table = pickle.load(f)

# def board_to_state_key(board):
#     # Convert 2D board to a single string key
#     return str(np.array(board).flatten().tolist())

# def choose_best_move(state_key):
#     # Map index 0-3 to directions
#     directions = ["UP", "DOWN", "LEFT", "RIGHT"]
#     if state_key in q_table:
#         best_action = np.argmax(q_table[state_key])
#         return directions[best_action]
#     else:
#         return np.random.choice(directions)  # fallback

# @app.route("/get_move", methods=["POST"])
# def get_move():
#     board = request.json["board"]
#     state_key = board_to_state_key(board)
#     move = choose_best_move(state_key)
#     return jsonify({"move": move})

# if __name__ == "__main__":
#     app.run(port=5000)
