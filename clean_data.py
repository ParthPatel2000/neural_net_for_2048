import os
import json
import math

data_location = "./raw_data/"
filename = "training_data.json"

outputLocation = "./processed_data/"
jsonObj = None


with open(os.path.join(data_location, filename), "r") as json_file:
    jsonObj = json.load(json_file)

direction_map = {
    "UP": 0,
    "DOWN": 1,
    "LEFT": 2,
    "RIGHT": 3
}

with open(os.path.join(outputLocation, filename.strip(".json") + ".csv"), "w") as f:
    count = 0
    for game in jsonObj:
        for move in game:
            normalized_board = [
                0 if val == 0 else int(math.log2(val)) for val in move["board"]]
            board_vals = ",".join(str(x) for x in normalized_board)
            direction = direction_map[move["direction"]]
            f.write(f"{board_vals},{direction}\n")
