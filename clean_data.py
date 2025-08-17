import os
import      json
import math

data_location = "./raw_data/"
filename = "training_data.json"

outputLocation = "./processed_data/"
jsonObj = []


with open("raw_data/test.ndjson", "r") as f:
    for line in f:
        game = json.loads(line)  # each line is one game's move history
        jsonObj.append(game)

print(jsonObj[0])  # print first game for sanity check

direction_map = {
    "UP": 0,
    "DOWN": 1,
    "LEFT": 2,
    "RIGHT": 3
}

with open(os.path.join(outputLocation, filename.strip(".json") + "_log2scale.csv"), "w") as f:
    games_kept = 0 
    games_distribution = {}
    for game in jsonObj:
        games_distribution[math.ceil(len(game)/100)] = games_distribution.get(math.ceil(len(game)/100), 0) + 1        
        if(math.ceil(len(game)/100) > 5):
            for move in game:
                normalized_board = [
                    0 if val == 0 else int(math.log2(val)) for val in move["board"]]
                board_vals = ",".join(str(x) for x in normalized_board)
                direction = direction_map[move["direction"]]
                f.write(f"{board_vals},{direction}\n")
            games_kept += 1

    print("Game Distribution(X100):", games_distribution)
    print("Total games Processed:", len(jsonObj))
    print("games Kept:", games_kept)
    print("Data cleaned and saved to CSV.")