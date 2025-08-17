import numpy as np
import random
import math
import copy
from typing import Dict, List, Tuple, Optional, Union

# Constants
# RANDOM_SEED = 42
# random.seed(RANDOM_SEED)
# np.random.seed(RANDOM_SEED)

IS_DEV = False

def log(*args, **kwargs):
    if IS_DEV:
        print(*args, **kwargs)

# Global weights for expectimax
weights = {
    'merges': 0.0,
    'emptiness': 1.0,
    'cluster': 1.0,
    'monotonicity': 1.0
}

def set_expecti_max_weights(new_weights: Dict[str, float]) -> None:
    """Update the weights for expectimax algorithm"""
    global weights
    weights.update(new_weights)

def move(direction: str, gameboard: List[List[int]]) -> Dict[str, Union[int, List[List[int]]]]:
    """
    Make a move in the specified direction
    Returns dict with 'score' and 'gameboard' keys
    """
    board_copy = [row[:] for row in gameboard]
    
    if direction == "UP":
        result = move_up(board_copy)
    elif direction == "DOWN":
        result = move_down(board_copy)
    elif direction == "LEFT":
        result = move_left(board_copy)
    elif direction == "RIGHT":
        result = move_right(board_copy)
    else:
        result = {'score': 0, 'gameboard': board_copy}
    
    return result

def move_up(gameboard: List[List[int]]) -> Dict[str, Union[int, List[List[int]]]]:
    """Move all tiles up"""
    added_score = 0
    cols = len(gameboard[0])
    
    for col in range(cols):
        rows = len(gameboard)
        trail_ptr = -1
        
        for row in range(rows):
            if gameboard[row][col] != 0:
                # Initialize trail pointer
                if trail_ptr == -1:
                    if row != 0:
                        gameboard[0][col] = gameboard[row][col]
                        gameboard[row][col] = 0
                    trail_ptr = 0
                else:
                    # Remove spaces and move tiles up
                    if gameboard[trail_ptr][col] == 0:
                        gameboard[trail_ptr][col] = gameboard[row][col]
                        gameboard[row][col] = 0
                    # Merge tiles if they're the same
                    elif gameboard[trail_ptr][col] == gameboard[row][col]:
                        gameboard[trail_ptr][col] *= 2
                        added_score += gameboard[trail_ptr][col]
                        gameboard[row][col] = 0
                        trail_ptr += 1
                    # Move pointer down
                    else:
                        trail_ptr += 1
                        if trail_ptr != row:
                            gameboard[trail_ptr][col] = gameboard[row][col]
                            gameboard[row][col] = 0
    
    log("This move merge score:", added_score)
    return {'score': added_score, 'gameboard': gameboard}

def transpose(gameboard: List[List[int]]) -> List[List[int]]:
    """Transpose the game board"""
    return [[gameboard[row][col] for row in range(len(gameboard))] 
            for col in range(len(gameboard[0]))]

def move_down(gameboard: List[List[int]]) -> Dict[str, Union[int, List[List[int]]]]:
    """Move all tiles down"""
    gameboard.reverse()
    result = move_up(gameboard)
    gameboard.reverse()
    return result

def move_left(gameboard: List[List[int]]) -> Dict[str, Union[int, List[List[int]]]]:
    """Move all tiles left"""
    transposed = transpose(gameboard)
    result = move_up(transposed)
    result['gameboard'] = transpose(result['gameboard'])
    return result

def move_right(gameboard: List[List[int]]) -> Dict[str, Union[int, List[List[int]]]]:
    """Move all tiles right"""
    transposed = transpose(gameboard)
    result = move_down(transposed)
    result['gameboard'] = transpose(result['gameboard'])
    return result

def get_move(gameboard: List[List[int]], model: str) -> Dict[str, str]:
    """
    Get the best move using the specified model
    """
    print(f"Getting move using model: {model}")
    
    if model == "maximizeScore":
        return maximize_score(gameboard)
    elif model == "maximizeMerges":
        return maximize_merges(gameboard)
    elif model == "clusterTiles":
        return maximize_closeness(gameboard)
    elif model == "monotonicity":
        return maximize_monotonicity(gameboard)
    elif model == "expectiMax":
        result = expecti_max(gameboard, 2, True)
        return result if result.get('direction') else {'direction': 'UP'}
    elif model == "neuralNet":
        # Placeholder for neural net implementation
        return {'direction': random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT'])}
    elif model == "pythonQ":
        # Placeholder for Q-learning implementation
        return {'direction': random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT'])}
    else:
        raise ValueError(f"Unknown model: {model}")

def count_non_zero(board: List[List[int]]) -> int:
    """Count non-zero tiles"""
    return sum(1 for row in board for val in row if val != 0)

def normalized_emptiness(gameboard: List[List[int]]) -> float:
    """Calculate normalized emptiness score"""
    current = count_non_zero(gameboard)
    max_empty = len(gameboard) * len(gameboard[0])
    return current / max_empty if max_empty > 0 else 0

def maximize_score(gameboard: List[List[int]]) -> Dict[str, str]:
    """Find move that maximizes immediate score"""
    best_move = None
    best_score = 0
    
    for direction in ["UP", "DOWN", "LEFT", "RIGHT"]:
        result = move(direction, gameboard)
        simulated_score = result['score']
        
        if simulated_score > best_score:
            best_score = simulated_score
            best_move = direction
        
        log(f"Simulated move {direction}: Score = {simulated_score}")
    
    return {'direction': best_move or random.choice(["UP", "DOWN", "LEFT", "RIGHT"])}

def maximize_merges(gameboard: List[List[int]]) -> Dict[str, str]:
    """Find move that maximizes number of merges"""
    best_move = None
    max_merges = 0
    
    for direction in ["UP", "DOWN", "LEFT", "RIGHT"]:
        result = move(direction, gameboard)
        
        log("Input Gameboard:", gameboard)
        log("Result Gameboard:", result['gameboard'])
        
        merges = count_non_zero(gameboard) - count_non_zero(result['gameboard'])
        
        if merges > max_merges:
            max_merges = merges
            best_move = direction
        
        log(f"Simulated move {direction}: Merges = {merges}")
    
    return {'direction': best_move or random.choice(["UP", "DOWN", "LEFT", "RIGHT"])}

def count_cluster_score(board: List[List[int]]) -> float:
    """Calculate clustering score for the board"""
    score = 0
    rows = len(board)
    cols = len(board[0])
    
    for r in range(rows):
        for c in range(cols):
            if board[r][c] == 0:
                continue
            
            val = math.log2(board[r][c])
            
            # Right
            if c + 1 < cols and board[r][c + 1] == board[r][c]:
                score += val
            
            # Down
            if r + 1 < rows and board[r + 1][c] == board[r][c]:
                score += val
            
            # Diagonal down-right
            if r + 1 < rows and c + 1 < cols and board[r + 1][c + 1] == board[r][c]:
                score += val * 0.5
            
            # Diagonal down-left
            if r + 1 < rows and c - 1 >= 0 and board[r + 1][c - 1] == board[r][c]:
                score += val * 0.5
    
    return score

def build_ideal_cluster_board(gameboard: List[List[int]]) -> List[List[int]]:
    """Build ideal clustering board for normalization"""
    rows = len(gameboard)
    cols = len(gameboard[0])
    
    # Count tiles
    counts = {}
    for row in gameboard:
        for val in row:
            if val != 0:
                counts[val] = counts.get(val, 0) + 1
    
    # Sort tile values high to low
    tile_values = sorted(counts.keys(), reverse=True)
    
    # Fill board grouping identical tiles together
    ideal = [[0] * cols for _ in range(rows)]
    r, c = 0, 0
    
    for val in tile_values:
        for _ in range(counts[val]):
            ideal[r][c] = val
            c += 1
            if c >= cols:
                c = 0
                r += 1
    
    return ideal

def normalized_cluster_score(board: List[List[int]]) -> float:
    """Calculate normalized cluster score"""
    current = count_cluster_score(board)
    ideal = build_ideal_cluster_board(board)
    max_cluster = count_cluster_score(ideal)
    return current / max_cluster if max_cluster > 0 else 0

def maximize_closeness(gameboard: List[List[int]]) -> Dict[str, str]:
    """Find move that maximizes tile clustering"""
    best_move = None
    max_cluster_score = 0
    
    for direction in ["UP", "DOWN", "LEFT", "RIGHT"]:
        result = move(direction, gameboard)
        log("Result Gameboard:", result['gameboard'])
        
        cluster_score = count_cluster_score(result['gameboard'])
        
        if cluster_score > max_cluster_score:
            max_cluster_score = cluster_score
            best_move = direction
        
        log(f"Simulated move {direction}: Cluster Score = {cluster_score}")
    
    return {'direction': best_move or random.choice(["UP", "DOWN", "LEFT", "RIGHT"])}

def calculate_monotonicity(gameboard: List[List[int]]) -> float:
    """Calculate monotonicity score"""
    rows = len(gameboard)
    cols = len(gameboard[0])
    score = 0
    
    for row in range(rows):
        for col in range(cols):
            if gameboard[row][col] != 0:
                current_tile = math.log2(gameboard[row][col])
                
                # Looking right
                if col + 1 < cols:
                    next_tile = math.log2(gameboard[row][col + 1]) if gameboard[row][col + 1] != 0 else 0
                    
                    if next_tile <= current_tile:
                        score += current_tile
                    else:
                        score -= current_tile
                
                # Looking down
                if row + 1 < rows:
                    next_tile = math.log2(gameboard[row + 1][col]) if gameboard[row + 1][col] != 0 else 0
                    
                    if next_tile <= current_tile:
                        score += current_tile
                    else:
                        score -= current_tile
    
    return score

def build_ideal_board(board: List[List[int]]) -> List[List[int]]:
    """Build ideal monotonic board for normalization"""
    rows = len(board)
    cols = len(board[0])
    tiles = sorted([val for row in board for val in row if val != 0], reverse=True)
    
    ideal = [[0] * cols for _ in range(rows)]
    idx = 0
    
    for r in range(rows):
        for c in range(cols):
            if idx < len(tiles):
                ideal[r][c] = tiles[idx]
                idx += 1
    
    return ideal

def normalized_monotonicity(gameboard: List[List[int]]) -> float:
    """Calculate normalized monotonicity score"""
    current = calculate_monotonicity(gameboard)
    ideal = build_ideal_board(gameboard)
    max_mono = calculate_monotonicity(ideal)
    
    if max_mono == 0:
        return 0
    
    return max(0, current / max_mono)

def maximize_monotonicity(gameboard: List[List[int]]) -> Dict[str, str]:
    """Find move that maximizes monotonicity"""
    best_move = None
    max_monotonicity_score = 0
    
    for direction in ["UP", "DOWN", "LEFT", "RIGHT"]:
        result = move(direction, gameboard)
        log("Result Gameboard:", result['gameboard'])
        
        monotonicity_score = calculate_monotonicity(result['gameboard'])
        
        if monotonicity_score > max_monotonicity_score:
            max_monotonicity_score = monotonicity_score
            best_move = direction
        
        log(f"Simulated move {direction}: Monotonicity Score = {monotonicity_score}")
    
    return {'direction': best_move or random.choice(["UP", "DOWN", "LEFT", "RIGHT"])}

def normalized_merges(old_board: List[List[int]], new_board: List[List[int]]) -> float:
    """Calculate normalized merge score"""
    merges = count_non_zero(old_board) - count_non_zero(new_board)
    max_possible = count_non_zero(old_board) // 2
    return merges / max_possible if max_possible > 0 else 0

def maximize_weighted_score(gameboard: List[List[int]]) -> Dict[str, Union[str, float]]:
    """Find move that maximizes weighted combination of heuristics"""
    max_overall_score = 0
    best_move = None
    
    for direction in ["UP", "DOWN", "LEFT", "RIGHT"]:
        result = move(direction, gameboard)
        
        log("Result Gameboard:", result['gameboard'])
        
        merges = normalized_merges(gameboard, result['gameboard'])
        cluster_score = normalized_cluster_score(result['gameboard'])
        monotonicity_score = normalized_monotonicity(result['gameboard'])
        emptiness_score = normalized_emptiness(result['gameboard'])
        
        overall_score = (merges * weights['merges'] + 
                        cluster_score * weights['cluster'] + 
                        monotonicity_score * weights['monotonicity'] + 
                        emptiness_score * weights['emptiness'])
        
        if overall_score > max_overall_score:
            max_overall_score = overall_score
            best_move = direction
        
        log(f"Simulated move {direction}: Merges = {merges}, "
            f"Cluster Score = {cluster_score}, Monotonicity Score = {monotonicity_score}")
        log(f"Overall Score = {overall_score}")
    
    return {
        'direction': best_move or random.choice(["UP", "DOWN", "LEFT", "RIGHT"]),
        'score': max_overall_score
    }

def expecti_max(board: List[List[int]], depth: int, is_max_node: bool) -> Union[float, Dict[str, Union[str, float]]]:
    """ExpectiMax algorithm implementation"""
    # Leaf node
    if depth == 0:
        return maximize_weighted_score(board)['score']
    
    # Max node - try all 4 moves
    if is_max_node:
        best_move = None
        max_heuristic_score = -float('inf')
        
        for direction in ["UP", "DOWN", "LEFT", "RIGHT"]:
            board_post_move = move(direction, board)['gameboard']
            score = expecti_max(board_post_move, depth - 1, False)
            
            if score is None:
                continue
            
            if score > max_heuristic_score:
                max_heuristic_score = score
                best_move = direction
        
        return {'direction': best_move, 'score': max_heuristic_score}
    
    # Chance node - simulate random tile placement
    else:
        empty_cells = []
        for r in range(len(board)):
            for c in range(len(board[0])):
                if board[r][c] == 0:
                    empty_cells.append((r, c))
        
        if not empty_cells:
            return None
        
        score = 0
        
        for r, c in empty_cells:
            # Simulate placing a 2
            new_2s_board = [row[:] for row in board]
            new_2s_board[r][c] = 2
            result = expecti_max(new_2s_board, depth, True)
            if result is not None:
                score += 0.9 / len(empty_cells) * result['score']
            
            # Simulate placing a 4
            new_4s_board = [row[:] for row in board]
            new_4s_board[r][c] = 4
            result = expecti_max(new_4s_board, depth, True)
            if result is not None:
                score += 0.1 / len(empty_cells) * result['score']
        
        return score

# Example usage and testing
if __name__ == "__main__":
    IS_DEV = False
    
    # Test gameboard
    gameboard = [
        [16, 8, 4, 2],
        [16, 8, 4, 2],
        [16, 8, 4, 2],
        [16, 8, 4, 2]
    ]
    
    best_move = get_move(gameboard, "expectiMax")
    print("Best Move:", best_move)