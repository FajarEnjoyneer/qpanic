import chess
import chess.pgn
import numpy as np
from typing import List, Tuple, Optional
from src.config import USERNAME
from src.encoder import encode_board

def process_pgn_file(filepath: str) -> List[Tuple[np.ndarray, float]]:
    """
    Reads a PGN file and extracts training data from all games within.
    Returns a list of (board_tensor, label) tuples.
    """
    training_data = []
    
    with open(filepath, 'r') as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            
            game_data = process_game(game)
            if game_data:
                training_data.extend(game_data)
                
    return training_data

def process_game(game: chess.pgn.Game) -> Optional[List[Tuple[np.ndarray, float]]]:
    """
    Process a single game.
    Returns a list of (board_tensor, label) tuples, or None if qpanic not found/invalid.
    """
    headers = game.headers
    white_player = headers.get("White", "")
    black_player = headers.get("Black", "")
    result_str = headers.get("Result", "*")

    if USERNAME not in white_player and USERNAME not in black_player:
        return None

    # Determine result from White's perspective initially
    if result_str == "1-0":
        game_result = 1.0
    elif result_str == "0-1":
        game_result = -1.0
    elif result_str == "1/2-1/2":
        game_result = 0.0
    else:
        return None  # Unknown result or still in progress

    # Determine qpanic's color and perspective multiplier
    # If qpanic is White (multiplier 1): Win(1) -> 1, Loss(-1) -> -1
    # If qpanic is Black (multiplier -1): Win(1=White wins) -> -1 (Loss for Black), Loss(-1=Black wins) -> 1 (Win for Black)
    if USERNAME in white_player:
        perspective_multiplier = 1.0
    else:
        perspective_multiplier = -1.0
    
    final_score = game_result * perspective_multiplier

    # Count total plies to calculate progressive reward
    # We can iterate once to count, or just store all positions and post-process
    # Let's verify legality and extract moves first
    board = game.board()
    boards_and_indices = [] # Stores (board_copy, ply_index)
    
    ply_index = 0
    for move in game.mainline_moves():
        if not board.is_legal(move):
            break # Stop if we encounter an illegal move (shouldn't happen in valid PGNs)
        
        # We want to evaluate the position *before* the move? 
        # Usually evaluation is for the position on the board.
        # If it's White to move, we eval White's position. 
        # If it's Black to move, we eval Black's position.
        # But wait, the label is "final result".
        # So we just encode the current board state.
        
        # NOTE: User requirement: "Extract per-position data"
        # We capture the board BEFORE the move is made.
        boards_and_indices.append((board.copy(), ply_index))
        
        board.push(move)
        ply_index += 1
    
    # Also include the final position? Usually yes.
    boards_and_indices.append((board.copy(), ply_index))
    
    total_plies = ply_index # The last index is effectively the total count if we strictly count moves made
    if total_plies == 0:
        return []

    game_samples = []
    
    for board_state, current_ply in boards_and_indices:
        # User Rule: label = final_result * (ply_index / total_plies)
        # Note: avoid division by zero if total_plies is somehow 0 (checked above)
        
        # We usually don't embrace the *very first* position (starting board) with 0 weight if ply is 0?
        # Actually ply 0 / total -> 0 label. Neutral start. Makes sense.
        
        weight = current_ply / total_plies
        label = final_score * weight
        
        feature = encode_board(board_state)
        game_samples.append((feature, label))

    return game_samples
