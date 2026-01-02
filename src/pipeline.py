import os
import glob
import numpy as np
from typing import List, Dict
import random
from src.config import DATA_DIR, GAME_DIR, PROCESSED_DATA_FILE, TRAIN_SPLIT_RATIO
from src.parser import process_pgn_file
import chess.pgn
from src.parser import process_game

def run_pipeline():
    """
    Main pipeline execution:
    1. Scan PGNs
    2. Process games
    3. Split data (by game)
    4. Save to disk
    """
    print(f"Scanning for PGN files in {GAME_DIR}...")
    pgn_files = glob.glob(os.path.join(GAME_DIR, "**/*.pgn"), recursive=True)
    
    if not pgn_files:
        print("No PGN files found!")
        return

    all_games_data: List[List] = [] 
    
    total_positions = 0
    valid_games_count = 0

    print(f"Found {len(pgn_files)} files. Processing...")
    
    for pgn_path in pgn_files:
         with open(pgn_path, 'r') as pgn_file:
            while True:
                try:
                    game = chess.pgn.read_game(pgn_file)
                except Exception as e:
                    print(f"Error reading game in {pgn_path}: {e}")
                    continue
                
                if game is None:
                    break
                
                game_samples = process_game(game)
                if game_samples:
                    all_games_data.append(game_samples)
                    valid_games_count += 1
                    total_positions += len(game_samples)

    print(f"Processed {valid_games_count} valid games with {total_positions} total positions.")

    if not all_games_data:
        print("No valid games processed.")
        return

    # Shuffle games for random split
    random.shuffle(all_games_data)

    split_idx = int(len(all_games_data) * TRAIN_SPLIT_RATIO)
    train_games = all_games_data[:split_idx]
    val_games = all_games_data[split_idx:]

    # Flatten for final dataset
    def flatten(games_list):
        X_list = []
        y_list = []
        for game_samples in games_list:
            for (board_tensor, label) in game_samples:
                X_list.append(board_tensor)
                y_list.append(label)
        
        if not X_list:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
            
        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

    print("Flattening and converting to numpy arrays...")
    train_X, train_y = flatten(train_games)
    val_X, val_y = flatten(val_games)

    print(f"Training set: {train_X.shape[0]} samples")
    print(f"Validation set: {val_X.shape[0]} samples")

    # Create output directory
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # Change extension to .npz
    output_path = os.path.join(DATA_DIR, "processed_dataset.npz")
    
    # Save as compressed numpy
    np.savez_compressed(
        output_path,
        train_X=train_X,
        train_y=train_y,
        val_X=val_X,
        val_y=val_y
    )

    print(f"Dataset saved to {output_path}")

if __name__ == "__main__":
    run_pipeline()
