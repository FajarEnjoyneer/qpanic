import chess
import numpy as np
from src.config import TENSOR_SHAPE

def encode_board(board: chess.Board) -> np.ndarray:
    """
    Encodes a python-chess board object into an (8, 8, 13) numpy array.
    Channels:
    0-5: White Pawn, Knight, Bishop, Rook, Queen, King
    6-11: Black Pawn, Knight, Bishop, Rook, Queen, King
    12: Side to move (1 for White, 0 for Black)
    """
    # Initialize 8x8x13 tensor with zeros
    tensor = np.zeros(TENSOR_SHAPE, dtype=np.float32)

    # Piece mapping to channel indices
    # White pieces: 1-6 -> channels 0-5
    # Black pieces: 1-6 -> channels 6-11
    piece_map = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5
    }

    # Iterate over all 64 squares
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            # Determine channel offset: 0 for White, 6 for Black
            offset = 0 if piece.color == chess.WHITE else 6
            channel = offset + piece_map[piece.piece_type]
            
            # Map square (0-63) to (row, col)
            # rank_index: 0 (rank 1) to 7 (rank 8)
            # file_index: 0 (file a) to 7 (file h)
            row = chess.square_rank(square)
            col = chess.square_file(square)
            
            tensor[row, col, channel] = 1.0

    # Channel 12: Side to move
    # 1.0 if White to move, 0.0 if Black to move
    if board.turn == chess.WHITE:
        tensor[:, :, 12] = 1.0
    
    return tensor
