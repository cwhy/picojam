import jax
import jax.numpy as jnp
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import logging
from typing import List, Optional, Tuple

# Pre-computed mappings for faster encoding
PIECE_TO_NUM = {
    'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,  # White pieces
    'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12  # Black pieces
}

def encode_fen_fast(fen: str) -> jnp.ndarray:
    """Optimized FEN encoding"""
    parts = fen.split()
    board_fen, side_to_move, castling, en_passant = parts[:4]
    
    # Parse board more efficiently
    board = np.zeros(64, dtype=np.float32)
    square_idx = 0
    
    for char in board_fen:
        if char == '/':
            continue
        elif char.isdigit():
            square_idx += int(char)  # Skip empty squares
        else:
            if square_idx < 64:  # Bounds check
                board[square_idx] = PIECE_TO_NUM.get(char, 0)
                square_idx += 1
    
    # Side to move (vectorized)
    side = np.array([1.0 if side_to_move == 'w' else 0.0], dtype=np.float32)
    
    # Castling rights (vectorized check)
    castling_rights = np.array([
        1.0 if 'K' in castling else 0.0,
        1.0 if 'Q' in castling else 0.0,
        1.0 if 'k' in castling else 0.0,
        1.0 if 'q' in castling else 0.0
    ], dtype=np.float32)
    
    # En passant (vectorized)
    en_passant_vec = np.zeros(8, dtype=np.float32)
    if en_passant != '-' and len(en_passant) >= 1:
        file_idx = ord(en_passant[0]) - ord('a')
        if 0 <= file_idx < 8:  # Bounds check
            en_passant_vec[file_idx] = 1.0
    
    # Concatenate efficiently
    features = np.concatenate([board, side, castling_rights, en_passant_vec])
    return jnp.array(features)

def encode_fen_batch(fens: List[str]) -> jnp.ndarray:
    """Encode multiple FENs at once with progress bar"""
    logging.info("Encoding FENs...")
    encoded = []
    for fen in tqdm(fens, desc="Encoding FENs"):
        encoded.append(encode_fen_fast(fen))
    return jnp.array(encoded)

def parse_evaluation(eval_str) -> Optional[float]:
    """Parse evaluation string, handling mate scores"""
    eval_str = str(eval_str).strip()
    
    if eval_str.startswith('M'):
        # Mate in N moves - convert to large numerical value
        try:
            mate_moves = int(eval_str[1:])
            # Positive mate scores for white advantage
            # Use 100 - mate_moves so mate in 1 = 99, mate in 10 = 90, etc.
            return 100.0 - mate_moves
        except ValueError:
            return None
    elif eval_str.startswith('-M'):
        # Mate against us - convert to large negative value
        try:
            mate_moves = int(eval_str[2:])
            # Negative mate scores for black advantage
            return -(100.0 - mate_moves)
        except ValueError:
            return None
    else:
        # Regular numerical evaluation
        try:
            return float(eval_str)
        except ValueError:
            return None

def load_data_from_hf(limit: int = 10000) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Load training data from Hugging Face dataset"""
    logging.info("Loading dataset from Hugging Face...")
    ds = load_dataset("bingbangboom/stockfish-evaluations", split="train")
    
    # Shuffle and limit data
    if limit is not None and limit < len(ds):
        ds = ds.shuffle(seed=42).select(range(limit))
    
    logging.info(f"Loaded {len(ds)} positions from dataset")
    
    # Collect FENs and evaluations first
    fens = []
    evaluations = []
    skipped = 0
    
    logging.info("Parsing positions...")
    for example in tqdm(ds, desc="Parsing positions"):
        try:
            fen = example['fen']
            evaluation_raw = example['evaluation']
            
            # Parse evaluation (handles mate scores)
            evaluation = parse_evaluation(evaluation_raw)
            if evaluation is None:
                skipped += 1
                continue
            
            fens.append(fen)
            evaluations.append(evaluation)
            
        except Exception as e:
            skipped += 1
            continue
    
    # Batch encode FENs (much faster)
    X = encode_fen_batch(fens)
    y = jnp.array(evaluations)
    
    logging.info(f"Successfully processed {len(X)} positions")
    if skipped > 0:
        logging.info(f"Skipped {skipped} positions due to parsing errors")
    
    return X, y

def get_sample_positions() -> List[Tuple[str, str]]:
    """Get a list of sample chess positions for evaluation"""
    return [
        ("Starting Position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("Sicilian Defense", "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2"),
        ("Queen's Gambit", "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2"),
        ("Scholar's Mate Setup", "rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 2 3"),
        ("French Defense", "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"),
        ("King's Indian Defense", "rnbqkb1r/pppppp1p/5np1/8/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3"),
        ("Ruy Lopez", "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"),
        ("Caro-Kann Defense", "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"),
    ]

def fen_to_board_visualization(fen: str) -> str:
    """Convert FEN to a simple ASCII board visualization"""
    parts = fen.split()
    board_fen = parts[0]
    
    board_str = ""
    rank = 8
    
    for char in board_fen:
        if char == '/':
            board_str += f" {rank}\n"
            rank -= 1
        elif char.isdigit():
            board_str += '. ' * int(char)
        else:
            board_str += char + ' '
    
    board_str += f" {rank}\n"
    board_str += "a b c d e f g h\n"
    
    return board_str

def analyze_position_features(fen: str) -> dict:
    """Analyze various features of a chess position"""
    encoded = encode_fen_fast(fen)
    
    # Extract different parts of the encoding
    board = encoded[:64]
    side_to_move = encoded[64]
    castling_rights = encoded[65:69]
    en_passant = encoded[69:77]
    
    # Count pieces
    piece_counts = {}
    for piece_name, piece_num in PIECE_TO_NUM.items():
        count = jnp.sum(board == piece_num)
        piece_counts[piece_name] = int(count)
    
    # Material count (rough approximation)
    white_material = (piece_counts.get('P', 0) * 1 + 
                     piece_counts.get('N', 0) * 3 + 
                     piece_counts.get('B', 0) * 3 + 
                     piece_counts.get('R', 0) * 5 + 
                     piece_counts.get('Q', 0) * 9)
    
    black_material = (piece_counts.get('p', 0) * 1 + 
                     piece_counts.get('n', 0) * 3 + 
                     piece_counts.get('b', 0) * 3 + 
                     piece_counts.get('r', 0) * 5 + 
                     piece_counts.get('q', 0) * 9)
    
    return {
        'side_to_move': 'White' if side_to_move == 1.0 else 'Black',
        'piece_counts': piece_counts,
        'white_material': white_material,
        'black_material': black_material,
        'material_balance': white_material - black_material,
        'castling_available': {
            'white_kingside': bool(castling_rights[0]),
            'white_queenside': bool(castling_rights[1]),
            'black_kingside': bool(castling_rights[2]),
            'black_queenside': bool(castling_rights[3])
        },
        'en_passant_file': jnp.argmax(en_passant) if jnp.any(en_passant) else None
    }