import jax.numpy as jnp
import numpy as np
from typing import List, Tuple, Dict
import json

class ChessEmbeddingEncoder:
    """
    Chess position encoder using embedding dictionary approach.
    Each piece state is mapped to a unique integer ID.
    
    Embedding ID structure:
    - piece_type (6 types: P,N,B,R,Q,K) 
    - position (65 squares: 0-63 for board, 64 for "dead")
    - color (2: white=0, black=1)
    - will_move_next (2: no=0, yes=1)
    
    Total vocabulary size: 6 * 65 * 2 * 2 = 1560 possible piece states
    Always encodes exactly 32 pieces, using "dead" position for missing pieces.
    """
    
    def __init__(self):
        # Piece type mappings
        self.piece_types = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5}
        self.type_to_piece = {v: k for k, v in self.piece_types.items()}
        
        # Position mappings (0-63 for board squares, 64 for "dead")
        self.dead_position = 64
        
        # Build embedding vocabulary
        self.vocab_size = 6 * 65 * 2 * 2  # 1560
        self.piece_to_id = {}
        self.id_to_piece = {}
        
        self._build_vocabulary()
    
    def _build_vocabulary(self):
        """Build the embedding vocabulary mapping"""
        embedding_id = 1  # Start from 1, reserve 0 for special use if needed
        
        for piece_type in range(6):  # 6 piece types
            for position in range(65):  # 65 positions (64 board + 1 dead)
                for color in range(2):  # 2 colors (white=0, black=1)
                    for will_move in range(2):  # 2 move states
                        
                        # Create piece state tuple
                        piece_state = (piece_type, position, color, will_move)
                        
                        # Map to embedding ID
                        self.piece_to_id[piece_state] = embedding_id
                        self.id_to_piece[embedding_id] = piece_state
                        
                        embedding_id += 1
        
        print(f"Built vocabulary with {len(self.piece_to_id)} piece states")
        print(f"Vocabulary size: {self.vocab_size}")
    
    def _position_to_square_index(self, rank: int, file: int) -> int:
        """Convert rank/file to square index (0-63)"""
        return rank * 8 + file
    
    def _square_index_to_position(self, square_idx: int) -> Tuple[int, int]:
        """Convert square index to rank/file"""
        if square_idx == self.dead_position:
            return (-1, -1)  # Dead position
        rank = square_idx // 8
        file = square_idx % 8
        return rank, file
    
    def encode_fen(self, fen: str, max_pieces: int = 32) -> jnp.ndarray:
        """
        Encode FEN string to list of embedding IDs
        
        Args:
            fen: FEN string
            max_pieces: Maximum number of pieces to encode (pad with zeros)
            
        Returns:
            Array of embedding IDs, shape (max_pieces,)
        """
        parts = fen.split()
        board_fen = parts[0]
        side_to_move = parts[1]  # 'w' or 'b'
        
        # Determine which color moves next
        white_moves_next = (side_to_move == 'w')
        
        pieces = []
        rank = 7  # Start from rank 8 (index 7)
        file = 0  # Start from file a (index 0)
        
        for char in board_fen:
            if char == '/':
                rank -= 1
                file = 0
            elif char.isdigit():
                file += int(char)  # Skip empty squares
            else:
                # Found a piece
                if char.upper() in self.piece_types:
                    # Get piece information
                    piece_type = self.piece_types[char.upper()]
                    position = self._position_to_square_index(rank, file)
                    color = 0 if char.isupper() else 1  # white=0, black=1
                    
                    # Determine if this piece's color moves next
                    will_move_next = 1 if (color == 0 and white_moves_next) or (color == 1 and not white_moves_next) else 0
                    
                    # Create piece state and get embedding ID
                    piece_state = (piece_type, position, color, will_move_next)
                    embedding_id = self.piece_to_id.get(piece_state, 0)
                    
                    pieces.append(embedding_id)
                    file += 1
        
        # Pad or truncate to max_pieces
        output = np.zeros(max_pieces, dtype=np.int32)
        num_pieces = min(len(pieces), max_pieces)
        if num_pieces > 0:
            output[:num_pieces] = pieces[:num_pieces]
        
        return jnp.array(output)
    
    def decode_embedding_ids(self, embedding_ids: jnp.ndarray) -> List[Tuple]:
        """
        Decode embedding IDs back to piece information
        
        Args:
            embedding_ids: Array of embedding IDs
            
        Returns:
            List of (piece_type, position, color, will_move_next) tuples
        """
        pieces = []
        
        for embedding_id in embedding_ids:
            embedding_id = int(embedding_id)
            
            # Skip padding (ID = 0)
            if embedding_id == 0:
                continue
                
            if embedding_id in self.id_to_piece:
                piece_info = self.id_to_piece[embedding_id]
                pieces.append(piece_info)
        
        return pieces
    
    def visualize_encoded_position(self, embedding_ids: jnp.ndarray) -> str:
        """Create ASCII visualization of the encoded position"""
        pieces = self.decode_embedding_ids(embedding_ids)
        
        # Create empty 8x8 board
        board = [['.' for _ in range(8)] for _ in range(8)]
        
        # Track which pieces will move next
        moving_pieces = set()
        
        # Place pieces on board
        for piece_type, position, color, will_move_next in pieces:
            rank, file = self._square_index_to_position(position)
            
            # Get piece symbol
            piece_char = self.type_to_piece[piece_type]
            if color == 1:  # Black pieces (lowercase)
                piece_char = piece_char.lower()
            
            board[7-rank][file] = piece_char  # Flip rank for display
            
            if will_move_next:
                moving_pieces.add((7-rank, file))
        
        # Create board string
        board_str = ""
        for rank_idx, rank in enumerate(board):
            board_str += f"{8-rank_idx} "
            for file_idx, piece in enumerate(rank):
                # Mark pieces that will move next with brackets
                if (rank_idx, file_idx) in moving_pieces:
                    board_str += f"[{piece}]"
                else:
                    board_str += f" {piece} "
            board_str += "\n"
        board_str += "   a  b  c  d  e  f  g  h\n"
        
        return board_str
    
    def analyze_encoding(self, embedding_ids: jnp.ndarray) -> Dict:
        """Analyze the encoded position"""
        pieces = self.decode_embedding_ids(embedding_ids)
        
        # Count pieces by type and color
        white_counts = {}
        black_counts = {}
        moving_pieces = 0
        
        piece_names = {0: 'Pawn', 1: 'Knight', 2: 'Bishop', 3: 'Rook', 4: 'Queen', 5: 'King'}
        
        for piece_type, position, color, will_move_next in pieces:
            piece_name = piece_names[piece_type]
            
            if color == 0:  # White
                white_counts[piece_name] = white_counts.get(piece_name, 0) + 1
            else:  # Black
                black_counts[piece_name] = black_counts.get(piece_name, 0) + 1
            
            if will_move_next:
                moving_pieces += 1
        
        # Determine side to move
        side_to_move = "White" if moving_pieces > 0 else "Unknown"
        if moving_pieces == 0:
            # Check if any pieces exist to infer
            if len(pieces) > 0:
                side_to_move = "Position encoded, but move info unclear"
        
        return {
            'total_pieces': len(pieces),
            'white_pieces': sum(white_counts.values()),
            'black_pieces': sum(black_counts.values()),
            'white_piece_counts': white_counts,
            'black_piece_counts': black_counts,
            'pieces_that_move_next': moving_pieces,
            'side_to_move': side_to_move,
            'encoding_efficiency': f"{len(pieces)}/32 slots used",
            'unique_embedding_ids': len(set(embedding_ids[embedding_ids != 0]))
        }
    
    def get_vocabulary_info(self) -> Dict:
        """Get information about the embedding vocabulary"""
        return {
            'vocabulary_size': self.vocab_size,
            'total_mapped_states': len(self.piece_to_id),
            'piece_types': 6,
            'board_positions': 64,
            'colors': 2,
            'move_states': 2,
            'reserved_for_padding': 1,
            'actual_vocab_size': len(self.piece_to_id) + 1  # +1 for padding
        }

# Example usage and testing
def test_embedding_encoding():
    """Test the embedding encoding with sample positions"""
    
    encoder = ChessEmbeddingEncoder()
    
    # Print vocabulary info
    vocab_info = encoder.get_vocabulary_info()
    print("=== Vocabulary Information ===")
    for key, value in vocab_info.items():
        print(f"{key}: {value}")
    
    test_positions = [
        ("Starting Position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("After 1.e4", "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"),
        ("Mid-game Position", "r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQ - 4 6")
    ]
    
    for name, fen in test_positions:
        print(f"\n=== {name} ===")
        print(f"FEN: {fen}")
        
        # Encode
        encoded = encoder.encode_fen(fen)
        print(f"Encoded shape: {encoded.shape}")
        print(f"Non-zero embedding IDs: {encoded[encoded != 0]}")
        
        # Analyze
        analysis = encoder.analyze_encoding(encoded)
        print("Analysis:")
        for key, value in analysis.items():
            print(f"  {key}: {value}")
        
        # Visualize
        print("Board visualization ([piece] = will move next):")
        print(encoder.visualize_encoded_position(encoded))

if __name__ == "__main__":
    test_embedding_encoding()