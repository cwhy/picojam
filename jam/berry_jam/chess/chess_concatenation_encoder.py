import jax.numpy as jnp
import numpy as np
from typing import List, Tuple, Dict
import json

class ChessConcatenationEncoder:
    """
    Chess position encoder using concatenation approach.
    Each piece state is represented as 4 separate values:
    [piece_type, position, color, will_move_next]
    
    Value ranges:
    - piece_type: 0-5 (P,N,B,R,Q,K) or 6 for padding/dead piece
    - position: 0-63 for board squares, 64 for "dead", 65 for padding
    - color: 0 (white), 1 (black), 2 for padding/dead piece
    - will_move_next: 0 (no), 1 (yes), 2 for padding/dead piece
    
    Output shape: (max_pieces, 4) where each row is [piece_type, position, color, will_move_next]
    """
    
    def __init__(self):
        # Piece type mappings
        self.piece_types = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5}
        self.type_to_piece = {v: k for k, v in self.piece_types.items()}
        
        # Special values for padding/dead pieces
        self.dead_piece_type = 6
        self.dead_position = 64
        self.padding_position = 65
        self.padding_color = 2
        self.padding_move = 2
        
        # Vocabulary sizes for each component
        self.piece_type_vocab = 7  # 6 piece types + 1 padding
        self.position_vocab = 66   # 64 board + 1 dead + 1 padding
        self.color_vocab = 3       # 2 colors + 1 padding
        self.move_vocab = 3        # 2 move states + 1 padding
        
        print(f"Component vocabulary sizes:")
        print(f"  Piece types: {self.piece_type_vocab}")
        print(f"  Positions: {self.position_vocab}")
        print(f"  Colors: {self.color_vocab}")
        print(f"  Move states: {self.move_vocab}")
    
    def _position_to_square_index(self, rank: int, file: int) -> int:
        """Convert rank/file to square index (0-63)"""
        return rank * 8 + file
    
    def _square_index_to_position(self, square_idx: int) -> Tuple[int, int]:
        """Convert square index to rank/file"""
        if square_idx >= 64:
            return (-1, -1)  # Dead/padding position
        rank = square_idx // 8
        file = square_idx % 8
        return rank, file
    
    def encode_fen(self, fen: str, max_pieces: int = 32) -> jnp.ndarray:
        """
        Encode FEN string to concatenated representation
        
        Args:
            fen: FEN string
            max_pieces: Maximum number of pieces to encode (pad with special values)
            
        Returns:
            Array of shape (max_pieces, 4) where each row is [piece_type, position, color, will_move_next]
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
                    
                    # Create piece representation as [piece_type, position, color, will_move_next]
                    piece_repr = [piece_type, position, color, will_move_next]
                    pieces.append(piece_repr)
                    file += 1
        
        # Create output array with padding
        output = np.full((max_pieces, 4), 
                        [self.dead_piece_type, self.padding_position, self.padding_color, self.padding_move], 
                        dtype=np.int32)
        
        # Fill with actual pieces
        num_pieces = min(len(pieces), max_pieces)
        if num_pieces > 0:
            output[:num_pieces] = pieces[:num_pieces]
        
        return jnp.array(output)
    
    def decode_concatenated_encoding(self, encoded_array: jnp.ndarray) -> List[Tuple]:
        """
        Decode concatenated representation back to piece information
        
        Args:
            encoded_array: Array of shape (max_pieces, 4)
            
        Returns:
            List of (piece_type, position, color, will_move_next) tuples for valid pieces
        """
        pieces = []
        
        for piece_row in encoded_array:
            piece_type, position, color, will_move_next = piece_row
            
            # Skip padding pieces
            if (piece_type == self.dead_piece_type or 
                position == self.padding_position or 
                color == self.padding_color):
                continue
                
            pieces.append((int(piece_type), int(position), int(color), int(will_move_next)))
        
        return pieces
    
    def visualize_encoded_position(self, encoded_array: jnp.ndarray) -> str:
        """Create ASCII visualization of the encoded position"""
        pieces = self.decode_concatenated_encoding(encoded_array)
        
        # Create empty 8x8 board
        board = [['.' for _ in range(8)] for _ in range(8)]
        
        # Track which pieces will move next
        moving_pieces = set()
        
        # Place pieces on board
        for piece_type, position, color, will_move_next in pieces:
            rank, file = self._square_index_to_position(position)
            
            if rank >= 0 and file >= 0:  # Valid board position
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
    
    def analyze_encoding(self, encoded_array: jnp.ndarray) -> Dict:
        """Analyze the encoded position"""
        pieces = self.decode_concatenated_encoding(encoded_array)
        
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
        side_to_move = "White" if moving_pieces > 0 else "Black"
        if moving_pieces == 0:
            # If no pieces are marked as moving, check the position
            if len(pieces) > 0:
                side_to_move = "Position encoded, but no pieces marked to move"
        
        return {
            'total_pieces': len(pieces),
            'white_pieces': sum(white_counts.values()),
            'black_pieces': sum(black_counts.values()),
            'white_piece_counts': white_counts,
            'black_piece_counts': black_counts,
            'pieces_that_move_next': moving_pieces,
            'side_to_move': side_to_move,
            'encoding_efficiency': f"{len(pieces)}/32 slots used"
        }
    
    def get_encoding_info(self) -> Dict:
        """Get information about the encoding format"""
        return {
            'encoding_type': 'concatenation',
            'output_shape': '(max_pieces, 4)',
            'component_meanings': ['piece_type', 'position', 'color', 'will_move_next'],
            'piece_type_range': f'0-5 (pieces), {self.dead_piece_type} (padding)',
            'position_range': f'0-63 (board), {self.dead_position} (dead), {self.padding_position} (padding)',
            'color_range': f'0 (white), 1 (black), {self.padding_color} (padding)',
            'move_range': f'0 (no), 1 (yes), {self.padding_move} (padding)',
            'vocabulary_sizes': {
                'piece_types': self.piece_type_vocab,
                'positions': self.position_vocab,
                'colors': self.color_vocab,
                'move_states': self.move_vocab
            }
        }
    
    def get_component_embeddings(self, encoded_array: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """
        Extract separate component arrays for embedding layers
        
        Args:
            encoded_array: Array of shape (max_pieces, 4)
            
        Returns:
            Dictionary with separate arrays for each component
        """
        return {
            'piece_types': encoded_array[:, 0],
            'positions': encoded_array[:, 1], 
            'colors': encoded_array[:, 2],
            'will_move': encoded_array[:, 3]
        }

# Example usage and testing
def test_concatenation_encoding():
    """Test the concatenation encoding with sample positions"""
    
    encoder = ChessConcatenationEncoder()
    
    # Print encoding info
    encoding_info = encoder.get_encoding_info()
    print("=== Encoding Information ===")
    for key, value in encoding_info.items():
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
        
        # Show first few piece encodings
        print("First 5 piece encodings [piece_type, position, color, will_move]:")
        for i in range(min(5, encoded.shape[0])):
            piece_data = encoded[i]
            if piece_data[0] != encoder.dead_piece_type:  # Skip padding
                piece_name = encoder.type_to_piece.get(int(piece_data[0]), 'Unknown')
                color_name = 'White' if piece_data[2] == 0 else 'Black' if piece_data[2] == 1 else 'Padding'
                move_status = 'Will move' if piece_data[3] == 1 else 'Will not move' if piece_data[3] == 0 else 'Padding'
                print(f"  Piece {i}: {piece_name} at position {piece_data[1]} ({color_name}, {move_status})")
        
        # Get component embeddings
        components = encoder.get_component_embeddings(encoded)
        print(f"Component arrays shapes:")
        for comp_name, comp_array in components.items():
            print(f"  {comp_name}: {comp_array.shape}")
        
        # Analyze
        analysis = encoder.analyze_encoding(encoded)
        print("Analysis:")
        for key, value in analysis.items():
            print(f"  {key}: {value}")
        
        # Visualize
        print("Board visualization ([piece] = will move next):")
        print(encoder.visualize_encoded_position(encoded))

if __name__ == "__main__":
    test_concatenation_encoding()