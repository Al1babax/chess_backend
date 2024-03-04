from typing import Tuple


class Piece:

    def __init__(self, position: Tuple[int, int], color: str, piece_type: str, moves=None,
                 first_move: bool = True, in_check: bool = False) -> None:
        """
        Initialize the piece
        :param position: Position in chess notation
        :param color: Color of the piece, either "w" or "b"
        :param piece_type: Type of the piece, either "P", "N", "B", "R", "Q", "K
        :param moves: List of possible moves of the piece in chess notation
        """
        # Position of the piece using chess notation
        self.position: Tuple[int, int] = position

        # Color of the piece
        self.color: str = color

        # Type of the piece
        self.piece_type: str = piece_type

        # Possible moves of the piece
        self.moves: list = []

        # If the piece is captured
        self.captured: bool = False

        # If it is the first move of the piece
        self.first_move: bool = True if piece_type == "P" and position[0] in [1, 6] else False

        # If the piece is king and is in check
        self.in_check: bool = False

        # If the piece is pinned
        self.pinned: bool = False


class Board:
    def __init__(self, fen_string: str, white_king: Tuple[int, int] = None, black_king: Tuple[int, int] = None,
                 board_history: list = None, white_moves: list = None, black_moves: list = None) -> None:
        # Fen string to initialize the board
        self.fen_string = fen_string

        # Make array of 8x8
        self.board = [[None for _ in range(8)] for _ in range(8)]

        # Kings positions
        self.white_king: Tuple[int, int] = white_king
        self.black_king: Tuple[int, int] = black_king

        # Turn
        self.turn = self.fen_string.split(" ")[1]

        # Castling rights
        self.castling_rights = self.fen_string.split(" ")[2]

        # En passant square
        self.en_passant_square = None if self.fen_string.split(" ")[3] == "-" else self.fen_string.split(" ")[3]

        # Halfmove clock
        self.halfmove_clock = int(self.fen_string.split(" ")[4])

        # Fullmove number
        self.fullmove_number = int(self.fen_string.split(" ")[5])

        # Flag that is turned on when checking if the move is valid
        self.checking_move_validity: bool = False

        # Moves
        self.black_moves = [] if black_moves is None else black_moves
        self.white_moves = [] if white_moves is None else white_moves

        # Board history
        self.board_history = [self.fen_string] if board_history is None else board_history

        # Recorded moves
        self.white_move_history = []
        self.black_move_history = []
