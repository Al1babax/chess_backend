from typing import List, Optional
import generate_moves
import numpy as np
import time


def measure_time(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter_ns()
        result = func(*args, **kwargs)
        end = time.perf_counter_ns()

        print(f"Time: {(end - start) / 1_000_000} ms")

        return result

    return wrapper


def pos_from_chess_notation(notation: str) -> tuple:
    """
    Convert chess notation to matrix notation
    :param notation: Chess notation
    :return: Tuple of the position in the matrix
    """
    # Make sure the piece is not pawn, if it is, it only has two letters
    if len(notation) == 2:
        row = 8 - int(notation[1])
        col = ord(notation[0]) - 97
    else:
        row = 8 - int(notation[2])
        col = ord(notation[1]) - 97

    return row, col


class Piece:

    def __init__(self, position: str, color: str, piece_type: str) -> None:
        """
        Initialize the piece
        :param position: Position in chess notation
        :param color: Color of the piece, either "w" or "b"
        :param piece_type: Type of the piece, either "P", "N", "B", "R", "Q", "K
        :param moves: List of possible moves of the piece in chess notation
        """
        # Position of the piece using chess notation
        self.position: str = position

        # Color of the piece
        self.color: str = color

        # Type of the piece
        self.piece_type: str = piece_type

        # Possible moves of the piece
        self.moves: np.ndarray = np.array([])

        # If the piece is captured
        self.captured: bool = False

        # If the piece is pinned
        self.is_pinned: bool = False

        # If it is the first move of the piece
        self.first_move: bool = True

        # If the piece is king and is in check
        self.in_check: bool = False


class Board:
    def __init__(self, fen_string: str) -> None:
        # TODO: cannot have custom fens atm than default

        # Fen string to initialize the board
        self.fen_string = fen_string

        # Make numpy array of 8x8
        self.board = np.empty((8, 8), dtype=object)

        # Kings
        self.white_king = "e1"
        self.black_king = "e8"

        # Turn
        self.turn = "w"

        # Castling rights
        self.castling_rights = "KQkq"

        # En passant square
        self.en_passant_square = "-"

        # Halfmove clock
        self.halfmove_clock = 0

        # Fullmove number
        self.fullmove_number = 1

        # Create board
        self.create_board()

        # Generate moves
        self.generate_piece_moves()

    def create_board(self) -> None:
        # Create the board so that white piece are first in the matrix
        fen_piece = self.fen_string.split(" ")[0]
        rows = fen_piece.split("/")

        for i, row in enumerate(rows):
            new_row = np.empty(8, dtype=object)
            for k, piece in enumerate(row):
                if piece.isdigit():
                    for _ in range(int(piece)):
                        new_row[k] = None

                    continue

                # Construct chess notation
                position = chr(97 + k) + str(8 - i)

                # Construct the piece color
                if piece.isupper():
                    color = "w"
                else:
                    color = "b"

                # Construct the piece type
                piece_type = piece.upper()

                # Create the piece
                new_row[k] = Piece(position, color, piece_type)

            self.board[i] = new_row

    def generate_piece_moves(self) -> None:
        """
        Generate the moves for all the pieces
        :return:
        """
        for row in range(8):
            for col in range(8):
                if self.board[row, col] is not None:
                    # Generate the moves for the piece
                    self.board[row, col].moves = generate_moves.generate(self.board[row, col], self.board)

    def is_valid(self, old, new) -> bool:
        """
        Check if the move  is valid
        :param new:
        :param old:
        :return: True if the move is valid else False
        """
        old_pos = pos_from_chess_notation(old)
        piece = self.board[old_pos[0]][old_pos[1]]

        # Check if the move is in the list of moves of the piece
        if new in piece.moves:
            return True
        else:
            return False

    def update_check(self, color) -> None:
        """
        Check if the color's king is in check by checking if opposite color's pieces have the king in their moves
        :return:
        """
        king_pos = pos_from_chess_notation(self.white_king if color == "w" else self.black_king)
        king_notation = self.white_king if color == "w" else self.black_king

        for row in range(8):
            for col in range(8):
                # If the piece is None, continue
                if self.board[row, col] is None:
                    continue

                # If the piece is of the same color, continue
                if self.board[row, col].color == color:
                    continue

                # If the piece is a king, continue
                if self.board[row, col].piece_type == "K":
                    continue

                # If the king is in the moves of the piece, set the king's in_check to True
                if king_notation in self.board[row, col].moves:
                    self.board[king_pos[0], king_pos[1]].in_check = True
                    return

    def update_pin(self) -> None:
        """
        Update the pin for all the pieces
        Check this by looping over all the pieces but the king, temporarily removing the king from the board
        and checking if the king goes into check
        :return:
        """
        pass

    @measure_time
    def update(self, piece_type: str, was_capture: bool) -> None:
        """
        Updates all the pieces if they are pinned, kings are in check, checkmate, stalemate, turn, fen string, etc..
        :param piece_type: Type of the piece
        :param was_capture: If the move was a capture
        :return:
        """
        # Generate the moves for all the pieces
        self.generate_piece_moves()

        # Update check for the opposite color
        self.update_check("w" if self.turn == "b" else "b")

        # Update clocks
        # If black moved, update fullmove number
        if self.turn == "b":
            self.fullmove_number += 1

        # If not pawn move or capture, update halfmove clock
        if piece_type == "P" or was_capture:
            self.halfmove_clock = 0
        else:
            self.halfmove_clock += 1

        # Change the turn
        self.turn = "w" if self.turn == "b" else "b"

    def move(self, old_pos, new_pos) -> None:
        """
        Move the piece
        Positions in chess notation as parameters
        :param old_pos: Old position of the piece
        :param new_pos: New position of the piece
        :return:
        """
        old = pos_from_chess_notation(old_pos)
        new = pos_from_chess_notation(new_pos)

        # Check if the move was capture
        was_capture = True if self.board[new[0], new[1]] is not None else False

        # Move the piece
        self.board[new[0], new[1]] = self.board[old[0], old[1]]
        self.board[old[0], old[1]] = None

        # Update the position of the piece
        self.board[new[0], new[1]].position = new_pos if len(new_pos) == 2 else new_pos[1:]

        # Make first move False, specifically for pawns
        self.board[new[0], new[1]].first_move = False

        # If the piece is a king, update the king position
        if self.board[new[0], new[1]].piece_type == "K":
            if self.board[new[0], new[1]].color == "w":
                self.white_king = f"{chr(97 + new[1])}{8 - new[0]}"
            else:
                self.black_king = f"{chr(97 + new[1])}{8 - new[0]}"

        # Update the board
        self.update(self.board[new[0], new[1]].piece_type, was_capture)


def print_moves_for_all_pieces(board) -> None:
    for row in range(8):
        for col in range(8):
            if board.board[row, col] is not None:
                print(f"Piece: {board.board[row, col].piece_type} {board.board[row, col].color}", end=" ")
                print(f"Position: {board.board[row, col].position}", end=" ")
                print(f"Moves: {board.board[row, col].moves}")


def main():
    # Create the board
    board = Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

    # Move white piece
    board.move("e2", "e4")

    board.move("d8", "e3")

    # Print the moves for all the pieces
    print_moves_for_all_pieces(board)

    print(board.board[7, 4].in_check)


if __name__ == "__main__":
    main()
