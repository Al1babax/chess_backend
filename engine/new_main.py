from typing import List, Optional
import generate_moves
import numpy as np
import time

# TODO: can_move maybe too slow, optimize it 6ms for 1 move


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

    def __init__(self, position: str, color: str, piece_type: str, moves: np.ndarray = np.array([]),
                 first_move: bool = True, in_check: bool = False) -> None:
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
        self.en_passant_square = None

        # Halfmove clock
        self.halfmove_clock = 0

        # Fullmove number
        self.fullmove_number = 1

        # Flag that is turned on when checking if the move is valid
        self.checking_move_validity: bool = False

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
                    self.board[row, col].moves = generate_moves.generate(self.board[row, col], self)

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

    def can_move(self, old_pos, new_pos) -> bool:
        """
        Checks from kings perspective whether it can hit enemies or not
        :param old_pos: in chess notation
        :param new_pos: in chess notation
        :return:
        """
        self.checking_move_validity = True

        # Get the piece and potential capture
        old = pos_from_chess_notation(old_pos)
        new = pos_from_chess_notation(new_pos)
        piece: Piece = self.board[old[0], old[1]]
        potential_capture = self.board[new[0], new[1]]

        # Move the piece
        self.board[new[0], new[1]] = piece
        self.board[old[0], old[1]] = None

        # King info
        if piece.piece_type != "K":
            king_position = self.white_king if piece.color == "w" else self.black_king
            king_pos = pos_from_chess_notation(king_position)
        else:
            king_position = new_pos
            king_pos = new

        # Generate every movement from kings perspective
        king_moves = generate_moves.generate_test(self.board[king_pos[0], king_pos[1]], self)

        # Flag to know whether move is valid or not
        flag = True

        # Loop through the kings move and find enemies
        for move in king_moves:
            move = pos_from_chess_notation(move)
            # If the move is not empty and the piece is of the opposite color and king position is in enemies moves
            if self.board[move[0], move[1]] is not None and self.board[move[0], move[1]].color != piece.color:
                # Generate move for that enemy piece
                enemy_moves = generate_moves.generate(self.board[move[0], move[1]], self)

                # If the king position is in the enemies moves, return True
                if king_position in enemy_moves:
                    flag = False
                    break

        # Revert the board back to original state
        self.board[new[0], new[1]] = potential_capture
        self.board[old[0], old[1]] = piece

        self.checking_move_validity = False

        return flag

    def is_check(self, color) -> bool:
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

                # If the kings position is in the moves of the piece, set the king's in_check to True
                if king_notation in self.board[row, col].moves:
                    self.board[king_pos[0], king_pos[1]].in_check = True
                    return True

        return False

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

        # Update check for both kings
        colors = ["w", "b"]

        for color in colors:
            king_pos = pos_from_chess_notation(self.white_king if color == "w" else self.black_king)

            if self.is_check(color):
                self.board[king_pos[0], king_pos[1]].in_check = True
            else:
                self.board[king_pos[0], king_pos[1]].in_check = False

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

        # If the move wes a pawn, it was a double move, update en passant square
        if self.board[new[0], new[1]].piece_type == "P" and abs(old[0] - new[0]) == 2:
            self.en_passant_square = f"{chr(97 + new[1])}{8 - new[0] - 1}" if self.board[new[0], new[1]].color == "w" else f"{chr(97 + new[1])}{8 - new[0] + 1}"
        else:
            self.en_passant_square = None

        # Handle castling, move the rook also
        if self.board[new[0], new[1]].piece_type == "K" and abs(old[1] - new[1]) == 2:
            # Queen side
            if new[1] == 2:
                self.board[new[0], 3] = self.board[new[0], 0]
                self.board[new[0], 0] = None
                # Update rook position and make first move False
                self.board[new[0], 3].position = "d1" if self.board[new[0], 3].color == "w" else "d8"
                self.board[new[0], 3].first_move = False
            else:
                self.board[new[0], 5] = self.board[new[0], 7]
                self.board[new[0], 7] = None
                # Update rook position and make first move False
                self.board[new[0], 5].position = "f1" if self.board[new[0], 5].color == "w" else "f8"
                self.board[new[0], 5].first_move = False

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
    board.move("f1", "d4")
    board.move("g1", "d5")
    board.move("b1", "d3")
    board.move("c1", "d6")

    # Do kingside castling
    board.move("e1", "c1")

    # Print the moves for all the pieces
    print_moves_for_all_pieces(board)



if __name__ == "__main__":
    main()
