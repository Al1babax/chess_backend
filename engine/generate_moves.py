from typing import List, Optional, Tuple
from new_main import Piece, Board, pos_from_chess_notation
import numpy as np


class Movement:
    # Class that look piece movement to all directions
    def __init__(self, piece: Piece, board: Board) -> None:
        self.piece: Piece = piece
        self.board: np.ndarray = board.board
        self.object: Board = board

    def move_up(self, distance: int = 8) -> np.ndarray:
        moves = np.array([], dtype=str)

        # Get the position of the piece
        position: Tuple[int, int] = pos_from_chess_notation(self.piece.position)
        color: str = self.piece.color

        for i in range(1, distance):
            # Check if the position is valid
            if position[0] - i >= 0:
                # Check if the square is empty
                if self.board[position[0] - i, position[1]] is None:
                    moves = np.append(moves, f"{chr(position[1] + 97)}{8 - (position[0] - i)}")
                # Check if the square is occupied by an enemy piece
                elif self.board[position[0] - i, position[1]].color != color:
                    moves = np.append(moves, f"{chr(position[1] + 97)}{8 - (position[0] - i)}")
                    break
                else:
                    break
            else:
                break

        return moves

    def move_down(self, distance: int = 8) -> np.ndarray:
        moves = np.array([], dtype=str)

        # Get the position of the piece
        position: Tuple[int, int] = pos_from_chess_notation(self.piece.position)
        color: str = self.piece.color

        for i in range(1, distance):
            # Check if the position is valid
            if position[0] + i < 8:
                # Check if the square is empty
                if self.board[position[0] + i, position[1]] is None:
                    moves = np.append(moves, f"{chr(position[1] + 97)}{8 - (position[0] + i)}")
                # Check if the square is occupied by an enemy piece
                elif self.board[position[0] + i, position[1]].color != color:
                    moves = np.append(moves, f"{chr(position[1] + 97)}{8 - (position[0] + i)}")
                    break
                else:
                    break
            else:
                break

        return moves

    def move_right(self, distance: int = 8) -> np.ndarray:
        moves = np.array([], dtype=str)

        # Get the position of the piece
        position: Tuple[int, int] = pos_from_chess_notation(self.piece.position)
        color: str = self.piece.color

        for i in range(1, distance):
            # Check if the position is valid
            if position[1] + i < 8:
                # Check if the square is empty
                if self.board[position[0], position[1] + i] is None:
                    moves = np.append(moves, f"{chr(position[1] + 97 + i)}{8 - position[0]}")
                # Check if the square is occupied by an enemy piece
                elif self.board[position[0], position[1] + i].color != color:
                    moves = np.append(moves, f"{chr(position[1] + 97 + i)}{8 - position[0]}")
                    break
                else:
                    break
            else:
                break

        return moves

    def move_left(self, distance: int = 8) -> np.ndarray:
        moves = np.array([], dtype=str)

        # Get the position of the piece
        position: Tuple[int, int] = pos_from_chess_notation(self.piece.position)
        color: str = self.piece.color

        for i in range(1, distance):
            # Check if the position is valid
            if position[1] - i >= 0:
                # Check if the square is empty
                if self.board[position[0], position[1] - i] is None:
                    moves = np.append(moves, f"{chr(position[1] + 97 - i)}{8 - position[0]}")
                # Check if the square is occupied by an enemy piece
                elif self.board[position[0], position[1] - i].color != color:
                    moves = np.append(moves, f"{chr(position[1] + 97 - i)}{8 - position[0]}")
                    break
                else:
                    break
            else:
                break

        return moves

    def move_up_right(self, distance: int = 8) -> np.ndarray:
        moves = np.array([], dtype=str)

        # Get the position of the piece
        position: Tuple[int, int] = pos_from_chess_notation(self.piece.position)
        color: str = self.piece.color

        for i in range(1, distance):
            # Check if the position is valid
            if position[0] - i >= 0 and position[1] + i < 8:
                # Check if the square is empty
                if self.board[position[0] - i, position[1] + i] is None:
                    moves = np.append(moves, f"{chr(position[1] + 97 + i)}{8 - (position[0] - i)}")
                # Check if the square is occupied by an enemy piece
                elif self.board[position[0] - i, position[1] + i].color != color:
                    moves = np.append(moves, f"{chr(position[1] + 97 + i)}{8 - (position[0] - i)}")
                    break
                else:
                    break
            else:
                break

        return moves

    def move_up_left(self, distance: int = 8) -> np.ndarray:
        moves = np.array([], dtype=str)

        # Get the position of the piece
        position: Tuple[int, int] = pos_from_chess_notation(self.piece.position)
        color: str = self.piece.color

        for i in range(1, distance):
            # Check if the position is valid
            if position[0] - i >= 0 and position[1] - i >= 0:
                # Check if the square is empty
                if self.board[position[0] - i, position[1] - i] is None:
                    moves = np.append(moves, f"{chr(position[1] + 97 - i)}{8 - (position[0] - i)}")
                # Check if the square is occupied by an enemy piece
                elif self.board[position[0] - i, position[1] - i].color != color:
                    moves = np.append(moves, f"{chr(position[1] + 97 - i)}{8 - (position[0] - i)}")
                    break
                else:
                    break
            else:
                break

        return moves

    def move_down_right(self, distance: int = 8) -> np.ndarray:
        moves = np.array([], dtype=str)

        # Get the position of the piece
        position: Tuple[int, int] = pos_from_chess_notation(self.piece.position)
        color: str = self.piece.color

        for i in range(1, distance):
            # Check if the position is valid
            if position[0] + i < 8 and position[1] + i < 8:
                # Check if the square is empty
                if self.board[position[0] + i, position[1] + i] is None:
                    moves = np.append(moves, f"{chr(position[1] + 97 + i)}{8 - (position[0] + i)}")
                # Check if the square is occupied by an enemy piece
                elif self.board[position[0] + i, position[1] + i].color != color:
                    moves = np.append(moves, f"{chr(position[1] + 97 + i)}{8 - (position[0] + i)}")
                    break
                else:
                    break
            else:
                break

        return moves

    def move_down_left(self, distance: int = 8) -> np.ndarray:
        moves = np.array([], dtype=str)

        # Get the position of the piece
        position: Tuple[int, int] = pos_from_chess_notation(self.piece.position)
        color: str = self.piece.color

        for i in range(1, distance):
            # Check if the position is valid
            if position[0] + i < 8 and position[1] - i >= 0:
                # Check if the square is empty
                if self.board[position[0] + i, position[1] - i] is None:
                    moves = np.append(moves, f"{chr(position[1] + 97 - i)}{8 - (position[0] + i)}")
                # Check if the square is occupied by an enemy piece
                elif self.board[position[0] + i, position[1] - i].color != color:
                    moves = np.append(moves, f"{chr(position[1] + 97 - i)}{8 - (position[0] + i)}")
                    break
                else:
                    break
            else:
                break

        return moves

    def move_knight(self) -> np.ndarray:
        moves = np.array([], dtype=str)

        # Get the position of the piece
        position: Tuple[int, int] = pos_from_chess_notation(self.piece.position)
        color: str = self.piece.color

        # Generate all the possible moves
        possible_moves = [
            (position[0] - 2, position[1] - 1),
            (position[0] - 2, position[1] + 1),
            (position[0] - 1, position[1] - 2),
            (position[0] - 1, position[1] + 2),
            (position[0] + 1, position[1] - 2),
            (position[0] + 1, position[1] + 2),
            (position[0] + 2, position[1] - 1),
            (position[0] + 2, position[1] + 1),
        ]

        for move in possible_moves:
            # Check if the move is valid
            if 0 <= move[0] < 8 and 0 <= move[1] < 8:
                # Check if the square is empty
                if self.board[move[0], move[1]] is None:
                    moves = np.append(moves, f"{chr(move[1] + 97)}{8 - move[0]}")
                # Check if the square is occupied by an enemy piece
                elif self.board[move[0], move[1]].color != color:
                    moves = np.append(moves, f"{chr(move[1] + 97)}{8 - move[0]}")

        return moves

    def pawn_capture(self) -> np.ndarray:
        """
        Pawn capture movement, only for pawns
        :return:
        """
        moves = np.array([], dtype=str)

        return moves

    def generate_pawn_movement(self) -> np.ndarray:
        distance = 3 if self.piece.first_move else 2

        # Generate forward movement, one or two squares
        if self.piece.color == "w":
            moves = self.move_up(distance)
        else:
            moves = self.move_down(distance)

        return moves

    def generate_knight_movement(self) -> np.ndarray:
        # Generate knight movement
        moves = self.move_knight()

        return moves

    def generate_bishop_movement(self) -> np.ndarray:
        # Generate bishop movement
        moves = np.array([], dtype=str)

        directions = [self.move_up_right, self.move_up_left, self.move_down_right, self.move_down_left]

        for direction in directions:
            moves = np.append(moves, direction())

        return moves

    def generate_rook_movement(self) -> np.ndarray:
        moves = np.array([], dtype=str)

        directions = [self.move_up, self.move_down, self.move_right, self.move_left]

        for direction in directions:
            moves = np.append(moves, direction())

        return moves

    def generate_queen_movement(self) -> np.ndarray:
        moves = np.array([], dtype=str)

        directions = [self.move_up, self.move_down, self.move_right, self.move_left, self.move_up_right,
                      self.move_up_left, self.move_down_right, self.move_down_left]

        for direction in directions:
            moves = np.append(moves, direction())

        return moves

    def generate_king_movement(self) -> np.ndarray:
        moves = np.array([], dtype=str)

        directions = [self.move_up, self.move_down, self.move_right, self.move_left, self.move_up_right,
                      self.move_up_left, self.move_down_right, self.move_down_left]

        for direction in directions:
            moves = np.append(moves, direction(2))

        return moves


def generate(piece: Piece, board) -> np.ndarray:
    """
    Generate the movement of the piece
    :param piece: Piece object
    :param board: List of the board with all the pieces and None if the square is empty
    :return:
    """
    movement_object = Movement(piece, board)
    moves = np.array([])

    # Generate moves based on the type of the piece
    match piece.piece_type:
        case "P":
            moves = movement_object.generate_pawn_movement()
        case "N":
            moves = movement_object.generate_knight_movement()
        case "B":
            moves = movement_object.generate_bishop_movement()
        case "R":
            moves = movement_object.generate_rook_movement()
        case "Q":
            moves = movement_object.generate_queen_movement()
        case "K":
            moves = movement_object.generate_king_movement()

    return moves


def main():
    pass


if __name__ == '__main__':
    main()
