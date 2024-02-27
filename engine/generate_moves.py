from typing import List, Optional, Tuple
from main import Piece, Board, pos_from_chess_notation


class Movement:
    # Class that look piece movement to all directions
    def __init__(self, piece: Piece, board: Board) -> None:
        self.piece: Piece = piece
        self.board: list = board.board
        self.object: Board = board
        self.is_pawn = self.piece.piece_type == "P"

    def move_up(self, distance: int = 8) -> list:
        moves = []

        # Get the position of the piece
        position: Tuple[int, int] = pos_from_chess_notation(self.piece.position)
        color: str = self.piece.color

        for i in range(1, distance):
            # Check if the position is valid
            if position[0] - i < 0:
                break

            is_enemy = self.board[position[0] - i][position[1]] and self.board[
                position[0] - i][position[1]].color != color

            # For pawn if square has enemy break
            if self.is_pawn and is_enemy:
                break

            # If square is not empty or enemy piece, break
            if not (self.board[position[0] - i][position[1]] is None or is_enemy):
                break

            if self.object.checking_move_validity:
                moves.append(f"{chr(position[1] + 97)}{8 - (position[0] - i)}")
            elif self.object.can_move(self.piece.position, f"{chr(position[1] + 97)}{8 - (position[0] - i)}"):
                moves.append(f"{chr(position[1] + 97)}{8 - (position[0] - i)}")

            # Break if enemy piece
            if is_enemy:
                break

        return moves

    def move_down(self, distance: int = 8) -> list:
        moves = []

        # Get the position of the piece
        position: Tuple[int, int] = pos_from_chess_notation(self.piece.position)
        color: str = self.piece.color

        for i in range(1, distance):
            # Check if the position is valid
            if position[0] + i >= 8:
                break

            is_enemy = self.board[position[0] + i][position[1]] and self.board[position[0] + i][position[1]].color != color

            # For pawn if square has enemy break
            if self.is_pawn and is_enemy:
                break

            # If square is not empty or enemy piece, break
            if not (self.board[position[0] + i][position[1]] is None or is_enemy):
                break

            if self.object.checking_move_validity:
                moves.append(f"{chr(position[1] + 97)}{8 - (position[0] + i)}")
            elif self.object.can_move(self.piece.position, f"{chr(position[1] + 97)}{8 - (position[0] + i)}"):
                moves.append(f"{chr(position[1] + 97)}{8 - (position[0] + i)}")

            # Break if enemy piece
            if is_enemy:
                break

        return moves

    def move_right(self, distance: int = 8) -> list:
        moves = []

        # Get the position of the piece
        position: Tuple[int, int] = pos_from_chess_notation(self.piece.position)
        color: str = self.piece.color

        for i in range(1, distance):
            # Check if the position is valid
            if position[1] + i >= 8:
                break

            is_enemy = self.board[position[0]][position[1] + i] and self.board[
                position[0]][position[1] + i].color != color

            # If square is not empty or enemy piece, break
            if not (self.board[position[0]][position[1] + i] is None or is_enemy):
                break

            if self.object.checking_move_validity:
                moves.append(f"{chr(position[1] + 97 + i)}{8 - position[0]}")
            elif self.object.can_move(self.piece.position, f"{chr(position[1] + 97 + i)}{8 - position[0]}"):
                moves.append(f"{chr(position[1] + 97 + i)}{8 - position[0]}")

            # Break if enemy piece
            if is_enemy:
                break

        return moves

    def move_left(self, distance: int = 8) -> list:
        moves = []

        # Get the position of the piece
        position: Tuple[int, int] = pos_from_chess_notation(self.piece.position)
        color: str = self.piece.color

        for i in range(1, distance):
            # Check if the position is valid
            if position[1] - i < 0:
                break

            is_enemy = self.board[position[0]][position[1] - i] and self.board[position[0]][position[1] - i].color != color

            # If square is not empty or enemy piece, break
            if not (self.board[position[0]][position[1] - i] is None or is_enemy):
                break

            if self.object.checking_move_validity:
                moves.append(f"{chr(position[1] + 97 - i)}{8 - position[0]}")
            elif self.object.can_move(self.piece.position, f"{chr(position[1] + 97 - i)}{8 - position[0]}"):
                moves.append(f"{chr(position[1] + 97 - i)}{8 - position[0]}")

            # Break if enemy piece
            if is_enemy:
                break

        return moves

    def move_up_right(self, distance: int = 8) -> list:
        moves = []

        # Get the position of the piece
        position: Tuple[int, int] = pos_from_chess_notation(self.piece.position)
        color: str = self.piece.color

        for i in range(1, distance):
            # Check if the position is valid
            if position[0] - i < 0 or position[1] + i >= 8:
                break

            is_enemy = self.board[position[0] - i][position[1] + i] and self.board[
                position[0] - i][position[1] + i].color != color

            # If square is not empty or enemy piece, break
            if not (self.board[position[0] - i][position[1] + i] is None or is_enemy):
                break

            if self.object.checking_move_validity:
                moves.append(f"{chr(position[1] + 97 + i)}{8 - (position[0] - i)}")
            elif self.object.can_move(self.piece.position, f"{chr(position[1] + 97 + i)}{8 - (position[0] - i)}"):
                moves.append(f"{chr(position[1] + 97 + i)}{8 - (position[0] - i)}")

            # Break if enemy piece
            if is_enemy:
                break

        return moves

    def move_up_left(self, distance: int = 8) -> list:
        moves = []

        # Get the position of the piece
        position: Tuple[int, int] = pos_from_chess_notation(self.piece.position)
        color: str = self.piece.color

        for i in range(1, distance):
            # Check if the position is valid
            if position[0] - i < 0 or position[1] - i < 0:
                break

            is_enemy = self.board[position[0] - i][position[1] - i] and self.board[
                position[0] - i][position[1] - i].color != color

            # If square is not empty or enemy piece, break
            if not (self.board[position[0] - i][position[1] - i] is None or is_enemy):
                break

            if self.object.checking_move_validity:
                moves.append(f"{chr(position[1] + 97 - i)}{8 - (position[0] - i)}")
            elif self.object.can_move(self.piece.position, f"{chr(position[1] + 97 - i)}{8 - (position[0] - i)}"):
                moves.append(f"{chr(position[1] + 97 - i)}{8 - (position[0] - i)}")

            # Break if enemy piece
            if is_enemy:
                break

        return moves

    def move_down_right(self, distance: int = 8) -> list:
        moves = []

        # Get the position of the piece
        position: Tuple[int, int] = pos_from_chess_notation(self.piece.position)
        color: str = self.piece.color

        for i in range(1, distance):
            # Check if the position is valid
            if position[0] + i >= 8 or position[1] + i >= 8:
                break

            is_enemy = self.board[position[0] + i][position[1] + i] and self.board[
                position[0] + i][position[1] + i].color != color

            # If square is not empty or enemy piece, break
            if not (self.board[position[0] + i][position[1] + i] is None or is_enemy):
                break

            if self.object.checking_move_validity:
                moves.append(f"{chr(position[1] + 97 + i)}{8 - (position[0] + i)}")
            elif self.object.can_move(self.piece.position, f"{chr(position[1] + 97 + i)}{8 - (position[0] + i)}"):
                moves.append(f"{chr(position[1] + 97 + i)}{8 - (position[0] + i)}")

            # Break if enemy piece
            if is_enemy:
                break

        return moves

    def move_down_left(self, distance: int = 8) -> list:
        moves = []

        # Get the position of the piece
        position: Tuple[int, int] = pos_from_chess_notation(self.piece.position)
        color: str = self.piece.color

        for i in range(1, distance):
            # Check if the position is valid
            if position[0] + i >= 8 or position[1] - i < 0:
                break

            is_enemy = self.board[position[0] + i][position[1] - i] and self.board[
                position[0] + i][position[1] - i].color != color

            # If square is not empty or enemy piece, break
            if not (self.board[position[0] + i][position[1] - i] is None or is_enemy):
                break

            if self.object.checking_move_validity:
                moves.append(f"{chr(position[1] + 97 - i)}{8 - (position[0] + i)}")
            elif self.object.can_move(self.piece.position, f"{chr(position[1] + 97 - i)}{8 - (position[0] + i)}"):
                moves.append(f"{chr(position[1] + 97 - i)}{8 - (position[0] + i)}")

            # Break if enemy piece
            if is_enemy:
                break

        return moves

    def move_knight(self) -> list:
        moves = []

        # Get the position of the piece
        position: Tuple[int, int] = pos_from_chess_notation(self.piece.position)

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
            if not (0 <= move[0] < 8 and 0 <= move[1] < 8):
                continue

            # Check if square is friendly
            if self.board[move[0]][move[1]] and self.board[move[0]][move[1]].color == self.piece.color:
                continue

            if self.object.checking_move_validity:
                moves.append(f"{chr(move[1] + 97)}{8 - move[0]}")
            elif self.object.can_move(self.piece.position, f"{chr(move[1] + 97)}{8 - move[0]}"):
                moves.append(f"{chr(move[1] + 97)}{8 - move[0]}")

        return moves

    def pawn_capture(self) -> list:
        """
        Pawn capture movement, only for pawns
        :return:
        """
        moves = []

        # Offset for captures
        row_offset = -1 if self.piece.color == "w" else 1
        col_offset = [-1, 1]

        # Get the position of the piece
        position: Tuple[int, int] = pos_from_chess_notation(self.piece.position)

        for col in col_offset:
            new_pos = (position[0] + row_offset, position[1] + col)
            if 0 <= new_pos[0] < 8 and 0 <= new_pos[1] < 8:
                if self.board[new_pos[0]][new_pos[1]] and self.board[new_pos[0]][new_pos[1]].color != self.piece.color:
                    if self.object.checking_move_validity:
                        moves.append(f"{chr(new_pos[1] + 97)}{8 - new_pos[0]}")
                    elif self.object.can_move(self.piece.position, f"{chr(new_pos[1] + 97)}{8 - new_pos[0]}"):
                        moves.append(f"{chr(new_pos[1] + 97)}{8 - new_pos[0]}")

        # Also check for en passant
        moves.extend(self.generate_en_passant(position))

        return moves

    def generate_en_passant(self, position: Tuple[int, int]) -> list:
        """
        Generate en passant movement
        :param position: row, col of the piece
        :return:
        """
        # CHeck that the en passant square is not None
        if not self.object.en_passant_square:
            return []

        # Get en passant square position
        en_passant_square = pos_from_chess_notation(self.object.en_passant_square)

        # Check if en passant square is occupied
        if self.board[en_passant_square[0]][en_passant_square[1]]:
            return []

        # Check if piece is not even 1 row away from the en passant square
        if abs(position[0] - en_passant_square[0]) != 1:
            return []

        # Check if piece is not even 1 col away from the en passant square
        if abs(position[1] - en_passant_square[1]) != 1:
            return []

        # If white turn, check if the en passant square is diagonal to the piece
        move = []
        if self.piece.color == "w":
            # Check if top left
            if position[0] - 1 == en_passant_square[0] and position[1] - 1 == en_passant_square[1]:
                move.append(f"{chr(en_passant_square[1] + 97)}{8 - en_passant_square[0]}")
            # Check if top right
            elif position[0] - 1 == en_passant_square[0] and position[1] + 1 == en_passant_square[1]:
                move.append(f"{chr(en_passant_square[1] + 97)}{8 - en_passant_square[0]}")
        # If black turn, check if the en passant square is diagonal to the piece
        else:
            # Check if bottom left
            if position[0] + 1 == en_passant_square[0] and position[1] - 1 == en_passant_square[1]:
                move.append(f"{chr(en_passant_square[1] + 97)}{8 - en_passant_square[0]}")
            # Check if bottom right
            elif position[0] + 1 == en_passant_square[0] and position[1] + 1 == en_passant_square[1]:
                move.append(f"{chr(en_passant_square[1] + 97)}{8 - en_passant_square[0]}")

        if len(move) == 0:
            return move

        # Make certain the move is valid too
        if self.object.checking_move_validity:
            return move
        elif self.object.can_move(self.piece.position, move[0]):
            return move
        else:
            return []

    def generate_pawn_movement(self) -> list:
        distance = 3 if self.piece.first_move else 2

        moves = []

        # Generate forward movement, one or two squares
        if self.piece.color == "w":
            moves.extend(self.move_up(distance))
        else:
            moves.extend(self.move_down(distance))

        # Generate capture movement
        moves.extend(self.pawn_capture())

        return moves

    def generate_knight_movement(self) -> list:
        # Generate knight movement
        moves = self.move_knight()

        return moves

    def generate_bishop_movement(self) -> list:
        # Generate bishop movement
        moves = []

        directions = [self.move_up_right, self.move_up_left, self.move_down_right, self.move_down_left]

        for direction in directions:
            moves.extend(direction())

        return moves

    def generate_rook_movement(self) -> list:
        moves = []

        directions = [self.move_up, self.move_down, self.move_right, self.move_left]

        for direction in directions:
            moves.extend(direction())

        return moves

    def generate_queen_movement(self) -> list:
        moves = []

        directions = [self.move_up, self.move_down, self.move_right, self.move_left, self.move_up_right,
                      self.move_up_left, self.move_down_right, self.move_down_left]

        for direction in directions:
            moves.extend(direction())

        return moves

    def generate_king_movement(self) -> list:
        moves = []

        directions = [self.move_up, self.move_down, self.move_right, self.move_left, self.move_up_right,
                      self.move_up_left, self.move_down_right, self.move_down_left]

        for direction in directions:
            moves.extend(direction(2))

        # Generate castling movement
        moves.extend(self.castling())

        return moves

    def castling(self) -> list:
        moves = []
        color: str = self.piece.color
        rights: str = self.object.castling_rights

        # Check if the king has moved
        if not self.piece.first_move:
            return moves

        # Make sure kings has some castling rights
        if color == "w" and "K" not in rights and "Q" not in rights:
            return moves
        if color == "b" and "k" not in rights and "q" not in rights:
            return moves

        # For kingside castling, make sure squares inbetween are empty
        if color == "w" and "K" in rights:
            if self.board[7][5] is None and self.board[7][6] is None:
                # Make sure kingside rook has not moved
                if self.board[7][7] and self.board[7][7].piece_type == "R" and self.board[7][7].first_move:
                    moves.append("g1")
        if color == "b" and "k" in rights:
            if self.board[0][5] is None and self.board[0][6] is None:
                # Make sure kingside rook has not moved
                if self.board[0][7] and self.board[0][7].piece_type == "R" and self.board[0][7].first_move:
                    moves.append("g8")

        # For queenside castling, make sure squares inbetween are empty
        if color == "w" and "Q" in rights:
            if self.board[7][1] is None and self.board[7][2] is None and self.board[7][3] is None:
                # Make sure queenside rook has not moved
                if self.board[7][0] and self.board[7][0].piece_type == "R" and self.board[7][0].first_move:
                    moves.append("c1")
        if color == "b" and "q" in rights:
            if self.board[0][1] is None and self.board[0][2] is None and self.board[0][3] is None:
                # Make sure queenside rook has not moved
                if self.board[0][0] and self.board[0][0].piece_type == "R" and self.board[0][0].first_move:
                    moves.append("c8")

        # Loop over these moves and validate them
        valid_moves = []
        for move in moves:
            if self.object.checking_move_validity:
                valid_moves.append(move)
            elif self.object.can_move(self.piece.position, move):
                valid_moves.append(move)

        return valid_moves


def generate(piece: Piece, board: Board) -> list:
    """
    Generate the movement of the piece
    :param piece: Piece object
    :param board: List of the board with all the pieces and None if the square is empty
    :return:
    """
    movement_object = Movement(piece, board)

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
        case _:
            moves = []

    return moves


def generate_test(piece: Piece, board) -> list:
    movement_object = Movement(piece, board)
    moves = []

    new_moves = movement_object.generate_queen_movement()
    if len(new_moves) > 0:
        moves.extend(new_moves)

    new_moves = movement_object.generate_knight_movement()
    if len(new_moves) > 0:
        moves.extend(new_moves)

    return moves


def main():
    pass


if __name__ == '__main__':
    main()
