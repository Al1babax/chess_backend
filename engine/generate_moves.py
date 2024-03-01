from typing import List, Optional, Tuple
from engine.utils import pos_from_chess_notation
from engine.classes import Piece, Board


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
        position: Tuple[int, int] = self.piece.position
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
                moves.append((position[0] - i, position[1]))
            elif self.object.can_move_2(self.piece.position, (position[0] - i, position[1])):
                moves.append((position[0] - i, position[1]))

            # Break if enemy piece
            if is_enemy:
                break

        return moves

    def move_down(self, distance: int = 8) -> list:
        moves = []

        # Get the position of the piece
        position: Tuple[int, int] = self.piece.position
        color: str = self.piece.color

        for i in range(1, distance):
            # Check if the position is valid
            if position[0] + i >= 8:
                break

            is_enemy = self.board[position[0] + i][position[1]] and self.board[position[0] + i][
                position[1]].color != color

            # For pawn if square has enemy break
            if self.is_pawn and is_enemy:
                break

            # If square is not empty or enemy piece, break
            if not (self.board[position[0] + i][position[1]] is None or is_enemy):
                break

            if self.object.checking_move_validity:
                moves.append((position[0] + i, position[1]))
            elif self.object.can_move_2(self.piece.position, (position[0] + i, position[1])):
                moves.append((position[0] + i, position[1]))

            # Break if enemy piece
            if is_enemy:
                break

        return moves

    def move_right(self, distance: int = 8) -> list:
        moves = []

        # Get the position of the piece
        position: Tuple[int, int] = self.piece.position
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
                moves.append((position[0], position[1] + i))
            elif self.object.can_move_2(self.piece.position, (position[0], position[1] + i)):
                moves.append((position[0], position[1] + i))

            # Break if enemy piece
            if is_enemy:
                break

        return moves

    def move_left(self, distance: int = 8) -> list:
        moves = []

        # Get the position of the piece
        position: Tuple[int, int] = self.piece.position
        color: str = self.piece.color

        for i in range(1, distance):
            # Check if the position is valid
            if position[1] - i < 0:
                break

            is_enemy = self.board[position[0]][position[1] - i] and self.board[position[0]][
                position[1] - i].color != color

            # If square is not empty or enemy piece, break
            if not (self.board[position[0]][position[1] - i] is None or is_enemy):
                break

            if self.object.checking_move_validity:
                moves.append((position[0], position[1] - i))
            elif self.object.can_move_2(self.piece.position, (position[0], position[1] - i)):
                moves.append((position[0], position[1] - i))

            # Break if enemy piece
            if is_enemy:
                break

        return moves

    def move_up_right(self, distance: int = 8) -> list:
        moves = []

        # Get the position of the piece
        position: Tuple[int, int] = self.piece.position
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
                moves.append((position[0] - i, position[1] + i))
            elif self.object.can_move_2(self.piece.position, (position[0] - i, position[1] + i)):
                moves.append((position[0] - i, position[1] + i))

            # Break if enemy piece
            if is_enemy:
                break

        return moves

    def move_up_left(self, distance: int = 8) -> list:
        moves = []

        # Get the position of the piece
        position: Tuple[int, int] = self.piece.position
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
                moves.append((position[0] - i, position[1] - i))
            elif self.object.can_move_2(self.piece.position, (position[0] - i, position[1] - i)):
                moves.append((position[0] - i, position[1] - i))

            # Break if enemy piece
            if is_enemy:
                break

        return moves

    def move_down_right(self, distance: int = 8) -> list:
        moves = []

        # Get the position of the piece
        position: Tuple[int, int] = self.piece.position
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
                moves.append((position[0] + i, position[1] + i))
            elif self.object.can_move_2(self.piece.position, (position[0] + i, position[1] + i)):
                moves.append((position[0] + i, position[1] + i))

            # Break if enemy piece
            if is_enemy:
                break

        return moves

    def move_down_left(self, distance: int = 8) -> list:
        moves = []

        # Get the position of the piece
        position: Tuple[int, int] = self.piece.position
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
                moves.append((position[0] + i, position[1] - i))
            elif self.object.can_move_2(self.piece.position, (position[0] + i, position[1] - i)):
                moves.append((position[0] + i, position[1] - i))

            # Break if enemy piece
            if is_enemy:
                break

        return moves

    def move_knight(self) -> list:
        moves = []

        # Get the position of the piece
        position: Tuple[int, int] = self.piece.position

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
                moves.append((move[0], move[1]))
            elif self.object.can_move_2(self.piece.position, (move[0], move[1])):
                moves.append((move[0], move[1]))

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
        position: Tuple[int, int] = self.piece.position

        for col in col_offset:
            new_pos = (position[0] + row_offset, position[1] + col)
            if 0 <= new_pos[0] < 8 and 0 <= new_pos[1] < 8:
                if self.board[new_pos[0]][new_pos[1]] and self.board[new_pos[0]][new_pos[1]].color != self.piece.color:
                    if self.object.checking_move_validity:
                        moves.append((new_pos[0], new_pos[1]))
                    elif self.object.can_move_2(self.piece.position, (new_pos[0], new_pos[1])):
                        moves.append((new_pos[0], new_pos[1]))

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
                move.append((en_passant_square[0], en_passant_square[1]))
            # Check if top right
            elif position[0] - 1 == en_passant_square[0] and position[1] + 1 == en_passant_square[1]:
                move.append((en_passant_square[0], en_passant_square[1]))
        # If black turn, check if the en passant square is diagonal to the piece
        else:
            # Check if bottom left
            if position[0] + 1 == en_passant_square[0] and position[1] - 1 == en_passant_square[1]:
                move.append((en_passant_square[0], en_passant_square[1]))
            # Check if bottom right
            elif position[0] + 1 == en_passant_square[0] and position[1] + 1 == en_passant_square[1]:
                move.append((en_passant_square[0], en_passant_square[1]))

        if len(move) == 0:
            return move

        # Make certain the move is valid too
        if self.object.checking_move_validity:
            return move
        elif self.object.can_move_2(self.piece.position, move[0]):
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
                    moves.append((7, 6))
        if color == "b" and "k" in rights:
            if self.board[0][5] is None and self.board[0][6] is None:
                # Make sure kingside rook has not moved
                if self.board[0][7] and self.board[0][7].piece_type == "R" and self.board[0][7].first_move:
                    moves.append((0, 6))

        # For queenside castling, make sure squares inbetween are empty
        if color == "w" and "Q" in rights:
            if self.board[7][1] is None and self.board[7][2] is None and self.board[7][3] is None:
                # Make sure queenside rook has not moved
                if self.board[7][0] and self.board[7][0].piece_type == "R" and self.board[7][0].first_move:
                    moves.append((7, 2))
        if color == "b" and "q" in rights:
            if self.board[0][1] is None and self.board[0][2] is None and self.board[0][3] is None:
                # Make sure queenside rook has not moved
                if self.board[0][0] and self.board[0][0].piece_type == "R" and self.board[0][0].first_move:
                    moves.append((0, 2))

        # Loop over these moves and validate them
        valid_moves = []
        for move in moves:
            if self.object.checking_move_validity:
                valid_moves.append(move)
            elif self.object.can_move_2(self.piece.position, move):
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


class AttackLines:
    """Class to get real attack lines and pinning attack lines from kings perspective to check move validity"""

    def __init__(self, king, board):
        self.attack_lines = []
        self.pinning_lines = []
        self.king: Piece = king
        self.king_position = king.position
        self.board: Board = board

    def generate_attack_lines(self):
        directions = [self.move_up, self.move_down, self.move_right, self.move_left, self.move_up_right,
                      self.move_up_left, self.move_down_right, self.move_down_left, self.move_knight]

        for direction in directions:
            direction()

    def normal_move(self, direction):
        attack_line = []
        possible_pin_piece = None
        pinning_line = False
        jumped_over = False
        enemy_found = False

        directions = {
            "up": (-1, 0),
            "down": (1, 0),
            "right": (0, 1),
            "left": (0, -1),
        }

        # Travel up from kings position, to find an enemy, can jump over ONE friendly piece and if that happens
        # The possible attack line becomes pinning line IF enemy is found after that, which can attack the king (queen, rook)
        # Update the friendly piece to be pinned and extend the classes attack_lines or pinning_lines accordingly

        # Temp location for the king, not to mess up the original for other functions in class
        temp_location = self.king_position

        while not enemy_found:
            # Update the temp location to move up
            temp_location = (temp_location[0] + directions[direction][0], temp_location[1] + directions[direction][1])

            # Check if the position is valid
            if not (0 <= temp_location[0] < 8 and 0 <= temp_location[1] < 8):
                break

            # Get the square
            square = self.board.board[temp_location[0]][temp_location[1]]

            # Check if the square is empty
            if square is None:
                attack_line.append(temp_location)
                continue

            # Check if the square has an enemy piece (rook or queen) if any other enemy then return
            if square.color != self.king.color:
                if square.piece_type not in ["R", "Q"]:
                    return
                elif square.piece_type in ["R", "Q"]:
                    attack_line.append(temp_location)
                    enemy_found = True
                    break
            # Check if the square has a friendly piece
            elif square.color == self.king.color:
                if jumped_over is False:
                    pinning_line = True
                    possible_pin_piece = (temp_location[0], temp_location[1])
                    jumped_over = True
                    attack_line.append(temp_location)
                    continue
                else:
                    break

        # If no enemy found, return
        if not enemy_found:
            return

        # If enemy found, extend the attack_lines or pinning_lines
        if pinning_line:
            # Update the friendly piece to be pinned
            self.board.board[possible_pin_piece[0]][possible_pin_piece[1]].pinned = True
            self.pinning_lines.append(attack_line)
        else:
            self.attack_lines.append(attack_line)

    def diagonal_move(self, direction):
        attack_line = []
        possible_pin_piece = None
        pinning_line = False
        jumped_over = False
        enemy_found = False
        turn = self.king.color

        directions = {
            "up_right": (-1, 1),
            "up_left": (-1, -1),
            "down_right": (1, 1),
            "down_left": (1, -1),
        }

        # Travel up from kings position, to find an enemy, can jump over ONE friendly piece and if that happens
        # The possible attack line becomes pinning line IF enemy is found after that, which can attack the king (bishop or queen)
        # Update the friendly piece to be pinned and extend the classes attack_lines or pinning_lines accordingly

        # Pawn can also be the enemy but only if it is 1 diagonal square away and black can only capture down and white up

        # Temp location for the king, not to mess up the original for other functions in class
        temp_location = self.king_position
        iterations = 0

        while not enemy_found:
            # Update the temp location to move up
            temp_location = (temp_location[0] + directions[direction][0], temp_location[1] + directions[direction][1])

            # Check if the position is valid
            if not (0 <= temp_location[0] < 8 and 0 <= temp_location[1] < 8):
                break

            # Get the square
            square = self.board.board[temp_location[0]][temp_location[1]]

            # Check if the square is empty
            if square is None:
                attack_line.append(temp_location)
                continue

            # Check if the square has an enemy piece (bishop or queen) if any other enemy then return
            if square.color != self.king.color:
                if square.piece_type not in ["B", "Q", "P"]:
                    return
                # Check if the square has an enemy piece (bishop or queen) if any other enemy then return
                elif square.piece_type in ["B", "Q"]:
                    attack_line.append(temp_location)
                    enemy_found = True
                    break
                # Check if piece is pawn and iterations is 0 meaning it is 1 diagonal square away
                # Also that if it is white king then black pawn can be found on up-left or up-right and vice versa
                elif square.piece_type == "P" and iterations == 0 and (
                        (temp_location[0] == self.king_position[0] - 1 and turn == "w") or (temp_location[0] ==
                                                                                            self.king_position[
                                                                                                0] + 1 and turn == "b")):
                    attack_line.append(temp_location)
                    enemy_found = True
                    break
            # Check if the square has a friendly piece
            elif square.color == self.king.color:
                if jumped_over is False:
                    pinning_line = True
                    possible_pin_piece = (temp_location[0], temp_location[1])
                    jumped_over = True
                    attack_line.append(temp_location)
                    continue
                else:
                    break

            iterations += 1

        # If no enemy found, return
        if not enemy_found:
            return

        # If enemy found, extend the attack_lines or pinning_lines
        if pinning_line:
            # Update the friendly piece to be pinned
            self.board.board[possible_pin_piece[0]][possible_pin_piece[1]].pinned = True
            self.pinning_lines.append(attack_line)
        else:
            self.attack_lines.append(attack_line)

    def knight_move(self):
        # Generate knight movement
        possible_moves = [
            (self.king_position[0] - 2, self.king_position[1] - 1),
            (self.king_position[0] - 2, self.king_position[1] + 1),
            (self.king_position[0] - 1, self.king_position[1] - 2),
            (self.king_position[0] - 1, self.king_position[1] + 2),
            (self.king_position[0] + 1, self.king_position[1] - 2),
            (self.king_position[0] + 1, self.king_position[1] + 2),
            (self.king_position[0] + 2, self.king_position[1] - 1),
            (self.king_position[0] + 2, self.king_position[1] + 1),
        ]

        for move in possible_moves:
            # Check if the move is valid
            if not (0 <= move[0] < 8 and 0 <= move[1] < 8):
                continue

            square = self.board.board[move[0]][move[1]]

            # Check if square is friendly
            if square and square.color == self.king.color:
                continue

            # If square is empty or enemy piece, add to attack lines
            if square and square.color != self.king.color and square.piece_type == "N":
                self.attack_lines.append([move])

    def move_up(self):
        self.normal_move("up")

    def move_down(self):
        self.normal_move("down")

    def move_right(self):
        self.normal_move("right")

    def move_left(self):
        self.normal_move("left")

    def move_up_right(self):
        self.diagonal_move("up_right")

    def move_up_left(self):
        self.diagonal_move("up_left")

    def move_down_right(self):
        self.diagonal_move("down_right")

    def move_down_left(self):
        self.diagonal_move("down_left")

    def move_knight(self):
        self.knight_move()


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


def generate_lines(king: Piece, board: Board) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Generate attack lines for the king from all possible movement.
    First do up, down, left, right and if you find enemy piece make sure that enemy piece can also attack the king.
    With up,dow,left,right only queen and rook can attack the king.
    For up-left, up-right, down-left, down-right only queen and bishop can attack the king OR pawn if max 1 square away
    right direction.
    For knight moves, only knight can attack the king.
    No need to generate moves for that enemy piece to know if it can attack based on previous logic.

    From all these function calls get back two different lists, one for direct attack lines and one for pinning attack
    lines.
    Also, when jumping over own piece once and finding threat behind it that can attack, make that own piece is_pinned for
    the object.
    :param king:
    :param board:
    :return:
    """
    attack_obj = AttackLines(king, board)
    attack_obj.generate_attack_lines()

    return attack_obj.attack_lines, attack_obj.pinning_lines


def main():
    pass


if __name__ == '__main__':
    main()
