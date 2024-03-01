import time
import multiprocessing
import cProfile
import pstats
from typing import Tuple, List, Optional
import datetime as dt

# My imports
import engine.generate_moves as generate_moves
from engine.chess_engine import Engine
from engine.classes import Piece
from engine.utils import pos_from_chess_notation


def measure_time(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter_ns()
        result = func(*args, **kwargs)
        end = time.perf_counter_ns()

        print(f"Time: {(end - start) / 1_000_000} ms")

        return result

    return wrapper


class Board:
    def __init__(self, fen_string: str, white_king: Tuple[int, int] = None, black_king: Tuple[int, int] = None,
                 board_history: list = None, white_moves: list = None, black_moves: list = None) -> None:
        # Fen string to initialize the board
        self.fen_string = fen_string

        # Make array of 8x8
        self.board: List[List[Optional[Piece]]] = [[None for _ in range(8)] for _ in range(8)]

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

        # Create board
        self.create_board()

        # Moves
        self.black_moves = [] if black_moves is None else black_moves
        self.white_moves = [] if white_moves is None else white_moves

        # Attack lines for move validation and check
        self.white_attack_lines = []
        self.white_pinned_lines = []
        self.black_attack_lines = []
        self.black_pinned_lines = []

        # Pinned pieces
        self.white_pinned_pieces = []
        self.black_pinned_pieces = []

        # Generate moves
        self.generate_piece_moves("w")
        self.generate_piece_moves("b")

        # Board history
        self.board_history = [self.fen_string] if board_history is None else board_history

        # Recorded moves
        self.white_move_history = []
        self.black_move_history = []

    def generate_fen_string(self) -> None:
        """
        Generate fen string from the board
        :return:
        """
        fen_string = ""
        for row in range(8):
            empty = 0
            for col in range(8):
                if self.board[row][col] is None:
                    empty += 1
                else:
                    if empty != 0:
                        fen_string += str(empty)
                        empty = 0

                    fen_string += self.board[row][col].piece_type if self.board[row][col].color == "w" else \
                        self.board[row][col].piece_type.lower()

            if empty != 0:
                fen_string += str(empty)
            if row != 7:
                fen_string += "/"

        fen_string += f" {self.turn} {self.castling_rights} " \
                      f"{'-' if self.en_passant_square is None else self.en_passant_square}" \
                      f" {self.halfmove_clock} {self.fullmove_number}"

        self.fen_string = fen_string

    def create_board(self) -> None:
        fen_piece = self.fen_string.split(" ")[0]
        rows = fen_piece.split("/")

        for i, row in enumerate(rows):
            new_row = [None for _ in range(8)]
            col_index = 0
            real_index = 0
            while col_index < 8:
                piece = row[real_index]
                if piece.isdigit():
                    for _ in range(int(piece)):
                        new_row[col_index] = None
                        col_index += 1

                    real_index += 1
                    continue

                # Construct chess notation
                position = (i, col_index)

                # Construct the piece color
                if piece.isupper():
                    color = "w"
                else:
                    color = "b"

                # Construct the piece type
                piece_type = piece.upper()

                # Create the piece
                new_row[col_index] = Piece(position, color, piece_type)

                # If piece was king update its position
                if piece_type == "K":
                    if color == "w":
                        self.white_king = position
                    else:
                        self.black_king = position

                col_index += 1
                real_index += 1

            self.board[i] = new_row

    def generate_piece_moves(self, color: str) -> None:
        """
        Generate the moves for only for the opponent's pieces
        :return:
        """
        # First make attack lines for the color
        king_piece: Piece = self.board[self.white_king[0]][self.white_king[1]] if color == "w" else \
            self.board[self.black_king[0]][self.black_king[1]]

        try:
            real_attacks, potential_attacks = generate_moves.generate_lines(king_piece, self)
        except Exception as e:
            # Loop through the board history and if there is no k or K in fen string, print that fen and the previous
            for fen in self.board_history:
                board = fen.split(" ")[0]
                if "k" not in board or "K" not in board:
                    print(f"Illegal move fen: {self.board_history[self.board_history.index(fen) - 2]}")
                    print(f"Previous fen: {self.board_history[self.board_history.index(fen) - 1]}")
                    print(f"Current fen: {fen}")
                    print(f"Color: {color}")
                    print_board(self)
                    break

            raise e

        if color == "w":
            self.white_attack_lines = real_attacks
            self.white_pinned_lines = potential_attacks
        else:
            self.black_attack_lines = real_attacks
            self.black_pinned_lines = potential_attacks

        if color == "w":
            self.white_moves = []
        else:
            self.black_moves = []

        # Reset pinned pieces
        # self.white_pinned_pieces = []
        # self.black_pinned_pieces = []

        for row in range(8):
            for col in range(8):
                if self.board[row][col] is None or self.board[row][col].color != color:
                    continue

                # # If piece was pinned put it to the pinned pieces list
                # if self.board[row][col].pinned:
                #     self.board[row][col].pinned = False
                #     if color == "w":
                #         self.white_pinned_pieces.append(self.board[row][col])
                #     else:
                #         self.black_pinned_pieces.append(self.board[row][col])

                # Update the moves for the piece class
                self.board[row][col].moves = generate_moves.generate(self.board[row][col], self)

                if len(self.board[row][col].moves) == 0:
                    continue

                for new_move in self.board[row][col].moves:
                    # Add third item to index 0 that is the type of the piece whose move it is
                    new_move = (self.board[row][col].position, new_move)

                    if color == "w":
                        self.white_moves.append(new_move)
                    else:
                        self.black_moves.append(new_move)

        # Reset pinned pieces
        # self.white_pinned_pieces = []
        # self.black_pinned_pieces = []

    def soft_generate_piece_moves(self, color: str) -> None:
        """
        Generate the moves for only for the opponent's pieces without validation
        :return:
        """
        self.checking_move_validity = True

        if color == "w":
            self.white_moves = []
        else:
            self.black_moves = []

        for row in range(8):
            for col in range(8):
                if self.board[row][col] is None or self.board[row][col].color != color:
                    continue

                self.board[row][col].moves = generate_moves.generate(self.board[row][col], self)

                if len(self.board[row][col].moves) == 0:
                    continue

                for new_move in self.board[row][col].moves:
                    # Add third item to index 0 that is the type of the piece whose move it is
                    new_move = (self.board[row][col].position, new_move)

                    if color == "w":
                        self.white_moves.append(new_move)
                    else:
                        self.black_moves.append(new_move)

        self.checking_move_validity = False

    def can_move(self, old_pos: Tuple[int, int], new_pos: Tuple[int, int]) -> bool:
        """
        Checks from kings perspective whether it can hit enemies or not
        :param old_pos: in chess notation
        :param new_pos: in chess notation
        :return:
        """
        self.checking_move_validity = True

        # Get the piece and potential capture
        piece: Piece = self.board[old_pos[0]][old_pos[1]]
        potential_capture = self.board[new_pos[0]][new_pos[1]]

        # Move the piece
        self.board[new_pos[0]][new_pos[1]] = piece
        self.board[old_pos[0]][old_pos[1]] = None

        # King info
        if piece.piece_type != "K":
            king_position = self.white_king if piece.color == "w" else self.black_king
        else:
            king_position = new_pos
            self.board[new_pos[0]][new_pos[1]].position = new_pos

        # Generate every movement from kings perspective
        king_moves = generate_moves.generate_test(self.board[king_position[0]][king_position[1]], self)

        # Flag to know whether move is valid or not
        flag = True

        # Loop through the kings move and find enemies
        for move in king_moves:
            # If the move is not empty and the piece is of the opposite color and king position is in enemies moves
            if self.board[move[0]][move[1]] is not None and self.board[move[0]][move[1]].color != piece.color:
                # Generate move for that enemy piece
                enemy_moves = generate_moves.generate(self.board[move[0]][move[1]], self)

                # If the king position is in the enemies moves, return True
                if king_position in enemy_moves:
                    flag = False
                    break

        # Revert the board back to original state
        self.board[new_pos[0]][new_pos[1]] = potential_capture
        self.board[old_pos[0]][old_pos[1]] = piece

        # Revert king position
        if piece.piece_type == "K":
            if piece.color == "w":
                self.white_king = old_pos
                self.board[old_pos[0]][old_pos[1]].position = old_pos
            else:
                self.black_king = old_pos
                self.board[old_pos[0]][old_pos[1]].position = old_pos

        self.checking_move_validity = False

        return flag

    def can_move_2(self, old_pos: Tuple[int, int], new_pos: Tuple[int, int]) -> bool:
        """
        1. Check if the piece is one of the pinned pieces, if so then it can only move to any square in that specific pin line
        Also if there is real attack line, then it is illegal to move
        2. Check for real attacks and if this piece is not pinned and can move to the real attack line, if not then it is illegal to move
        3. In any other case the move is legal
        :param old_pos:
        :param new_pos:
        :return:
        """
        # Get piece
        piece = self.board[old_pos[0]][old_pos[1]]

        if piece is None:
            # Print board and old and new position if piece is None
            print_board(self)
            print(f"Old: {old_pos} New: {new_pos}")
            raise Exception("Piece is None")

        # Get attack lines for the color
        attack_lines = self.white_attack_lines if piece.color == "w" else self.black_attack_lines
        pinned_lines = self.white_pinned_lines if piece.color == "w" else self.black_pinned_lines

        # If piece is king, then have to recalculate attack lines
        if piece.piece_type == "K":
            # check if new kings position is 1 square away from enemy king position
            if piece.color == "w":
                if abs(self.black_king[0] - new_pos[0]) <= 1 and abs(self.black_king[1] - new_pos[1]) <= 1:
                    return False
            else:
                if abs(self.white_king[0] - new_pos[0]) <= 1 and abs(self.white_king[1] - new_pos[1]) <= 1:
                    return False

            # Move king momentarily to new position, generate new attack lines and check if it is safe
            potential_capture = self.board[new_pos[0]][new_pos[1]]
            king_old_position = piece.position
            self.board[new_pos[0]][new_pos[1]] = piece
            self.board[old_pos[0]][old_pos[1]] = None
            piece.position = new_pos

            at_lines, _ = generate_moves.generate_lines(piece, self)

            self.board[new_pos[0]][new_pos[1]] = potential_capture
            self.board[old_pos[0]][old_pos[1]] = piece
            piece.position = king_old_position

            if len(at_lines) == 0:
                return True
            else:
                return False

        # If attack lines and pinned lines empty move is legal
        if len(attack_lines) == 0 and len(pinned_lines) == 0:
            return True

        # Find if the pieces old position is on the pinned line, if so then the piece is pinned
        pin_line = None
        is_pinned = False

        for line in pinned_lines:
            for pos in line:
                if pos == old_pos:
                    pin_line = line
                    is_pinned = True
                    break

        # Check if there are any attack lines and if the piece is pinned
        if len(attack_lines) != 0 and is_pinned:
            return False

        # Check if the move is in the pinned lines
        if is_pinned and len(attack_lines) == 0:
            if pin_line is None:
                print(f"Piece: {piece.piece_type} {piece.color} Old: {old_pos} New: {new_pos}")
                print(f"Attack lines: {attack_lines}")
                print(f"Pinned lines: {pinned_lines}")
                print(f"White pinned pieces: {self.white_pinned_pieces}")
                print(f"Black pinned pieces: {self.black_pinned_pieces}")
                print(f"Move: {old_pos} -> {new_pos}")
                print_board(self)
                print(f"FEN: {self.fen_string}")
                raise Exception("Pin line not found, but piece was pinned")

            # If the move is not in the pinned line, return False
            if new_pos in pin_line:
                return True
            else:
                return False

        if not is_pinned and len(attack_lines) == 0:
            return True

        # Make sure the piece can move to block the attack
        if not is_pinned and len(attack_lines) == 1 and new_pos in attack_lines[0]:
            return True
        elif not is_pinned and len(attack_lines) == 1 and new_pos not in attack_lines[0]:
            return False

        # King has to move to safety if more than 1 attack line
        if len(attack_lines) > 1 and piece.piece_type != "K":
            return False
        elif len(attack_lines) > 1 and piece.piece_type == "K":
            # Move king momentarily to new position, generate new attack lines and check if it is safe
            potential_capture = self.board[new_pos[0]][new_pos[1]]
            king_old_position = piece.position
            self.board[new_pos[0]][new_pos[1]] = piece
            self.board[old_pos[0]][old_pos[1]] = None
            piece.position = new_pos

            at_lines, _ = generate_moves.generate_lines(piece, self)

            self.board[new_pos[0]][new_pos[1]] = potential_capture
            self.board[old_pos[0]][old_pos[1]] = piece
            piece.position = king_old_position

            if len(at_lines) == 0:
                return True
            else:
                return False

        print(f"Piece: {piece.piece_type} {piece.color} Old: {old_pos} New: {new_pos}")
        print(f"Attack lines: {attack_lines}")
        print(f"Pinned lines: {pinned_lines}")
        print(f"White pinned pieces: {self.white_pinned_pieces}")
        print(f"Black pinned pieces: {self.black_pinned_pieces}")
        print_board(self)
        print(f"FEN: {self.fen_string}")
        raise Exception("Illegal move")

    def is_check(self, color) -> bool:
        """
        Check if the color's king is in check by checking if opposite color's pieces have the king in their moves
        :return:
        """
        king_pos = self.white_king if color == "w" else self.black_king

        for row in range(8):
            for col in range(8):
                # If the piece is None, continue
                if self.board[row][col] is None:
                    continue

                # If the piece is of the same color, continue
                if self.board[row][col].color == color:
                    continue

                # If the piece is a king, continue
                if self.board[row][col].piece_type == "K":
                    continue

                # If the kings position is in the moves of the piece, set the king's in_check to True
                for move in self.board[row][col].moves:
                    if king_pos == move[1]:
                        return True

        return False

    def is_check_2(self) -> bool:
        """
        Check if it is check for whose turn it is, by checking attack lines
        :return:
        """
        # Get attack lines for turn
        attack_lines = self.white_attack_lines if self.turn == "w" else self.black_attack_lines

        # If the attack lines are empty, return False
        if len(attack_lines) == 0:
            return False
        else:
            return True

    def is_checkmate(self, color) -> bool:
        """
        Check if the color is in checkmate
        :param color: Color of the king
        :return: True if the color is in checkmate else False
        """
        # If the king is not in check, return False
        if not self.is_check_2():
            return False

        # Loop through all the pieces
        for row in range(8):
            for col in range(8):
                # If the piece is None, continue
                if self.board[row][col] is None:
                    continue

                # If the piece is not of the same color, continue
                if self.board[row][col].color != color:
                    continue

                # If the piece has moves, return False
                if len(self.board[row][col].moves) > 0:
                    return False

        return True

    def is_stalemate(self, color) -> bool:
        """
        Check if the color is in stalemate
        :param color: Color of the king
        :return: True if the color is in stalemate else False
        """
        # If the king is in check, return False
        if self.is_check_2():
            return False

        # Loop through all the pieces
        for row in range(8):
            for col in range(8):
                # If the piece is None, continue
                if self.board[row][col] is None:
                    continue

                # If the piece is not of the same color, continue
                if self.board[row][col].color != color:
                    continue

                # If the piece has moves, return False
                if len(self.board[row][col].moves) > 0:
                    return False

        return True

    def update(self, piece_type: str, was_capture: bool) -> None:
        """
        Updates all the pieces if they are pinned, kings are in check, checkmate, stalemate, turn, fen string, etc..
        :param piece_type: Type of the piece
        :param was_capture: If the move was a capture
        :return:
        """
        # Generate the moves for all the pieces
        self.generate_piece_moves("w" if self.turn == "b" else "b")

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

        # Generate fen string
        self.generate_fen_string()

        # Update board history
        self.board_history.append(self.fen_string)

    def pawn_promotion(self, position: Tuple[int, int], piece_type: str = "Q") -> None:
        """
        Promote the pawn to a piece
        :param position: Position of the pawn
        :param piece_type: Type of the piece to promote to, default to queen
        :return:
        """
        self.board[position[0]][position[1]].piece_type = piece_type

    def undo_move(self) -> None:
        """
        Pop the previous fen string and reconstruct the board
        :return:
        """
        # Pop the latest fen string
        self.board_history.pop()

        # Get previous fen string
        previous_fen = self.board_history[-1]

        # Save init vars
        board_save = self.board_history

        # Update the board
        self.__init__(previous_fen)

        # Restore init vars
        self.board_history = board_save

        # Find the new locations for white and black king positions
        for row in range(8):
            for col in range(8):
                if self.board[row][col] is not None:
                    if self.board[row][col].piece_type == "K":
                        if self.board[row][col].color == "w":
                            self.white_king = self.board[row][col].position
                        else:
                            self.black_king = self.board[row][col].position

    def move(self, old_pos, new_pos) -> None:
        """
        Move the piece
        Positions in chess notation as parameters
        :param old_pos: Old position of the piece
        :param new_pos: New position of the piece
        :return:
        """

        # Check if the move was capture
        was_capture = True if self.board[new_pos[0]][new_pos[1]] is not None else False

        # Move the piece
        self.board[new_pos[0]][new_pos[1]] = self.board[old_pos[0]][old_pos[1]]
        self.board[old_pos[0]][old_pos[1]] = None

        # Update the position of the piece
        self.board[new_pos[0]][new_pos[1]].position = new_pos

        # Make first move False, specifically for pawns
        self.board[new_pos[0]][new_pos[1]].first_move = False

        # If the piece is a king, update the king position
        if self.board[new_pos[0]][new_pos[1]].piece_type == "K":
            if self.board[new_pos[0]][new_pos[1]].color == "w":
                self.white_king = (new_pos[0], new_pos[1])
            else:
                self.black_king = (new_pos[0], new_pos[1])

        # If the move wes a pawn, it was a double move, update en passant square
        if self.board[new_pos[0]][new_pos[1]].piece_type == "P" and abs(old_pos[0] - new_pos[0]) == 2:
            self.en_passant_square = f"{chr(97 + new_pos[1])}{8 - new_pos[0] - 1}" if self.board[new_pos[0]][new_pos[
                1]].color == "w" else f"{chr(97 + new_pos[1])}{8 - new_pos[0] + 1}"
        else:
            self.en_passant_square = None

        # Handle castling, move the rook also
        if self.board[new_pos[0]][new_pos[1]].piece_type == "K" and abs(old_pos[1] - new_pos[1]) == 2:
            # Queen side
            if new_pos[1] == 2:
                self.board[new_pos[0]][3] = self.board[new_pos[0]][0]
                self.board[new_pos[0]][0] = None
                # Update rook position and make first move False
                self.board[new_pos[0]][3].position = (7, 3) if self.board[new_pos[0]][3].color == "w" else (0, 3)
                self.board[new_pos[0]][3].first_move = False
            else:
                self.board[new_pos[0]][5] = self.board[new_pos[0]][7]
                self.board[new_pos[0]][7] = None
                # Update rook position and make first move False
                self.board[new_pos[0]][5].position = (7, 5) if self.board[new_pos[0]][5].color == "w" else (0, 5)
                self.board[new_pos[0]][5].first_move = False

        # Remove castling rights if king or rook moved
        if self.board[new_pos[0]][new_pos[1]].piece_type == "K" or self.board[new_pos[0]][new_pos[1]].piece_type == "R":
            if self.board[new_pos[0]][new_pos[1]].color == "w":
                if self.board[new_pos[0]][new_pos[1]].piece_type == "K":
                    self.castling_rights = self.castling_rights.replace("K", "")
                    self.castling_rights = self.castling_rights.replace("Q", "")
                elif new_pos[1] == 0:
                    self.castling_rights = self.castling_rights.replace("Q", "")
                elif new_pos[1] == 7:
                    self.castling_rights = self.castling_rights.replace("K", "")
            else:
                if self.board[new_pos[0]][new_pos[1]].piece_type == "K":
                    self.castling_rights = self.castling_rights.replace("k", "")
                    self.castling_rights = self.castling_rights.replace("q", "")
                elif new_pos[1] == 0:
                    self.castling_rights = self.castling_rights.replace("q", "")
                elif new_pos[1] == 7:
                    self.castling_rights = self.castling_rights.replace("k", "")

        # If pawn reached promotion, promote it to queen
        if self.board[new_pos[0]][new_pos[1]].piece_type == "P" and (new_pos[0] == 0 or new_pos[0] == 7):
            self.pawn_promotion(new_pos)

        # Record the move for the color
        if self.turn == "w":
            self.white_move_history.append((old_pos, new_pos))
        else:
            self.black_move_history.append((old_pos, new_pos))

        # Update the board
        self.update(self.board[new_pos[0]][new_pos[1]].piece_type, was_capture)


class Game:

    def __init__(self) -> None:
        self.board = Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        self.white_move_history: list = []
        self.black_move_history: list = []
        self.game_over = False

    def is_threefold_repetition(self) -> bool:
        """
        Check if the game is over by threefold repetition
        Count how many times the current board is in the board history
        :return: True if the game is over else False
        """

        # Loop through white boards
        unique_board_counts = {}

        for board in self.board.board_history:
            board = board.split(" ")[0]
            if board in unique_board_counts:
                unique_board_counts[board] += 1

                # Check if counts go to 3
                if unique_board_counts[board] == 3:
                    return True

            else:
                unique_board_counts[board] = 1

        # Delete dict from memory
        del unique_board_counts

        return False

    def move(self, old_pos, new_pos) -> None:
        self.board.move(old_pos, new_pos)

        if self.board.turn == "w":
            self.white_move_history.append((old_pos, new_pos))
        else:
            self.black_move_history.append((old_pos, new_pos))

        # Check if the game is over
        if self.board.is_checkmate("w"):
            self.game_over = True
            print("Black wins")
        elif self.board.is_checkmate("b"):
            self.game_over = True
            print("White wins")
        elif self.board.is_stalemate("w") or self.board.is_stalemate("b"):
            self.game_over = True
            print("Stalemate")
        elif self.is_threefold_repetition():
            self.game_over = True
            print("Threefold repetition")

    def undo_move(self) -> None:
        self.board.undo_move()

        if self.board.turn == "w":
            self.white_move_history.pop()
        else:
            self.black_move_history.pop()


def print_moves_for_all_pieces(board) -> None:
    for row in range(8):
        for col in range(8):
            if board.board[row][col] is not None:
                print(f"Piece: {board.board[row][col].piece_type} {board.board[row][col].color}", end=" ")
                print(f"Position: {board.board[row][col].position}", end=" ")
                print(f"Moves: {board.board[row][col].moves}")


def print_board(board) -> None:
    print("\n")
    print("#" * 50)

    for row in range(8):
        for col in range(8):
            if board.board[row][col] is not None:
                print(
                    f"{board.board[row][col].piece_type.upper() if board.board[row][col].color == 'w' else board.board[row][col].piece_type.lower():^4}",
                    end=" ")
            else:
                print("None", end=" ")
        print("")


def game_test():
    game = Game()
    engine = Engine(game)

    # print(engine.move_generation_test(1))
    # for move in game.board.white_moves:
    #     old_pos = f"{chr(97 + move[0][1])}{8 - move[0][0]}"
    #     new_pos = f"{chr(97 + move[1][1])}{8 - move[1][0]}"
    #     print(f"Old: {old_pos} New: {new_pos}")

    # print(len(game.board.white_moves))
    # print(engine.move_generation_test(3))


def run_game() -> int:
    game = Game()
    engine = Engine(game)

    index = 0

    while not game.game_over:
        # Make copy of the game.board for engine
        move = engine.random_move()
        game.move(move[0], move[1])

        # # If index is divisible by 2, un-do move
        # if index % 4 == 0:
        #     game.un_do_move()

        index += 1

    print(f"Turns played: {index // 2}")

    return len(game.black_move_history)


def run_multiple_games(count: int) -> float:
    game_moves = []

    for _ in range(count):
        move_counter = run_game()
        game_moves.append(move_counter)

    return sum(game_moves) / len(game_moves)


def run_multiprocessing():
    games = 1_000
    games_per_core = games // 16
    start_time = time.perf_counter_ns()

    # Run run_multiple_games in parallel with 16 cores, combine the results for time and average moves
    with multiprocessing.Pool(16) as pool:
        results = pool.map(run_multiple_games, [games_per_core] * 16)

    average_moves = 0

    for result in results:
        average_moves += result

    end_time = time.perf_counter_ns()

    # Current time in format: 2021_08_29_20_00_00
    current_time = dt.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    # Save the results to a file
    save_path = f"logs/{current_time}.log"

    with open(save_path, "w+") as file:
        file.write(f"Results:\n")
        file.write(f"[Games] {games}\n")
        file.write(f"[Time] {(end_time - start_time) / 1_000_000_000: .2f} s\n")
        file.write(f"[Average turns] {average_moves / 16: .2f}\n")
        file.write(f"[Average time per game] {(end_time - start_time) / games / 1_000_000: .2f} ms\n")
        file.write(
            f"[Average time per move] {(end_time - start_time) / (average_moves * 2 * games) / 1_000_000: .4f} ms\n")

    print("\n" + "#" * 50)
    print("Results:")
    print(f"[Games] {games}")
    print(f"[Time] {(end_time - start_time) / 1_000_000_000: .2f} s")
    print(f"[Average turns] {average_moves / 16: .2f}")
    print(f"[Average time per game] {(end_time - start_time) / games / 1_000_000: .2f} ms")
    print(f"[Average time per move] {(end_time - start_time) / (average_moves * 2 * games) / 1_000_000: .4f} ms")
    print("#" * 50)


def profiling():
    with cProfile.Profile() as pr:
        # test()
        # run_game()
        game_test()

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats()


def check_board_size():
    import sys

    # Get the size in kilobytes
    board = Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    size = sys.getsizeof(board) / 1024
    print(f"Size: {size} KB")


def main():
    # game_test()
    # run_game()
    # run_multiprocessing()
    profiling()
    # check_board_size()
    # run_multiple_games(100)
    pass


if __name__ == "__main__":
    main()
