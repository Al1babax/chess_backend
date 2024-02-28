import time
import multiprocessing
import cProfile
import pstats
from typing import Tuple, List
import datetime as dt

# My imports
import generate_moves
from engine.chess_engine import Engine


# TODO: After piece moves, check if that new position is direct attack to enemy king or potential attack to enemy king
# TODO: IF save save it to black or white direct or potential attack list (piece_location, [list_of_square_attacks])
# TODO: Also if it was potential attack make  that piece on the way to be pinned

# TODO: Change can_move function to check the enemies direct_attack list and potential attack list


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
        self.first_move: bool = True if piece_type == "P" and position[1] in [2, 7] else False

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

        # Create board
        self.create_board()

        # Attack lines
        self.white_direct_attack: List[tuple, list] = [(), []]
        self.white_potential_attack: List[tuple, list] = [(), []]
        self.black_direct_attack: List[tuple, list] = [(), []]
        self.black_potential_attack: List[tuple, list] = [(), []]

        # Moves
        self.black_moves = [] if black_moves is None else black_moves
        self.white_moves = [] if white_moves is None else white_moves

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
        if color == "w":
            self.white_moves = []
        else:
            self.black_moves = []

        for row in range(8):
            for col in range(8):
                if self.board[row][col] is None or self.board[row][col].color != color:
                    continue

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

    def is_checkmate(self, color) -> bool:
        """
        Check if the color is in checkmate
        :param color: Color of the king
        :return: True if the color is in checkmate else False
        """
        # If the king is not in check, return False
        if not self.is_check(color):
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
        if self.is_check(color):
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
        self.soft_generate_piece_moves(self.turn)

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

    def update_attack_lines(self, piece: Piece):
        """
        Update the attack lines for the piece
        1. Generate direct attacks by generating the possible moves for the piece, if move could attack the king, save it
        2. Generate potential attacks by generating the possible moves for the piece, if move could attack the king jumping over one enemy piece, save it
        :param piece:
        :return:
        """
        pass

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

        # Piece that was moved
        prev_piece_position = self.white_moves[-1][1] if self.turn == "b" else self.black_moves[-1][1]
        prev_piece = self.board[prev_piece_position[0]][prev_piece_position[1]]

        # If prev_piece_position in that colors attack lines remove it
        if prev_piece.color == "w":
            for attack_line in self.white_direct_attack:
                if prev_piece_position in attack_line[0]:
                    self.white_direct_attack.remove(attack_line)

            for attack_line in self.white_potential_attack:
                if prev_piece_position in attack_line[0]:
                    self.white_potential_attack.remove(attack_line)
        else:
            for attack_line in self.black_direct_attack:
                if prev_piece_position in attack_line[0]:
                    self.black_direct_attack.remove(attack_line)

            for attack_line in self.black_potential_attack:
                if prev_piece_position in attack_line[0]:
                    self.black_potential_attack.remove(attack_line)

        # Save previous attack lines
        attack_lines = [self.white_direct_attack.copy(), self.white_potential_attack.copy(),
                        self.black_direct_attack.copy(), self.black_potential_attack.copy()]

        # Update the board
        self.__init__(previous_fen)

        # Restore init vars
        self.board_history = board_save

        # Restore previous attack lines
        self.white_direct_attack = attack_lines[0]
        self.white_potential_attack = attack_lines[1]
        self.black_direct_attack = attack_lines[2]
        self.black_potential_attack = attack_lines[3]

        # Update the attack lines for the piece
        self.update_attack_lines(prev_piece)

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
                self.board[new_pos[0]][3].position = "d1" if self.board[new_pos[0]][3].color == "w" else "d8"
                self.board[new_pos[0]][3].first_move = False
            else:
                self.board[new_pos[0]][5] = self.board[new_pos[0]][7]
                self.board[new_pos[0]][7] = None
                # Update rook position and make first move False
                self.board[new_pos[0]][5].position = "f1" if self.board[new_pos[0]][5].color == "w" else "f8"
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
        self.board = Board("r1bqkbnr/p1p1pp1p/1p1Pn1p1/8/1B3Q2/8/PPP1PPPP/RN2KBNR w KQkq - 0 1")
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


def test():
    game = Game()
    engine = Engine(game)

    print(engine.best_move())


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
        run_game()

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
    # test()
    # run_game()
    # run_multiprocessing()
    profiling()
    # check_board_size()

    pass


if __name__ == "__main__":
    main()
