from engine import movement
from typing import List
import random

# TODO: Write testing for is_pinned and castling all cases
# TODO: Finnish is_checkmate
# TODO: Finnish is_stalemate
# TODO: [BONUS] add threefold repetition rule by checking if the same board state has been repeated 3 times


class Board:
    def __init__(self):
        # Letter to index
        self.lti: dict = {chr(97 + i): i for i in range(8)}
        self.itl: dict = {i: chr(97 + i) for i in range(8)}
        self.board: List[str] = []
        self.white_moves: List[str] = []
        self.black_moves: List[str] = []

        self.init_board()

    def init_board(self) -> None:
        # Init board using FEN, instead of 8 have multiple 1's
        self.board = ["rnbqkbnr", "pppppppp", "11111111", "11111111", "11111111", "11111111", "PPPPPPPP", "RNBQKBNR",
                      "w", "KQkq", "-", "0", "1"]

    def is_pinned(self, row: int, col: int) -> bool:
        """
        Check if a piece is pinned
        :param row: The row of the piece
        :param col: The column of the piece
        :return: True if the piece is pinned, False otherwise
        """
        # Make certain piece is not protecting the king from check
        piece = self.board[row][col]

        # Remove piece momentarily and check if the king is in check
        self.board[row] = self.board[row][:col] + "1" + self.board[row][col + 1:]
        if self.is_check():
            self.board[row] = self.board[row][:col] + piece + self.board[row][col + 1:]
            return True

    def is_check(self) -> bool:
        """
        Check if a player is in check from kings perspective
        "move" the king to all possible positions to see if there is an enemy
        :return: True if the player is in check, False otherwise
        """
        turn = self.board[-5]
        enemy_pieces = "rnbqkp" if turn == "w" else "RNBQKP"
        king_pos = None

        # Find the king
        for row in range(8):
            for col in range(8):
                if turn == "w" and self.board[row][col] in "K":
                    king_pos = (row, col)
                elif turn == "b" and self.board[row][col] in "k":
                    king_pos = (row, col)

        # IF king not find raise exception
        if not king_pos:
            raise ValueError("King not found")

        # Make a list of candidates which base on all possible movement and if there is enemy piece in the end of list
        # then the king is in check
        movements = [movement.move_up, movement.move_down, movement.move_right, movement.move_left,
                     movement.move_top_right, movement.move_top_left, movement.move_bottom_right,
                     movement.move_bottom_left]

        candidates = []

        # Loop over the movements and add the candidates
        for move in movements:
            new_candidates = move(self.board, king_pos[0], king_pos[1], enemy_pieces)
            if new_candidates:
                # Add only the last candidate, because it only stops for 3 reasons:
                # 1. It hits a piece
                # 2. It hits the end of the board
                # 3. It hits a piece of the same color
                candidates.append(new_candidates[-1])

        # Knight moves
        new_candidates = movement.move_knight(self.board, king_pos[0], king_pos[1], enemy_pieces)
        if new_candidates:
            candidates.extend(new_candidates)

        # Loop over the candidates and check if there is an enemy piece
        for candidate in candidates:
            row = 8 - int(candidate[2])
            col = self.lti[candidate[1]]
            board_piece = self.board[row][col]

            # Make certain the board piece is an enemy piece
            if board_piece not in enemy_pieces:
                continue

            # Make sure the enemy piece can move to the king
            if self.is_valid_move((row, col), king_pos):
                return True

        return False

    def is_checkmate(self, color: str) -> bool:
        """
        Check if a player is in checkmate
        :param color: The color of the player
        :return: True if the player is in checkmate, False otherwise
        """
        # IF king is in check and after doing any possible move the king is still in check it is checkmate
        if self.is_check() and not self.generate_moves():
            return True

    def is_stalemate(self, color: str) -> bool:
        """
        Check if a player is in stalemate
        :param color: The color of the player
        :return: True if the player is in stalemate, False otherwise
        """
        pass

    def generate_bishop_moves(self) -> list:
        """
        Generate all possible bishop moves, use chess notation
        :return: A list of all possible bishop moves
        """
        turn = self.board[-5]
        rook_positions = []
        moves = []
        enemy_pieces = "rnbqkp" if turn == "w" else "RNBQKP"

        # Find the bishops
        for row in range(8):
            for col in range(8):
                if turn == "w" and self.board[row][col] in "B":
                    rook_positions.append((row, col))
                elif turn == "b" and self.board[row][col] in "b":
                    rook_positions.append((row, col))

        # Generate moves
        for pos in rook_positions:
            row, col = pos
            movements = [movement.move_top_right, movement.move_top_left, movement.move_bottom_right,
                         movement.move_bottom_left]

            for move in movements:
                new_moves = move(self.board, row, col, enemy_pieces)
                if new_moves:
                    moves.extend(new_moves)

        return moves

    def generate_queen_moves(self) -> list:
        """
        Generate all possible queen moves, use chess notation
        :return: A list of all possible queen moves
        """
        turn = self.board[-5]
        queen_positions = []
        moves = []
        enemy_pieces = "rnbqkp" if turn == "w" else "RNBQKP"

        # Find the queens
        for row in range(8):
            for col in range(8):
                if turn == "w" and self.board[row][col] in "Q":
                    queen_positions.append((row, col))
                elif turn == "b" and self.board[row][col] in "q":
                    queen_positions.append((row, col))

        # Generate moves
        for pos in queen_positions:
            row, col = pos
            movements = [movement.move_top_right, movement.move_top_left, movement.move_bottom_right,
                         movement.move_bottom_left, movement.move_up, movement.move_down, movement.move_right,
                         movement.move_left]

            for move in movements:
                new_moves = move(self.board, row, col, enemy_pieces)
                if new_moves:
                    moves.extend(new_moves)

        return moves

    def generate_knight_moves(self) -> list:
        """
        Generate all possible knight moves, use chess notation
        :return: A list of all possible knight moves
        """
        turn = self.board[-5]
        knight_positions = []
        moves = []
        enemy_pieces = "rnbqkp" if turn == "w" else "RNBQKP"

        # Find the knights
        for row in range(8):
            for col in range(8):
                if turn == "w" and self.board[row][col] in "N":
                    knight_positions.append((row, col))
                elif turn == "b" and self.board[row][col] in "n":
                    knight_positions.append((row, col))

        # Generate moves
        for pos in knight_positions:
            row, col = pos

            # Get knight moves
            moves.extend(movement.move_knight(self.board, row, col, enemy_pieces))

        return moves

    def generate_rook_moves(self) -> list:
        """
        Generate all possible rook moves, use chess notation
        :return: A list of all possible rook moves
        """
        turn = self.board[-5]
        rook_positions = []
        moves = []
        enemy_pieces = "rnbqkp" if turn == "w" else "RNBQKP"

        # Find the rooks
        for row in range(8):
            for col in range(8):
                if turn == "w" and self.board[row][col] in "R":
                    rook_positions.append((row, col))
                elif turn == "b" and self.board[row][col] in "r":
                    rook_positions.append((row, col))

        # Generate moves
        for pos in rook_positions:
            row, col = pos
            movements = [movement.move_up, movement.move_down, movement.move_right, movement.move_left]

            for move in movements:
                new_moves = move(self.board, row, col, enemy_pieces)
                if new_moves:
                    moves.extend(new_moves)

        return moves

    def generate_king_moves(self) -> list:
        """
        Generate all possible king moves, use chess notation
        :return: A list of all possible king moves
        """
        turn = self.board[-5]
        king_positions = []
        moves = []
        enemy_pieces = "rnbqkp" if turn == "w" else "RNBQKP"

        # Find the kings
        for row in range(8):
            for col in range(8):
                if turn == "w" and self.board[row][col] in "K":
                    king_positions.append((row, col))
                elif turn == "b" and self.board[row][col] in "k":
                    king_positions.append((row, col))

        # Generate moves
        for pos in king_positions:
            row, col = pos
            movements = [movement.move_top_right, movement.move_top_left, movement.move_bottom_right,
                         movement.move_bottom_left, movement.move_up, movement.move_down, movement.move_right,
                         movement.move_left]

            # Get king moves
            for move in movements:
                new_moves = move(self.board, row, col, enemy_pieces, 1)
                # Make certain new move is not in check
                # TODO: make sure the king is not in check after the move
                if new_moves:
                    moves.extend(new_moves)

            # add castling moves
            if turn == "w":
                if "K" in self.board[-4] and self.board[7][5] == "1" and self.board[7][6] == "1":
                    moves.append("Kg1")
                if "Q" in self.board[-4] and self.board[7][3] == "1" and self.board[7][2] == "1" and self.board[7][
                    1] == "1":
                    moves.append("Kc1")
            else:
                if "k" in self.board[-4] and self.board[0][5] == "1" and self.board[0][6] == "1":
                    moves.append("Kg8")
                if "q" in self.board[-4] and self.board[0][3] == "1" and self.board[0][2] == "1" and self.board[0][
                    1] == "1":
                    moves.append("Kc8")

        return moves

    def generate_pawn_moves(self) -> list:
        """
        Generate all possible pawn moves, use chess notation
        :return: A list of all possible pawn moves
        """
        turn = self.board[-5]
        pawn_positions = []
        moves = []
        enemy_pieces = "rnbqkp" if turn == "w" else "RNBQKP"

        # Find the pawns
        for row in range(8):
            for col in range(8):
                if turn == "w" and self.board[row][col] in "P":
                    pawn_positions.append((row, col))
                elif turn == "b" and self.board[row][col] in "p":
                    pawn_positions.append((row, col))

        # Generate moves
        for pos in pawn_positions:
            row, col = pos

            # Get pawn moves
            moves.extend(movement.move_pawn(self.board, row, col, enemy_pieces))

        return moves

    def generate_moves(self) -> list:
        """
        Generate all possible moves, use chess notation
        :return: A list of all possible moves
        """
        moves = []
        funcs = [self.generate_pawn_moves, self.generate_knight_moves, self.generate_bishop_moves,
                 self.generate_rook_moves, self.generate_queen_moves, self.generate_king_moves]

        for func in funcs:
            moves.extend(func())

        return moves

    def is_valid_move(self, old_pos: tuple, new_pos: tuple) -> bool:
        """
        Make sure the move is within generated moves for that piece
        :param old_pos:
        :param new_pos:
        :return:
        """
        # Generate all possible moves for the piece in old_pos
        piece = self.board[old_pos[0]][old_pos[1]]

        if piece in "pP":
            moves = self.generate_pawn_moves()
        elif piece in "nN":
            moves = self.generate_knight_moves()
        elif piece in "bB":
            moves = self.generate_bishop_moves()
        elif piece in "rR":
            moves = self.generate_rook_moves()
        elif piece in "qQ":
            moves = self.generate_queen_moves()
        elif piece in "kK":
            moves = self.generate_king_moves()
        else:
            return False

        # Move the piece momentarily and check if the king is in check
        self.board[old_pos[0]] = self.board[old_pos[0]][:old_pos[1]] + "1" + self.board[old_pos[0]][old_pos[1] + 1:]
        self.board[new_pos[0]] = self.board[new_pos[0]][:new_pos[1]] + piece + self.board[new_pos[0]][
                                                                               new_pos[1] + 1:]
        check_status = self.is_check()

        self.board[old_pos[0]] = self.board[old_pos[0]][:old_pos[1]] + piece + self.board[old_pos[0]][
                                                                               old_pos[1] + 1:]
        self.board[new_pos[0]] = self.board[new_pos[0]][:new_pos[1]] + "1" + self.board[new_pos[0]][
                                                                                 new_pos[1] + 1:]
        if check_status:
            return False

        # Check if the new position is in the list of moves
        for move in moves:
            # if pawn
            if piece in "pP":
                if move == f"{self.itl[new_pos[1]]}{8 - new_pos[0]}":
                    return True
            else:
                if move[1:] == f"{self.itl[new_pos[1]]}{8 - new_pos[0]}":
                    return True

        return False

    def castle_move(self, old_pos: str, new_pos: str) -> None:
        castling_side = "k" if new_pos[0] in "fg" else "q"

        # If whites turn make castling side capital
        if self.board[-5] == "w":
            castling_side = castling_side.upper()

        # Make sure castling is possible in FEN
        if self.board[-4].find(castling_side) == -1:
            raise ValueError("Invalid move")

        # If castling is kingside make sure the squares between king and rook are empty for black or white
        if self.board[-5] == "w" and castling_side == "k":
            for i in range(5, 7):
                if self.board[7][i] != "1":
                    raise ValueError("Invalid move")
        elif self.board[-5] == "b" and castling_side == "k":
            for i in range(5, 7):
                if self.board[0][i] != "1":
                    raise ValueError("Invalid move")

        # If castling is queenside make sure the squares between king and rook are empty for black or white
        if self.board[-5] == "w" and castling_side == "q":
            for i in range(1, 4):
                if self.board[7][i] != "1":
                    raise ValueError("Invalid move")
        elif self.board[-5] == "b" and castling_side == "q":
            for i in range(1, 4):
                if self.board[0][i] != "1":
                    raise ValueError("Invalid move")

        # If move was made for king move it and the rook to new position for black and white for both kingside and
        # queenside
        if self.board[-5] == "w":
            if castling_side == "K":
                self.board[7] = self.board[7][:4] + "1RK1"
            else:
                self.board[7] = "1" + self.board[7][1] + "KR1" + self.board[7][5:]
        else:
            if castling_side == "k":
                self.board[0] = self.board[0][:4] + "1rk1"
            else:
                self.board[0] = "1" + self.board[0][1] + "kr1" + self.board[0][5:]

        # Remove castling from FEN
        if self.board[-5] == "w":
            self.board[-4] = self.board[-4].replace("K", "")
            self.board[-4] = self.board[-4].replace("Q", "")
        else:
            self.board[-4] = self.board[-4].replace("k", "")
            self.board[-4] = self.board[-4].replace("q", "")

    def move(self, old_pos: str, new_pos: str) -> None:
        row = 8 - int(old_pos[1])
        col = int(self.lti[old_pos[0]])
        o_p = (row, col)

        row = 8 - int(new_pos[1])
        col = int(self.lti[new_pos[0]])
        n_p = (row, col)

        print(f"Moving {self.board[o_p[0]][o_p[1]]} from {old_pos} to {new_pos}")

        # Validate move
        if not self.is_valid_move(o_p, n_p):
            raise ValueError("Invalid move")

        # Save the piece
        piece = self.board[o_p[0]][o_p[1]]

        # If move is castling, move the king and rook
        if piece in "kK" and abs(o_p[1] - n_p[1]) > 1:
            self.castle_move(old_pos, new_pos)

            # Increase whole move counter if black moved
            if self.board[-5] == "b":
                self.board[-1] = str(int(self.board[-1]) + 1)

            # Remove en passant if the move is not a pawn move
            if piece not in "pP":
                self.board[-3] = "-"

            # Change turn THIS NEEDS TO BE LAST
            self.board[-5] = "w" if self.board[-5] == "b" else "b"

            return

        # If move is pawn and is en passant, remove the enemy pawn
        if piece in "pP" and new_pos == self.board[-3]:
            en_passant_row = n_p[0]
            en_passant_col = n_p[1]
            if piece == "p":
                # Make the piece below en passant square empty
                remove_piece_row = en_passant_row - 1
                remove_piece_col = en_passant_col
                self.board[remove_piece_row] = self.board[remove_piece_row][:remove_piece_col] + "1" + \
                                               self.board[remove_piece_row][remove_piece_col + 1:]
            else:
                # Make the piece above en passant square empty
                remove_piece_row = en_passant_row + 1
                remove_piece_col = en_passant_col
                self.board[remove_piece_row] = self.board[remove_piece_row][:remove_piece_col] + "1" + \
                                               self.board[remove_piece_row][remove_piece_col + 1:]

        # Remove the piece from the old position
        self.board[o_p[0]] = self.board[o_p[0]][:o_p[1]] + "1" + self.board[o_p[0]][o_p[1] + 1:]

        # If the move is a pawn move or a capture, reset the counter
        if piece in "pP" or self.board[n_p[0]][n_p[1]] != "1":
            self.board[-2] = "0"
        else:
            self.board[-2] = str(int(self.board[-2]) + 1)

        # Add the piece to the new position
        self.board[n_p[0]] = self.board[n_p[0]][:n_p[1]] + piece + self.board[n_p[0]][n_p[1] + 1:]

        # Increase whole move counter if black moved
        if self.board[-5] == "b":
            self.board[-1] = str(int(self.board[-1]) + 1)

        # Check if move was king to eliminate castling
        if piece in "kK":
            if piece == "k":
                self.board[-4] = self.board[-4].replace("k", "")
                self.board[-4] = self.board[-4].replace("q", "")
            else:
                self.board[-4] = self.board[-4].replace("K", "")
                self.board[-4] = self.board[-4].replace("Q", "")

        # Check if rook was in its initial position and moved to eliminate castling
        if piece in "rR":
            if piece == "r":
                if o_p == (7, 0):
                    self.board[-4] = self.board[-4].replace("q", "")
                elif o_p == (7, 7):
                    self.board[-4] = self.board[-4].replace("k", "")
            else:
                if o_p == (0, 0):
                    self.board[-4] = self.board[-4].replace("Q", "")
                elif o_p == (0, 7):
                    self.board[-4] = self.board[-4].replace("K", "")

        # Enemy pawn
        enemy_pawn = "p" if self.board[-5] == "w" else "P"

        # Handle en passant by checking if new position has enemy pawn in the same column right or left
        #TODO: Fix out of boundary error
        if piece in "pP" and (
                self.board[n_p[0]][n_p[1] + 1] == enemy_pawn or self.board[n_p[0]][n_p[1] - 1] == enemy_pawn):
            # Also make sure the square behind enemy pawn is empty
            if piece == "p" and self.board[n_p[0] + 1][n_p[1] + 1] == "1":
                self.board[-3] = f"{self.itl[n_p[1]]}{8 - n_p[0] + 1}"
            elif piece == "p" and self.board[n_p[0]][n_p[1] - 1] == "1":
                self.board[-3] = f"{self.itl[n_p[1]]}{8 - n_p[0] - 1}"
            elif piece == "P" and self.board[n_p[0] - 1][n_p[1] + 1] == "1":
                self.board[-3] = f"{self.itl[n_p[1]]}{8 - n_p[0] - 1}"
            elif piece == "P" and self.board[n_p[0] - 1][n_p[1] - 1] == "1":
                self.board[-3] = f"{self.itl[n_p[1]]}{8 - n_p[0] + 1}"

        # Remove en passant if the move is not a pawn move
        if piece not in "pP":
            self.board[-3] = "-"

        # Change turn THIS NEEDS TO BE LAST
        self.board[-5] = "w" if self.board[-5] == "b" else "b"


class Engine:
    def __init__(self):
        self.board = Board()

    def move(self, old_pos: str, new_pos: str) -> None:
        self.board.move(old_pos, new_pos)
        print(self.board.board)

    def get_moves(self) -> List[str]:
        return self.board.generate_moves()

    def do_random_move(self) -> None:
        # DOES NOT WORK YET, random move only have destination, not source
        # DO random move using random library
        moves = self.get_moves()
        move = random.choice(moves)
        self.move(move[:2], move[2:])


def test_castling(engine: Engine):
    # Use pytest to test castling
    engine.move("e2", "e3")
    engine.move("e7", "e5")
    engine.move("g1", "f3")
    engine.move("d7", "d5")
    engine.move("f1", "c4")
    engine.move("d8", "d6")
    engine.move("e1", "g1")


if __name__ == "__main__":
    engine = Engine()

    engine.move('e2', 'e4')
    engine.move('e7', 'e5')
    engine.move('g1', 'f3')
    engine.move('g8', 'f6')
    engine.move('f1', 'c4')
    engine.move('f8', 'c5')
    engine.move('e1', 'g1')

    print(engine.board.board)
