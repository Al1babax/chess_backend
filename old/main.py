from tests import movement
from typing import List
import random


# TODO: Write testing for every basic piece movement (necessary to fix other stuff)
# TODO: Write testing for is_pinned and castling all cases
# TODO: Fix is_checkmate
# TODO: Fix is_stalemate
# TODO: [BONUS] add threefold repetition rule by checking if the same board state has been repeated 3 times
# TODO: Rook cannot initiate castling if it has moved


class Board:
    def __init__(self, custom_fen: List[str] = None) -> None:
        # Letter to index
        self.lti: dict = {chr(97 + i): i for i in range(8)}
        self.itl: dict = {i: chr(97 + i) for i in range(8)}
        self.board: List[str] = []
        self.white_moves: List[str] = []
        self.black_moves: List[str] = []
        self.valid_moves = []
        self.valid_enemy_moves = []

        if custom_fen is None:
            self.init_board()
        else:
            self.board = custom_fen

        # Make valid moves
        self.valid_moves = self.generate_moves()

        self.board[-5] = "b"
        self.valid_enemy_moves = self.generate_moves()
        self.board[-5] = "w"

    def init_board(self) -> None:
        # Init board using FEN, instead of 8 have multiple 1's
        self.board = ["rnbqkbnr", "pppppppp", "11111111", "11111111", "11111111", "11111111", "PPPPPPPP", "RNBQKBNR",
                      "w", "KQkq", "-", "0", "1"]

    def fen(self) -> str:
        # Convert the board to FEN
        fen = ""
        for row in self.board[:8]:
            empty = 0
            for char in row:
                if char == "1":
                    empty += 1
                else:
                    if empty:
                        fen += str(empty)
                        empty = 0
                    fen += char
            if empty:
                fen += str(empty)
            fen += "/"

        fen = fen[:-1]
        fen += f" {self.board[-5]} {self.board[-4]} {self.board[-3]} {self.board[-2]} {self.board[-1]}"

        return fen

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

        return False

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
                    break
                elif turn == "b" and self.board[row][col] in "k":
                    king_pos = (row, col)
                    break

        # IF king not find raise exception
        if not king_pos:
            print(f"King is not found in the following board, turn: {self.board[-5]}")
            print(self.board)
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
            candidate = candidate.split(",")[1]

            row = 8 - int(candidate[2])
            col = self.lti[candidate[1]]
            board_piece = self.board[row][col]

            # Make certain the board piece is an enemy piece
            if board_piece not in enemy_pieces:
                continue

            # Make sure the enemy piece can move to the king
            # Generate all movement for that enemy piece and check if one of the reaches the king
            enemy_piece = self.board[row][col]
            enemy_piece = "" if enemy_piece in "pP" else enemy_piece

            for move in self.valid_enemy_moves:
                if enemy_piece in "pP":
                    if move.split(",")[1] == f"{self.itl[king_pos[1]]}{8 - king_pos[0]}":
                        return True
                else:
                    if move.split(",")[1][1:] == f"{self.itl[king_pos[1]]}{8 - king_pos[0]}":
                        return True

        return False

    def is_checkmate(self) -> bool:
        """
        Check if a player is in checkmate
        :param color: The color of the player
        :return: True if the player is in checkmate, False otherwise
        """
        # IF king is in check and after doing any possible move the king is still in check it is checkmate
        if self.is_check() and not self.generate_moves():
            return True

        return False

    def is_stalemate(self) -> bool:
        """
        Check if a player is in stalemate
        King is not in check, but all the moves king can make lead to check
        :param color: The color of the player
        :return: True if the player is in stalemate, False otherwise
        """
        # IF king is not in check and after doing any possible move the king is still in check it is stalemate
        if not self.is_check() and not self.generate_moves():
            return True

        return False

    def generate_bishop_moves(self, is_check_call: bool = False) -> list:
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
                if not new_moves:
                    continue

                if is_check_call:
                    moves.extend(new_moves)
                    continue

                for new_move in new_moves:
                    if not self.is_pinned(8 - int(new_move.split(",")[1][2]), self.lti[new_move.split(",")[1][1]]):
                        moves.append(new_move)

        return moves

    def generate_queen_moves(self, is_check_call: bool = False) -> list:
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
                if not new_moves:
                    continue

                if is_check_call:
                    moves.extend(new_moves)
                    continue

                for new_move in new_moves:
                    if not self.is_pinned(8 - int(new_move.split(",")[1][2]), self.lti[new_move.split(",")[1][1]]):
                        moves.append(new_move)

        return moves

    def generate_knight_moves(self, is_check_call: bool = False) -> list:
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
            new_moves = movement.move_knight(self.board, row, col, enemy_pieces)

            if not new_moves:
                continue

            if is_check_call:
                moves.extend(new_moves)
                continue

            for move in new_moves:
                if not self.is_pinned(8 - int(move.split(",")[1][2]), self.lti[move.split(",")[1][1]]):
                    moves.append(move)

        return moves

    def generate_rook_moves(self, is_check_call: bool = False) -> list:
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
                if not new_moves:
                    continue

                if is_check_call:
                    moves.extend(new_moves)
                    continue

                for new_move in new_moves:
                    if not self.is_pinned(8 - int(new_move.split(",")[1][2]), self.lti[new_move.split(",")[1][1]]):
                        moves.append(new_move)

        return moves

    def generate_king_moves(self, is_check_call: bool = False) -> list:
        """
        Generate all possible king moves, use chess notation
        :return: A list of all possible king moves
        """

        def check_king_safety(old_pos: tuple, new_pos: tuple) -> bool:
            """
            Check if the king is safe after a move
            :param old_pos:
            :param new_pos:
            :return: True if king is safe, False otherwise
            """
            # Make certain king is not under check
            # Move king to new position
            piece = self.board[old_pos[0]][old_pos[1]]

            self.board[old_pos[0]] = self.board[old_pos[0]][:old_pos[1]] + "1" + self.board[old_pos[0]][old_pos[1] + 1:]
            self.board[new_pos[0]] = self.board[new_pos[0]][:new_pos[1]] + piece + self.board[new_pos[0]][
                                                                                   new_pos[1] + 1:]
            is_king_safe = True

            if self.is_check():
                is_king_safe = False

            # Move king back to old position
            self.board[old_pos[0]] = self.board[old_pos[0]][:old_pos[1]] + piece + self.board[old_pos[0]][
                                                                                   old_pos[1] + 1:]
            self.board[new_pos[0]] = self.board[new_pos[0]][:new_pos[1]] + "1" + self.board[new_pos[0]][new_pos[1] + 1:]

            return is_king_safe

        turn = self.board[-5]
        king_position = None
        moves = []
        enemy_pieces = "rnbqkp" if turn == "w" else "RNBQKP"

        # Find the kings
        for row in range(8):
            for col in range(8):
                if turn == "w" and self.board[row][col] in "K":
                    king_position = (row, col)
                    break
                elif turn == "b" and self.board[row][col] in "k":
                    king_position = (row, col)
                    break

        # Generate moves
        row, col = king_position
        movements = [movement.move_top_right, movement.move_top_left, movement.move_bottom_right,
                     movement.move_bottom_left, movement.move_up, movement.move_down, movement.move_right,
                     movement.move_left]

        # Get king moves
        for move in movements:
            new_moves = move(self.board, row, col, enemy_pieces, 1)
            # Make certain new move is not in check
            if not new_moves:
                continue

            if is_check_call:
                moves.extend(new_moves)
                continue

            for new_move in new_moves:
                new_row = 8 - int(new_move.split(",")[1][2])
                new_col = self.lti[new_move.split(",")[1][1]]

                if not check_king_safety((row, col), (new_row, new_col)):
                    continue

                moves.append(new_move)

            # add castling moves
            if turn == "w":
                if "K" in self.board[-4] and self.board[7][5] == "1" and self.board[7][6] == "1":
                    if check_king_safety((row, col), (7, 6)):
                        moves.append("Ke1,Kg1")
                if "Q" in self.board[-4] and self.board[7][3] == "1" and self.board[7][2] == "1" and self.board[7][
                    1] == "1":
                    if check_king_safety((row, col), (7, 2)):
                        moves.append("Ke1,Kc1")
            else:
                if "k" in self.board[-4] and self.board[0][5] == "1" and self.board[0][6] == "1":
                    if check_king_safety((row, col), (0, 6)):
                        moves.append("Ke8,Kg8")
                if "q" in self.board[-4] and self.board[0][3] == "1" and self.board[0][2] == "1" and self.board[0][
                    1] == "1":
                    if check_king_safety((row, col), (0, 2)):
                        moves.append("Ke8,Kc8")

        return moves

    def generate_pawn_moves(self, is_check_call: bool = False) -> list:
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
            new_moves = movement.move_pawn(self.board, row, col, enemy_pieces)

            if not new_moves:
                continue

            if is_check_call:
                moves.extend(new_moves)
                continue

            for move in new_moves:
                new_row = 8 - int(move.split(",")[1][1])
                new_col = self.lti[move.split(",")[1][0]]
                if not self.is_pinned(new_row, new_col):
                    moves.append(move)

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

    def is_valid_move(self, old_pos: tuple, new_pos: tuple, is_check_call: bool = False) -> bool:
        """
        Make sure the move is within generated moves for that piece
        :param is_check_call:
        :param old_pos:
        :param new_pos:
        :return:
        """
        # Generate all possible moves for the piece in old_pos
        piece = self.board[old_pos[0]][old_pos[1]]

        # if piece in "pP":
        #     moves = self.generate_pawn_moves()
        # elif piece in "nN":
        #     moves = self.generate_knight_moves()
        # elif piece in "bB":
        #     moves = self.generate_bishop_moves()
        # elif piece in "rR":
        #     moves = self.generate_rook_moves()
        # elif piece in "qQ":
        #     moves = self.generate_queen_moves()
        # elif piece in "kK":
        #     moves = self.generate_king_moves()
        # else:
        #     return False

        # Move the piece momentarily and check if the king is in check
        if not is_check_call:
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
        for move in self.valid_moves:
            # Take only the destination of a move
            move = move.split(",")[1]

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
        # Generate valid moves
        self.valid_moves = self.generate_moves()

        # IF pawn move it does not have the piece as prefix
        if len(old_pos) == 3:
            o_p = (8 - int(old_pos[2]), self.lti[old_pos[1]])
            n_p = (8 - int(new_pos[2]), self.lti[new_pos[1]])
        else:
            o_p = (8 - int(old_pos[1]), self.lti[old_pos[0]])
            n_p = (8 - int(new_pos[1]), self.lti[new_pos[0]])

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

            # Check if checkmate
            if self.is_checkmate():
                raise ValueError("Checkmate")
            elif self.is_stalemate():
                raise ValueError("Stalemate")

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
        if piece in "pP" and ((n_p[1] + 1 < 8 and self.board[n_p[0]][n_p[1] + 1] == enemy_pawn) or (
                n_p[1] + 1 >= 0 and self.board[n_p[0]][n_p[1] - 1] == enemy_pawn)) and abs(o_p[0] - n_p[0]) == 2:
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

        # Check if checkmate
        if self.is_checkmate():
            raise ValueError("Checkmate")
        elif self.is_stalemate():
            raise ValueError("Stalemate")

        # Change turn THIS NEEDS TO BE LAST
        self.board[-5] = "w" if self.board[-5] == "b" else "b"


class Engine:
    def __init__(self, custom_fen: List[str] = None) -> None:
        self.board = Board(custom_fen)
        self.moves = []

    def move(self, old_pos: str, new_pos: str) -> None:
        self.board.move(old_pos, new_pos)
        print(self.board.board)
        print(self.board.fen())

    def get_moves(self) -> List[str]:
        return self.board.generate_moves()

    def do_random_move(self) -> None:
        # DO random move using random library
        moves = self.get_moves()
        move = random.choice(moves).split(",")
        self.move(move[0], move[1])

        # Record move
        # FOR DEBUGGING
        # self.moves.append(move)
        # print(self.moves)


def fen_string_to_array(fen: str) -> List[str]:
    # Convert FEN to array and make the numbers all 1
    board = []
    new_row = ""
    for char in fen[:-2]:
        if char == " " or char == "/":
            board.append(new_row)
            new_row = ""
            continue

        if char.isdigit():
            new_row += "1" * int(char)
        else:
            new_row += char

    board.append("0")
    board.append("1")

    return board


def test_castling(engine: Engine):
    # Use pytest to test castling
    engine.move("e2", "e3")
    engine.move("e7", "e5")
    engine.move("g1", "f3")
    engine.move("d7", "d5")
    engine.move("f1", "c4")
    engine.move("d8", "d6")
    engine.move("e1", "g1")


def random_50_test():
    engine = Engine()

    for _ in range(200):
        engine.do_random_move()


def pretty_print_fen(board_state):
    # Mapping for piece characters
    piece_mapping = {
        '1': '[  ]',
        'k': '[♚]',
        'q': '[♛]',
        'r': '[♜]',
        'b': '[♝]',
        'n': '[♞]',
        'p': '[♟]',
        'K': '[♔]',
        'Q': '[♕]',
        'R': '[♖]',
        'B': '[♗]',
        'N': '[♘]',
        'P': '[♙]'
    }

    # Print the chessboard
    for row in board_state[:8][::-1]:
        for char in row:
            print(f'{piece_mapping.get(char, char): <3}', end='')
        print()

    # Print turn, castling, and en passant information
    print(f'Turn: {board_state[8]}')
    print(f'Castling: {board_state[9]}')
    print(f'En Passant: {board_state[10]}')
    print(f'Halfmove Clock: {board_state[11]}')
    print(f'Fullmove Number: {board_state[12]}')


if __name__ == "__main__":
    fen_string = "rnb1kb1r/p2pq1pp/2p2n2/8/3P4/8/4K1PP/2B4R w kq - 3 7"
    engine = Engine(fen_string_to_array(fen_string))
    print(engine.board.valid_moves)
    print(engine.board.valid_enemy_moves)

    # print(engine.board.valid_moves)

    # print(engine.board.generate_king_moves())
    # print(engine.board.is_check())

    # random_50_test()
