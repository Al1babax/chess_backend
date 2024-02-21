from typing import List

"""
All functions takes the following parameters:
    board: list of strings (FEN notation)
    row: int
    col: int
    enemy_pieces: str
    distance: int
"""

itl = {i: chr(97 + i) for i in range(8)}


def move_top_right(board: List[str], row: int, col: int, enemy_pieces: str, distance: int = 8):
    moves = []
    piece = board[row][col].upper().upper()
    piece = "" if piece in "pP" else piece

    for i in range(1, distance + 1):
        if row + i < 8 and col + i < 8:
            if board[row + i][col + i] == "1":
                moves.append(f"{piece}{itl[col + i]}{8 - (row + i)}")
            elif board[row + i][col + i] in enemy_pieces:
                moves.append(f"{piece}{itl[col + i]}{8 - (row + i)}")
                break
            else:
                break
        else:
            break

    return moves


def move_top_left(board: List[str], row: int, col: int, enemy_pieces: str, distance: int = 8):
    moves = []
    piece = board[row][col].upper().upper()
    piece = "" if piece in "pP" else piece

    for i in range(1, distance + 1):
        if row + i < 8 and col - i >= 0:
            if board[row + i][col - i] == "1":
                moves.append(f"{piece}{itl[col - i]}{8 - (row + i)}")
            elif board[row + i][col - i] in enemy_pieces:
                moves.append(f"{piece}{itl[col - i]}{8 - (row + i)}")
                break
            else:
                break
        else:
            break

    return moves


def move_bottom_right(board: List[str], row: int, col: int, enemy_pieces: str, distance: int = 8):
    moves = []
    piece = board[row][col].upper().upper()
    piece = "" if piece in "pP" else piece

    for i in range(1, distance + 1):
        if row - i >= 0 and col + i < 8:
            if board[row - i][col + i] == "1":
                moves.append(f"{piece}{itl[col + i]}{8 - (row - i)}")
            elif board[row - i][col + i] in enemy_pieces:
                moves.append(f"{piece}{itl[col + i]}{8 - (row - i)}")
                break
            else:
                break
        else:
            break

    return moves


def move_bottom_left(board: List[str], row: int, col: int, enemy_pieces: str, distance: int = 8):
    moves = []
    piece = board[row][col].upper().upper()
    piece = "" if piece in "pP" else piece

    for i in range(1, distance + 1):
        if row - i >= 0 and col - i >= 0:
            if board[row - i][col - i] == "1":
                moves.append(f"{piece}{itl[col - i]}{8 - (row - i)}")
            elif board[row - i][col - i] in enemy_pieces:
                moves.append(f"{piece}{itl[col - i]}{8 - (row - i)}")
                break
            else:
                break
        else:
            break

    return moves


def move_up(board: List[str], row: int, col: int, enemy_pieces: str, distance: int = 8):
    moves = []
    piece = board[row][col].upper()
    piece = "" if piece in "pP" else piece

    for i in range(1, distance + 1):
        if row + i < 8:
            if board[row + i][col] == "1":
                moves.append(f"{piece}{itl[col]}{8 - (row + i)}")
            elif board[row + i][col] in enemy_pieces:
                moves.append(f"{piece}{itl[col]}{8 - (row + i)}")
                break
            else:
                break
        else:
            break

    return moves


def move_down(board: List[str], row: int, col: int, enemy_pieces: str, distance: int = 8):
    moves = []
    piece = board[row][col].upper()
    piece = "" if piece in "pP" else piece

    for i in range(1, distance + 1):
        if row - i >= 0:
            if board[row - i][col] == "1":
                moves.append(f"{piece}{itl[col]}{8 - (row - i)}")
            elif board[row - i][col] in enemy_pieces:
                moves.append(f"{piece}{itl[col]}{8 - (row - i)}")
                break
            else:
                break
        else:
            break

    return moves


def move_right(board: List[str], row: int, col: int, enemy_pieces: str, distance: int = 8):
    moves = []
    piece = board[row][col].upper()
    piece = "" if piece in "pP" else piece

    for i in range(1, distance + 1):
        if col + i < 8:
            if board[row][col + i] == "1":
                moves.append(f"{piece}{itl[col + i]}{8 - row}")
            elif board[row][col + i] in enemy_pieces:
                moves.append(f"{piece}{itl[col + i]}{8 - row}")
                break
            else:
                break
        else:
            break

    return moves


def move_left(board: List[str], row: int, col: int, enemy_pieces: str, distance: int = 8):
    moves = []
    piece = board[row][col].upper()
    piece = "" if piece in "pP" else piece

    for i in range(1, distance + 1):
        if col - i >= 0:
            if board[row][col - i] == "1":
                moves.append(f"{piece}{itl[col - i]}{8 - row}")
            elif board[row][col - i] in enemy_pieces:
                moves.append(f"{piece}{itl[col - i]}{8 - row}")
                break
            else:
                break
        else:
            break

    return moves


def move_knight(board: List[str], row: int, col: int, enemy_pieces: str):
    def move_knight_helper(row_offset: int, col_offset: int) -> None:
        if 0 <= row + row_offset < 8 and 0 <= col + col_offset < 8:
            if board[row + row_offset][col + col_offset] == "1":
                moves.append(f"{piece}{itl[col + col_offset]}{8 - (row + row_offset)}")
            elif board[row + row_offset][col + col_offset] in enemy_pieces:
                moves.append(f"{piece}{itl[col + col_offset]}{8 - (row + row_offset)}")

    moves = []
    piece = board[row][col].upper()
    piece = "" if piece in "pP" else piece

    # Check all possible moves for the knight
    offsets = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]
    for offset in offsets:
        row_offset, col_offset = offset
        move_knight_helper(row_offset, col_offset)

    return moves


def move_pawn(board: List[str], row: int, col: int, enemy_pieces: str):
    moves = []
    piece = board[row][col].upper()
    piece = "" if piece in "pP" else piece

    if board[-5] == "w":
        if row - 1 > 0 and board[row - 1][col] == "1":
            moves.append(f"{piece}{itl[col]}{8 - (row - 1)}")
            if row == 6 and board[row - 2][col] == "1":
                moves.append(f"{piece}{itl[col]}{8 - (row - 2)}")
        if row - 1 > 0 and col + 1 < 8 and board[row - 1][col + 1] in enemy_pieces:
            moves.append(f"{piece}{itl[col + 1]}{8 - (row - 1)}")
        if row - 1 > 0 and col - 1 >= 0 and board[row - 1][col - 1] in enemy_pieces:
            moves.append(f"{piece}{itl[col - 1]}{8 - (row - 1)}")
    else:
        if row + 1 < 8 and board[row + 1][col] == "1":
            moves.append(f"{piece}{itl[col]}{8 - (row + 1)}")
            if row == 1 and board[row + 2][col] == "1":
                moves.append(f"{piece}{itl[col]}{8 - (row + 2)}")
        if row + 1 < 8 and col + 1 < 8 and board[row + 1][col + 1] in enemy_pieces:
            moves.append(f"{piece}{itl[col + 1]}{8 - (row + 1)}")
        if row + 1 < 8 and col - 1 >= 0 and board[row + 1][col - 1] in enemy_pieces:
            moves.append(f"{piece}{itl[col - 1]}{8 - (row + 1)}")

    # Also check for en passant the square where pawn can land is in the FEN notation
    possible_en_passant = board[-3]

    # Check if en passant is possible
    if possible_en_passant != "-":
        en_passant_row = 8 - int(possible_en_passant[1])
        en_passant_col = ord(possible_en_passant[0]) - 97

        if board[-5] == "b":
            if en_passant_row == row + 1 and (col + 1 == en_passant_col or col - 1 == en_passant_col):
                moves.append(possible_en_passant)
        else:
            if en_passant_row == row - 1 and (col + 1 == en_passant_col or col - 1 == en_passant_col):
                moves.append(possible_en_passant)

    return moves
