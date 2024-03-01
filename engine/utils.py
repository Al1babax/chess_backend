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
