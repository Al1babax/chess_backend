from typing import List, Optional
from new_main import Piece, Board


class Movement:
    # Class that look piece movement to all directions
    def __init__(self, piece: Piece, board: List[List[Optional[Piece]]]) -> None:
        self.piece = piece
        self.board = board

    def move_up(self, distance: int = 8) -> List[str]:
        pass

    def move_down(self, distance: int = 8) -> List[str]:
        pass

    def move_right(self, distance: int = 8) -> List[str]:
        pass

    def move_left(self, distance: int = 8) -> List[str]:
        pass

    def move_up_right(self, distance: int = 8) -> List[str]:
        pass

    def move_up_left(self, distance: int = 8) -> List[str]:
        pass

    def move_down_right(self, distance: int = 8) -> List[str]:
        pass

    def move_down_left(self, distance: int = 8) -> List[str]:
        pass

    def move_knight(self) -> List[str]:
        pass

    def pawn_capture(self) -> List[str]:
        pass

    def generate_pawn_movement(self) -> List[str]:
        pass

    def generate_knight_movement(self) -> List[str]:
        pass

    def generate_bishop_movement(self) -> List[str]:
        pass

    def generate_rook_movement(self) -> List[str]:
        pass

    def generate_queen_movement(self) -> List[str]:
        pass

    def generate_king_movement(self) -> List[str]:
        pass


def generate(piece: Piece, board: List[List[Optional[Piece]]]):
    """
    Generate the movement of the piece
    :param piece: Piece object
    :param board: List of the board with all the pieces and None if the square is empty
    :return:
    """
    movement_object = Movement(piece, board)
    moves = []

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
