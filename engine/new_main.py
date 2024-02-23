from typing import List, Optional
import generate_moves


class Piece:

    def __init__(self, position: str, color: str, piece_type: str) -> None:
        """
        Initialize the piece
        :param position: Position in chess notation
        :param color: Color of the piece, either "w" or "b"
        :param piece_type: Type of the piece, either "P", "N", "B", "R", "Q", "K
        :param moves: List of possible moves of the piece in chess notation
        """
        # Position of the piece using chess notation
        self.position: str = position

        # Color of the piece
        self.color: str = color

        # Type of the piece
        self.piece_type: str = piece_type

        # Possible moves of the piece
        self.moves: List[str] = []

        # If the piece is captured
        self.captured: bool = False

        # If the piece is pinned
        self.is_pinned: bool = False

        # If it is the first move of the piece
        self.first_move: bool = True

        # If the piece is king and is in check
        self.in_check: bool = False


class Board:
    def __init__(self, fen_string: str) -> None:
        # Fen string to initialize the board
        self.fen_string = fen_string

        # Board that have pieces and if the square is empty it is None
        self.board: List[List[Optional[Piece]]] = []

        # Create the board
        self.create_board()

    def create_board(self) -> None:
        # Create the board so that white piece are first in the matrix
        fen_piece = self.fen_string.split(" ")[0]
        rows = fen_piece.split("/")

        for i, row in enumerate(rows):
            new_row = []
            for k, piece in enumerate(row):
                if piece.isdigit():
                    for _ in range(int(piece)):
                        new_row.append(None)

                    continue

                # Construct chess notation
                position = chr(97 + k) + str(8 - i)

                # Construct the piece color
                if piece.isupper():
                    color = "w"
                else:
                    color = "b"

                # Construct the piece type
                piece_type = piece.upper()

                # Create the piece
                new_row.append(Piece(position, color, piece_type))

            self.board.append(new_row)

    def generate_piece_moves(self) -> None:
        """
        Generate the moves for all the pieces
        :return:
        """
        for row in self.board:
            for piece in row:
                if piece is not None:
                    # Generate the moves for the piece
                    piece.moves = generate_moves.generate(piece, self.board)

    def is_valid(self, move: str) -> bool:
        """
        Check if the move is valid
        :param move: Move in chess notation
        :return: True if the move is valid else False
        """
        row = 8 - int(move[1])
        col = ord(move[0]) - 97

        # Check if the move is in the list of moves of the piece
        if move in self.board[row][col].moves:
            return True
        else:
            return False

    def move(self, old_pos, new_pos) -> None:
        """
        Move the piece
        :param old_pos: Old position of the piece
        :param new_pos: New position of the piece
        :return:
        """
        pass


class Game:
    def __init__(self) -> None:
        pass


def main():
    # Create the board
    board = Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

    print(board.board[0])

    # Create the game
    game = Game()


if __name__ == "__main__":
    main()
