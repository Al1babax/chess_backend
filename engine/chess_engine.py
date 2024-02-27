import random


class Engine:

    def __init__(self, game) -> None:
        self.board = game.board

    def eval(self):
        """
        Evaluate the board
        The higher the value, the better the position for white
        :return:
        """
        piece_values = {
            "P": 1.00,
            "N": 3.5,
            "B": 3.5,
            "R": 5.25,
            "Q": 10,
        }

        # Get value of the color's pieces
        value = 0
        for row in range(8):
            for col in range(8):
                if self.board.board[row][col] is not None:
                    if self.board.board[row][col].color == "w":
                        value += piece_values[self.board.board[row][col].symbol]
                    else:
                        value -= piece_values[self.board.board[row][col].symbol]

        return value

    def gather_all_moves(self):
        # Gather all moves from pieces for whose turn it is
        moves = []
        turn = self.board.turn

        for row in range(8):
            for col in range(8):
                if self.board.board[row][col] is not None and self.board.board[row][col].color == turn:
                    moves += self.board.board[row][col].moves

        return moves

    def search(self, depth):
        """

        :param depth:
        :return:
        """
        # Gather all moves
        moves = self.gather_all_moves()

        # If depth is 0, return the evaluation
        if depth == 0:
            return self.eval()

        # If there are no moves, return the evaluation
        if len(moves) == 0:
            return self.eval()

        # If depth is 1, return the evaluation
        if depth == 1:
            return max([self.eval() for move in moves])

        # If depth is more than 1, return the maximum evaluation of the next depth
        return max([self.search(depth - 1) for move in moves])

    def random_move(self) -> tuple:
        """
        Select a random piece for a color and select random move from its moves, if it does not have any moves, select another piece
        :param game: Game object
        :return: Tuple of old position and new position
        """
        # Get turn
        turn = self.board.turn

        # Get all the pieces of the color
        pieces = []
        for row in range(8):
            for col in range(8):
                if self.board.board[row][col] is not None and self.board.board[row][col].color == turn:
                    pieces.append(self.board.board[row][col])

        # Select random piece
        pieces = [piece for piece in pieces if len(piece.moves) > 0]
        random_piece = pieces[random.choice(range(len(pieces)))]

        # Select random move
        move = random_piece.moves[random.choice(range(len(random_piece.moves)))]

        # Perform the move
        return random_piece.position, move
