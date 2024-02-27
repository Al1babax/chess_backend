import random
# TODO: Optimize the search minmax algorithm, also add alpha-beta pruning after that


class Engine:

    def __init__(self, game) -> None:
        self.board = game.board
        self.search_depth = 2

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
            "K": 0
        }

        # Get value of the color's pieces
        value = 0
        for row in range(8):
            for col in range(8):
                if self.board.board[row][col] is not None:
                    if self.board.board[row][col].color == "w":
                        value += piece_values[self.board.board[row][col].piece_type]
                    else:
                        value -= piece_values[self.board.board[row][col].piece_type]

        return value

    def gather_all_moves(self):
        # Gather all moves from pieces for whose turn it is
        moves = []
        turn = self.board.turn

        for row in range(8):
            for col in range(8):
                if self.board.board[row][col] is not None and self.board.board[row][col].color == turn:
                    for move in self.board.board[row][col].moves:
                        # print(f"Piece: {self.board.board[row][col].piece_type} {self.board.board[row][col].position} Move: {move}")
                        moves.append((self.board.board[row][col].position, move))

        return moves

    def search(self, depth: int) -> int:
        """
        Search by doing the most optimal move based on whose turn it is.
        Searching in rotation first most optimal move for whose turn it is and on next recursion level, most optimal
        move for the other color.
        Base cases are when depth goes to 0 and then return evaluation of the board.
        OR when there are no moves for the color, which basically means either checkmate or stalemate and return - or + infinity for eval.
        :param depth:
        :return:
        """
        # Base case
        if depth == 0:
            return self.eval()

        # Get all moves
        moves = self.gather_all_moves()

        # Base case
        if len(moves) == 0:
            if self.board.is_check() and self.board.turn == "w":
                # White is in checkmate
                return -10000
            elif self.board.is_check() and self.board.turn == "b":
                # Black is in checkmate
                return 10000
            else:
                return 0

        # Search
        if self.board.turn == "w":
            best_move = -10000
            for move in moves:
                self.board.move(move[0], move[1])
                best_move = max(best_move, self.search(depth - 1))
                self.board.undo_move()

            return best_move
        else:
            best_move = 10000
            for move in moves:
                self.board.move(move[0], move[1])
                best_move = min(best_move, self.search(depth - 1))
                self.board.undo_move()

            return best_move

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

    def best_move(self) -> tuple:
        """
        Select the best move for the color
        :return: Tuple of old position and new position
        """

        # Get all the moves
        moves = self.gather_all_moves()

        # Search for the best move
        turn = self.board.turn
        best_val = 0
        best_move_index = 0
        for i, move in enumerate(moves):
            print(f"Progress: {i + 1}/{len(moves)}")
            self.board.move(move[0], move[1])
            value = self.search(self.search_depth)
            self.board.undo_move()
            if turn == "w":
                if value > best_val:
                    best_val = value
                    best_move_index = i
            else:
                if value < best_val:
                    best_val = value
                    best_move_index = i

        return moves[best_move_index]
