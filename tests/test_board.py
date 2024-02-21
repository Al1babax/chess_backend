import pytest
from engine import main


def move_pieces_for_castling(engine):
    # Move all the pieces away from the queenside and kingside black and white
    # Move all pawns two squares forward
    engine.move("a2", "a4")
    engine.move("a7", "a5")
    engine.move("b2", "b4")
    engine.move("b7", "b5")
    engine.move("c2", "c4")
    engine.move("c7", "c5")
    engine.move("d2", "d4")
    engine.move("d7", "d5")
    engine.move("e2", "e4")
    engine.move("e7", "e5")
    engine.move("f2", "f4")
    engine.move("f7", "f5")
    engine.move("g2", "g4")
    engine.move("g7", "g5")
    engine.move("h2", "h4")
    engine.move("h7", "h5")

    # Move the knights and bishops and queens
    engine.move("b1", "a3")
    engine.move("b8", "a6")
    engine.move("c1", "b2")
    engine.move("c8", "b7")
    engine.move("d1", "c2")
    engine.move("d8", "c7")
    engine.move("f1", "e2")
    engine.move("f8", "e7")
    engine.move("g1", "f3")
    engine.move("g8", "f6")


def test_white_kingside_castling():
    engine = main.Engine()

    move_pieces_for_castling(engine)

    # Perform kingside castling
    engine.move("e1", "g1")

    # Check that FEN got updated correctly
    assert "K" not in engine.board.board[-4] and "Q" not in engine.board.board[-4]

    # Check FEN board
    assert engine.board.board[:8] == ['rnbqk11r', 'pppp1ppp', '11111n11', '11b1p111', '11B1P111', '11111N11',
                                      'PPPP1PPP', 'RNBQ1RK1']


def test_black_kingside_castling():
    engine = main.Engine()

    move_pieces_for_castling(engine)

    # Perform kingside castling
    engine.move("e1", "g1")

    # Perform kingside castling
    engine.move("e8", "g8")

    # Check that FEN got updated correctly
    assert "k" not in engine.board.board[-4] and "q" not in engine.board.board[-4]

    # Check FEN board
    assert engine.board.board[:8] == ['rnbq1rk1', 'pppp1ppp', '11111n11', '11b1p111', '11B1P111', '11111N11',
                                      'PPPP1PPP', 'RNBQ1RK1']


def test_white_queenside_castling():
    pass
