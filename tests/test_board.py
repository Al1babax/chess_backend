import engine.main as main
import engine.generate_moves as generate_moves


def test_attack_lines():
    """
    Test that right kind of attack lines and potential attack lines are generated from kings perspective for move validation
    :return:
    """
    fen = ["4kb2/pp2p1pp/2n2n2/1Bp2p2/q1NKQ1Pr/2P5/PbP1PpPP/R1B1r1NR w - c6 0 1",
           "r5nr/pbp2kpp/1p6/2bp2q1/2B1Pp2/2N3P1/PPP1n2P/R1BQ1RK1 w - - 0 1"]

    answers = [(['c5', 'c6'], ['c4', 'b4', 'a4', 'c3', 'b2']),
               (['f2', 'e3', 'd4', 'c5', 'e2'], ['g2', 'g3', 'g4', 'g5'])]

    for f in fen:
        board = main.Board(f)
        ra, pa = generate_moves.generate_lines(board.board[board.white_king[0]][board.white_king[1]], board)
        ra = [(chr(97 + move[1]) + str(8 - move[0])) for move in ra]
        pa = [(chr(97 + move[1]) + str(8 - move[0])) for move in pa]

        assert ra == answers[fen.index(f)][0]
        assert pa == answers[fen.index(f)][1]

