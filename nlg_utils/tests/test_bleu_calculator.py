import numpy as np

from ..bleu_calculator import BLEUScoreCalculator, SmoothingFunction


EOS = -1


def test_trivial():
    ref = [
        [0, 1, 2, EOS, 0],   # seqlen = 3
        [0, 1, 1, 2, EOS],   # seqlen = 4
    ]
    cand = ref
    calculator = BLEUScoreCalculator(ref, eos_idx=EOS, max_gram=5)
    np.testing.assert_array_almost_equal(
        calculator.bleu_score(cand),
        [
            [1, 1, 1, 0, 0],  # seqlen = 3
            [1, 1, 1, 1, 0],  # seqlen = 4
        ],
    )


def test_with_clipped():
    ref = [
        [0, 1, 2, 3, 4],  # has 1 `1`
        [0, 1, 1, 2, 3],  # has 2 `1`
    ]
    cand = [[1, 1, 1, 1, 1]]  # has 5 `1`
    calculator = BLEUScoreCalculator(ref, eos_idx=EOS, max_gram=5)
    np.testing.assert_array_almost_equal(
        calculator.calculate_batch_precision(cand),
        [[2 / 5, 1 / 4, 0, 0, 0]],
        # min(2, 5) / 5, min(1, 4) / 4
    )
    np.testing.assert_array_almost_equal(
        calculator.bleu_score(cand),
        [[2 / 5, (2 / 5 * 1 / 4) ** 0.5, 0, 0, 0]],
    )


def test_long():
    ref = [[0, 1, 2, 0, 1, 2, 3, 0, 1, 2]]
    cand = [[0, 1, 0, 1, 2, 1, 0, 1, 2, 3]]
    calculator = BLEUScoreCalculator(ref, eos_idx=EOS, max_gram=4)
    np.testing.assert_array_almost_equal(
        calculator.calculate_batch_precision(cand),
        [[
            9 / 10,  # 4 `1` in cand, clipped by 3 `1` in ref
            6 / 9,   # 2 `10` + 1 `21` not in ref
            3 / 8,   # 2 `012` + 1 `123`
            1 / 7,   # 1 `0123`
        ]],
    )


def test_with_brevity_penalty():
    ref = [
        [0, 1, 2, EOS, 0, 0, 0, 0, 0, 0],  # L = 3
        [0, 1, 2, 3, 4, EOS, 0, 0, 0, 0],  # L = 5
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # L = 10
    ]
    cand = [
        [0, 1, EOS, 0, 0, 0, 0, 0],  # L = 2
        [0, 1, 2, 3, EOS, 0, 0, 0],  # L = 4
        [0, 1, 2, 3, 4, 5, 6, EOS],  # L = 7
        [0, 1, 2, 3, 4, 5, 6, 7],    # L = 8
    ]
    calculator = BLEUScoreCalculator(ref, eos_idx=EOS, max_gram=1)
    np.testing.assert_array_almost_equal(
        calculator.bleu_score(cand),
        [
            [np.exp(1 - 3 / 2)],
            [1],  # 4 is closest to 3 & 5, choose 3 since it's tied and no penalty applied
            [1],  # 7 is closest to 5, which is shorter so no penalty applied
            [np.exp(1 - 10 / 8)],
        ],
    )


def test_smoothing():
    ref = [[0, 1, 2, 3, 4]]
    smoothing = SmoothingFunction.fuzz_smoothing
    calculator = BLEUScoreCalculator(ref, eos_idx=EOS, max_gram=4, smoothing=smoothing)
    np.testing.assert_array_almost_equal(
        calculator.bleu_score([[1, 2, 0, 3, 4]]),
        [[
            1.,
            (1 * 2 / 4) ** (1 / 2),
            (1 * 2 / 4 * 0.1 / 3) ** (1 / 3),  # no matching 3-gram
            (1 * 2 / 4 * 0.1 / 3 * 0.1 / 2) ** (1 / 4),  # no matching 3/4-gram
        ]],
    )
