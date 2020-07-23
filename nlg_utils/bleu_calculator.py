from collections import Counter, defaultdict
from typing import Callable, Mapping

import numpy as np
from tqdm import tqdm

from .utils import DirectoryHelper, PickleCache


class BLEUScoreCalculator:

    INT_DTYPE = np.int32

    def __init__(
            self,
            references: np.ndarray,
            max_gram: int,
            *,
            eos_idx: int = None,
            smoothing: Callable = None,
            verbose: bool = False,
            cache_dir: DirectoryHelper = None,
        ):
        self.max_gram = max_gram
        self.eos_idx = eos_idx
        self.smoothing = smoothing
        self.verbose = verbose
        self.cache_dir = DirectoryHelper(cache_dir)
        self.cache_dir.makedirs()

        self._initialize_tables(references)

    def _initialize_tables(self, references):
        references = np.asarray(references, dtype=self.INT_DTYPE)
        ref_lengths = get_seqlens(references, eos_idx=self.eos_idx)
        self.ref_counters = [
            self._create_ngram_counter(n, references, ref_lengths)
            for n in range(1, self.max_gram + 1)
        ]
        self.brevity_penalty = get_brevity_penalty_table(
            ref_lengths,
            maxlen=references.shape[1],
        )

    @PickleCache.tofile(path=lambda self, n, *_: self.cache_dir.get_path(f'{n}-gram.pkl'))
    def _create_ngram_counter(self, n, references, seqlens) -> Mapping:
        iterator = zip(references, seqlens)
        if self.verbose:
            print(f"Building {n}-gram table...")
            iterator = tqdm(iterator, total=len(references), unit='sample')
        max_count = {}
        for ref, length in iterator:
            for gram, cnt in Counter(get_hashable_ngrams(ref[:length], n)).items():
                if cnt > max_count.get(gram, 0):
                    max_count[gram] = cnt
        return max_count

    def bleu_score(self, candidates: np.ndarray) -> np.ndarray:
        candidates = np.asarray(candidates, dtype=self.INT_DTYPE)
        cand_lens = get_seqlens(candidates, eos_idx=self.eos_idx)

        precisions = self.calculate_batch_precision(candidates, cand_lens)  # shape (N, max_grams)
        mean_precision = np.power(
            np.cumprod(precisions, axis=1),
            1. / np.arange(1, self.max_gram + 1),  # to perform geometric mean
        )  # shape (N, max_gram)
        batch_bleu = mean_precision * self.brevity_penalty[cand_lens, np.newaxis]
        return batch_bleu

    def calculate_batch_precision(self, candidates, seqlens=None) -> np.ndarray:
        candidates = np.asarray(candidates, dtype=self.INT_DTYPE)
        if seqlens is None:
            seqlens = get_seqlens(candidates, eos_idx=self.eos_idx)

        clipped_count = np.array([
            self._calculate_clipped_count(cand[:length])
            for cand, length in zip(candidates, seqlens)
        ])  # shape (N, max_gram)

        # Total number of n-grams = Length - (n - 1)
        total_count = seqlens[:, np.newaxis] - np.arange(self.max_gram)  # shape (N, max_gram)
        if self.smoothing:
            clipped_count, total_count = self.smoothing(clipped_count, total_count)
        return safe_divide(clipped_count, total_count)  # avoid zero division

    def _calculate_clipped_count(self, candidate: np.ndarray):
        return [self._clipped_count_n(candidate, n) for n in range(1, self.max_gram + 1)]

    def _clipped_count_n(self, candidate, n):
        ref_counter = self.ref_counters[n - 1]
        counter = defaultdict(int)
        for gram in get_hashable_ngrams(candidate, n):
            # NOTE faster than dict.get and min
            if gram in ref_counter and counter[gram] < ref_counter[gram]:
                counter[gram] += 1
        return sum(counter.values())


class SmoothingFunction:
    # Reference: http://acl2014.org/acl2014/W14-33/pdf/W14-3346.pdf

    @staticmethod  # smoothing1
    def fuzz_smoothing(numerator, denominator, eps: float = 0.1):
        numerator = np.maximum(numerator, eps)
        return numerator, denominator

    @staticmethod  # smoothing2
    def add1_smoothing(numerator, denominator):
        shift = np.ones_like(numerator)
        shift[:, 0] = 0  # don't add 1-gram
        return numerator + shift, denominator + shift


def get_brevity_penalty_table(ref_lengths, maxlen):
    possible_lengths = np.arange(maxlen + 1)
    closest_lengths = get_closest_values(possible_lengths, target=ref_lengths)
    brevity_penalty = np.minimum(
        np.exp(1. - safe_divide(closest_lengths, possible_lengths)),
        1.,
    )
    return brevity_penalty


def safe_divide(a, b):
    return a / np.maximum(b, 1)


def get_seqlens(data: np.ndarray, eos_idx: int = None):
    data = np.asarray(data, dtype=np.int)
    maxlen = data.shape[1]
    if eos_idx is None:
        return np.full([data.shape[0], *data.shape[2:]], maxlen)
    end_mask = np.equal(data, eos_idx)
    return np.where(
        np.any(end_mask, axis=1),
        np.argmax(end_mask, axis=1),  # position of eos
        maxlen,  # pad length
    )


def get_closest_values(arr: np.ndarray, target: np.ndarray):
    # arr shape (M), target shape (N)
    target = np.unique(target)
    diff = np.abs(arr[:, np.newaxis] - target)  # shape (M, N)
    closest_ids = np.argmin(diff, axis=-1)  # shape (M), value in [0, N)
    return target[closest_ids]  # shape (M)


def get_hashable_ngrams(sequence, n):
    if n == 1:  # items already hashable
        return sequence

    width = sequence.itemsize * n
    bytes_sequence = sequence.tobytes()  # hashable
    return [
        bytes_sequence[start: start + width]
        for start in range(0, len(bytes_sequence) - width + 1, sequence.itemsize)
    ]
