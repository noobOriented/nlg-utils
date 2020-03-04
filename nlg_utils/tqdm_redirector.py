import builtins
import sys

from tqdm import tqdm
from tqdm.contrib import DummyTqdmFile


class TqdmRedirector:

    # ports before being redirected
    STDOUT, STDERR, PRINT = sys.stdout, sys.stderr, builtins.print
    TQDMOUT, TQDMERR = DummyTqdmFile(sys.stdout), DummyTqdmFile(sys.stderr)
    STREAMS_TO_REDIRECT = {None, STDOUT, STDERR, TQDMOUT, TQDMERR}

    @classmethod
    def enable(cls):

        def safe_print(*values, sep=' ', end='\n', file=None, flush=False):
            if file in cls.STREAMS_TO_REDIRECT:
                # NOTE tqdm (v4.40.0) can't support end != '\n' and flush
                tqdm.write(sep.join(str(v) for v in values), file=cls.STDOUT)
            else:
                cls.PRINT(*values, sep=sep, end=end, file=file, flush=flush)

        sys.stdout, sys.stderr, builtins.print = cls.TQDMOUT, cls.TQDMERR, safe_print

    @classmethod
    def disable(cls):
        sys.stdout, sys.stderr, builtins.print = cls.STDOUT, cls.STDERR, cls.PRINT
