import os
import sys
from unittest.mock import patch

from tqdm_redirector import TqdmRedirector


class TestTqdmRedirector:

    def test_before_enable(self):
        assert print == TqdmRedirector.PRINT  # noqa
        assert sys.stdout == TqdmRedirector.STDOUT

    def test_redirect_ports_after_enable(self):
        TqdmRedirector.enable()
        with patch('tqdm.tqdm.write') as tqdm_write:
            print('a', 'b', 'c')
            tqdm_write.assert_called_with('a b c', file=TqdmRedirector.STDOUT)
        assert sys.stdout == TqdmRedirector.TQDMOUT

    def test_dont_affect_fileIO(self, tmpdir):
        # won't interfere file IO
        filepath = os.path.join(tmpdir, 'output_stream')
        with open(filepath, 'w') as f_out:
            print('a', 'b', 'c', file=f_out)
        with open(filepath, 'r') as f_in:
            assert f_in.readlines() == ['a b c\n']

    def test_after_disable(self):
        TqdmRedirector.disable()
        assert print == TqdmRedirector.PRINT  # noqa
        assert sys.stdout == TqdmRedirector.STDOUT
