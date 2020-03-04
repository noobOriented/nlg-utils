import os
from unittest.mock import Mock

import numpy as np
import pytest

from ..utils import PickleCache, NumpyCache


def equal(x, y):
    return np.array_equal(x, y) if isinstance(x, np.ndarray) else x == y


@pytest.mark.parametrize(
    'cacher, data, filename',
    [
        (PickleCache, {'a': 1, 'b': 2, 'c': [3, 4]}, 'test.pkl'),
        (NumpyCache, np.random.choice(100, size=[100]), 'test.npz'),
    ],
)
def test_cache(tmpdir, cacher, data, filename):
    filename = os.path.join(tmpdir, filename)
    create = Mock(return_value=data)
    wrapped_create = cacher.tofile(filename)(create)

    assert equal(wrapped_create(), data)
    assert os.path.isfile(filename)  # save to file
    assert create.call_count == 1

    assert equal(wrapped_create(), data)
    assert create.call_count == 1  # load from file, don't create again


def test_cache_dynamic(tmpdir):
    data = '123'
    create = Mock(return_value=data)
    wrapped_create = PickleCache.tofile(
        path=lambda key: (
            os.path.join(tmpdir, key) if key.endswith('.pkl')
            else None
        ),
    )(create)

    assert wrapped_create('a.pkl') == data
    assert os.path.isfile(os.path.join(tmpdir, 'a.pkl'))  # save to file
    assert create.call_count == 1

    assert wrapped_create('a.pkl') == data
    assert create.call_count == 1  # load from file, don't create again

    assert wrapped_create('b.pkl') == data
    assert create.call_count == 2  # different key, create again

    assert wrapped_create('c') == data
    assert not os.path.isfile(os.path.join(tmpdir, 'c'))  # condition return False
