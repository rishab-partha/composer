# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0
import pytest

from composer.utils import LocalEvalClient


@pytest.mark.parametrize('code_results', [('def add_1(x):\n    return x + 1', True),
                                          ('def add_1(x):\n    return y + 1', False),
                                          ('def add_1(x):\n    while True:\n        x += 1', False),
                                          ('def add_1(x): return x + 2', False)])
def test_local_invoke(code_results):
    """Test invocation function for LocalEvalClient with code that succeeds, fails compilation, times out, and is incorrect.
    """
    eval_client = LocalEvalClient()
    assert eval_client.invoke({
        'code': code_results[0],
        'input': '(1,)',
        'output': '2',
        'entry_point': 'add_1'
    }) == code_results[1]
