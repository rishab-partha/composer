# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2023 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Abstract class for utilities that access and run code on serverless eval clients."""

import abc
from typing import Dict, List

__all__ = ['EvalClient']


class EvalClient(abc.ABC):
    """Abstract class for implementing eval clients, such as LambdaEvalClient."""

    def invoke(self, payload: List[List[List[Dict[str, str]]]]) -> List[List[List[bool]]]:
        """Invoke a provided batch of dictionary payload to the client.

        For code generation, the payload is a list of list of lists of JSONs with the following attributes:
            {
                'code': <code to be evaluated>,
                'input': <test input>,
                'output': <test output>,
                'entry_point': <entry point>,
                'language': <language>,

            }

        The JSON is formatted as [[[request]]] so that the client can batch requests. The outermost list is for the generations of a
        given prompt, the middle list is for the beam generations of a given prompt, and the innermost list is for each test cases.
        Args:
            payload: the materials of the batched HTTPS request to the client organized by prompt, beam generation, and test case.

        Returns:
            Whether the test case passed or failed.
        """
        del payload  # unused
        raise NotImplementedError(f'{type(self).__name__}.invoke is not implemented')

    def close(self):
        """Close the object store."""
        pass
