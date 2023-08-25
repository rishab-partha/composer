# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""MCLI compatible eval client."""
import logging
import os
from typing import Dict, List
import time
import numpy as np

import mcli

from composer.utils.eval_client.eval_client import EvalClient

__all__ = ['MCLIEvalClient']
log = logging.getLogger(__name__)


class MCLIEvalClient(EvalClient):
    """Utility for creating a client for and invoking an AWS Lambda through MCLI."""

    def __init__(self, backoff: int = 3, num_retries: int = 5) -> None:
        """Checks that the requisite environment variables are in the EvalClient.

        There must be MOSAICML_PLATFORM to be on MCLI.
        """
        if os.environ.get('MOSAICML_PLATFORM', False) == False:
            raise Exception('Cannot use MCLI eval without being on MosaicML Platform.')
        log.debug('Running code eval through MCLI.')
        self.backoff = backoff
        self.num_retries = num_retries

    def invoke(self, payload: List[List[List[Dict[str, str]]]]) -> List[List[List[bool]]]:
        """Invoke a batch of provided payloads for code evaluations."""
        num_beams = len(payload[0])
        num_tests = [len(generation_payload[0]) for generation_payload in payload]
        cum_tests = (np.cumsum([0] + num_tests[:-1])*num_beams).tolist()
        test_cases = [test_case for generation_payload in payload for beam_payload in generation_payload for test_case in beam_payload]
        ret_helper = [False] * len(test_cases)
        for i in range(self.num_retries):
            try:
                ret_helper = mcli.get_code_eval_output(test_cases)
                break
            except mcli.MAPIException as e:
                if i == self.num_retries - 1:
                    log.error(f'Failed to get code eval output after {self.num_retries} retries. Error: {e}')
                log.warning(f'Failed to get code eval output, retrying in {self.backoff**i} seconds.')
                time.sleep(self.backoff**i) 

        ret = [[[ret_helper[cum_tests[i] + j * num_tests[i] + k] for k in range(num_tests[i])] for j in range(num_beams)] for i in range(len(payload))]
        return ret