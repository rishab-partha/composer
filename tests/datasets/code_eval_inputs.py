import contextlib
import os
import random
from pathlib import Path

import pytest
import transformers
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from composer import Evaluator
from composer.core import DataSpec
from composer.datasets.in_context_learning_evaluation import (InContextLearningCodeEvalDataset,
                                                              _get_fewshot_sample_idxs, _make_padded_input,
                                                              get_icl_task_dataloader)
from composer.loggers import InMemoryLogger
from composer.metrics import (InContextLearningLMAccuracy, InContextLearningMultipleChoiceAccuracy,
                              InContextLearningQAAccuracy)
from composer.models import HuggingFaceModel
from composer.trainer import Trainer
from composer.utils import dist, reproducibility

def get_code_eval_inputs():
    dataset_uri = 'human_eval.jsonl'
    num_fewshot = 1
    prompt_string = 'Please code:\n'
    num_evals = 1

    local_data = os.path.join(os.path.dirname(__file__), 'local_data')

    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
    dataset_uri = f'{local_data}/{dataset_uri}'
    batch_size = 4
    seqlen = 1024

    dl = get_icl_task_dataloader('code_evaluation',
                                 dataset_uri,
                                 tokenizer,
                                 batch_size,
                                 max_seq_len=seqlen,
                                 pad_tok_id=tokenizer.eos_token_id,
                                 num_fewshot=num_fewshot,
                                 prompt_string=prompt_string,
                                 example_delimiter='\n',
                                 question_prelimiter='Complete the code: \n',
                                 destination_path=str(f'icl_{num_fewshot}.jsonl'),
                                 num_evals=num_evals)
    assert isinstance(dl, DataSpec)

    assert isinstance(dl.dataloader, DataLoader)  # pyright

    return dl