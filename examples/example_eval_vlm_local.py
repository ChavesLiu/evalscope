# Copyright (c) Alibaba, Inc. and its affiliates.

"""
1. Installation
eval-scope: pip install llmuses[vlmeval]>=0.4.3

2. Deploy judge model

3. Run eval task
"""
from llmuses.backend.vlm_eval_kit import VLMEvalKitBackendManager
from llmuses.run import run_task
from llmuses.summarizer import Summarizer
from llmuses.utils.logger import get_logger

logger = get_logger()


def run_swift_eval():

    # List all datasets
    print(f'** All models from VLMEvalKit backend: {VLMEvalKitBackendManager.list_supported_VLMs().keys()}')
    print(f'** All datasets from VLMEvalKit backend: {VLMEvalKitBackendManager.list_supported_datasets()}')

    # Prepare the config

    # Option 1: Use dict format
    # task_cfg = {'eval_backend': 'VLMEvalKit',
    #             'eval_config': {'LOCAL_LLM': 'qwen2-7b-instruct',
    #                             'OPENAI_API_BASE': 'http://localhost:8866/v1/chat/completions',
    #                             'OPENAI_API_KEY': 'EMPTY',
    #                             'data': ['SEEDBench_IMG', 'ChartQA_TEST'],
    #                             'limit': 20,
    #                             'mode': 'all',
    #                             'model': [{'model_path': '../models/internlm-xcomposer2d5-7b', # path/to/model_dir
    #                                         'name': 'XComposer2d5'}],                          # model name for VLMEval config
    #                             'nproc': 1,
    #                             'rerun': True,
    #                             'work_dir': 'output'}}

    # Option 2: Use yaml file
    task_cfg = "examples/tasks/eval_vlm_local.yaml"

    # Run task
    run_task(task_cfg=task_cfg)

    # [Optional] Get the final report with summarizer
    logger.info('>> Start to get the report with summarizer ...')
    report_list = Summarizer.get_report_from_cfg(task_cfg)
    logger.info(f'\n>> The report list: {report_list}')


if __name__ == '__main__':
    run_swift_eval()