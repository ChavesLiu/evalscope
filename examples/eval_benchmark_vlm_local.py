# Copyright (c) Alibaba, Inc. and its affiliates.

"""
1. Installation
EvalScope: pip install evalscope[vlmeval]

2. Deploy judge model

3. Run eval task
"""
from evalscope.backend.vlm_eval_kit import VLMEvalKitBackendManager
from evalscope.run import run_task
from evalscope.summarizer import Summarizer
from evalscope.utils.logger import get_logger

logger = get_logger()


def run_swift_eval():

    # List all datasets
    print(f'** All models from VLMEvalKit backend: {VLMEvalKitBackendManager.list_supported_models().keys()}')
    print(f'** All datasets from VLMEvalKit backend: {VLMEvalKitBackendManager.list_supported_datasets()}')

    # Prepare the config

    # Option 1: Use dict format
    task_cfg_dict = dict(
    eval_backend='OpenCompass',
    eval_config={
        'datasets': ["humaneval","triviaqa","commonsenseqa","tydiqa", "mmlu","cmmlu", "math", "ceval",'ARC_c', 'gsm8k'],
        'models': [
            {'path': 'qwen2-0_5b-instruct', 
            'openai_api_base': 'http://127.0.0.1:8000/v1/chat/completions', 
            'is_chat': True,
            'batch_size': 16},
        ],
        'work_dir': 'outputs/qwen2_eval_result',
        'limit': 10,
        },
    )
    task_cfg = {'eval_backend': 'VLMEvalKit',
                'eval_config': {'LOCAL_LLM': 'qwen2-7b-instruct',
                                'OPENAI_API_BASE': 'http://localhost:8866/v1/chat/completions',
                                'OPENAI_API_KEY': 'EMPTY',
                                'data': ['SEEDBench_IMG', 'ChartQA_TEST'],
                                'limit': 20,
                                'mode': 'all',
                                'model': [{'model_path': '../models/internlm-xcomposer2d5-7b', # path/to/model_dir
                                            'name': 'XComposer2d5'}],                          # model name for VLMEval config
                                'nproc': 1,
                                'rerun': True,
                                'work_dir': 'output'}}

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
