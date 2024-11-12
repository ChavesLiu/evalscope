# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
import pprint
from evalscope.run import run_task
from evalscope.config import TaskConfig, registry_tasks


"""
数据格式：
{"query": "中国的首都是哪里？", "response": "中国的首都是北京"}
{"query": "世界上最高的山是哪座山？", "response": "是珠穆朗玛峰"}
{"query": "为什么北极见不到企鹅？", "response": "因为企鹅大多生活在南极"}
"""

def run_x_task(task_cfg, name, eval_model_path):
    # Prepare the config
    work_spaces = "/chubao/tj-data-ssd-03/liuchengwei/workspaces/evals_farm/evalscope"  # 工作目录
    task_cfg.update({
        # 'model_args': {'revision': None, 'precision': torch.float16, 'device_map': 'auto'},
        'generation_config': {'do_sample': False, 'repetition_penalty': 1.0, 'max_new_tokens': 2048},
        'eval_type': 'checkpoint',                 # 评测类型，需保留，固定为checkpoint
        'model': eval_model_path,  # 模型路径
        'template_type': 'qwen',                   # 模型模板类型
        # 'work_dir': work_spaces,
        'outputs': f"{work_spaces}/outputs/{name}",
        # 'limit': 100,
        'mem_cache': False,
        'use_cache': True,
        'eval_backend': 'Native',
        # 'eval_backend': 'VLMEvalKit',
        'debug': False,
    })
    print("task_cfg:")
    pprint.pprint(task_cfg)
    
    # Run task
    run_task(task_cfg=task_cfg)



def run_query_recom_eval():
    """
    query推荐评测
    """
    # 1. 配置自定义数据集和模型
    # 自定义模型
    eval_model_path = '/chubao/tj-train-ssd-21/liuchengwei/models/saves/sft-query_recom_v1_qwen2_7B_dashu-v5_test11_all_10281127/checkpoint-1180'
    # 自定义数据
    name = "query_recom"
    dataset_dir = '/chubao/tj-train-ssd-21/liuchengwei/datasets/query_recom_lcw/query_recom_eval'
    data_pattern = 'general_qa'
    TaskConfig.registry(
        name=name,                  # 任务名称
        data_pattern=data_pattern,  # 数据格式
        dataset_dir=dataset_dir,    # 数据集路径
        # subset_list=['chatlog_query_recom_v9_10W_test_transqa',],    # 评测数据集名称，上述 example.jsonl
        # subset_list=['test',],    # 评测数据集名称，上述 example.jsonl
    )
    
    # 2. 配置任务，通过任务名称获取配置
    task_cfg = registry_tasks[name]
    
    # 3. 执行评测
    run_x_task(task_cfg=task_cfg,
        name=name,
        eval_model_path=eval_model_path)



def run_user_portrait_eval(subtask=None):
    """
    用户画像评测
    """
    if not subtask:
        return
    # 1. 配置自定义数据集和模型
    # 自定义模型
    eval_model_path = '/chubao/tj-train-ssd-21/liuchengwei/models/reserve/memory/v2/UP_sft_v2.0_4.8kl1_4.6kl2_1.5wl3_5.9kl1up_10112045/checkpoint-862'
    # 自定义数据
    name = f"user_portrait_{subtask}"
    dataset_dir = f'/chubao/tj-train-ssd-21/liuchengwei/datasets/user_portrait_new_lt/user_portrait_eval/{subtask}'
    data_pattern = 'general_qa'
    TaskConfig.registry(
        name=name,                  # 任务名称
        data_pattern=data_pattern,  # 数据格式
        dataset_dir=dataset_dir,    # 数据集路径
        # subset_list=['head','l1_summary_to_l2_summary/',],    # 评测数据集名称，上述 example.jsonl
    )
    
    # 2. 配置任务，通过任务名称获取配置
    task_cfg = registry_tasks[name]
    
    # 3. 执行评测
    run_x_task(task_cfg=task_cfg,
        name=name,
        eval_model_path=eval_model_path)


def run_benchmark_eval(subtask=None):
    """
    用户画像评测
    """
    if not subtask:
        return
    # 1. 配置自定义数据集和模型
    # 自定义模型
    eval_model_path = '/chubao/tj-train-ssd-21/liuchengwei/models/reserve/memory/v2/UP_sft_v2.0_4.8kl1_4.6kl2_1.5wl3_5.9kl1up_10112045/checkpoint-862'
    # 自定义数据
    name = f"user_portrait_{subtask}"
    dataset_dir = f'/chubao/tj-train-ssd-21/liuchengwei/datasets/user_portrait_new_lt/user_portrait_eval/{subtask}'
    data_pattern = 'general_qa'
    TaskConfig.registry(
        name=name,                  # 任务名称
        data_pattern=data_pattern,  # 数据格式
        dataset_dir=dataset_dir,    # 数据集路径
        # subset_list=['head','l1_summary_to_l2_summary/',],    # 评测数据集名称，上述 example.jsonl
    )
    
    # 2. 配置任务，通过任务名称获取配置
    task_cfg = registry_tasks[name]
    
    # 3. 执行评测
    run_x_task(
        task_cfg=task_cfg,
        name=name,
        eval_model_path=eval_model_path)


if __name__ == '__main__':    
    # run_query_recom_eval()
    # tasknames = ["l1_summary_to_l2_summary","l2_summary_to_l3_tag","user_data_to_l1_summary","user_portrait_l1_summary"]
    tasknames = ["user_data_to_l1_summary","user_portrait_l1_summary"]
    for taskname in tasknames:
        run_user_portrait_eval(taskname)
    
    # CUDA_VISIBLE_DEVICES=0 python3 examples/query_memory_eval_custom_llm_data.py
    
    """
    python -m evalscope.run \
        --model /chubao/tj-train-ssd-21/liuchengwei/models/saves/sft-query_recom_v1_qwen2_7B_dashu-v5_test11_all_10281127/checkpoint-1180 \
        --work-dir ./ \
        --dataset-dir /chubao/tj-train-ssd-21/liuchengwei/datasets/eval_data/opensource_data/ \
        --outputs ./outputs/arc \
        --template-type qwen \
        --datasets arc 
        # --dataset-hub Local \
        
    {'dataset_args': {'general_qa': {'local_path': '/chubao/tj-train-ssd-21/liuchengwei/datasets/query_recom_lcw/',
                                     'subset_list': ['chatlog_query_recom_v9_10W_test_transqa_head']}},
     'dataset_hub': 'Local',
     'datasets': ['general_qa'],
     'debug': False,
     'dry_run': False,
     'eval_backend': 'Native',
     'eval_type': 'checkpoint',
     'generation_config': {
         'do_sample': False,
         'max_new_tokens': 4096,
         'repetition_penalty': 1.0
         },
     'limit': 10,
     'mem_cache': False,
     'model': '/chubao/tj-train-ssd-21/liuchengwei/models/saves/sft-query_recom_v1_qwen2_7B_dashu-v5_test11_all_10281127/checkpoint-1180',
     'model_args': {
         'device_map': 'auto',
         'precision': torch.float16,
         'revision': None
         },
     'outputs': '/chubao/tj-data-ssd-03/liuchengwei/workspaces/evals_farm/evalscope/outputs/query_recom',
     'stage': 'all',
     'template_type': 'qwen',
     'use_cache': True}    
    # ---------------------------------------------------
    {'dataset_args': {},
     'dataset_hub': 'ModelScope',
     'datasets': ['cmmlu'],
     'debug': False,
     'dry_run': False,
     'eval_backend': 'Native',
     'eval_type': 'checkpoint',
     'generation_config': {
         'do_sample': False,
         'max_new_tokens': 4096,
         'repetition_penalty': 1.0
         },
     'limit': 10,
     'mem_cache': False,
     'model': '/chubao/tj-train-ssd-21/liuchengwei/models/saves/sft-query_recom_v1_qwen2_7B_dashu-v5_test11_all_10281127/checkpoint-1180',
     'model_args': {'device_map': 'auto',
                    'precision': torch.float16,
                    'revision': None},
     'outputs': '/chubao/tj-data-ssd-03/liuchengwei/workspaces/evals_farm/evalscope/outputs/cmmlu',
     'stage': 'all',
     'template_type': 'qwen',
     'use_cache': True}
    """