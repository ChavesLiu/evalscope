#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File    :   transform_data_qa.py
# Time    :   2024/11/06 20:37:57
# Author  :   liuchengwei/lingcheng 
# Email   :   liuchengwei007@ke.com
# Desc    :   None

import os
import json
import shutil


def json_loads(json_str):
    return json.loads(json_str)

def mkdirs(path, clear=False):
    if not path:
        return

    if not os.path.exists(path):
        os.makedirs(path)
    elif clear:
        shutil.rmtree(path)
        os.makedirs(path)

def jsonl_load(file_path):
    def has_intent():
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                json_loads(f.readline())
                return False
            except Exception:
                return True

    if not has_intent():
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                yield json_loads(line)
    else:
        lines = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip()
                lines.append(line)
                if line in ['}', ']']:
                    yield json_loads('\n'.join(lines))
                    lines = []
        if lines:
            yield json_loads('\n'.join(lines))


def json_dumps(data, indent=None):
    return json.dumps(data, ensure_ascii=False, indent=indent)

def append_to_file(data_string, file_path):
    mkdirs(os.path.dirname(file_path))
    with open(file_path, "a", encoding="utf8") as f:
        f.write(str(data_string) + "\n")

def append_to_jsonl(data, file_path, indent=None):
    json_string = json_dumps(data, indent)
    append_to_file(json_string, file_path)


def trans_qa(datapath, outpath):
    for line in jsonl_load(datapath):
        system = line["system"]
        conversations = line["conversations"]
        query = conversations[0]["value"]
        response = conversations[1]["value"]
        prompt = system+query
        # prompt = f"'<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n'"
        if response == "":
            continue
        trans_qa = {
            "query": prompt,
            "response": response #+"<|im_end|>"
            }
        append_to_jsonl(trans_qa, outpath)

def del_response_isnull(datapath, outpath):
    for line in jsonl_load(datapath):
        if line["response"]:
            append_to_jsonl(line, outpath)


if __name__ == '__main__':
    # # query recom
    # datapath = "/chubao/tj-train-ssd-21/liuchengwei/datasets/query_recom_lcw/chatlog_query_recom_v9_10W_test.jsonl"
    # # 用户画像
    datapath_lst = [
        "# The directory `/chubao/tj-train-ssd-21/liuchengwei/datasets/user_portrait_new_lt/` seems to
        # be related to datasets for user portrait analysis. The Python script you provided is
        # processing data from this directory, specifically performing transformations on the data
        # related to user portraits. The script is reading JSON files from this directory, extracting
        # relevant information such as system, conversations, queries, and responses, and then
        # creating new JSON files with transformed question-answer pairs.
        /chubao/tj-train-ssd-21/liuchengwei/datasets/user_portrait_new_lt/l1_summary_to_l2_summary_sft_data_test.jsonl",
        "/chubao/tj-train-ssd-21/liuchengwei/datasets/user_portrait_new_lt/l2_summary_to_l3_tag_sft_data_test.jsonl",
        "/chubao/tj-train-ssd-21/liuchengwei/datasets/user_portrait_new_lt/user_data_to_l1_summary_sft_data_test.jsonl",
        "/chubao/tj-train-ssd-21/liuchengwei/datasets/user_portrait_new_lt/user_portrait_l1_summary_update_sft_data_test.jsonl",
    ]
    # datapath_lst=[]
    for datapath in datapath_lst:
        outpath = datapath.replace(".jsonl", "_transqa.jsonl")
        trans_qa(datapath, outpath)
    pass