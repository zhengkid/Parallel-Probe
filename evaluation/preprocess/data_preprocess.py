import os
from huggingface_hub import hf_hub_download
from datasets import load_dataset
import json

filename = "Qwen3-1.7B__aime24__br128__bg5120k__fr500__0121_1455.jsonl"
local_file_path = hf_hub_download(
    repo_id="EfficientReasoning/Qwen3-1.7B-AIME24-br128-bg5120k-fr500",
    filename=filename,
    repo_type="dataset",
)

print(f"文件已下载到本地: {local_file_path}")

with open(local_file_path, 'r', encoding='utf-8') as f:
    datas=json.load(f)
filtered_datas = []
for data in datas:
    assert len(data['final_answers_trace']) == len(data['probe_matrix_mxn']) == len(data['branch_tokens'])
    print()
    # break
    filtered_datas.append({
        'question': data['question'],
        'final_answers_trace': data['final_answers_trace'],
        "each_branch": [(i, j, k) for i, j ,k in zip(data['probe_matrix_mxn'], data['branch_tokens'], data['final_answers_trace']) ],
        'gold_answer': data['gold_answer'],
        "probe_freq": data['probe_freq']
    })
json.dump(filtered_datas, open(f"{filename.replace('.jsonl', '')}_filtered.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)