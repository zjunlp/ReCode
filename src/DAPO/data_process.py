import argparse
import os
import json
from datasets import Dataset

from verl.utils.hdfs_io import copy, makedirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="src/DAPO/verl/data")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_dict = {}
    with open("data/api_update.jsonl", "rb") as f:
        for line in f:
            line_dict = json.loads(line.decode("utf-8").strip())
            for k, v in line_dict.items():
                if k not in data_dict:
                    data_dict[k] = [v]
                else:
                    data_dict[k].append(v)

    train_dataset = Dataset.from_dict(data_dict)
    
    def make_map_fn(split):
        def process_fn(example, idx):
            data = {
                "data_source": "api_update",
                "prompt": [
                    {
                        "role": "system",
                        "content": "You are a helpful coding assistant. Your task is to transform the old version of the code into the new version specified, based on the update information. You first thinks about the reasoning process in the mind and then provides the solution.",
                    },
                    {
                        "role": "user",
                        "content": f"""
{example["dependency"]} performed an API update in version {example["new_version"][2:]}, and the update content includes:
<doc>
{example["update_info"]}
</doc>
The old version of the code is:
```python
{example["old_code"]}
```

Show your work in <think> </think> tags. And return the final code in <answer> </answer>, the code within <answer></answer> should be enclosed in ```python ``` tags.
"""
                    },
                    {
                        "role": "assistant",
                        "content": "Let me solve this step by step.\n<think>"
                    }
                ],
                "ability": "api_update",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": example["new_code"],
                },
                "extra_info": {
                    "index": idx,
                }
            }
            return data

        return process_fn
    
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)