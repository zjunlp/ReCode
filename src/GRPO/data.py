import json
from tqdm import tqdm
from torch.utils.data import Dataset


class APIUpdateDataset(Dataset):
    def __init__(self, path):
        self.lines = []
        with open(path, "rb") as f:
            for line in f:
                self.lines.append(json.loads(line.decode("utf-8").strip()))
        self.tokenized = False
        
    def tokenize(self, tokenizer):
        self.tokenized_lines = []
        for i, line in tqdm(enumerate(self.lines)):
            line_dict = generate_prompt(line, tokenizer)
            line_dict["ids"] = i
            self.tokenized_lines.append(line_dict)
        
        self.tokenized = True
        
    def __getitem__(self, idx):
        assert self.tokenized is True, "Dataset must be tokenized before being used."
        return self.tokenized_lines[idx]
    
    def __len__(self):
        return len(self.lines)
        

def generate_prompt(data_line, tokenizer):
    rl_prefix =[{
        "role": "system",
        "content": "You are a helpful coding assistant. Your task is to transform the old version of the code into the new version specified, based on the update information. You first thinks about the reasoning process in the mind and then provides the solution."
    },
    {
        "role": "user",
        "content": f"""
{data_line["dependency"]} performed an API update in version {data_line["new_version"][2:]}, and the update content includes:
<doc>
{data_line["update_info"]}
</doc>
The old version of the code is:
```python
{data_line["old_code"]}
```

Show your work in <think> </think> tags. And return the final code in <answer> </answer>, the code within <answer></answer> should be enclosed in ```python ``` tags.
""",
    },
    {
        "role": "assistant",
        "content": "Let me solve this step by step.\n<think>"
    }
    ]
    
    return {"prompt": tokenizer.apply_chat_template(rl_prefix, tokenize=False, continue_final_message=True, chat_template="{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}{% endfor %}"), "target": data_line['new_code']}
    
        