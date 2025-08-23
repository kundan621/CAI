import pandas as pd
import torch
import math
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model

# Load dataset
file_path = 'makemytrip_qa_dataset_mini.json'
df = pd.read_json(file_path)

# Prepare training data
system_prompt = "You are a helpful assistant that provides financial data from MakeMyTrip reports."
training_data = [
    {"text": f"<|system|>\n{system_prompt}</s>\n<|user|>\n{row['question']}</s>\n<|assistant|>\n{row['answer']}</s>"}
    for _, row in df.iterrows()
]

# Tokenizer and preprocess
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)

def preprocess(example):
    tokens = tokenizer(example['text'], truncation=True, padding=False)
    text = example['text']
    assistant_index = text.find("<|assistant|>")
    prefix_ids = tokenizer(text[:assistant_index], add_special_tokens=False)['input_ids']
    prefix_len = len(prefix_ids)
    labels = tokens['input_ids'].copy()
    labels[:prefix_len] = [-100] * prefix_len
    tokens['labels'] = labels
    return tokens

from datasets import Dataset
dataset = Dataset.from_list(training_data)
tokenized_dataset = dataset.map(preprocess, remove_columns=["text"])

# LoRA and MoE classes
class LoraLinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, lora_alpha=16, lora_dropout=0.05, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.scaling = lora_alpha / r if r > 0 else 1.0
        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False) if bias else None
        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros((r, in_features)))
            self.lora_B = nn.Parameter(torch.zeros((out_features, r)))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_A, self.lora_B, self.lora_dropout = None, None, None
    def forward(self, x):
        result = nn.functional.linear(x, self.weight, self.bias)
        if self.r > 0 and self.lora_A is not None and self.lora_B is not None and self.lora_dropout is not None:
            lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
            result = result + self.scaling * lora_out
        return result

class MoELoRALinear(nn.Module):
    def __init__(self, base_linear, r, num_experts=2, k=1, lora_alpha=16, lora_dropout=0.05):
        super().__init__()
        self.base_linear = base_linear
        self.num_experts = num_experts
        self.k = k
        self.experts = nn.ModuleList([
            LoraLinear(
                in_features=base_linear.in_features,
                out_features=base_linear.out_features,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout
            ) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(base_linear.in_features, num_experts)
    def forward(self, x):
        base_out = self.base_linear(x)
        gate_scores = torch.softmax(self.gate(x), dim=-1)
        expert_out = 0
        for i, expert in enumerate(self.experts):
            expert_out += gate_scores[..., i:i+1] * expert(x)
        return base_out + expert_out

def replace_proj_with_moe_lora(model, r=8, num_experts=2, k=1, lora_alpha=16, lora_dropout=0.05):
    for layer in model.model.layers:
        for proj_name in ["up_proj", "down_proj"]:
            old = getattr(layer.mlp, proj_name)
            moe = MoELoRALinear(
                base_linear=old,
                r=r,
                num_experts=num_experts,
                k=k,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            ).to(next(old.parameters()).device)
            setattr(layer.mlp, proj_name, moe)
    return model

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    device_map="cpu",
    trust_remote_code=True,
)
base_model.config.use_cache = False
base_model.config.pretraining_tp = 1
model = replace_proj_with_moe_lora(
    base_model,
    r=8,
    num_experts=1,
    k=1,
    lora_alpha=16,
    lora_dropout=0.05
)
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)

def generate_finetuned_answer(query, model, tokenizer):
    system_prompt = "You are a helpful assistant that provides financial data from MakeMyTrip reports."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer_start_token = '<|assistant|>'
    answer_start_index = decoded_output.rfind(answer_start_token)
    if answer_start_index != -1:
        generated_answer = decoded_output[answer_start_index + len(answer_start_token):].strip()
        if generated_answer.endswith('</s>'):
            generated_answer = generated_answer[:-len('</s>')].strip()
    else:
        generated_answer = "Could not extract answer from model output."
    return generated_answer
