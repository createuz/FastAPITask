from fastapi import FastAPI
from transformers import LlamaForCausalLM, AutoTokenizer
from pydantic import BaseModel
import torch

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
model = LlamaForCausalLM.from_pretrained(model_name, use_auth_token='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')
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token='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')

tokenizer.add_special_tokens({'pad_token': '[PAD]'})

device = torch.device("cpu")
model.to(device)
model.resize_token_embeddings(len(tokenizer))

app = FastAPI()


class Prompt(BaseModel):
    prompt: str


@app.get("/generate")
async def generate_get(prompt: str):
    inputs = tokenizer.encode_plus(prompt, return_tensors='pt', padding=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=100, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"response": response}

# @app.get("/generate")
# async def generate(prompt: str):
#     inputs = tokenizer.encode(prompt, return_tensors='pt').to(device)
#     outputs = model.generate(inputs, max_length=100, num_return_sequences=1)
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return {"response": response}
