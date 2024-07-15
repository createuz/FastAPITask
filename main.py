# from fastapi import FastAPI, Request
# from transformers import LlamaForCausalLM, AutoTokenizer
# import torch
#
# # Model and tokenizer loading
# model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
# model = LlamaForCausalLM.from_pretrained(model_name, token='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')
# tokenizer = AutoTokenizer.from_pretrained(model_name, token='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')
#
# # Use CPU instead of GPU
# device = torch.device("cpu")
# model.to(device)
#
# app = FastAPI()
#
#
# @app.get("/generate")
# async def generate(prompt: str):
#     inputs = tokenizer.encode(prompt, return_tensors='pt').to(device)
#     outputs = model.generate(inputs, max_length=100, num_return_sequences=1)
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return {"response": response}


from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}
