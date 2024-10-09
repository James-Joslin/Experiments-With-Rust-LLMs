import os
os.environ['HF_HOME'] = './network/'
from transformers.models.phi.modeling_phi import PhiForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
import torch

base_model_id = "Qwen/Qwen2-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(base_model_id, trust_remote_code=True, torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
model.eval()

base_path = './network/hub/models--Qwen--Qwen2-0.5B-Instruct/snapshots/c540970f9e29518b1d8f06ab8b24cba66ad77b6d'
onnx_path = os.path.join(base_path, "qwen2-0-5.onnx")

prompt = "Hello world\nAnswer:"
sample_inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
id_tensor : torch.Tensor = sample_inputs["input_ids"]

torch.onnx.export(
    model,                           # The model to be exported
    id_tensor, #(input_ids, ),     # Input tuple to the model
    onnx_path,                       # Path where the ONNX model will be saved
    export_params=True,              # Store the trained parameter weights inside the model file
    opset_version=14,                # The ONNX version to export the model to
    do_constant_folding=True,        # Whether to execute constant folding for optimization
    input_names=["input"], #, "attention_mask"],       # Input names
    output_names=["output"],         # Output name
    dynamic_axes={
        "input": {0: "batch_size", 1: "sequence_length"},   # Dynamic axes for input
        # "attention_mask": {0: "batch_size", 1: "sequence_length"},  # Dynamic axes for attention mask
        "output": {0: "batch_size", 1: "sequence_length"}       # Dynamic axes for output
    }
)
onnx_model = onnx.load(onnx_path)
onnx.save(onnx_model, "./qwen2-0-5.onnx", save_as_external_data=True, location="./qwen2-0-5.onnx.data")
onnx_model = onnx.load("./qwen2-0-5.onnx")

suffix = "\nAnswer:"
print("Enter prompt:")
_input = input()
prompt = _input + suffix

sample_inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
id_tensor : torch.Tensor = sample_inputs["input_ids"]
input_ids : list[np.ndarray] = id_tensor.numpy()

# Number of tokens to generate
num_tokens_to_generate = 256
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
ort_sess = ort.InferenceSession("./qwen2-0-5.onnx", sess_options=sess_options)
# past_three_tokens = []
# Loop for each token to generate
for _ in range(num_tokens_to_generate):
    # Run the model
    outputs : np.ndarray = ort_sess.run(None, {'input': input_ids})[0]
    
    # Get the logits for the last token
    last_token_logits = outputs[0, -1, :]
    
    # Get the token id of the most likely token
    most_likely_token_id = np.argmax(last_token_logits)
    
    # past_three_tokens.append(int(most_likely_token_id))
    # if len(past_three_tokens) > 3:
    #     past_three_tokens = past_three_tokens[-3:]
        
    # if past_three_tokens == [198, 198, 3109]:
    #     break
    
    if most_likely_token_id == 151643 or most_likely_token_id == 151645:
        break
    
    # Append the most likely token id to the input sequence
    input_ids = np.concatenate([input_ids, [[most_likely_token_id]]], axis=1)
    
    result = tokenizer.batch_decode(torch.tensor(most_likely_token_id).int().unsqueeze(0))[0]
    
    print(result, end="")