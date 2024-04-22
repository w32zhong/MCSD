import time
import torch
import sys
sys.path.insert(0, './model/llama_tree_attn')
from modeling_llama import LlamaForCausalLM
from tokenization_llama import LlamaTokenizer
#from model.llama_tree_attn import LlamaForCausalLM, LlamaTokenizer
sys.path.insert(0, './inference')
#from inference.generate import SpeculativeGenerator
from generate import SpeculativeGenerator

drafter_path = "JackFram/llama-68m"
#drafter_path = "JackFram/llama-160m"
target_path = "NousResearch/Llama-2-7b-chat-hf"

draft_model = LlamaForCausalLM.from_pretrained(
    drafter_path,
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_8bit=True
)
target_model = LlamaForCausalLM.from_pretrained(
    target_path,
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_8bit=True
)
tokenizer = LlamaTokenizer.from_pretrained(target_path)

generator = SpeculativeGenerator(
    draft_model,
    target_model,
    eos_token_id=tokenizer.eos_token_id,
    k_config=(4, 2, 2),
    max_length=1900,
    draft_model_temp=1,
    target_model_temp=1,
    replacement=False,
    speculative_sampling=True,
    tree_attn=True,
    tokenizer=tokenizer
)
print('models loaded')

prompt_text = "[INST] tell me something interesting about the solar eclipse in April 2024. [/INST]"
inputs = tokenizer(prompt_text, return_tensors="pt") #.to("cuda")
input_ids = inputs.input_ids

start_time = time.time()
with torch.no_grad():
    output = generator.generate(input_ids)
time_delta = time.time() - start_time

cnt_tokens = output.sequences.shape[-1] - output.init_input_len
print('e2e speed:', time_delta, cnt_tokens, cnt_tokens / time_delta)

#output_text = tokenizer.batch_decode(
#    output.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False
#)[0]
#print("Output:\n{}".format(output_text))
