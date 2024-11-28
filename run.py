import transformers

import QEfficient

from QEfficient import QEFFAutoModelForCausalLM as AutoModelForCausalLM

model_name = "meta-llama/Llama-3.2-3B-Instruct"

# Init tokenizer, can change cache_dir according to the real path to model tokenizer.

tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, cache_dir="cache_dir")


# Init Transformers model optimized for AI100, can change cache_dir according to the real path to Transformers model.

qeff_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name, cache_dir="cache_dir/hub")


# Export to ONNX

qeff_model.export()

# Convert ONNX model to AI100 format

qeff_model.compile(

    num_cores=8,

    mxfp6=True,

    mxint8=True,

    device_group=[0],

    prompt_len=512,

    ctx_len=1024,

    mos=1,

    aic_enable_depth_first=True,

    batch_size=1

)

 

# Testing Generation

 

# Path to compiled model for AI080

qpc_path = "qeff_cache/meta-llama/Llama-3.2-3B-Instruct/qpc_8cores_1bs_256pl_256cl_1mos_1devices_mxfp6/qpcs "


# Testing prompt

prompt = "Đến Việt Nam nên đi đâu chơi?"

 

# Adding to prompt template

messages = [

    {"role": "system", "content": "You're a good assistant"},

    {"role": "user", "content": prompt}

]

text = tokenizer.apply_chat_template(

    messages,

    tokenize=False,

    add_generation_prompt=True

)


# Start the inference and benchmark

execinfo = QEfficient.cloud_ai_100_exec_kv(tokenizer=tokenizer, qpc_path=qpc_path, prompt=text, device_id=[0], generation_len=1024, stream=False, automation=True)