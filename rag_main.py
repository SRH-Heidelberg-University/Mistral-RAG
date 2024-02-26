import torch
import transformers
import re
from torch import cuda, bfloat16
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,pipeline
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA
from prompt_template import chain_type_kwargs
from vectordb import vectorstore
from tokens_and_model import hf_token,model_id


#download the model with below configurations
device = "cuda"

bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)
model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    use_auth_token=hf_token
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True, #bolean flag saying we are downloading from HF
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto',  #will map model to device, i.e. gpu or cpu
    use_auth_token=hf_token,
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2"
)

tokenizer =AutoTokenizer.from_pretrained(
    model_id,
    token=hf_token
)


#text generation pipeline
pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        use_cache=True,
        device_map="auto",
        max_length=500,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
)


llm = HuggingFacePipeline(pipeline=pipeline,model_kwargs={'temperature':0.1})

#retrieving top 2 relevant information with sources
rag_pipeline = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type='stuff',
    retriever=vectorstore.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs
)
  

def ask_mistral(question):
  x= rag_pipeline(question)

  x['result'] = re.sub(r"\n|#\*\`", "",x['result'])

  return x


# output= ask_mistral('Explain decision tree and random forest')
# print(output)
