import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline, ChatHuggingFace
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

embedding_model_name = "BAAI/bge-small-en-v1.5"
llm_model_name = "Qwen/Qwen2.5-0.5B-Instruct"

# 1. Initialize the BGE embedding model locally on MPS
embedding_model = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs={"device": "mps"},          
    encode_kwargs={"normalize_embeddings": True} 
)

tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
model = AutoModelForCausalLM.from_pretrained(
    llm_model_name,
    torch_dtype=torch.float16 if torch.backends.mps.is_available() else torch.float32
).to("mps")

# 3. Create the standard pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    do_sample=True
)

# 4. Wrap it as a LangChain base LLM, then turn it into a Chat Model
llm = HuggingFacePipeline(pipeline=pipe)
chat_model = ChatHuggingFace(llm=llm)

# 5. Use it like a true Chat Model with messages!
messages = [
    SystemMessage(content="You are a helpful and concise assistant."),
    HumanMessage(content="What are embeddings in simple terms?")
]

response = chat_model.invoke(messages)
print(response.content)