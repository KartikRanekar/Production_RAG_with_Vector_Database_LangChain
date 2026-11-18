import os
from dotenv import load_dotenv
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

client = Groq(api_key = os.environ.get("GROQ_API_KEY"))
'''
conversation = client.chat.completions.create(
    model = "openai/gpt-oss-20b",
    messages = [
        {"role" : "user", "content" : "You are a helpful assisstant."},
        {"role": "user", "content" : "What is the capital of France ?"},
    ],
)

print(conversation.choices[0].message.content)
'''
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector = embeddings.embed_query("Your text string goes here")

print(vector)
print("Dimensions:", len(vector))

