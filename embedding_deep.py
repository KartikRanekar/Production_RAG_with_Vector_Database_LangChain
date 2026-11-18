from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import numpy as np

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview"
    )

def basic_embeddings():
    try:
        single_embedding = embeddings.embed_query(text="What is Machine Learning?")
        print(f"The Length of embedding is: {len(single_embedding)}")
        print("Vector Dimensions:\n", len(single_embedding))
        print(f"First 5 values {single_embedding[:5]}")
        print(f"Vector norm: {np.linalg.norm(single_embedding):.4f}")

    except Exception as e:
        print("Error:", e)
    
def batch_embeddings():
    text = [        
        "What is Machine Learning ?",
        "Explain the concept of overfitting in ML.",
        "How does a neural network work?",
    ]

    batch_embedding = embeddings.embed_documents(text)
    for i ,emb in enumerate(batch_embedding):
       print(f"Text {i+1} - Vector dimensions: {len(emb)}")    
       print(f"Text {i+1} - First 5 values: {emb[:5]}")
       print(f"Text {i+1} - Vector norm: {np.linalg.norm(emb):.4f}")

def similiarity_search():
    docs = [
        "Python is a Programming Language.",
        "JavaScript is used for web development",
        "Machine learning is a subset of AI",
        "Deep Learning uses Neural Networks",
        "Cats are popular pets"       
    ]

    query  = "What is programming language ?"

    doc_vector = embeddings.embed_documents(docs)
    query_vector = embeddings.embed_query(query)

    def cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    similiarities = [cosine_similarity(query_vector, doc_vector[i]) for i in range(len(docs))] 
    ranked_docs = sorted(zip(docs, similiarities), key=lambda x: x[1], reverse=True)

    for doc, score in ranked_docs:
        print(f"Similiarity score of {doc}: {score:.4f}")
        print()
if __name__ == "__main__":
    #basic_embeddings()
    #batch_embeddings()
    similiarity_search()



