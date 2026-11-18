import tempfile
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

KNOWLEDGE_BASE = """
LangChain is a framework for developing applications powered by
large language models (LLMs). It was created by Harrison Chase
in October 2022.

LangChain provides components such as models, prompts, chains,
agents, and memory. It helps developers build applications that
combine language models with external data and tools.

LangGraph is a framework built for creating stateful, multi-actor
applications using language models. It focuses on state management,
cycles, loops, human-in-the-loop interactions, and persistence.

LangGraph is used to build complex AI agent workflows where multiple
agents can collaborate, make decisions, and maintain state throughout
the execution process.
"""

embeddings_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview"
    )

def create_kb():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500, 
        chunk_overlap = 50
        )
        
    doc = Document(
        page_content=KNOWLEDGE_BASE,
        metadata ={"source": "langchain_knowlegde_base.md"}
        )

    chunks = splitter.split_documents([doc])

    vector_stores = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_directory=tempfile.mkdtemp(),
    )
    print(len(chunks))
    return vector_stores

def basic_rag():
    vector_store = create_kb()
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":2})
    llm = ChatGroq(
        model = "openai/gpt-oss-20b",
        temperature=0.2,
    )

    #RAG prompt templates
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the based only on the context
        {context}
        
        Question: {question}
        
        Answer:
        
        Make sure to answer in concise manner and if you don't know the answer, just say I don't know.""" 
    )
    #Format retrieved document
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])
    #RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question" : RunnablePassthrough()}
        |prompt
        |llm
        |StrOutputParser()
    )

    #Test RAG chain
    #Test
    questions = [
        "What is LangChain ?",
        "Who created LangChain ?",
        "Who is LangGraph used for ?",
    ]

    print("Basic RAG Demo:\n")
    
    for q in questions:
        answer = rag_chain.invoke(q)
        print(f"Q: {q}")
        print(f"A: {answer}\n")

if __name__ == "__main__":
    basic_rag()

