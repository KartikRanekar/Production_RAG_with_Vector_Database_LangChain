
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

load_dotenv(Path(__file__).resolve().parent / ".env")

def main():

    # Test Grok
    if os.getenv("GROQ_API_KEY"):
        try:
            llm = ChatGroq(
                model = "openai/gpt-oss-20b",
                temperature = 0
            )
            response = llm.invoke("Say setup complete in one word")
            print("Grok response:", response.content)
        except Exception as e:
            print("Grok error:", repr(e))
    else:
        print("Grok API key missing!")

    # Test Gemini
    if os.getenv("GOOGLE_API_KEY"):
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-4-hydrogen",
                temperature=0
            )
            response = llm.invoke("Say setup complete in one word")
            print("Gemini response:", response)
        except Exception as e:
            print("Gemini error:", e)
    else:
        print("Gemini API key missing!")


if __name__ == "__main__":
    main()