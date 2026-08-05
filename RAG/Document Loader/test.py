from langchain_community.document_loaders import TextLoader
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate



data = TextLoader(r"D:\gen-ai-with-muhammad-javed\RAG\Document Loader\notes.txt",  encoding="utf-8")

docs = data.load()

print(docs[0])


