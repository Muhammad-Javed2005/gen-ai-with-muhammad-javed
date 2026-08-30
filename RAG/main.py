from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate 
from langchain_community.document_loaders import TextLoader


load_dotenv()

data = TextLoader(r"D:\gen-ai-with-muhammad-javed\RAG\Document Loader\notes.txt", encoding="utf-8")
docs = data.load()


prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are an AI that summarizes the text."),
    ("human", "{data}")
])

# Model Initialize
model = ChatMistralAI(
    model="mistral-small-latest"
)

chain = prompt_template | model
response = chain.invoke({"data": docs[0].page_content})

print(response.content)