from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate 
from langchain_community.document_loaders import PyPDFLoader


load_dotenv()

data = PyPDFLoader(r"D:\gen-ai-with-muhammad-javed\RAG\Document Loader\GRU.pdf")
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