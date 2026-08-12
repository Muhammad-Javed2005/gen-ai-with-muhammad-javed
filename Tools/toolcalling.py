from dotenv import load_dotenv
load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain.tools import tool 
from langchain_core.messages import HumanMessage
from rich import print 

@tool
def get_text_length(text: str) -> int:
    """Returns the number of characters in a given text"""
    return len(text)

tools = {
    "get_text_length": get_text_length
}

llm = ChatMistralAI(model="mistral-small-2506")

llm_with_tool = llm.bind_tools([get_text_length])

messages = []
prompt = input("You: ")
query = HumanMessage(prompt)
messages.append(query)

result = llm_with_tool.invoke(messages)
messages.append(result)
if result.tool_call:
    tool_call = result.tool_calls[0]
    tool_call = tool_call["name"]


    final_result = llm_with_tool.invoke(messages)
    print(final_result.content)





# FIX: Sirf tabhi dobara invoke karein AGAR LLM ne Tool call kiya ho
if result.tool_calls:
    # 1. Tool execution
    tool_call = result.tool_calls[0]
    tool_name = tool_call["name"]
    tool_message = tools[tool_name].invoke(tool_call)
    
    # 2. Tool ka result messages list me add karein
    messages.append(tool_message)
    
    # 3. Phir se LLM ko dein taake wo final answer bana sake
    # 3. Phir se LLM ko dein taake wo final aswer bana saka 
    final_result = llm_with_tool.invoke(messages)
    print(final_result.content)
else:
    # Agar koi tool call nahi hua (e.g., "Hi"), toh direct print karein
    print(result.content)
    print(result.content)
    # Agar kio tool call nahi hua (e.g., "Hi") , toh direct print karein
    # Agar kio tool call nahi hua (e.g., "Hi") , toh direct print karein 

