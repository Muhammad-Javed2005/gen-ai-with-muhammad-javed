import streamlit as st
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Load Environment Variables
load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="AI Mood Chatbot",
    page_icon="🤖",
    layout="centered"
)

# Title
st.title("🤖 AI Mood Chatbot")
st.caption("Chat with AI in different moods using LangChain + Mistral AI")

# Sidebar
st.sidebar.header("⚙️ Settings")

mood = st.sidebar.selectbox(
    "Choose AI Mood",
    ["😡 Angry", "😂 Funny", "😢 Sad"]
)

temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.9,
    step=0.1
)

# Mood Prompt
if mood == "😡 Angry":
    system_prompt = (
        "You are an angry AI Agent. "
        "You respond aggressively and with frustration."
    )

elif mood == "😂 Funny":
    system_prompt = (
        "You are a Funny AI Agent. "
        "You respond with humor and jokes."
    )

else:
    system_prompt = (
        "You are a Sad AI Agent. "
        "You respond with sadness and empathy."
    )

# Load Model
model = ChatMistralAI(
    model="mistral-small-2506",
    temperature=temperature
)

# Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content=system_prompt)
    ]

# Reset when mood changes
if st.session_state.messages[0].content != system_prompt:
    st.session_state.messages = [
        SystemMessage(content=system_prompt)
    ]

# Display Chat
for msg in st.session_state.messages[1:]:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)

    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)

# Chat Input
user_input = st.chat_input("Type your message...")

if user_input:

    st.session_state.messages.append(
        HumanMessage(content=user_input)
    )

    with st.chat_message("user"):
        st.write(user_input)

    response = model.invoke(st.session_state.messages)

    st.session_state.messages.append(
        AIMessage(content=response.content)
    )

    with st.chat_message("assistant"):
        st.write(response.content)

