# streamlit
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.utilities import GoogleSerperAPIWrapper
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    trim_messages
)

import json
import dotenv
import os
import streamlit as st

# отримати сам ключ
api_key = st.secrets.get('GEMINI_API_KEY')

st.title("English Language Teacher Chat Bot")

# Ініціалізація великої мовної моделі (LLM)
llm = ChatGoogleGenerativeAI(
    model='gemini-3-flash-preview',
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# Ініціалізація історії повідомлень
if 'history' not in st.session_state:
    st.session_state['history'] = [
        SystemMessage(
            """
            Ти -- ввічливий бот-помічник для вивчення англійської мови.
            Інструкції:
            1. Якщо користувач надсилає слово або фразу: надай переклад українською та приклад використання у реченні англійською.
            2. Якщо користувач надсилає речення: надай переклад та поясни граматику (часи, структури на кшталт there is/are, пасивний стан тощо).
            """
        )
    ]

for message in st.session_state['history']:
    if isinstance(message, SystemMessage):
        continue

    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

if user_query := st.chat_input("Ваше повідомлення"):
    with st.chat_message("user"):
        st.markdown(user_query)

    st.session_state['history'].append(HumanMessage(user_query))

    with st.chat_message("assistant"):
        response = llm.invoke(st.session_state['history'])
        st.markdown(response.content)

    st.session_state['history'].append(response)