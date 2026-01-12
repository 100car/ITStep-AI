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

# основний заголовок
st.title("IT-Step chat-bot")

# простий текст
st.markdown("Простий текст. Можливо опис вашого застосунку")


# отримати сам ключ
api_key = st.secrets.get('GEMINI_API_KEY')

# створення чат моделі
# Велика мовна модель(llm)

llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',  # назва моделі
    google_api_key=api_key,    # ваша API
)
print(api_key)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",  # назва моделі
    google_api_key=api_key
)

# чат бот

# історія повідомлень
# session_state -- сховище для зберігання даних
# session_state -- словник

# якщо це перший запуск і історії повідомленнь ще немає
if 'messages' not in st.session_state:
    # добавляємо system message
    st.session_state['messages'] = [
        SystemMessage('Ти ввічливий чат бот')
    ]

# запит від користувача
user_text = st.chat_input("Ваше повідомлення")

# usert_text може бути None якщо користувач нічого не ввів

# якщо користувач щось написав
if user_text:
    human_massege = HumanMessage(user_text)

    st.session_state['messages'].append(human_massege)

    response = llm.invoke(st.session_state['messages'])

    st.session_state['messages'].append(response)


# вивід історії спідкування
for message in st.session_state['messages']:
    text = message.content

    # визначити чиє повідомлення
    if isinstance(message, HumanMessage):
        role = 'human'
    elif isinstance(message, AIMessage):
        role = 'ai'
    else:  # SystemMessage пропускаємо
        continue

    # вивід повідомлення з міткою
    with st.chat_message(role):
        st.markdown(text)
