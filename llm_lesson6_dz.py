"""
LangChain. Частина 6
Завдання 1 — додавання великого файлу в існуючу векторну базу
1) Читаємо файл data/lesson_rag/huge_file.txt
2) Розбиваємо його на блоки (між блоками 2 порожніх рядки)
3) Перший рядок кожного блоку — назва блоку
4) Створює Document для кожного блоку з метаданими:
   - filename       — назва файлу
   - block_title    — назва блоку
   - block_index    — номер блоку
   - source_path    — шлях до файлу
5) Генерує UUID для кожного блоку
6) Додає документи в існуючу Pinecone-базу "tokardz" з класної роботи
7) Дописує ID у ids.json (без дублікатів)
8) Створює Agent + Tool("search"):
   - Tool робить similarity_search у Pinecone і повертає знайдені шматки
   - Agent сам вирішує, коли викликати tool, і формує фінальну відповідь
"""

import json
import os
import re
from uuid import uuid4

import dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)

from langchain.agents import Tool, initialize_agent, AgentType


# -----------------------------
# НАЛАШТУВАННЯ
# -----------------------------
INDEX_NAME = "tokardz"
HUGE_FILE_PATH = "data/lesson_rag/huge_file.txt"
IDS_JSON_PATH = "ids.json"

# Скільки документів повертати при пошуку
TOP_K = 4


# -----------------------------
# ДОПОМІЖНІ ФУНКЦІЇ
# -----------------------------
def split_into_blocks(text: str) -> list[str]:
    """
    Розбиває текст на блоки.
    Розділювач — ДВА порожніх рядки.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    blocks = re.split(r"\n\s*\n\s*\n+", text)
    return [b.strip() for b in blocks if b.strip()]


def get_block_title(block: str) -> str:
    """
    Повертає перший непорожній рядок блоку — назву блоку.
    """
    for line in block.splitlines():
        line = line.strip()
        if line:
            return line
    return "Без назви"


def load_ids(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_ids(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def format_search_results(docs: list[Document]) -> str:
    """
    Форматуємо результати пошуку так, щоб агенту було зручно їх читати.
    """
    if not docs:
        return "Нічого не знайдено."

    parts = []
    for i, d in enumerate(docs, start=1):
        meta = d.metadata or {}
        title = meta.get("block_title", "—")
        idx = meta.get("block_index", "—")
        filename = meta.get("filename", "—")
        text = d.page_content.strip().replace("\n", " ")

        # щоб відповідь не була гігантською — беремо короткий фрагмент
        snippet = text[:700] + ("..." if len(text) > 700 else "")

        parts.append(
            f"[{i}] файл: {filename} | блок: {idx} | назва: {title}\n"
            f"    фрагмент: {snippet}"
        )
    return "\n\n".join(parts)


# -----------------------------
# ОСНОВНА ЛОГІКА
# -----------------------------
def main():
    dotenv.load_dotenv()

    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("Не знайдено GEMINI_API_KEY у .env")
    if not os.getenv("PINECONE_API_KEY"):
        raise RuntimeError("Не знайдено PINECONE_API_KEY у .env")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    if not pc.has_index(INDEX_NAME):
        raise RuntimeError(f'Індекс "{INDEX_NAME}" не існує. Його треба було в лабораторній роботі створити.')

    index = pc.Index(INDEX_NAME)

    # Векторне сховище LangChain
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)

    # -----------------------------
    # 1) Додаємо huge_file.txt блоками (без дублікатів)
    # -----------------------------
    if not os.path.exists(HUGE_FILE_PATH):
        raise FileNotFoundError(f"Файл не знайдено: {HUGE_FILE_PATH}")

    with open(HUGE_FILE_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    blocks = split_into_blocks(text)
    print(f"Знайдено блоків у huge_file: {len(blocks)}")

    filename_only = os.path.basename(HUGE_FILE_PATH)

    # Формуємо документи + айді
    documents: list[Document] = []
    ids: list[str] = []

    for i, block in enumerate(blocks, start=1):
        title = get_block_title(block)
        documents.append(
            Document(
                page_content=block,
                metadata={
                    "filename": filename_only,
                    "block_title": title,
                    "block_index": i,
                    "source_path": HUGE_FILE_PATH,
                },
            )
        )
        ids.append(str(uuid4()))

    # Оновлюємо ids.json і залишаємо тільки нові блоки
    id_map = load_ids(IDS_JSON_PATH)

    new_docs: list[Document] = []
    new_ids: list[str] = []

    for doc, doc_id in zip(documents, ids):
        key = f"{HUGE_FILE_PATH}::block_{doc.metadata['block_index']:03d}::{doc.metadata['block_title']}"
        if key in id_map:
            continue
        id_map[key] = doc_id
        new_docs.append(doc)
        new_ids.append(doc_id)

    if new_docs:
        vector_store.add_documents(new_docs, ids=new_ids)
        save_ids(IDS_JSON_PATH, id_map)
        print(f"✅ Додано нових блоків у Pinecone: {len(new_docs)}")
        print(f"✅ Оновлено {IDS_JSON_PATH}: +{len(new_ids)} ID")
    else:
        print("ℹ️ Нових блоків немає — huge_file вже був доданий раніше.")

    # -----------------------------
    # 2) Tool(search) для агента
    # -----------------------------
    def search_tool(query: str) -> str:
        """
        Інструмент для агента: шукає у векторній базі та повертає релевантні фрагменти.
        """
        docs = vector_store.similarity_search(query, k=TOP_K)
        return format_search_results(docs)

    tools = [
        Tool(
            name="search",
            func=search_tool,
            description=(
                "Використовуй цей інструмент, щоб знайти інформацію у векторній базі "
                "(документи з файлів та huge_file). "
                "Вхід: рядок-пошуковий запит. Вихід: релевантні фрагменти з метаданими."
            ),
        )
    ]

    # -----------------------------
    # 3) Агент initialize_agent + Tool
    # -----------------------------
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,  # щоб було видно, коли агент викликає Tool
        handle_parsing_errors=True,
    )

    # -----------------------------
    # 4) Діалог з агентом
    # -----------------------------
    print("\n" + "=" * 80)
    print("🤖 Агент готовий. Пиши питання (або 'exit' для виходу).")
    print("=" * 80)

    while True:
        user_q = input("\nТи: ").strip()
        if user_q.lower() in {"exit", "quit", "q"}:
            print("Бувай 🙂")
            break

        # Підказка агенту: завжди користуйся tool для фактів з бази
        prompt = (
            "Ти відповідаєш тільки на основі інформації, яку знайдеш через інструмент search.\n"
            f"Питання: {user_q}"
        )

        answer = agent.run(prompt)
        print("\nАгент:", answer)


if __name__ == "__main__":
    main()
