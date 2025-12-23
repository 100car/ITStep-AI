# Завдання 1
# Напишіть чат бота, з інструментом по рекомендації ресторанів.
# Для цього скористайтесь GoogleSerperAPIWrapper(type="places")
# Інструмент повинен отримувати запит для пошуку та повертати таку інформацію про ресторани:
#  назва
#  посилання на сайт(якщо є)
#  рейтинг


# створення агентів
# агент -- чат-бот(llm) + інструменти

import os
import dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import GoogleSerperAPIWrapper
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    trim_messages, BaseMessage
)

# завантаження апі ключа
dotenv.load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")

# створити llm
llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    api_key=gemini_api_key,
)

# інструмент -- функція
# обов'язкова документація

def recommended_restaurants(restaurant_criteria: str) -> dict:
    """
    Рекомендує ресторани, що відповідають заданим вимогам (критеріям).

    Аргумент: вимога (критерій) ресторану

    Результат: повертає список ресторанів, відсортованих по рейтингу, у яких є:
    - рейтинг
    - посилання на сайт
    """

    # перевірка на порожній запит
    if not isinstance(restaurant_criteria, str) or not restaurant_criteria.strip():
        return {
            "query": restaurant_criteria,
            "restaurants": [],
            "error": "Порожній запит"
        }

    restaurant_info = {
        "query": restaurant_criteria,
        "restaurants": []
    }

    # пошук через Serper Places
    places_searcher = GoogleSerperAPIWrapper(
        serper_api_key=serper_api_key,
        type="places"
    )

    results = places_searcher.results(restaurant_criteria)
    places = results.get("places", []) if isinstance(results, dict) else []

    for place in places:
        title = place.get("title") or place.get("name")
        rating = place.get("rating")
        website = place.get("website")

        # ❗ пропускаємо ресторани без сайту або без рейтингу
        if not title or rating is None or not website:
            continue

        restaurant_info["restaurants"].append({
            "name": title,
            "website": website,
            "rating": rating
        })

    # сортування за рейтингом (спадання)
    restaurant_info["restaurants"].sort(
        key=lambda x: float(x["rating"]),
        reverse=True
    )

    return restaurant_info


# створення агента
agent = create_react_agent(
    model=llm,  # мовна модель
    tools=[recommended_restaurants]
)

# історія повідомлень + інструкції

messages = [
    SystemMessage(
        """
        Ти ввічлий чат-бот. Твоя задача давати інформативні та чіткі відповіді
        на запити користувача.

        У тебе є доступ до таких інструментів:
        * recommended_restaurants — рекомендації ресторанів (назва, сайт, рейтинг)
        """
    )
]

while True:
    user_query = input("Ви: ")

    if user_query == '':
        break

    # переводимо str рядок у  HumanMessage
    human_message = HumanMessage(user_query)

    # добавляємо повідослення користувача до історії
    messages.append(human_message)

    # застосування агента
    # треба передавати словник
    input_data = {
        "messages": messages
    }

    response = agent.invoke(input_data)
    # response -- словник з усією історією + відповідь моделі

    # отримання всіє історії повідомлень
    messages = response['messages']

    # отримати фінальну відповідь моделі
    answear = messages[-1]
    print(answear.content)

    # виведемння всієї історії
    print()
    print("Історія")

    for message in messages:
        print(repr(message))