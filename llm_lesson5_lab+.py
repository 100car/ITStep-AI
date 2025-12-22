# Завдання 1
# Напишіть функцію яка перевіряє складність паролю:
#  кількість символів(>8)
#  наявність хоча б однієї літери\цифри\спеціального символу
#  наявність літер в різних регістрах
# Функція повертає тест з описом паролю(що добре, а що погано)
# На основі цієї функції створіть агента.
from sympy.solvers.diophantine.diophantine import length


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

def product(a: int, b: int) -> int:
    """
    Множить 2 цілих числа то повертає їхній добуток

    :param a: перше число
    :param b: друге число
    :return: добуток чисел
    """
    print("hello from product")
    return a * b


def get_weather(city: str, time: str) -> str:
    """
    Повертає інформацію про погоду у місті в певний час доби

    :param city: назва міста
    :param time: час доби(наприклад ранок, вечір, 10:30, 4 години дня)
    :return: інформація про погоду
    """
    print("hello from get_weather")
    return f"У {city} о {time} буде сонячно"


def check_password(parol: str) -> dict:
    """
    Перевірка паролю на складність.

    Аргумент: parol - пароль користувача

    Функція повертає словник, яка вказує складність паролю:

    """

    password_info = {}
    ###################### кількість символів(>8) ####################
    if len(parol) <= 8:
        password_info["Довжина паролю"] = f"Погано: довжина пароля менша або дорівнює 8 ({len(parol)})"
    else:
        password_info["Довжина паролю"] = f"Добре: довжина пароля більше 8 ({len(parol)})"

    ###################### наявність хоча б однієї літери\цифри\спеціального символу ####################
    flag_alpha = False
    flag_number = False
    flag_special = False

    for alpha in parol:
        if alpha.isalpha():
            flag_alpha = True
        elif alpha.isdigit():
            flag_number = True
        else:
            flag_special = True

    password_info["Чи є буква?"] = flag_alpha
    password_info["Чи є цифра?"] = flag_number
    password_info["Чи є спец. символ?"] = flag_special

    return password_info


# інструмент для пошуку в інтернеті
searcher = GoogleSerperAPIWrapper(serper_api_key=serper_api_key)


def search(query: str) -> str:
    """
    Шукає інформацію в інтернеті за запитом користувача

    :param query: запит користувача
    :return: результати пошуку
    """

    results = searcher.results(query)
    print(results)  # результати пошуку

    return results


# створення агента
agent = create_react_agent(
    model=llm,  # мовна модель
    tools=[check_password]
)

# історія повідомлень + інструкції

messages = [
    SystemMessage(
        """
        Ти ввічлий чат-бот. Твоя задача давати інформативні та чіткі відповіді
        на запити користувача.

        У тебе є доступ до таких інструментів:
        * product
        * get_weather
        * search -- завжди давай посилання на новини
        * check_password - перевірка паролю
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



