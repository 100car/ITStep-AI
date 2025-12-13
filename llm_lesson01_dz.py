# LLM
# Large Language Model
# велика мовна модель

# завантеження api key як змінну середовища

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
import os

# завантаження даних з файлу .env
load_dotenv()

# сам api key
api_key = os.getenv('GEMINI_API_KEY')

# сама модель LMM
import langchain
from langchain_google_genai import GoogleGenerativeAI

# створення моделі, параметри креативності
llm = GoogleGenerativeAI(
    model='gemini-2.5-flash-lite',   # назва моделі
    api_key=api_key,
    top_k=10,   # вибрати випадково наступне слово з 10 з найбільшою ймовірністю
    top_p=0.8,  # залишити ті слова, сума ймовірностей яких не менше 80%, та вибирати серед них
    temperature=0  # вища температура -- відсотки стають більш однаковими
)

# temperature
# 0 - 0.3    -- низька креативність(відповіді як по методичці)
# 0.7 - 1.2  -- середня креативність(відповідає як людина)
# 1.5-1.7    -- висока креативність(вигадає щось цікаве або збреше)
# >2         -- випадкові слова

# запуск моделі
with open("data/lesson9/return_policy.txt", 'r', encoding='utf-8') as file:
    return_policy = file.read()

chat_history = f"Instruction: Відповідай на питання користувачів щодо повернення товару, враховуючи правила:\n{return_policy}\n"

print("Чат-бот готовий! Пишіть питання (порожній рядок для виходу).")

while True:
    user_input = input("Ви: ").strip()
    if user_input == "":
        print("Чат завершено.")
        break

    # додаємо повідомлення користувача до історії
    chat_history += f"Human: {user_input}\nAI: "

    try:
        # отримуємо відповідь від моделі
        response = llm.invoke(chat_history)

        # друкуємо відповідь
        print("Бот:", response.strip())

        # додаємо відповідь бота до історії
        chat_history += response.strip() + "\n"

    except Exception as e:
        print("Помилка при запиті до моделі:", e)
        break

