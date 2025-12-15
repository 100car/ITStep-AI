import os
import dotenv

from typing import List
from pydantic import BaseModel, Field
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser

# завантаження апі ключа
dotenv.load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# створити llm
llm = GoogleGenerativeAI(
    model='gemini-2.5-flash-lite',
    api_key=api_key,
)


# Завдання 1
# Напишіть модель для генерації персонального плану тренувань:
# Перший ланцюг отримує мету тренування(схуднення, набір м’язів, тощо) та повертає список вправ

# структура відповіді
class ExerciseList(BaseModel):
    exercise_goal: str = Field(description="Мета тренувань")
    exercise_list: List[str] = Field(description="Список вправ")


# створення парсера
parser = PydanticOutputParser(pydantic_object=ExerciseList)

# інструкція для llm як має виглядати відповідь
instructions = parser.get_format_instructions()

prompt = PromptTemplate.from_template(
    """
    Ти - фітнес-тренер. Твоя задача згенерувати список вправ для досягнення заданої мети тренувань.

    ### МЕТА
    {goal}

    ### ФОРМАТ ВІДПОВІДІ
    {instructions}
    """,
    partial_variables={"instructions": instructions}  # передаємо інструкції
)

chain = prompt | llm | parser

goal = input("Введіть мету тренувань: ")

response = chain.invoke({
    "goal": goal,
})

print(f"Мета тренувань: {response.exercise_goal}")
print("Список вправ:")
for exercise in response.exercise_list:
    print(f"- {exercise}")

# Другий ланцюг отримує список вправ, рівень підготовки користувача(низький, середній,
# професіонал) та кількість часу на тиждень (в годинах) і повертає план тренувань

class TrainingPlan(BaseModel):
    plan: List[str] = Field(description="План тренувань на тиждень")
    fitness_level: str = Field(description="Рівень підготовки користувача")
    hours_per_week: int = Field(description="Кількість годин на тиждень (ЦІЛЕ число)")

parser2 = PydanticOutputParser(pydantic_object=TrainingPlan)
instructions = parser2.get_format_instructions()

prompt2 = PromptTemplate.from_template(
    """
    Ти — професійний фітнес-тренер.
    
    Твоя задача — скласти ПЛАН ТРЕНУВАНЬ НА ТИЖДЕНЬ
    СТРОГО на основі переданих даних.
    
    ОБОВʼЯЗКОВІ ПРАВИЛА:
    - НЕ вигадуй нові значення
    - НЕ змінюй fitness_level
    - hours_per_week ПОВИННО бути ЦІЛИМ числом (int)
    - hours_per_week = значенню, яке передав користувач
    - НЕ обчислюй години самостійно
    - Відповідь ТІЛЬКИ у вказаному JSON-форматі
    
    ### ВХІДНІ ДАНІ
    Вправи:
    {exercises}
    
    Рівень підготовки:
    {fitness_level}
    
    Годин на тиждень:
    {hours_per_week}
    
    ### ФОРМАТ ВІДПОВІДІ
    {instructions}
    """,
    partial_variables={"instructions": instructions}
)


chain2 = prompt2 | llm | parser2

fitness_level = input("Введіть рівень підготовки користувача (низький, середній, професіонал): ")
hours_per_week = int(input("Введіть кількість годин на тиждень: "))

response2 = chain2.invoke({
    "exercises": ", ".join(response.exercise_list),
    "fitness_level": fitness_level,
    "hours_per_week": hours_per_week
})


plan = response2.plan

print("План тренувань:")
for day in response2.plan:
    print("-", day)

print("Рівень:", response2.fitness_level)
print("Годин на тиждень:", response2.hours_per_week)


