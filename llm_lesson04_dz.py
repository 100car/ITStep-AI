import os
import dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

dotenv.load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not found")

chat_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    api_key=api_key,
)

summary_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    api_key=api_key,
)

base_system = SystemMessage(
    "Ти — ввічливий чат-бот. Відповідай коротко та по суті."
)

messages = [base_system]


def summarize_history(msgs):
    dialog = [m for m in msgs if isinstance(m, (HumanMessage, AIMessage))]
    prompt = [
        SystemMessage(
            "Підсумуй ВСЮ попередню розмову в декількох реченнях. "
            "Збережи якомога більше деталей."
        )
    ] + dialog

    summary = summary_llm.invoke(prompt)

    return SystemMessage(
        "Короткий підсумок попередньої розмови (використовуй як памʼять):\n"
        + summary.content
    )


while True:
    user_input = input("Ви: ").strip()
    if not user_input:
        break

    messages.append(HumanMessage(user_input))

    dialog_msgs = [m for m in messages if isinstance(m, (HumanMessage, AIMessage))]

    if len(dialog_msgs) > 4:
        summary_msg = summarize_history(messages)
        print(f"\n--- ПІДСУМОК ПОПЕРЕДНЬОЇ РОЗМОВИ ---\n{summary_msg.content}\n")
        last_human = messages[-1]
        messages = [base_system, summary_msg, last_human]

    response = chat_llm.invoke(messages)
    messages.append(response)

    print(f"AI: {response.content}")

# AI: 36 дюймів.
# Ви: Володимир - син Івана.
# AI: Зрозумів.
# Ви: Олександр - брат Володимира. Як звати тата Олександра?
#
# --- ПІДСУМОК ПОПЕРЕДНЬОЇ РОЗМОВИ ---
# Короткий підсумок попередньої розмови (використовуй як памʼять):
# Іван.
#
# AI: Тата Олександра звати Володимир.
# Ви: Чому так?
# AI: Володимир є братом Олександра, отже, вони мають спільного тата.
# Ви: Але Володимир - син Івана.
#
# --- ПІДСУМОК ПОПЕРЕДНЬОЇ РОЗМОВИ ---
# Короткий підсумок попередньої розмови (використовуй як памʼять):
# Вибачте, я помилився. Якщо Володимир є братом Олександра, то вони мають спільного батька. Але ви також зазначили, що Володимир є сином Івана. Це означає, що **Іван є татом Володимира**.
#
# Оскільки Володимир є братом Олександра, то **Іван також є татом Олександра**.
#
# AI: Так, це вже було зазначено.
