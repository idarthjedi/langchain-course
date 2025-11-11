from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

load_dotenv()

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a professor for doctoral students, reviewing a paragraph provided by the user. Generate a critique and recommendations for the user's input."
            "Always provide detailed recommendations, including effectiveness, efficiency, APA 7th style recommendations, critical thought, clarity, etc.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a doctoral candidate tasked with writing quality and concise paragraphs. "
            "Generate the best paragraph possible for the user's request"
            "If the user provides a critique, respond with a revised version of your previous attempts.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

llm = ChatOllama(model="gpt-oss:120b")
# llm = ChatOpenAI(model="gpt-5")

generate_chain = generation_prompt | llm
reflect_chain = reflection_prompt | llm
