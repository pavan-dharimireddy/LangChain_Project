# pip install langchain-community wikipedia
from langchain.agents import create_agent
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
# pyrefly: ignore[missing-import]
from langchain_mistralai import ChatMistralAI
import os
from dotenv import load_dotenv
load_dotenv()

llm = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0.6,
    api_key=os.getenv("MISTRAL_API_KEY"),
)

tools = [WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())]
agent = create_agent(model=llm, tools=tools)
result = agent.invoke({"messages": [{"role": "user", "content": "When was Elon Musk born? What is his age in 2023?"}]})
print(result["messages"][-1].content)
