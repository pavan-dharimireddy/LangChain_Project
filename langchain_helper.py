import os

from dotenv import load_dotenv
# pyrefly: ignore[missing-import]
# from langchain_mistralai import ChatMistralAI   # Mistral (kept for reference, now using Groq)
from langchain_groq import ChatGroq
# OLD: from langchain.llms import OpenAI
# OLD: from langchain.prompts import PromptTemplate   (removed in LangChain 1.x)
# pyrefly: ignore[missing-import]
from langchain_core.prompts import PromptTemplate
# OLD: from langchain.chains import LLMChain, SimpleSequentialChain   (removed in LangChain 1.x)

# pyrefly: ignore[missing-import]
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough



load_dotenv()  # reads GROQ_API_KEY (and MISTRAL_API_KEY, if you switch back) from .env




# OLD (OpenAI):
# llm = OpenAI(temperature=0.6)
# name = llm("I want to open a restaurant for Indian food. Suggest a fancy name for this.")
# print(name)
# Mistral version (kept for reference):
# llm = ChatMistralAI(
#     model="mistral-small-latest",
#     temperature=0.6,
#     api_key=os.getenv("MISTRAL_API_KEY"),
# )

# Groq version (active):
llm = ChatGroq(
    model="openai/gpt-oss-120b",  # llama-3.3-70b-versatile is not available on this key
    temperature=0.6,
    api_key=os.getenv("GROQ_API_KEY"),
)

'''

# Direct call (chat models use .invoke() and return a message; text is in .content)
# response = llm.invoke("I want to open a restaurant for Indian food. Suggest a fancy name for this.")
# print(response.content)

prompt = PromptTemplate(
    input_variables=["cuisine"],
    template="I want to open a restaurant for {cuisine} food. Suggest only one fancy name for this.",
)

# Prompt template + direct call
# response = llm.invoke(prompt.format(cuisine="Italian"))
# print(response.content)

# OLD (LLMChain):
# chain = LLMChain(llm=llm, prompt=prompt)
# print(chain.run("Mexican"))
# NEW: pipe the prompt into the model (LCEL)
name_chain = prompt | llm
# print(chain.invoke({"cuisine": "Mexican"}).content)


prompt_items = PromptTemplate(
    input_variables = ['restaurant_name'],
    template = "Suggest some menu items for {restaurant_name}. Return them as a comma separated string"
)

# chain_items = LLMChain(llm=llm, prompt=prompt_items)
menu_chain = prompt_items | llm


print(menu_chain.invoke({"restaurant_name": "Paneer palace"}))

=====================================   this is completed response from LLM ==============================
content='Here are some suggested menu items for Paneer Palace:\n\nPaneer Tikka, Paneer Butter Masala, Shahi Paneer,
Paneer Do Pyaza, Kadhai Paneer, Paneer Pasanda, Paneer Bhurji, Malai Kofta, Paneer Pakora, Paneer Tikka Masala,
 Paneer Angara, Paneer Jalfrezi, Paneer Makhani, Paneer Handi, Paneer Chilli, Paneer 65,
 Paneer Biryani, Paneer Fried Rice, Paneer Naan, Paneer Kulcha'
 additional_kwargs={} response_metadata={'token_usage': {'prompt_tokens': 33, 'total_tokens': 174, 'completion_tokens': 141,
 'prompt_tokens_details': {'cached_tokens': 0}, 'completion_tokens_details': {'reasoning_tokens': 0}, 'service_tier': 'standard'},
  'model_name': 'mistral-small-latest', 'model': 'mistral-small-latest', 'finish_reason': 'stop', 'model_provider': 'mistralai'}
   id='lc_run--01a182c5-c4d3-744b-8050-1430b35eef7f-0' tool_calls=[] invalid_tool_calls=[] usage_metadata={'input_tokens': 33, 'output_tokens': 141, 'total_tokens': 174}


# OLD (SimpleSequentialChain, removed in LangChain 1.x):
# chain = SimpleSequentialChain(chains=[name_chain, menu_chain], verbose=True)
# print(chain.run("Indian"))
# NEW: chain the steps with | (LCEL). The first step's text output is
# mapped into the {restaurant_name} variable the second prompt expects.
# Why the lambda? It connects the two steps:
#   - StrOutputParser() gives a plain string, e.g. "The Royal Tandoor"
#   - menu_chain's prompt has {restaurant_name}, so it needs a dict:
#     {"restaurant_name": "The Royal Tandoor"}
#   - The lambda wraps the string into that dict.
# SimpleSequentialChain did this wrapping automatically; with | we say
# which variable receives the value.
chain = (
    name_chain
    | StrOutputParser()
    | (lambda name: {"restaurant_name": name})
    | menu_chain
)
print(chain.invoke({"cuisine": "Indian"}).content)
'''



# ============================ Sequential Chain =========================================
''' -- old format
prompt = PromptTemplate(
    input_variables=["cuisine"],
    template="I want to open a restaurant for {cuisine} food. Suggest only one fancy name for this.",
)

name_chain = LLMChain(llm=llm, prompt=prompt,output_key="restaurant_name")

prompt_items = PromptTemplate(
    input_variables = ['restaurant_name'],
    template = "Suggest some menu items for {restaurant_name}. Return them as a comma separated string"
)

menu_chain = LLMChain(llm=llm, prompt=prompt_items,output_key="menu_items")

chain = SequentialChain(
    chains = [name_chain, menu_chain],
    input_variables = ["cuisine"],
    output_variables = ["restaurant_name","menu_items"]
    )

print(chain({"cuisine": "Italian"}))
'''

# NEW: modern LCEL version of SequentialChain

'''
prompt = PromptTemplate(
    input_variables=["cuisine"],
    template="I want to open a restaurant for {cuisine} food. Suggest only one fancy name for this.",
)
prompt_items = PromptTemplate(
    input_variables=["restaurant_name"],
    template="Suggest some menu items for {restaurant_name}. Return them as a comma separated string",
)

name_chain = prompt | llm | StrOutputParser()
menu_chain = prompt_items | llm | StrOutputParser()
'''

'''
# What is RunnablePassthrough?
# A LangChain building block that passes its input through unchanged.
#   RunnablePassthrough().invoke({"cuisine": "Italian"})
#   -> {"cuisine": "Italian"}   (same thing back)
# On its own it does nothing; it is useful for its .assign() method:
#   .assign(key=some_runnable) takes the incoming dict, runs some_runnable
#   on it, and returns the same dict with one new key added:
#   RunnablePassthrough.assign(restaurant_name=name_chain)
#   input:  {"cuisine": "Italian"}
#   output: {"cuisine": "Italian", "restaurant_name": "Alba d'Oro"}
# The original keys stay, so each step reads from a growing dict and
# adds its own output (the same job output_key + SequentialChain did).
#
# RunnablePassthrough.assign keeps the existing keys and adds a new one.
# Each .assign() is the replacement for one output_key in the old chains:
#   output_key="restaurant_name"  ->  .assign(restaurant_name=name_chain)
#   output_key="menu_items"       ->  .assign(menu_items=menu_chain)
# .pick() plays the role of output_variables (keeps only those keys).
'''
'''
chain = (
    RunnablePassthrough.assign(restaurant_name=name_chain)
    | RunnablePassthrough.assign(menu_items=menu_chain)
).pick(["restaurant_name", "menu_items"])

print(chain.invoke({"cuisine": "Italian"}))
'''



def generate_restaurant_name_and_items(cuisine):
    prompt = PromptTemplate(
        input_variables=["cuisine"],
        template="I want to open a restaurant for {cuisine} food. Suggest only one fancy name for this.",
    )
    prompt_items = PromptTemplate(
        input_variables=["restaurant_name"],
        template="Suggest some menu items for {restaurant_name}. Return them as a comma separated string",
    )

    name_chain = prompt | llm | StrOutputParser()
    menu_chain = prompt_items | llm | StrOutputParser()

    chain = (
        RunnablePassthrough.assign(restaurant_name=name_chain)
        | RunnablePassthrough.assign(menu_items=menu_chain)
    ).pick(["restaurant_name", "menu_items"])

    response = chain.invoke({"cuisine": cuisine})
    return response

if __name__ == "__main__":
    print(generate_restaurant_name_and_items("Italian"))

