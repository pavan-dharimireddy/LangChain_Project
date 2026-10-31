# Project: Restaurant Idea Generator (LangChain + Mistral + Streamlit)

A learning project that builds an LLM app with LangChain 1.x, using Mistral instead of OpenAI.

## Files
- `langchain_helper.py`: LLM logic. `generate_restaurant_name_and_items(cuisine)` builds two LCEL chains
  (`prompt | llm | StrOutputParser()`), joins them with `RunnablePassthrough.assign(...)`, and `.pick()`s
  `restaurant_name` and `menu_items`. The big commented blocks are learning notes (old vs new syntax). Keep them.
- `main.py`: Streamlit UI. Sidebar picks a cuisine, shows the name as a header, splits `menu_items` on commas.
- `agent.py`: standalone agent demo with `create_agent` + Wikipedia tool.
- `README.md`: full concept notes and old-vs-modern syntax table.

## Conventions
- Use modern LangChain 1.x syntax (LCEL, `.invoke()`, `create_agent`). Old OpenAI-era code stays as comments marked `OLD:`.
- Model: `ChatMistralAI(model="mistral-small-latest", temperature=0.6)`; key `MISTRAL_API_KEY` comes from `.env`. Never print or commit the key.
- Explain changes in simple English; this is a learning project.

## Run
- `pip install -r requirements.txt`
- `streamlit run main.py`
- `python agent.py`
