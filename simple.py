#!/usr/bin/env python
# coding: utf-8

# In[1]:


from langchain_together import ChatTogether
import requests
import torch
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_together import ChatTogether
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage, SystemMessage


# In[2]:


TOGETHER_API_KEY = ""

chat = ChatTogether(
    together_api_key= TOGETHER_API_KEY,
    model="mistralai/Mistral-7B-Instruct-v0.1",
)
## BIGGER MODEL
# model="mistralai/Mixtral-8x7B-Instruct-v0.1"


# ## Getting the model to do a web search

# In[3]:


# Determine if a web search is neded 

def needs_web_search(query, db_results):
    """Return True if database results are empty or clearly insufficient."""
    if len(db_results )== 0:
        return True

    decision_prompt = f"""The following user query was asked: '{query}'.
    The retrieved database results were:

    {db_results}

    Based on this, is the database context sufficient to fully answer the query?
    Respond with 'yes' or 'no'."""

    decision = chat.invoke(decision_prompt, max_tokens=10)
    return "yes" in decision.content.lower()


# Do the web search 
def do_web_search(query):
    api_key = ""
    cse_id = ""
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={api_key}&cx={cse_id}"
    response = requests.get(url)
    results = response.json()
    snippets = " ".join(item.get("snippet", "") for item in results.get("items", []))
    return snippets


# In case there's the need for a web search, the search is done and the result is added to the prompt
def run_query(prompt):
    entries, new_query = get_dbandentries(prompt, threshold = 0.7)
    if needs_web_search(prompt, entries):
        context = do_web_search(prompt)
        prompt = "PROMPT: " + prompt + "\nCONTEXT: " + context
    return prompt


# ### Implement web search as a tool

# In[4]:


# 1. Define the Tool's Functionality

def do_web_search(query):
    print("\nWeb search is being used")
    api_key = ""
    cse_id = ""
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={api_key}&cx={cse_id}"
    response = requests.get(url)
    results = response.json()
    snippets = " ".join(item["snippet"] for item in results.get("items", []))
    return snippets

# 2. Wrap the Function as a LangChain Tool

web_tool = Tool(
    name="do_web_search",
    func=run_query,
    description="Use this tool to search the web for information ONLY when the query requires current, real-time information, or knowledge beyond your internal capabilities (e.g., recent events, specific URLs, current status of something). Do not use it for general knowledge questions.",
)


# ### Implement database search as a tool

# In[5]:


# 1. Define the Tool's Functionality

# Use the mps on a mac
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Import the data
nutrition = pd.read_csv("epi_r.csv").head(1000)[['title','calories','protein','fat','sodium']].dropna()
recipes = pd.read_csv("dataset.csv").head(1000)[['title','ingredients','directions']].dropna()

# Combine the relevant data into a single column so it can be processed by the sentence transformer
recipes["combined"] = "TITLE: " + recipes["title"] + " INGREDIENTS: " + recipes["ingredients"] + " DIRECTIONS: " + recipes["directions"]
nutrition["combined"] = "TITLE: " + nutrition["title"]	+ " CALORIES: " + nutrition["calories"].astype(str) + " PROTEIN: " + nutrition["protein"].astype(str) + " FAT: " + 	nutrition["fat"].astype(str) + " SODIUM: " + nutrition["sodium"].astype(str)

# Use the sentence transformer model to get the embeddings

embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device) # Use the all-MiniLM-L6-v2 sentence transformer model
embeddings_recipes = embedder.encode(recipes["combined"].tolist(), convert_to_numpy=True)
embeddings_nutrition = embedder.encode(nutrition["combined"].tolist(), convert_to_numpy=True)

# Create the faiss index

index_recipes = faiss.IndexFlatL2(embeddings_recipes.shape[1])
index_recipes.add(embeddings_recipes)
index_nutrition = faiss.IndexFlatL2(embeddings_nutrition.shape[1])
index_nutrition.add(embeddings_nutrition)


# In[6]:


# Create a dictionary contatining the databases

databases = {
    "recipes": {
        "df": recipes,
        "text": recipes["combined"],
        "index": index_recipes,
        "embeddings": embeddings_recipes,
    },
    "nutrition": {
        "df": nutrition,
        "text": nutrition["combined"],
        "index": index_nutrition,
        "embeddings": embeddings_nutrition,
    }
}

# Implement a function that finds the relevant databases

def find_relevant_databases(query, threshold = 0.7):
    query_emb = embedder.encode([query], convert_to_numpy=True)
    relevant = []
    for name, db in databases.items():
        db_embeddings = db["embeddings"]
        similarity_scores = cosine_similarity(query_emb, db_embeddings)
        max_similarity_score = np.max(similarity_scores)
        #print("NAME AND MAX SIMILARITY: ",name, max_similarity_score)
        if max_similarity_score > threshold:
            relevant.append(name)
    return relevant, query_emb

# Implement a function that takes the name of the databse, the embedded query and gets the top 3 most relecant results

def get_entries(relevant, query_emb):
    entries = []
    context = "CONTEXT: "
    for dbname in relevant:
        if dbname == "recipes":
            D, I = index_recipes.search(query_emb, k=3)
            for idx in I[0]:
                entry = recipes.iloc[idx]
                context += recipes.iloc[idx]["combined"]
                entries.append((idx, recipes.iloc[idx] ))
        if dbname == "nutrition":
            D, I = index_nutrition.search(query_emb, k=3)
            for idx in I[0]:
                entry = nutrition.iloc[idx]
                entries.append((idx, nutrition.iloc[idx]))
                context += nutrition.iloc[idx]["combined"]
    if entries:
        df_entries = pd.DataFrame([entry[1] for entry in entries])
        print("\nRelevant database Entries:")
        print(df_entries[["title", "ingredients", "directions"]].reset_index(drop=True))
    else:
        print("\nNo relevant database entries found.")

    return entries, context

# Implement the function that contains the tool's functionality

def get_dbandentries(query, threshold = 0.7):
    print("\nDatabase search is being used")
    relevant, query_emb = find_relevant_databases(query, threshold)
    entries, context = get_entries(relevant, query_emb)
    new_query = f"{query}\n\nContext:\n{context}"
    return entries, new_query


# 2. Wrap the Function as a LangChain Tool

db_tool = Tool(
    name="do_db_search",
    func=get_dbandentries,
    description="Use this tool to search the database for information. This tool should be used specifically when the user asks about macronutrient content of foods or requests recipes. Do not use it for other types of questions.",
)

tools = [web_tool, db_tool]


# ## Experimental components

# ### Advanced prompting techniques

# #### Meta prompting

# In[9]:


prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are an expert cooking assistant. Your job is to answer user questions with accurate and complete information related to cooking and nutrition.

    Follow these steps carefully:

    1. If the question is unrelated to cooking or nutrition, politely say that you can't answer it.

    2. If the question is about a cooking recipe or nutrition:
       a. First, use the database search tool to retrieve relevant recipes or nutritional entries.
       b. If the database contains relevant information, use that CONTEXT to generate a full, natural-sounding answer.
          - For recipe questions, include a complete list of ingredients and step-by-step directions.
          - For nutrition questions, explain clearly and helpfully based on the data.

    3. If the database does not contain a relevant answer:
       a. Use the web search tool to gather helpful information.
       b. Write a complete, well-structured answer based on the web CONTEXT.
          - For recipes, write a full recipe (ingredients and instructions) in conversational tone.
          - Do not include search result links — only the helpful information.
          - Rephrase, rewrite, and summarize where necessary to sound natural.

    4. If no useful information is found even after searching the web, politely explain that you cannot answer the question.

    Always respond clearly, naturally, and helpfully.
    """),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"), # Important for the agent's thinking process
])

# 5. Create the Agent
agent = create_tool_calling_agent(chat, tools, prompt)

# 6. Create the Agent Executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False) # verbose=True shows the agent's steps


# In[11]:


# Function to check if the query is cooking or nutrition related using the model
def is_cooking_related(query):
    system_msg = SystemMessage(content="You are a helpful assistant. Determine if the following question is related to cooking or nutrition. Reply only 'yes' if the question is related or 'no' if the question is not related.")
    user_msg = HumanMessage(content=query)

    classification = chat.invoke([system_msg, user_msg])
    #print(classification.content.strip())
    return "yes" in classification.content.strip().lower()

# --- Test with Cooking Relevance Filtering ---
def run_query_with_filter(query):
    print(is_cooking_related(query))
    if is_cooking_related(query):
        print(f"\n--- Running query: {query} ---")
        result = agent_executor.invoke({"input": query})
        print("\n--- Result ---")
        print(result['output'])
    else:
        print(f"\n--- Query blocked ---\nSorry, I can only answer cooking or nutrition-related questions.")

# Example queries
run_query_with_filter("What is the population of Paris?")
run_query_with_filter("Tell me a chocolate cake recipe.")
run_query_with_filter("How to make Frikadeller?")


# In[12]:


query = "Give calories, protein and fat for Boudin Blanc Terrine with Red Onion Confit?"
find_relevant_databases(query, threshold = 0.7)


# In[ ]:


nutrition.head()

