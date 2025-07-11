# app.py
import streamlit as st
import io
import sys
from contextlib import redirect_stdout
from langchain_core.messages import HumanMessage, SystemMessage
from simple import chat, agent_executor  # replace with your actual script filename

def is_cooking_related(query):
    system_msg = SystemMessage(content="You are a helpful assistant. Determine if the following question is related to cooking or nutrition. Reply only 'yes' if the question is related or 'no' if the question is not related.")
    user_msg = HumanMessage(content=query)
    classification = chat.invoke([system_msg, user_msg])
    return "yes" in classification.content.strip().lower()

def run_query_with_filter_streamlit(query):
    f = io.StringIO()
    with redirect_stdout(f):  # Capture all prints
        if is_cooking_related(query):
            print(f"\n--- Running query: {query} ---")
            result = agent_executor.invoke({"input": query})
            print("\n--- Result ---")
            print(result['output'])
        else:
            print("\n--- Query blocked ---")
            print("⚠️ Sorry, I can only answer cooking or nutrition-related questions.")
    return f.getvalue()

# Streamlit UI
st.set_page_config(page_title="Cooking & Nutrition Assistant", layout="centered")
st.title("🍲 Cooking & Nutrition Assistant")
st.write("Ask a question related to recipes or nutrition.")

query = st.text_input("Your question:", placeholder="e.g., Suggest a high protein breakfast")

if st.button("Submit"):
    with st.spinner("Processing..."):
        try:
            output = run_query_with_filter_streamlit(query)
            st.text_area("Output Log", value=output, height=400)
        except Exception as e:
            st.error(f"Error: {e}")
