# A chatbot that runs a local LLM through Ollama and remembers the conversation.
# Before running:  ollama pull llama3.2:1b
# Run with:        streamlit run chatbot.py

import ollama
import streamlit as st

MODEL = "llama3.2:1b"

st.title("My Local Chatbot")
st.caption(f"Model: {MODEL} (running on this computer with Ollama)")

# st.session_state keeps data between reruns, so we store the chat history there.
# Each message is a dict like {"role": "user", "content": "hi"}.
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a friendly helper. Keep answers short and simple."}
    ]

# Show all old messages (but not the hidden system message)
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

question = st.chat_input("Ask me anything")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # Send the WHOLE history, so the model remembers what was said before
    response = ollama.chat(model=MODEL, messages=st.session_state.messages)
    answer = response["message"]["content"]

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)

if st.button("Clear chat"):
    st.session_state.messages = st.session_state.messages[:1]  # keep only the system message
    st.rerun()
