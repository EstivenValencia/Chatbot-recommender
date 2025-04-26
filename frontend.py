import streamlit as st
import requests
import uuid

# ----------------- Config -----------------
DEFAULT_API_URL = "http://localhost:5000"
API_URL = st.sidebar.text_input("URL del servicio LagunAI:", DEFAULT_API_URL)

# ----------------- Session State -----------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "history" not in st.session_state:
    # history: list of dicts {role: "user"|"assistant", message:str, events:list}
    st.session_state.history = []

# ----------------- Header -----------------
st.title("LagunAI Chatbot")
st.caption("Tu guía de ocio en Bilbao ✨")

# ----------------- Display Past Messages -----------------
for turn in st.session_state.history:
    role = turn["role"]
    with st.chat_message(role):
        st.markdown(turn["message"])
        if role == "assistant" and turn.get("events"):
            with st.expander("Ver eventos sugeridos"):
                for e in turn["events"]:
                    st.write(f"- **{e['titulo']}** ({e['categoria']}) el {e['fecha']} en {e['ubicacion']}")

# ----------------- Chat Input -----------------
user_query = st.chat_input("Send a message…")

if user_query:
    # Show user bubble immediately
    with st.chat_message("user"):
        st.markdown(user_query)

    # Append user message to history
    st.session_state.history.append({"role": "user", "message": user_query})

    # Call backend
    payload = {"session_id": st.session_state.session_id, "query": user_query}
    try:
        res = requests.post(f"{API_URL}/query_friendly", json=payload, timeout=30)
        if res.status_code == 200:
            data = res.json()
            assistant_msg = data.get("response", "(sin respuesta)")
            events = data.get("events", [])
        else:
            assistant_msg = f"(Error {res.status_code}: {res.text})"
            events = []
    except requests.exceptions.RequestException as e:
        assistant_msg = f"(No se pudo conectar al servicio: {e})"
        events = []

    # Show assistant bubble
    with st.chat_message("assistant"):
        st.markdown(assistant_msg)
        if events:
            with st.expander("Ver eventos sugeridos"):
                for e in events:
                    st.write(f"- **{e['titulo']}** ({e['categoria']}) el {e['fecha']} en {e['ubicacion']}")

    # Append assistant message to history
    st.session_state.history.append({"role": "assistant", "message": assistant_msg, "events": events})

# ----------------- Footer -----------------
st.markdown("---")
st.write("LagunAI • © 2025 • Eventos en Bilbao")