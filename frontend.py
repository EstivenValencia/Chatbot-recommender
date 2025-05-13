import streamlit as st
import requests
import uuid

# ================= Streamlit App =================
# ----------------- Config -----------------
DEFAULT_API_URL = "http://localhost:5000"
API_URL = st.sidebar.text_input("URL del servicio LagunAI:", DEFAULT_API_URL)

# ----------------- Session State -----------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "history" not in st.session_state:
    st.session_state.history = []

# ----------------- Initial Name Prompt -----------------
if "user_name" not in st.session_state:
    # Ask for name without calling API
    prompt = "¡Hola soy guru!. Me puede indicar tu nombre?"
    st.session_state.history.append({"role": "assistant", "message": prompt, "events": []})

# ----------------- Header -----------------
st.title("LagunAI Chatbot")
st.caption("Tu guía de ocio en Bilbao ✨")

# ----------------- Show History -----------------
for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["message"])
        if turn.get("events"):
            with st.expander("Ver eventos sugeridos"):
                for e in turn["events"]:
                    st.write(f"- **{e['titulo']}** ({e['categoria']}) el {e['fecha']} en {e['ubicacion']}")

# ----------------- Chat Input -----------------
user_input = st.chat_input("Escribe tu mensaje...")
if user_input:
    # Display user message
    st.session_state.history.append({"role": "user", "message": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # If name not yet set, treat this input as name and immediately call API
    if "user_name" not in st.session_state:
        name = user_input.strip()
        st.session_state.user_name = name
        # Immediately call API now that name is known
        payload = {
            "session_id": st.session_state.session_id,
            "query": name,
            "user_name": name
        }
        try:
            res = requests.post(f"{API_URL}/query_friendly", json=payload, timeout=30)
            res.raise_for_status()
            data = res.json()
        except Exception as e:
            st.error(f"Error al conectar al servicio: {e}")
            # Fallback greeting
            data = {"response": f"Encantado de conocerte, {name}!", "events": [], "set_user_name": name}

        assistant_msg = data.get("response", f"Encantado de conocerte, {name}!")
        events = data.get("events", [])
        with st.chat_message("assistant"):
            st.markdown(assistant_msg)
            if events:
                with st.expander("Ver eventos sugeridos"):
                    for e in events:
                        st.write(f"- **{e['titulo']}** ({e['categoria']}) el {e['fecha']} en {e['ubicacion']}")
        st.session_state.history.append({"role": "assistant", "message": assistant_msg, "events": events})

    else:
        # Now user_name is set; always include it in payload
        payload = {
            "session_id": st.session_state.session_id,
            "query": user_input,
            "user_name": st.session_state.user_name
        }
        try:
            res = requests.post(f"{API_URL}/query_friendly", json=payload, timeout=30)
            res.raise_for_status()
            data = res.json()
        except Exception as e:
            st.error(f"Error al conectar al servicio: {e}")
            data = {"response": "(sin respuesta)", "events": []}

        assistant_msg = data.get("response", "(sin respuesta)")
        events = data.get("events", [])

        with st.chat_message("assistant"):
            st.markdown(assistant_msg)
            if events:
                with st.expander("Ver eventos sugeridos"):
                    for e in events:
                        st.write(f"- **{e['titulo']}** ({e['categoria']}) el {e['fecha']} en {e['ubicacion']}")
        st.session_state.history.append({"role": "assistant", "message": assistant_msg, "events": events})

# ----------------- Footer -----------------
st.markdown("---")
st.write("LagunAI • © 2025 • Eventos en Bilbao")