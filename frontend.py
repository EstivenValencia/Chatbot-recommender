import uuid

import requests
import streamlit as st

# ================= Streamlit App =================
# ------------- Configuración -----------------
DEFAULT_API_URL = "http://localhost:5000"

# Sidebar: URL del servicio LagunAI
API_URL = st.sidebar.text_input(
    "URL del servicio LagunAI:", DEFAULT_API_URL
)

# ------------- Estado de Sesión ---------------
if "session_id" not in st.session_state:
    # Identificador único para la sesión actual
    st.session_state.session_id = str(uuid.uuid4())

if "history" not in st.session_state:
    # Historial de mensajes (role: assistant/user, message, events)
    st.session_state.history = []

# ------------- Solicitud Inicial de Nombre ---
if "user_name" not in st.session_state:
    # Pregunta inicial sin llamar a la API
    greeting = "¡Hola! Soy Guru. ¿Cómo te llamas?"
    st.session_state.history.append({
        "role": "assistant",
        "message": greeting,
        "events": []
    })

# ================ Cabecera ====================
st.title("LagunAI Chatbot")
st.caption("Tu guía de ocio en Bilbao ✨")

# ================ Mostrar Historial ===========
for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["message"])
        if turn.get("events"):
            # Desplegable con lista de eventos sugeridos
            with st.expander("Ver eventos sugeridos"):
                for event in turn["events"]:
                    st.write(
                        f"- **{event['titulo']}** ({event['categoria']}) "
                        f"el {event['fecha']} en {event['ubicacion']}"
                    )

# ================ Entrada de Chat ============
user_input = st.chat_input("Escribe tu mensaje...")

if user_input:
    # Añadir mensaje del usuario al historial
    st.session_state.history.append({
        "role": "user",
        "message": user_input
    })
    with st.chat_message("user"):
        st.markdown(user_input)

    # Construir payload para API
    payload = {
        "session_id": st.session_state.session_id,
        "query": user_input,
        "user_name": st.session_state.get("user_name")
    }

    # Si aún no tenemos nombre, usamos este input como nombre
    if "user_name" not in st.session_state:
        st.session_state.user_name = user_input.strip()
        payload.update({"user_name": st.session_state.user_name})

    # Llamada al servicio LagunAI
    try:
        response = requests.post(
            f"{API_URL}/query_friendly", json=payload, timeout=30
        )
        response.raise_for_status()
        data = response.json()
    except Exception as err:
        st.error(f"Error al conectar al servicio: {err}")
        data = {"response": "(sin respuesta)", "events": []}

    # Mostrar respuesta del asistente
    assistant_msg = data.get("response", "(sin respuesta)")
    events = data.get("events", [])

    with st.chat_message("assistant"):
        st.markdown(assistant_msg)
        if events:
            with st.expander("Ver eventos sugeridos"):
                for event in events:
                    st.write(
                        f"- **{event['titulo']}** ({event['categoria']}) "
                        f"el {event['fecha']} en {event['ubicacion']}"
                    )

    # Guardar turno en historial
    st.session_state.history.append({
        "role": "assistant",
        "message": assistant_msg,
        "events": events
    })

# ================= Pie de Página ===============
st.markdown("---")
st.write("LagunAI • © 2025 • Eventos en Bilbao")