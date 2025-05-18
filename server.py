import os
import re
from datetime import datetime, timedelta

import dateparser
import isodate
import pandas as pd
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from py_heideltime import heideltime
from google import genai
from config import TOKEN_GEMINI

from models import (
    connect_db,
    create_table,
    insert_data,
    query_data,
    get_embeddings,
    get_user_profile,
    insert_user_profile,
    load_model_missing_categories,
    load_model_normalization,
    normalize_category,
    ranking_missing_categories,
)

###########################################
#        CONFIGURACIÓN GLOBAL             #
###########################################

QUESTIONS_NUMBER = 3

# Definir y cargar API Key de Google
os.environ['GOOGLE_API_KEY'] = TOKEN_GEMINI
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    raise ValueError("No se encontró la variable de entorno 'GOOGLE_API_KEY'.")
client = genai.Client(api_key=api_key)

# Carga de modelos de normalización y categorización
(
    MODEL,
    EMBS_N,
    CANONICAL_LABELS_N,
    SUBCANONICAL_LABELS_N,
    RULE_MAP,
    THRESHOLD
) = load_model_normalization()

BM25, TFIDF, X_TFIDF, LABELS_M = load_model_missing_categories()

# Instrucciones del sistema para RAG y perfil
system_instruction = (
    """
    Eres 'guru', un asistente experto y amigable especializado en
    actividades de ocio y recreación en Bilbao y sus alrededores.
    Tu conocimiento se basa EXCLUSIVAMENTE en el contexto proporcionado.

    **Instrucción**:
    No calcules fechas dentro de la respuesta, haz la respuesta amigable,
    pero solo haz referencia a «Fecha de referencia» si no está vacía,
    **debes empezar tu respuesta** con un saludo que incluya exactamente
    esa fecha. Debes analizar las solicitudes del usuario y determinar si
    contienen una categoría especifica y recomendar el evento que mejor
    se adapte a esa categoría. En caso de que las solicitud sean ambiguas
    y no se especifique una categoría, la recomendación debe adaptarse al
    perfil del usuario.
    """
)

profile_instruction = f(
    """
    Eres 'guru', un asistente que construye un perfil del usuario preguntando
    sus gustos y preferencias de ocio.
    Para ello deberás preguntarle sobre cuales de las siguientes categorias
    son de su preferencia {CANONICAL_LABELS_N}. En caso de seleccionar la
    categoría 'Exposiciones & Ferias', utiliza estas subcategorias
    {SUBCANONICAL_LABELS_N}. Para cada categoría y subcategoría debes crear
    una breve descripción de máximo 10 palabras, luego haz preguntas abiertas
    sobre ubicación, temáticas específicas, artistas, géneros, etc.
    """
)

###########################################
#            FUNCIONES AUXILIARES         #
###########################################

def format_history(history):
    """
    Formatea el historial de conversación para inclusión en prompts.

    Args:
        history (list): Lista de turnos con 'query' y 'response'.

    Returns:
        str: Texto formateado con líneas de Usuario y Asistente.
    """
    return "\n".join(
        f"Usuario: {turn['query']}\nAsistente: {turn['response']}"
        for turn in history
    )


def parse_date_expression(text, reference_date=None):
    """
    Detecta expresiones de fecha en un texto usando HeidelTime y Dateparser.

    Args:
        text (str): Texto de entrada a analizar.
        reference_date (datetime, optional): Fecha de referencia para valores relativos.

    Returns:
        str|None: Fecha ISO (YYYY-MM-DD) detectada o None si no se encuentra.
    """
    if reference_date is None:
        reference_date = datetime.today()

    timexs = heideltime(text, language="Spanish", dct=reference_date.date().isoformat())
    if not timexs:
        return None

    timex = timexs[0]
    ttype, value, span = timex['type'], timex['value'], timex['text']

    try:
        # Duraciones
        if ttype == 'DURATION':
            delta = isodate.parse_duration(value)
            return (reference_date + delta).date().isoformat()

        # Fechas explícitas
        if ttype == 'DATE':
            if re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
                return value
            # Semanas ISO
            m = re.fullmatch(r'(\d{4})-W(\d{2})(?:-(\d))?', value)
            if m:
                year, week = int(m.group(1)), int(m.group(2))
                weekday = int(m.group(3)) if m.group(3) else 1
                dt = datetime.fromisocalendar(year, week, weekday)
                return dt.date().isoformat()
            # Otros alias
            dt = dateparser.parse(value, languages=['es'])
            return dt.date().isoformat() if dt else None

        # Horas y conjuntos
        if ttype in ('TIME', 'SET'):
            dt = dateparser.parse(
                span,
                settings={'RELATIVE_BASE': reference_date},
                languages=['es'],
            )
            return dt.date().isoformat() if dt else None

        # Fallback genérico
        dt = dateparser.parse(text, languages=['es'])
        return dt.date().isoformat() if dt else None

    except Exception:
        return None


def get_profile_response(query, history, prefs):
    """
    Genera la siguiente pregunta para construir el perfil del usuario.

    Args:
        query (str): Entrada actual del usuario.
        history (list): Historial de interacciones.
        prefs (dict): Preferencias y estado de perfil.

    Returns:
        str: Pregunta generada por el modelo.
    """
    prompt = (
        f"{profile_instruction}\n"
        f"Historial de perfil hasta ahora:\n{format_history(history)}\n"
        f"Respuesta del usuario o nombre:\n{query}\n"
        "Siguiente pregunta de guru para profundizar en su perfil:"  
    )
    response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    text = getattr(response, 'text', '').strip() or "Lo siento, no pude generar una pregunta."
    history.append({'query': query, 'response': text})
    return text


def get_rag_response(
    query, profile, history, date_filter=None, category_filter=None, limit=1
):
    """
    Genera una respuesta RAG usando contexto recuperado de la base de datos.

    Args:
        query (str): Consulta del usuario.
        profile (str): Resumen de perfil del usuario.
        history (list): Historial de la sesión.
        date_filter (str|None): Filtro de fecha en ISO.
        category_filter (str|None): Filtro de categoría.
        limit (int): Número máximo de eventos a recuperar.

    Returns:
        tuple: (texto de respuesta, filas crudas de eventos)
    """
    q_emb = get_embeddings(query)
    conn, cur = connect_db()
    rows, raw_rows = query_data(cur, q_emb, limit, date=date_filter)
    conn.close()

    context = "\n".join(rows) if rows else "No se encontraron eventos con esos criterios."
    prompt = (
        f"{system_instruction}\n"
        f"--- Fecha de referencia ---\n{date_filter or 'ninguna'}\n"
        f"--- Perfil ---\n{profile}\n"
        f"--- Historial ---\n{format_history(history)}\n"
        f"--- Contexto Recuperado ---\n{context}\n"
        f"--- Pregunta ---\n{query}\n--- Respuesta ---\n"
    )
    response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    text = getattr(response, 'text', '').strip() or "Lo siento, no pude generar respuesta."
    history.append({'query': query, 'response': text})
    return text, raw_rows

###########################################
#           CONFIGURACIÓN DE API          #
###########################################

app = FastAPI()
user_sessions = {}

class QueryRequest(BaseModel):
    """
    Modelo de petición de consulta que incluye sesión y parámetros.

    session_id: Identificador único de la sesión.
    query: Texto de la consulta.
    user_name: Nombre del usuario (solo en inicialización).
    limit: Límite de eventos a recuperar.
    """
    session_id: str
    query: str
    user_name: str | None = None
    limit: int = 5

@app.post('/init_db')
async def init_db():
    """
    Inicializa el esquema de la base de datos.

    Creates tables if they do not exist.
    """
    try:
        conn, cur = connect_db()
        create_table(cur)
        conn.close()
        return JSONResponse(content={'message': 'Esquema inicializado correctamente.'})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/add_events')
async def add_events(file: UploadFile = File(...)):
    """
    Procesa un CSV de eventos, normaliza y almacena en la base de datos.

    Args:
        file (UploadFile): Archivo CSV con columnas Título, Descripción, Categoría.

    Returns:
        dict: Mensaje de éxito con número de eventos añadidos.
    """
    try:
        df = pd.read_csv(file.file).head(500)
        # Clasificar categorías faltantes
        mask = df['Categoría'].isna()
        df.loc[mask, 'Categoría'] = df.loc[mask, 'Título'].apply(
            lambda title: ranking_missing_categories(
                title, bm25=BM25, tfidf=TFIDF, X_tfidf=X_TFIDF, labels=LABELS_M, k=10
            )
        )
        # Normalizar categorías
        df['Categoría'] = df['Categoría'].apply(
            lambda x: normalize_category(
                x, RULE_MAP, THRESHOLD, CANONICAL_LABELS_N, MODEL, EMBS_N
            )
        )
        # Construir texto para embeddings
        df['text'] = df['Categoría'] + ': ' + df['Título'] + ': ' + df['Descripción']

        embeddings = get_embeddings(df['text'].tolist())
        conn, cur = connect_db()
        insert_data(cur, df, embeddings)
        conn.close()
        return {'message': f'{len(df)} eventos añadidos correctamente.'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/query_friendly')
async def query_friendly(req: QueryRequest):
    """
    Endpoint principal para gestionar interacciones de usuario.

    Maneja identificación, construcción de perfil y consultas RAG.
    """
    sess = user_sessions.setdefault(req.session_id, {'history': [], 'prefs': {}})
    prefs, hist = sess['prefs'], sess['history']

    # 1) Identificación de usuario
    if 'user_name' not in prefs:
        if not req.user_name:
            raise HTTPException(status_code=400, detail="Se requiere user_name para identificación.")
        prefs['user_name'] = req.user_name
        conn, cur = connect_db()
        profile = get_user_profile(cur, req.user_name)
        conn.close()
        # Usuario existente
        if profile:
            prefs['profile_summary'] = profile
            prefs['profile_complete'] = True
            return {
                'response': f"¡Bienvenido de nuevo, {req.user_name}! ¿En qué puedo ayudarte hoy?",
                'events': []
            }
        # Nuevo usuario: iniciar perfil
        prefs['profile_qas'] = []
        next_q = get_profile_response(req.query, hist, prefs)
        return {'response': next_q, 'events': []}

    # 2) Fase de perfil
    if not prefs.get('profile_complete'):
        prefs['profile_qas'].append({
            'question': hist[-1]['response'],
            'answer': req.query
        })
        if len(prefs['profile_qas']) < QUESTIONS_NUMBER:
            next_q = get_profile_response(req.query, hist, prefs)
            return {'response': next_q, 'events': []}
        # Perfil completo: generar resumen
        qa_text = '\n'.join(f"P: {qa['question']}\nR: {qa['answer']}" for qa in prefs['profile_qas'])
        summary_prompt = (
            f"Genera un perfil de usuario en base a estas preguntas y respuestas:\n{qa_text}"
            " No puedes inventar información que no de el usuario..."
        )
        summary_resp = client.models.generate_content(
            model="gemini-2.0-flash", contents=summary_prompt
        )
        profile_summary = summary_resp.text.strip() or ""
        conn, cur = connect_db()
        insert_user_profile(cur, prefs['user_name'], profile_summary, prefs['profile_qas'][0]['answer'])
        conn.close()
        prefs['profile_complete'] = True
        prefs['profile_summary'] = profile_summary
        return {
            'response': f"¡Perfecto! He creado tu perfil:\n{profile_summary}\n\n¿En qué más puedo ayudarte?",
            'events': []
        }

    # 3) Consulta RAG
    detected = parse_date_expression(req.query)
    if detected:
        prefs['date'] = detected
        response, rows = get_rag_response(
            req.query, prefs['profile_summary'], hist,
            date_filter=prefs.get('date'),
            category_filter=prefs.get('category'),
            limit=req.limit
        )
    else:
        response, rows = get_rag_response(
            req.query, prefs['profile_summary'], hist,
            date_filter=None,
            category_filter=prefs.get('category'),
            limit=req.limit
        )

    events = [
        {'id': r[0], 'categoria': r[1], 'titulo': r[2], 'ubicacion': r[3], 'fecha': str(r[4])}
        for r in rows
    ]
    return {'response': response, 'events': events}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=5000)
