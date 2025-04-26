import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import pandas as pd
from datetime import datetime, timedelta
from py_heideltime import heideltime
import dateparser
import isodate
from google import genai
import re

# Importar helpers de la base de datos y embeddings
from models import get_embeddings, connect_db, create_table, insert_data, query_data

# Definir variable de entorno directamente en el script (no recomendado para producción)
os.environ['GOOGLE_API_KEY'] = 'AIzaSyAStou5qqNC-770W79MV79Im752WOVeytg'

# Carga segura de API Key y configuración de genai (Gemini) usando el cliente de Google
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    raise ValueError("No se encontró la variable de entorno 'GOOGLE_API_KEY'.")
# Crear instancia de cliente
client = genai.Client(api_key=api_key)

# # Instrucciones del sistema
# system_instruction = """Eres 'LagunAI', un asistente experto y amigable especializado en actividades de ocio y recreación en Bilbao y sus alrededores. Tu conocimiento se basa EXCLUSIVAMENTE en la información de contexto que se te proporciona en cada turno.

# Instrucciones clave:
# 1.  Tu base de datos contiene eventos con título, categoría, fecha, descripción y ubicación. A veces recibirás actividades filtradas por fecha, otras veces listados de categorías/fechas, y otras veces actividades encontradas por similitud de texto.
# 2.  Responde a las preguntas del usuario basándote ÚNICAMENTE en el 'Contexto Recuperado' proporcionado. NO inventes información ni uses conocimiento externo.
# 3.  Si el contexto indica explícitamente que no se encontró información (por ejemplo, para una fecha específica), informa al usuario de eso. Si el contexto proviene de una búsqueda semántica, puedes mencionar que son resultados similares al tema general de la consulta.
# 4.  Considera el 'Historial de Conversación' para entender el flujo del diálogo, pero basa tu respuesta factual SIEMPRE en el 'Contexto Recuperado' actual.
# 5.  Sé amable, conciso y directo en tus respuestas. Menciona detalles clave como fechas, lugares y descripciones cuando sea relevante según el contexto recibido.
# 6.  Estás operando en Bilbao, País Vasco.
# """

# Instrucciones del sistema RAG
system_instruction = """
Eres 'LagunAI', un asistente experto y amigable especializado en
actividades de ocio y recreación en Bilbao y sus alrededores.
Tu conocimiento se basa EXCLUSIVAMENTE en el contexto proporcionado.
"""

# Instrucciones del sistema Perfil
profile_instruction = """
Eres 'LagunAI', un asistente que construye un perfil del usuario preguntando sus gustos y preferencias de ocio.
Haz preguntas abiertas para conocer categorías de eventos, preferencias de horario, ubicaciones favoritas y estilo de actividades.
Guarda cada respuesta en el perfil de la sesión (prefs).
"""

# Formateo del historial
def format_history(history):
    return "\n".join([
        f"Usuario: {turn['query']}\nAsistente: {turn['response']}" for turn in history
    ])

# Función para detectar y resolver fechas en texto usando heideltime y dateparserimport re

def parse_date_expression(text, reference_date=None):
    if reference_date is None:
        reference_date = datetime.today()

    dct = reference_date.date().isoformat()
    timexs = heideltime(text, language="Spanish", dct=dct)
    print("Timexs:", timexs)
    if not timexs:
        return None

    timex = timexs[0]
    ttype, value, span = timex["type"], timex["value"], timex["text"]

    try:
        if ttype == "DURATION":
            delta = isodate.parse_duration(value)
            return (reference_date + delta).date().isoformat()

        if ttype == "DATE":
            print("Valor:", value)
            # ──► si ya viene como AAAA-MM-DD, úsalo sin más
            if '-' in value:
                return value
            # si viniera en otro formato, intenta parsearlo
            dt = dateparser.parse(value, languages=["es"])
            print("O aqui")
            return dt.date().isoformat() if dt else None

        if ttype in ("TIME", "SET"):
            dt = dateparser.parse(
                span,
                settings={"RELATIVE_BASE": reference_date},
                languages=["es"],
            )
            return dt.date().isoformat() if dt else None

        # Fallback genérico
        dt = dateparser.parse(text, languages=["es"])
        return dt.date().isoformat() if dt else None

    except Exception as e:
        print("Error al parsear fecha:", e)
        return None

# Generar respuesta de perfil construyendo preguntas
def get_profile_response(query, history, prefs):
    # Construir prompt para perfil
    prompt = f"""{profile_instruction}

Historial de perfil hasta ahora:
{format_history(history)}

Pregunta del usuario:
{query}

Siguiente pregunta de LagunAI para profundizar en su perfil:
"""
    response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    text = response.text.strip() if getattr(response, 'text', None) else "Lo siento, no pude generar una pregunta."
    history.append({'query': query, 'response': text})
    return text

# RAG + generación con Gemini
def get_rag_response(query, history, date_filter=None, category_filter=None, limit=5):
    q_emb = get_embeddings(query)
    conn, cur = connect_db()
    rows = query_data(cur, q_emb, limit, date=date_filter)
    conn.close()
    if category_filter:
        rows = [r for r in rows if r[1].lower()==category_filter.lower()][:limit]
    context = ("".join([f"- {r[2]} ({r[1]}) el {r[4]} en {r[3]}\n" for r in rows])
               if rows else "No se encontraron eventos con esos criterios.")
    prompt = f"""{system_instruction}

--- Historial ---
{format_history(history)}

--- Contexto Recuperado ---
{context}

--- Pregunta ---
{query}

--- Respuesta ---
"""
    response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    text = response.text.strip() if getattr(response, 'text', None) else "Lo siento, no pude generar respuesta."
    history.append({'query': query, 'response': text})
    return text, rows

# Crear la aplicación FastAPI
app = FastAPI()
user_sessions = {}  # session_id -> {'history': [], 'prefs': {}}

class QueryRequest(BaseModel):
    session_id: str
    query: str
    limit: int = 5

@app.post('/init_db')
async def init_db():
    try:
        conn, cur = connect_db(); create_table(cur); conn.close()
        return JSONResponse(content={'message':'Esquema inicializado correctamente.'})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/add_events')
async def add_events(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        df['text']=df['categoria']+': '+df['titulo']+': '+df['descripcion']
        embeddings=get_embeddings(df['text'].tolist())
        conn,cur=connect_db(); insert_data(cur,df,embeddings); conn.close()
        return {'message':f'{len(df)} eventos añadidos correctamente.'}
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))

@app.post('/query_friendly')
async def query_friendly(req: QueryRequest):
    sess = user_sessions.setdefault(req.session_id, {'history': [], 'prefs': {}})
    prefs = sess['prefs']; hist = sess['history']
    # Detectar fecha relativa o explícita
    detected = parse_date_expression(req.query)
    # if req.date:
    #     detected = req.date
    print("Fecha: ",detected)
    if detected:
        prefs['date'] = detected
        # RAG response
        response, rows = get_rag_response(req.query, hist, prefs.get('date'), prefs.get('category'), req.limit)
    else:
        # Perfil response hasta completar 5 preguntas
        count = prefs.get('profile_count', 0)
        if count < 2:
            prefs['profile_count'] = count + 1
            response = get_profile_response(req.query, hist, prefs)
        else:
            # Perfil complete
            prefs['profile_complete'] = True
            response = (
                "¡Fantástico! Ya conozco tus gustos y preferencias. "
                "Ahora estoy listo para recomendarte los mejores eventos. "
                "¿Qué fecha te interesa?"
            )
        rows = []

    # Formatear eventos
    events = [
        {'id': r[0], 'categoria': r[1], 'titulo': r[2], 'ubicacion': r[3], 'fecha': str(r[4])}
        for r in rows
    ]
    return {'response': response, 'events': events}

if __name__=='__main__':
    uvicorn.run(app,host='0.0.0.0',port=5000)