import pandas as pd
import psycopg2
import requests
from tqdm import tqdm
from datetime import datetime, timedelta

# Configuración global
VECTOR_SIZE = 768
BATCH_SIZE = 50
URL_EMBEDDING_MODEL = "https://9a4a-35-204-14-185.ngrok-free.app/embed"
DELTA = 5 # Días máximos para la busqueda de eventos

DB_PARAMS = {
    "host": "localhost",
    "port": 5432,
    "user": "estiven",
    "password": "estiven123",
    "dbname": "events_db"
}

# Funciones auxiliares
def get_embeddings(texts, url=URL_EMBEDDING_MODEL, batch_size=BATCH_SIZE):
    is_single_text = False
    if isinstance(texts, str):
        texts = [texts]
        is_single_text = True

    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Obteniendo embeddings"):
        batch = texts[i:i+batch_size]
        response = requests.post(url, json={"texts": batch})
        if response.status_code == 200:
            batch_embeddings = response.json()["embeddings"]
            all_embeddings.extend(batch_embeddings)
        else:
            raise Exception(f"Error en batch {i//batch_size}: {response.status_code} - {response.text}")

    return all_embeddings[0] if is_single_text else all_embeddings

def connect_db():
    conn = psycopg2.connect(**DB_PARAMS)
    conn.autocommit = True
    cur = conn.cursor()
    return conn, cur

def create_table(cur, vector_size=VECTOR_SIZE, ):
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS eventos (
            id SERIAL PRIMARY KEY,
            titulo TEXT,
            categoria TEXT,
            fecha DATE,
            descripcion TEXT,
            ubicacion TEXT,
            embedding VECTOR({vector_size})
        );
    """)

    # Crear IVFFLAT para mayor rapidez en las consultas basado en la distancia L2
    cur.execute("""CREATE INDEX ON eventos
    USING ivfflat (embedding vector_l2_ops)
    WITH (lists = 100);""")

    # 4. Ejecutar ANALYZE para que PostgreSQL recoja estadísticas
    cur.execute("ANALYZE eventos;")

def insert_data(cur, df, embeddings):
    for i, row in df.iterrows():
        cur.execute(
            """
            INSERT INTO eventos (titulo, categoria, fecha, descripcion, ubicacion, embedding)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                row['titulo'],
                row['categoria'],
                row['fecha'],
                row['descripcion'],
                row['ubicacion'],
                embeddings[i]
            )
        )

def query_data(cur, query_embedding, limit=5, date=None):
    if date:
        start_date = date
        end_date = (datetime.fromisoformat(date) + timedelta(days=DELTA)).date().isoformat()

        cur.execute(
            """
            SELECT id, categoria, titulo, ubicacion, fecha
            FROM eventos
            WHERE fecha >= %s AND fecha <= %s
            ORDER BY embedding <-> %s::vector
            LIMIT %s
            """,
            (start_date, end_date, list(query_embedding), limit)
        )
    else:
        cur.execute(
            """
            SELECT id, categoria, titulo, ubicacion, fecha
            FROM eventos
            ORDER BY embedding <-> %s::vector
            LIMIT %s
            """,
            (list(query_embedding), limit)
        )
    return cur.fetchall()


