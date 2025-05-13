import pandas as pd
import psycopg2
import requests
from tqdm import tqdm
from datetime import datetime, timedelta
import json
import numpy as np
from sentence_transformers import SentenceTransformer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise        import cosine_similarity
from sklearn.preprocessing           import MinMaxScaler
from sklearn.metrics                 import f1_score
from rank_bm25                       import BM25Okapi
import numpy as np
import pickle
import re


# Configuración global
VECTOR_SIZE = 768
BATCH_SIZE = 50
URL_EMBEDDING_MODEL = "https://99e3-34-16-226-98.ngrok-free.app"
DELTA = 1 # Días máximos para la busqueda de eventos

DB_PARAMS = {
    "host": "localhost",
    "port": 5432,
    "user": "estiven",
    "password": "estiven123",
    "dbname": "events_db"
}

def load_model_normalization():

    with open("config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    canonical_labels = cfg["canonical_labels"]
    rule_map         = cfg["rule_map"]
    threshold        = cfg["threshold"]

    # 2) cargar embeddings
    can_embs = np.load("can_embs.npy")

    # 3) cargar modelo
    model = SentenceTransformer.load("sbert_model")
    return model, can_embs, canonical_labels, rule_map, threshold

def normalize_category(cat_str, rule_map, threshold, canonical_labels, model, can_embs):
    low = cat_str.lower()
    for token, root in rule_map.items():
        if token in low:
            return root
    emb = model.encode(cat_str, normalize_embeddings=True)
    sims = can_embs @ emb
    idx  = int(np.argmax(sims))
    return canonical_labels[idx] if sims[idx] >= threshold else "Varios / Otros"

def load_model_missing_categories(path: str = "hybrid_pipeline.pkl"):
    """
    Carga los artefactos necesarios para inferencia del pipeline híbrido:
      - bm25:  modelo BM25Okapi
      - tfidf: vectorizador TF-IDF
      - X_tfidf: matriz TF-IDF del set de entrenamiento
      - labels: lista de etiquetas asociadas
    
    Args:
        path: ruta al archivo pickle generado en el entrenamiento.
    
    Returns:
        bm25, tfidf, X_tfidf, labels
    """
    with open(path, "rb") as f:
        artifacts = pickle.load(f)
    
    bm25    = artifacts["bm25"]
    tfidf   = artifacts["tfidf"]
    X_tfidf = artifacts["X_tfidf"]
    labels  = artifacts["labels"]
    
    return bm25, tfidf, X_tfidf, labels

WORD_RE = re.compile(r'[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+', flags=re.UNICODE)

def preprocess(text):
    """
    Tokeniza el texto extrayendo sólo palabras y pasando a minúscula.
    """
    return [w.lower() for w in WORD_RE.findall(text)]

def ranking_missing_categories(
    title: str,
    bm25,
    tfidf,
    X_tfidf,
    labels,
    k: int = 5
) -> str:
    """
    Predice la categoría de un nuevo título usando el pipeline híbrido BM25 + TF-IDF.

    Args:
        title: texto del título a clasificar.
        bm25: objeto BM25Okapi entrenado sobre los títulos de train.
        tfidf: vectorizador TfidfVectorizer ajustado sobre train.
        X_tfidf: matriz TF-IDF del set de entrenamiento.
        labels: lista de etiquetas (misma longitud que X_tfidf) para cada muestra de train.
        k: número de vecinos más cercanos a considerar.

    Returns:
        La etiqueta predicha (str).
    """
    # 1) Tokenizar y puntuar con BM25
    tokens    = preprocess(title)              # tu función de preprocesado
    bm_scores = bm25.get_scores(tokens)        # array de tamaño len(train)

    # 2) Vectorizar con TF-IDF y puntuar con similaridad coseno
    q_vec     = tfidf.transform([title])       # (1, n_features)
    tf_scores = cosine_similarity(q_vec, X_tfidf).flatten()

    # 3) Normalizar ambas puntuaciones en [0,1]
    scaler_bm = MinMaxScaler()
    bm_n      = scaler_bm.fit_transform(bm_scores.reshape(-1,1)).flatten()

    scaler_tf = MinMaxScaler()
    tf_n      = scaler_tf.fit_transform(tf_scores.reshape(-1,1)).flatten()

    # 4) Sumar las señales y quedarse con los k top
    final     = bm_n + tf_n
    topk_idx  = np.argsort(final)[::-1][:k]

    # 5) Votación ponderada por la puntuación final
    agg = {}
    for idx in topk_idx:
        cat = labels[idx]
        agg[cat] = agg.get(cat, 0) + final[idx]

    # 6) Devolver la categoría con mayor voto
    return max(agg.items(), key=lambda x: x[1])[0]

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
    print("Se obtiene la fecha: ",date)
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

def insert_user_profile(cur, user_name: str, prefs: list[str]):
    """
    Inserta o actualiza el perfil de usuario en la tabla user_profiles.
    """
    profile_text = "\n".join(prefs)
    profile_date = datetime.today().date()
    cur.execute(
        """
        INSERT INTO user_profiles (user_name, profile_date, profile_text)
        VALUES (%s, %s, %s)
        ON CONFLICT (user_name) DO UPDATE
        SET profile_date = EXCLUDED.profile_date,
            profile_text = EXCLUDED.profile_text;
        """,
        (user_name, profile_date, profile_text)
    )


def get_user_profile(cur, user_name: str) -> dict | None:
    """
    Recupera el perfil de usuario; devuelve None si no existe.
    """
    cur.execute(
        "SELECT user_name, profile_date, profile_text FROM user_profiles WHERE user_name=%s",
        (user_name,)
    )
    row = cur.fetchone()
    if not row:
        return None
    return {
        'user_name': row[0],
        'profile_date': row[1],
        'profile_text': row[2]
    }