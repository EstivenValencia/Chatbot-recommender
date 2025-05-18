import json
import pickle
import re
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd  # Included for potential future use
import psycopg2
import requests
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score  # Imported for evaluation
from sklearn.model_selection import train_test_split  # Imported for experimentation
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from config import URL_EMBEDDING_MODEL


# ====================== CONFIGURACIÓN GLOBAL ======================
VECTOR_SIZE = 768
BATCH_SIZE = 50
DELTA_DAYS = 1  # Días máximos para rango de fecha

DB_PARAMS = {
    "host": "localhost",
    "port": 5432,
    "user": "estiven",
    "password": "estiven123",
    "dbname": "events_db"
}

WORD_RE = re.compile(r'[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+', flags=re.UNICODE)

# ====================== MODELO DE NORMALIZACIÓN ====================

def load_model_normalization(config_path: str = "Models/config.json"):
    """
    Carga configuración y modelo para normalizar categorías.

    Args:
        config_path (str): Ruta al JSON con canonical_labels, rule_map, threshold.

    Returns:
        tuple: (SentenceTransformer, np.ndarray, list, list, dict, float)
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    canonical_labels = cfg["canonical_labels"]
    rule_map = cfg["rule_map"]
    threshold = cfg["threshold"]

    subcanonical_labels = [
        "Guggenheim & Colecciones",
        "Deporte & Cultura Deportiva",
        "Arte & Arte Moderno",
    ]

    can_embs = np.load("Models/can_embs.npy")
    model = SentenceTransformer(
        "paraphrase-multilingual-MiniLM-L12-v2"
    )
    return model, can_embs, canonical_labels, subcanonical_labels, rule_map, threshold


def normalize_category(
    category: str,
    rule_map: dict,
    threshold: float,
    canonical_labels: list,
    model: SentenceTransformer,
    canonical_embeddings: np.ndarray
) -> str:
    """
    Normaliza una cadena de categoría a etiquetas canónicas.

    Args:
        category (str): Texto de la categoría original.
        rule_map (dict): Mapa de tokens a etiquetas.
        threshold (float): Umbral de similitud para embeddings.
        canonical_labels (list): Etiquetas canónicas.
        model (SentenceTransformer): Modelo de embeddings.
        canonical_embeddings (np.ndarray): Embeddings de etiquetas.

    Returns:
        str: Etiqueta normalizada o 'Varios / Otros'.
    """
    low = category.lower()
    for token, root in rule_map.items():
        if token in low:
            return root

    emb = model.encode(category, normalize_embeddings=True)
    similarities = canonical_embeddings @ emb
    idx = int(np.argmax(similarities))
    return (
        canonical_labels[idx]
        if similarities[idx] >= threshold
        else "Varios / Otros"
    )

# ============ PIPELINE DE CATEGORÍAS FALTANTES (BM25 + TF-IDF) ===========

def load_model_missing_categories(
    path: str = "Models/hybrid_pipeline.pkl"
):
    """
    Carga artefactos del pipeline híbrido de categorización.

    Args:
        path (str): Archivo pickle con bm25, tfidf, X_tfidf, labels.

    Returns:
        tuple: (BM25Okapi, TfidfVectorizer, np.ndarray, list)
    """
    with open(path, "rb") as f:
        artifacts = pickle.load(f)
    return (
        artifacts["bm25"],
        artifacts["tfidf"],
        artifacts["X_tfidf"],
        artifacts["labels"]
    )


def preprocess_text(text: str) -> list[str]:
    """
    Tokeniza un texto extrayendo solo palabras y minúsculas.

    Args:
        text (str): Cadena de entrada.

    Returns:
        list[str]: Lista de tokens.
    """
    return [w.lower() for w in WORD_RE.findall(text)]


def rank_missing_categories(
    title: str,
    bm25: BM25Okapi,
    tfidf: TfidfVectorizer,
    X_tfidf: np.ndarray,
    labels: list[str],
    k: int = 5
) -> str:
    """
    Predice categoría de título usando BM25 y TF-IDF.

    Args:
        title (str): Texto del título.
        bm25 (BM25Okapi): Modelo BM25.
        tfidf (TfidfVectorizer): Vectorizador TF-IDF.
        X_tfidf (np.ndarray): Matriz TF-IDF de train.
        labels (list): Etiquetas asociadas.
        k (int): Vecinos a considerar.

    Returns:
        str: Etiqueta más votada.
    """
    # BM25
    tokens = preprocess_text(title)
    bm_scores = bm25.get_scores(tokens)

    # TF-IDF
    q_vec = tfidf.transform([title])
    tf_scores = cosine_similarity(q_vec, X_tfidf).flatten()

    # Normalizar
    bm_norm = MinMaxScaler().fit_transform(bm_scores.reshape(-1, 1)).flatten()
    tf_norm = MinMaxScaler().fit_transform(tf_scores.reshape(-1, 1)).flatten()

    # Agregar señales
    final_scores = bm_norm + tf_norm
    top_indices = np.argsort(final_scores)[::-1][:k]

    # Votación
    agg = {}
    for idx in top_indices:
        cat = labels[idx]
        agg[cat] = agg.get(cat, 0) + final_scores[idx]

    return max(agg.items(), key=lambda x: x[1])[0]

# ================= EMBEDDINGS Y CONEXIÓN BD ==================

def get_embeddings(
    texts: list[str] | str,
    url: str = URL_EMBEDDING_MODEL,
    batch_size: int = BATCH_SIZE
) -> list or list[list]:
    """
    Obtiene embeddings desde un servicio externo en batches.

    Args:
        texts (list[str] | str): Texto(s) a embedding.
        url (str): Endpoint del servicio.
        batch_size (int): Tamaño de lote.

    Returns:
        list or list[list]: Embedding(s) resultante(s).
    """
    single = isinstance(texts, str)
    data = [texts] if single else texts

    embeddings = []
    for i in tqdm(range(0, len(data), batch_size), desc="Embeddings"):
        batch = data[i : i + batch_size]
        resp = requests.post(url, json={"texts": batch})
        if resp.status_code != 200:
            raise ConnectionError(
                f"Error batch {i//batch_size}: {resp.status_code}"
            )
        embeddings.extend(resp.json()["embeddings"])

    return embeddings[0] if single else embeddings


def connect_db():
    """
    Conecta a PostgreSQL usando parámetros globales.

    Returns:
        tuple: (conn, cursor)
    """
    conn = psycopg2.connect(**DB_PARAMS)
    conn.autocommit = True
    return conn, conn.cursor()


def create_event_table(
    cursor,
    vector_size: int = VECTOR_SIZE
) -> None:
    """
    Crea tabla 'eventos' con extensión vector y su índice.

    Args:
        cursor: Cursor de psycopg2.
        vector_size (int): Dimensión del vector embedding.
    """
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS eventos (
            id SERIAL PRIMARY KEY,
            titulo TEXT,
            categoria TEXT,
            fecha DATE,
            descripcion TEXT,
            ubicacion TEXT,
            embedding VECTOR({vector_size})
        );
    """
    )
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_eventos_embedding
        ON eventos USING ivfflat (embedding vector_l2_ops)
        WITH (lists = 100);
        """
    )
    cursor.execute("ANALYZE eventos;")


def insert_events(
    cursor,
    df: pd.DataFrame,
    embeddings: list
) -> None:
    """
    Inserta datos de eventos y embeddings en la BD.

    Args:
        cursor: Cursor activo.
        df (pd.DataFrame): DataFrame con columnas Título, Categoría, Fecha, Descripción, Lugar.
        embeddings (list): Lista de vectores para cada fila.
    """
    for row, emb in zip(df.itertuples(index=False), embeddings):
        cursor.execute(
            """
            INSERT INTO eventos (
                titulo, categoria, fecha, descripcion, ubicacion, embedding
            ) VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                row.Título,
                row.Categoría,
                row.Fecha,
                row.Descripción,
                row.Lugar,
                emb,
            ),
        )


# ================= CONSULTA DE DATOS ====================

def build_date_range(date_str: str) -> tuple[str, str]:
    """
    Construye rango de fecha (inicio, fin) desde YYYY-MM o YYYY-MM-DD.

    Args:
        date_str (str): Fecha en formato ISO parcial/completo.

    Returns:
        tuple[str, str]: Fecha inicio y fin en ISO.
    """
    if re.fullmatch(r"\d{4}-\d{2}", date_str):
        start = datetime.fromisoformat(date_str + "-01").date()
        end = (start + relativedelta(months=1) - timedelta(days=1))
    else:
        start = datetime.fromisoformat(date_str).date()
        end = start + timedelta(days=DELTA_DAYS)
    return start.isoformat(), end.isoformat()


def query_events(
    cursor,
    embedding: list,
    limit: int = 5,
    date: str | None = None
) -> tuple[list[str], list[tuple]]:
    """
    Recupera eventos más cercanos por embedding y opcional filtro de fecha.

    Args:
        cursor: Cursor de psycopg2.
        embedding (list): Vector de consulta.
        limit (int): Máximo resultados.
        date (str|None): Fecha ISO para filtrar.

    Returns:
        tuple: (formatted_strings, raw_rows)
    """
    if date:
        start, end = build_date_range(date)
        sql = """
            SELECT id, categoria, titulo, ubicacion, fecha, descripcion
            FROM eventos
            WHERE fecha BETWEEN %s AND %s
            ORDER BY embedding <-> %s::vector
            LIMIT %s
        """
        params = (start, end, embedding, limit)
    else:
        sql = """
            SELECT id, categoria, titulo, ubicacion, fecha, descripcion
            FROM eventos
            ORDER BY embedding <-> %s::vector
            LIMIT %s
        """
        params = (embedding, limit)

    cursor.execute(sql, params)
    rows = cursor.fetchall()
    formatted = [
        f"- titulo: {titulo} categoria: {categoria} fecha: {fecha} descripcion: {descripcion}"
        for (_, categoria, titulo, ubicacion, fecha, descripcion) in rows
    ]
    return formatted, rows

# ================ PERFIL DE USUARIO ==================

def insert_user_profile(
    cursor,
    user_name: str,
    profile_lines: list[str],
    category: str
) -> None:
    """
    Inserta o actualiza perfil de usuario en la tabla perfil_usuario.

    Args:
        cursor: Cursor de psycopg2.
        user_name (str): Nombre clave.
        profile_lines (list[str]): Líneas del perfil.
        category (str): Categoría principal.
    """
    text = "\n".join(profile_lines)
    today = datetime.today().date()
    cursor.execute(
        """
        INSERT INTO perfil_usuario (nombre, fecha_creacion, perfil, categoria)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (nombre) DO UPDATE SET
            fecha_creacion = EXCLUDED.fecha_creacion,
            perfil = EXCLUDED.perfil;
        """,
        (user_name, today, text, category)
    )


def get_user_profile(cursor, user_name: str) -> dict | None:
    """
    Obtiene perfil de usuario; None si no existe.

    Args:
        cursor: Cursor de psycopg2.
        user_name (str): Nombre de usuario.

    Returns:
        dict | None: Datos de perfil.
    """
    cursor.execute(
        "SELECT nombre, fecha_creacion, perfil FROM perfil_usuario WHERE nombre = %s",
        (user_name,),
    )
    row = cursor.fetchone()
    if not row:
        return None
    return {"user_name": row[0], "profile_date": row[1], "profile_text": row[2]}


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
                row['Título'],
                row['Categoría'],
                row['Fecha'],
                row['Descripción'],
                row['Lugar'],
                embeddings[i]
            )
        )























# import re
# from datetime import datetime, timedelta
# from dateutil.relativedelta import relativedelta


# # Configuración global
# VECTOR_SIZE = 768
# BATCH_SIZE = 50
# URL_EMBEDDING_MODEL = "https://8963-34-9-207-28.ngrok-free.app/embed"
# DELTA = 1 # Días máximos para la busqueda de eventos

# DB_PARAMS = {
#     "host": "localhost",
#     "port": 5432,
#     "user": "estiven",
#     "password": "estiven123",
#     "dbname": "events_db"
# }

# def load_model_normalization():

#     with open("Models/config.json", "r", encoding="utf-8") as f:
#         cfg = json.load(f)
#     canonical_labels = cfg["canonical_labels"]
#     rule_map         = cfg["rule_map"]
#     threshold        = cfg["threshold"]

#     # Subcategorias
#     subcanonical_labels = ["Guggenheim & Colecciones",
#     "Deporte & Cultura Deportiva",
#     "Arte & Arte Moderno",
#     ]

#     # 2) cargar embeddings
#     can_embs = np.load("Models/can_embs.npy")

#     # 3) cargar modelo
#     model = SentenceTransformer.load("paraphrase-multilingual-MiniLM-L12-v2")
#     return model, can_embs, canonical_labels, subcanonical_labels, rule_map, threshold

# def normalize_category(cat_str, rule_map, threshold, canonical_labels, model, can_embs):
#     low = cat_str.lower()
#     for token, root in rule_map.items():
#         if token in low:
#             return root
#     emb = model.encode(cat_str, normalize_embeddings=True)
#     sims = can_embs @ emb
#     idx  = int(np.argmax(sims))
#     return canonical_labels[idx] if sims[idx] >= threshold else "Varios / Otros"

# def load_model_missing_categories(path: str = "Models/hybrid_pipeline.pkl"):
#     """
#     Carga los artefactos necesarios para inferencia del pipeline híbrido:
#       - bm25:  modelo BM25Okapi
#       - tfidf: vectorizador TF-IDF
#       - X_tfidf: matriz TF-IDF del set de entrenamiento
#       - labels: lista de etiquetas asociadas
    
#     Args:
#         path: ruta al archivo pickle generado en el entrenamiento.
    
#     Returns:
#         bm25, tfidf, X_tfidf, labels
#     """
#     with open(path, "rb") as f:
#         artifacts = pickle.load(f)
    
#     bm25    = artifacts["bm25"]
#     tfidf   = artifacts["tfidf"]
#     X_tfidf = artifacts["X_tfidf"]
#     labels  = artifacts["labels"]
    
#     return bm25, tfidf, X_tfidf, labels

# WORD_RE = re.compile(r'[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+', flags=re.UNICODE)

# def preprocess(text):
#     """
#     Tokeniza el texto extrayendo sólo palabras y pasando a minúscula.
#     """
#     return [w.lower() for w in WORD_RE.findall(text)]

# def ranking_missing_categories(
#     title: str,
#     bm25,
#     tfidf,
#     X_tfidf,
#     labels,
#     k: int = 5
# ) -> str:
#     """
#     Predice la categoría de un nuevo título usando el pipeline híbrido BM25 + TF-IDF.

#     Args:
#         title: texto del título a clasificar.
#         bm25: objeto BM25Okapi entrenado sobre los títulos de train.
#         tfidf: vectorizador TfidfVectorizer ajustado sobre train.
#         X_tfidf: matriz TF-IDF del set de entrenamiento.
#         labels: lista de etiquetas (misma longitud que X_tfidf) para cada muestra de train.
#         k: número de vecinos más cercanos a considerar.

#     Returns:
#         La etiqueta predicha (str).
#     """
#     # 1) Tokenizar y puntuar con BM25
#     tokens    = preprocess(title)              # tu función de preprocesado
#     bm_scores = bm25.get_scores(tokens)        # array de tamaño len(train)

#     # 2) Vectorizar con TF-IDF y puntuar con similaridad coseno
#     q_vec     = tfidf.transform([title])       # (1, n_features)
#     tf_scores = cosine_similarity(q_vec, X_tfidf).flatten()

#     # 3) Normalizar ambas puntuaciones en [0,1]
#     scaler_bm = MinMaxScaler()
#     bm_n      = scaler_bm.fit_transform(bm_scores.reshape(-1,1)).flatten()

#     scaler_tf = MinMaxScaler()
#     tf_n      = scaler_tf.fit_transform(tf_scores.reshape(-1,1)).flatten()

#     # 4) Sumar las señales y quedarse con los k top
#     final     = bm_n + tf_n
#     topk_idx  = np.argsort(final)[::-1][:k]

#     # 5) Votación ponderada por la puntuación final
#     agg = {}
#     for idx in topk_idx:
#         cat = labels[idx]
#         agg[cat] = agg.get(cat, 0) + final[idx]

#     # 6) Devolver la categoría con mayor voto
#     return max(agg.items(), key=lambda x: x[1])[0]

# # Funciones auxiliares
# def get_embeddings(texts, url=URL_EMBEDDING_MODEL, batch_size=BATCH_SIZE):
#     is_single_text = False
#     if isinstance(texts, str):
#         texts = [texts]
#         is_single_text = True

#     all_embeddings = []
#     for i in tqdm(range(0, len(texts), batch_size), desc="Obteniendo embeddings"):
#         batch = texts[i:i+batch_size]
#         response = requests.post(url, json={"texts": batch})
#         if response.status_code == 200:
#             batch_embeddings = response.json()["embeddings"]
#             all_embeddings.extend(batch_embeddings)
#         else:
#             raise Exception(f"Error en batch {i//batch_size}: {response.status_code} - {response.text}")

#     return all_embeddings[0] if is_single_text else all_embeddings

# def connect_db():
#     conn = psycopg2.connect(**DB_PARAMS)
#     conn.autocommit = True
#     cur = conn.cursor()
#     return conn, cur




# DELTA = 0   # si el parámetro DELTA deja de tener sentido para meses, puedes ignorarlo

# def build_date_range(date_str):
#     # YYYY-MM
#     if re.fullmatch(r'\d{4}-\d{2}', date_str):
#         # día 1 del mes
#         start = datetime.fromisoformat(date_str + '-01').date()
#         # siguiente mes menos un día
#         end = (start + relativedelta(months=1) - timedelta(days=1))
#     else:
#         # YYYY-MM-DD
#         start = datetime.fromisoformat(date_str).date()
#         end = (start + timedelta(days=DELTA))
#     return start.isoformat(), end.isoformat()


# def query_data(cur, query_embedding, limit=5, date=None):
#     print("Se obtiene la fecha: ", date)
#     # 1) Prepara fechas si hay filtro
#     if date:
#         start_date, end_date = build_date_range(date)
#         #start_date = date
#         #end_date = (datetime.fromisoformat(date) + timedelta(days=DELTA)).date().isoformat()
#         sql = """
#             SELECT id, categoria, titulo, ubicacion, fecha, descripcion
#             FROM eventos
#             WHERE fecha >= %s AND fecha <= %s
#             ORDER BY embedding <-> %s::vector
#             LIMIT %s
#         """
#         params = (start_date, end_date, list(query_embedding), limit)
#     else:
#         sql = """
#             SELECT id, categoria, titulo, ubicacion, fecha, descripcion
#             FROM eventos
#             ORDER BY embedding <-> %s::vector
#             LIMIT %s
#         """
#         params = (list(query_embedding), limit)

#     # 2) Ejecuta y recoge tuplas
#     cur.execute(sql, params)
#     rows = cur.fetchall()
#     print("ROWS: ",rows)
#     # 3) Mapea cada tupla a la cadena deseada
#     formatted = [
#         f"-titulo: {titulo} categoria: {categoria} fecha: {fecha} descripcion: {descripcion}"
#         for (_id, categoria, titulo, ubicacion, fecha, descripcion) in rows
#     ]
#     print(formatted)
#     return formatted, rows

# def insert_user_profile(cur, user_name: str, prefs: list[str], category):
#     """
#     Inserta o actualiza el perfil de usuario en la tabla user_profiles.
#     """
#     profile_text = "\n".join(prefs)
#     profile_date = datetime.today().date()
#     cur.execute(
#         """
#         INSERT INTO perfil_usuario (nombre, fecha_creacion, perfil, categoria)
#         VALUES (%s, %s, %s, %s)
#         ON CONFLICT (nombre) DO UPDATE
#         SET fecha_creacion = EXCLUDED.fecha_creacion,
#             perfil = EXCLUDED.perfil;
#         """,
#         (user_name, profile_date, profile_text, category)
#     )


# def get_user_profile(cur, user_name: str) -> dict | None:
#     """
#     Recupera el perfil de usuario; devuelve None si no existe.
#     """
#     cur.execute(
#         "SELECT nombre, fecha_creacion, perfil FROM perfil_usuario WHERE nombre=%s",
#         (user_name,)
#     )
#     row = cur.fetchone()
#     if not row:
#         return None
#     return {
#         'user_name': row[0],
#         'profile_date': row[1],
#         'profile_text': row[2]
#     }"