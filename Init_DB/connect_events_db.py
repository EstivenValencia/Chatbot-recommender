import psycopg2

# Parámetros de conexión
conn = psycopg2.connect(
    host="localhost",
    port=5432,
    user="estiven",
    password="estiven123",
    dbname="events_db"  # inicialmente conectamos a la base por defecto "postgres"
)

# Crear un cursor para ejecutar SQL
cur = conn.cursor()

# Asegúrate de crear la extensión de pgvector
cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

# Crear tabla eventos
cur.execute("""
CREATE TABLE IF NOT EXISTS eventos (
    id SERIAL PRIMARY KEY,
    categoria TEXT,
    titulo TEXT,
    lugar TEXT,
    fecha DATE,
    hora_inicio TEXT,
    embedding VECTOR(768)  -- ajusta 768 al tamaño real de tu modelo
);
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS perfil_usuario (
    nombre    TEXT PRIMARY KEY,  -- el nombre del usuario
    fecha_creacion DATE     NOT NULL, -- la fecha de creación del perfil
    perfil TEXT     NOT NULL,  -- texto libre con la información de gustos, intereses, etc.
    categoria     TEXT     NOT NULL  -- la categoria de gusto del usuario
);
""")

# Confirmar los cambios
conn.commit()