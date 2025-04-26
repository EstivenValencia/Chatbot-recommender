import psycopg2

# Parámetros de conexión
conn = psycopg2.connect(
    host="localhost",
    port=5432,
    user="estiven",
    password="estiven123",
    dbname="postgres"  # inicialmente conectamos a la base por defecto "postgres"
)

conn.autocommit = True

# Crear un cursor para ejecutar SQL
cur = conn.cursor()

# Crear una nueva base de datos
cur.execute("CREATE DATABASE events_db;")

# Confirmar los cambios
conn.commit()

# Cerrar conexión
cur.close()
conn.close()

print("Base de datos 'events_db' creada exitosamente.")
