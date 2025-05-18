# Proyecto de Minería de Datos - Sistema de Eventos

Un framework conversacional que consolida y organiza automáticamente la oferta de eventos culturales de Bilbao (título, fecha, categoría, descripción) mediante web scraping, asistentes RAG y etiquetadores temporales multilenguaje. Ofrece:

* Un chat que recupera con precisión eventos según fechas y perfil de usuario.

* Reducción de la brecha informativa y mejora de la accesibilidad, especialmente para mayores.

* Segmentación eficaz del público y optimización de la planificación de agendas y programaciones futuras.

## Prerrequisitos

Antes de comenzar, asegúrate de tener instalado lo siguiente:

*   **Python**: Versión 3.12.2.
*   **Docker**: Para ejecutar la base de datos PostgreSQL en un contenedor. (Instrucciones de instalación: [Docker](https://docs.docker.com/get-docker/))

## Instalación

Sigue estos pasos para configurar el entorno de desarrollo:

### 1. Creación de ambiente
```bash
python -m venv .env
source .env/bin/activate
```
### 2. Instalación de dependencias
```bash
pip install -r requirements.txt
```
### 1. Configuración de la Base de Datos (PostgreSQL + pgvector con Docker)

Utilizaremos Docker para levantar una instancia de PostgreSQL con la extensión `pgvector`. Esto solo se debe realizar la primera
vez cuando se crea la base de datos, en usos siguientes se deberá seguir directamente a la sección de `uso`

**a. Iniciar el Contenedor de la Base de Datos:**

Abre una terminal en la raíz de tu proyecto (donde está el `docker-compose.yml`) y ejecuta:
```bash
docker-compose up -d
```
Esto descargará la imagen (si no la tienes) e iniciará el contenedor de PostgreSQL en segundo plano. El usuario será `estiven`, la contraseña `estiven123` y se conectará al puerto `5432` de tu `localhost`.

**c. Crear la Base de Datos `events_db`:**

Una vez que el contenedor de Docker esté en ejecución, ejecuta el script para crear la base de datos `events_db`.

```bash
python Init_DB/create_db.py
```

**d. Crear las Tablas y la Extensión `vector`:**

Luego, ejecuta el script para crear las tablas necesarias y la extensión `vector` dentro de la base de datos `events_db`.

```bash
python Init_DB/connect_events_db.py
```

## Uso 

Deberá crear un archivo `config.py` vacio en este nivel del directorio.y deberá definir dos variables str `URL_EMBEDDING_MODEL` y `TOKEN_GEMINI` que deberá dejar vacias por ahora. 

**a. Servidor embedding COLAB**

Debido a las limitaciones de recursos para ejecutar el proyecto, el modulo encargado de calcular los embeddings de los modelos
se encuentra definido dentro de un notebook que se puede ejecutar en colab, para ello se debe subir el notebook  `Servidor_embedding.ipynb`, al ejecutarlo encontrará en el penultima celda la salida `Ngrok URL: https://8963-34-9-207-28.ngrok-free.app` que corresponde
a la URL del servidor, esta URL debe reemplazarse en la variable `URL_EMBEDDING_MODEL` 

También deberá contar con el token de gemini, ya que este proyecto utiliza el modelo más liviano de google, este deberá estar definido dentro de la variable `TOKEN_GEMINI` del archivo config.

**b. Inciar base de datos**

Ejecute la siguiente línea para iniciar el servidor backend

```bash
docker-compose up -d
```

**b. Inciar servidor backend**

Ejecute la siguiente línea para iniciar el servidor backend

```bash
python server.py
```

**b. Inciar servidor frontend**

Ejecute la siguiente línea para iniciar el servidor frontend

```bash
streamlit run frontend.py
```

Si todo funciona correctamente deberá ver la aplicación desplegada en la url: http://localhost:8501/. A medida que el web scrapping descargue nuevos eventos estos serán almacenados y gestionados con el servidor embedding para su recuperación. 