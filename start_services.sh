echo "Iniciando base de datos postgres..."
docker-compose up -d

echo "URL para acceder a la base de datos: http://localhost:5050/login?next=/"