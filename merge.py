import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from datetime import datetime, timedelta
from playwright.sync_api import sync_playwright
import time
import os

# ========== ElCorreo ==========
def parse_spanish_date(fecha_str):
    meses = {"enero": "01", "febrero": "02", "marzo": "03", "abril": "04", "mayo": "05", "junio": "06",
             "julio": "07", "agosto": "08", "septiembre": "09", "octubre": "10", "noviembre": "11", "diciembre": "12"}
    match = re.match(r"(\d{1,2}) de (\w+) de (\d{4})", fecha_str.lower())
    if match:
        dia, mes_nombre, anio = match.groups()
        mes = meses.get(mes_nombre)
        if mes:
            return datetime.strptime(f"{anio}-{mes}-{int(dia):02d}", "%Y-%m-%d")
    return None

def scrape_elcorreo():
    eventos = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto("https://agenda.elcorreo.com/eventos/bilbao/listado.html?pag=1")
        page.wait_for_selector("article")
        pagination_links = page.query_selector_all("ul.voc-pagination li a")
        last_page_num = 1
        for link in pagination_links:
            href = link.get_attribute("href")
            if href and "pag=" in href:
                match = re.search(r"pag=(\d+)", href)
                if match:
                    last_page_num = max(last_page_num, int(match.group(1)))

        for current_page in range(1, last_page_num + 1):
            page.goto(f"https://agenda.elcorreo.com/eventos/bilbao/listado.html?pag={current_page}")
            page.wait_for_selector("article")
            articles = page.query_selector_all("article")
            for art in articles:
                try:
                    categoria = art.query_selector(".voc-agenda-antetitulo").inner_text().strip()
                    titulo = art.query_selector(".voc-agenda-titulo2").inner_text().strip()
                    descripcion_elem = art.query_selector(".voc-agenda-noticia")
                    descripcion = descripcion_elem.inner_text().strip() if descripcion_elem else "No disponible"
                    etiquetas = art.query_selector_all(".voc-agenda-lugar")
                    valores = art.query_selector_all(".voc-agenda-dia")
                    lugar = hora = precio = "No disponible"
                    fechas = []
                    for label, value in zip(etiquetas, valores):
                        texto = label.inner_text().lower()
                        contenido = value.inner_text().strip()
                        if "lugar" in texto:
                            lugar = contenido
                        elif "fecha" in texto:
                            fechas_detectadas = re.findall(r"\d{1,2} de \w+ de \d{4}", contenido)
                            if len(fechas_detectadas) == 1:
                                fecha = parse_spanish_date(fechas_detectadas[0])
                                if fecha:
                                    fechas.append(fecha)
                            elif len(fechas_detectadas) >= 2:
                                inicio = parse_spanish_date(fechas_detectadas[0])
                                fin = parse_spanish_date(fechas_detectadas[1])
                                if inicio and fin:
                                    fechas += [inicio + timedelta(days=d) for d in range((fin - inicio).days + 1)]
                        elif "hora" in texto:
                            hora = contenido
                        elif "precio" in texto:
                            precio = contenido
                    for fecha in fechas:
                        eventos.append({
                            "Categoría": categoria,
                            "Título": titulo,
                            "Descripción": descripcion,
                            "Lugar": lugar,
                            "Fecha": fecha.strftime("%Y-%m-%d"),
                            "Hora": hora,
                            "Precio": precio
                        })
                except Exception as e:
                    print("❌ Error ElCorreo:", e)
        browser.close()
    return eventos

# ========== VisitBiscay ==========
def expandir_fechas_visitbiscay(evento):
    try:
        inicio = datetime.strptime(evento.get("Fecha inicio"), "%d/%m/%Y")
    except:
        return []
    try:
        fin = datetime.strptime(evento.get("Fecha fin"), "%d/%m/%Y")
    except:
        fin = inicio
    return [{**evento, "Fecha": f.strftime("%Y-%m-%d")} for f in [inicio + timedelta(days=i) for i in range((fin - inicio).days + 1)]]

def scrape_visitbiscay():
    eventos = []
    base_url = "https://www.visitbiscay.eus/es/agenda?p_p_id=buscadorEventos_INSTANCE_9Usis7q2715w&_buscadorEventos_INSTANCE_9Usis7q2715w_cur={}"
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, slow_mo=30)
        page = browser.new_page()
        for page_num in range(1, 32):
            print(f"📄 VisitBiscay página {page_num}")
            page.goto(base_url.format(page_num))
            try:
                page.wait_for_selector(".card-evento", timeout=10000)
            except:
                print("⏳ Timeout esperando eventos.")
                break
            cards = page.query_selector_all(".card-evento")
            if not cards:
                print("✅ No se encontraron más tarjetas.")
                break
            for card in cards:
                try:
                    categoria = card.query_selector(".categoria.bgcolor")
                    titulo = card.query_selector("h4.card-title a")
                    lugar = card.query_selector('ul.tags.mb-3 li.tag-donde')
                    fecha_li = card.query_selector('li:has(.icon-calendar)')
                    hora_li = card.query_selector('li:has(.icon-horario)')
                    fechas = re.findall(r"\d{2}/\d{2}/\d{4}", fecha_li.inner_text()) if fecha_li else []
                    hora = hora_li.inner_text().split("·")[0].replace("Hora inicio", "").strip() if hora_li else "No disponible"
                    evento = {
                        "Categoría": categoria.inner_text().strip() if categoria else "No disponible",
                        "Título": titulo.inner_text().strip() if titulo else "No disponible",
                        "Descripción": "",
                        "Lugar": lugar.inner_text().strip() if lugar else "No disponible",
                        "Fecha inicio": fechas[0] if len(fechas) > 0 else "01/01/1900",
                        "Fecha fin": fechas[1] if len(fechas) > 1 else fechas[0] if fechas else "01/01/1900",
                        "Hora": hora,
                        "Precio": "No disponible"
                    }
                    eventos.extend(expandir_fechas_visitbiscay(evento))
                except Exception as e:
                    print("❌ Error VisitBiscay:", e)
        browser.close()
    return eventos

# ========== Kulturklik ==========
def parsear_fecha_texto(texto):
    texto = texto.lower()
    meses = {
        "enero": "01", "febrero": "02", "marzo": "03", "abril": "04", "mayo": "05", "junio": "06", "julio": "07", "agosto": "08",
        "septiembre": "09", "octubre": "10", "noviembre": "11", "diciembre": "12"
    }
    fechas = re.findall(r"(\d{1,2}) de (\w+) (\d{4})", texto)
    resultado = []
    for dia, mes, anio in fechas:
        mes_num = meses.get(mes.strip(), "01")
        resultado.append(f"{int(dia):02d}/{mes_num}/{anio}")
    return resultado

def expandir_fechas_kulturklik(evento, fechas):
    try:
        inicio = datetime.strptime(fechas[0], "%d/%m/%Y")
    except:
        return []
    try:
        fin = datetime.strptime(fechas[1], "%d/%m/%Y") if len(fechas) > 1 else inicio
    except:
        fin = inicio
    return [{**evento, "Fecha": f.strftime("%Y-%m-%d")} for f in [inicio + timedelta(days=i) for i in range((fin - inicio).days + 1)]]

def scrape_kulturklik(url, max_eventos):
    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'html.parser')
    eventos = []
    total_procesados = 0
    for li in soup.select("li"):
        if total_procesados >= max_eventos:
            break
        try:
            tipo = li.select_one("p.filtro")
            titulo = li.select_one("h4.clase_acortar a")
            detalles = li.select("ul.detalles li.clase_acortar")
            lugar = detalles[0].text.strip() if len(detalles) > 0 else "No disponible"
            hora = detalles[1].text.strip() if len(detalles) > 1 else "No disponible"
            fecha_texto = detalles[2].text.strip() if len(detalles) > 2 else "No disponible"
            fechas = parsear_fecha_texto(fecha_texto)
            evento = {
                "Categoría": tipo.text.strip() if tipo else "No disponible",
                "Título": titulo.text.strip() if titulo else "No disponible",
                "Descripción": "",
                "Lugar": lugar,
                "Hora": hora,
                "Precio": "No disponible"
            }
            eventos_expandidos = expandir_fechas_kulturklik(evento, fechas)
            if eventos_expandidos:
                eventos.extend(eventos_expandidos)
                total_procesados += 1
        except Exception as e:
            print("❌ Error Kulturklik:", e)
    return eventos

# ========== MAIN ==========
def ejecutar_scraping():
    print("🕒 Ejecutando scraping...")
    eventos1 = scrape_elcorreo()
    print(f"✔️ ElCorreo     : {len(eventos1)} eventos")

    eventos2 = scrape_visitbiscay()
    print(f"✔️ VisitBiscay  : {len(eventos2)} eventos")

    eventos3 = scrape_kulturklik(
        "https://www.kulturklik.euskadi.eus/webkklik00-shagenda/es/aa58aPublicoWar/agenda/sacarAgendaDia?eventosParaFecha=08/05/2025&locale=es",
        max_eventos=200
    )
    print(f"✔️ Kulturklik   : {len(eventos3)} eventos")

    df = pd.DataFrame(eventos1 + eventos2 + eventos3)
    columnas = ["Categoría", "Título", "Descripción", "Lugar", "Fecha", "Hora", "Precio"]
    df = df[columnas]

    # Cargar eventos anteriores si existen
    archivo_salida = "eventos_unificados.csv"
    if os.path.exists(archivo_salida):
        df_anterior = pd.read_csv(archivo_salida)
        # Identificar eventos nuevos
        merged = df.merge(df_anterior, on=["Título", "Fecha"], how="left", indicator=True)
        nuevos_eventos = merged[merged["_merge"] == "left_only"]
        if not nuevos_eventos.empty:
            nuevos_eventos[columnas].to_csv("eventos_nuevos.csv", index=False)
            print(f"📥 Nuevos eventos encontrados: {len(nuevos_eventos)}")
        else:
            print("📭 No hay eventos nuevos.")
    else:
        # Primera vez: guardar todos como nuevos también
        df.to_csv("eventos_nuevos.csv", index=False)
        print(f"📥 Todos los eventos se consideran nuevos: {len(df)}")

    # Guardar todos los eventos
    df.to_csv(archivo_salida, index=False)
    print(f"✅ Total eventos guardados: {len(df)} en '{archivo_salida}'")

if __name__ == "__main__":
    while True:
        try:
            ejecutar_scraping()
        except Exception as e:
            print(f"❌ Error durante el scraping: {e}")
        print("⏳ Esperando 24 horas para la próxima ejecución...")
        time.sleep(24 * 60 * 60)

