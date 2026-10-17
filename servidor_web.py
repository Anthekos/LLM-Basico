"""Servidor local para la interfaz web del modelo de caracteres."""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
from pypdf import PdfReader, PdfWriter
from pypdf.generic import (
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
)
from pypdf.errors import PdfReadError

from atencion_causal import ModeloAtencionCausal
from embedding_basico import EmbeddingBasico
from pdf_a_corpus import sanitizar_texto
from tokenizador_basico import TokenizadorBasico


DIRECTORIO_WEB = Path(__file__).parent / "web"
RUTA_CORPUS = Path(__file__).parent / "corpus.txt"
MAXIMO_PDF_BYTES = 12 * 1024 * 1024
MAXIMO_CARACTERES_CHAT = 1200
TEXTO_INICIAL = (
    "Una red neuronal aprende relaciones entre datos mediante ejemplos. "
    "Cada capa transforma una representacion y ajusta sus pesos para reducir "
    "el error. La atencion causal permite que cada caracter consulte el "
    "contexto anterior sin mirar el futuro. Un modelo de lenguaje pequeno "
    "predice el siguiente caracter usando probabilidades aprendidas. "
) * 3

EJEMPLOS = {
    "red-neuronal": {
        "titulo": "Introduccion a redes neuronales",
        "descripcion": "Un texto corto sobre capas, pesos y aprendizaje.",
        "texto": (
            "Una red neuronal es un modelo que aprende patrones a partir de "
            "ejemplos. Una capa recibe valores, los combina con pesos y aplica "
            "una funcion de activacion. Durante el entrenamiento, la red "
            "compara su respuesta con el objetivo y ajusta sus pesos para "
            "reducir el error. Las capas ocultas aprenden representaciones "
            "intermedias. Una red de caracteres puede aprender a predecir "
            "cual sera el siguiente caracter de una secuencia."
        ),
    },
    "atencion-causal": {
        "titulo": "Atencion causal y lenguaje",
        "descripcion": "Un ejemplo simple de contexto y prediccion.",
        "texto": (
            "Un modelo de lenguaje predice el siguiente elemento de una "
            "secuencia. La atencion causal compara cada posicion con las "
            "posiciones anteriores. Una mascara impide consultar el futuro. "
            "Las consultas, las llaves y los valores permiten construir "
            "representaciones del contexto. La salida se transforma en "
            "probabilidades y el muestreo selecciona el siguiente caracter."
        ),
    },
}


class ModeloWeb:
    """Agrupa el corpus, el modelo y la sincronizacion de inferencia."""

    def __init__(self) -> None:
        self.tokenizador = TokenizadorBasico()
        self.bloqueo = threading.Lock()
        self.corpus = TEXTO_INICIAL
        if RUTA_CORPUS.is_file():
            texto_guardado = RUTA_CORPUS.read_text(encoding="utf-8")
            if texto_guardado.strip():
                self.corpus += "\n" + sanitizar_texto(texto_guardado, self.tokenizador)
        self.modelo = self._entrenar(self.corpus)

    @staticmethod
    def _entrenar(corpus: str) -> ModeloAtencionCausal:
        tokenizador = TokenizadorBasico()
        inicializador = EmbeddingBasico(
            tokenizador=tokenizador,
            dimension=12,
            semilla=17,
        )
        inicializador.entrenar_inicializacion_red(
            corpus, epocas=8, tasa_aprendizaje=0.03
        )
        modelo = ModeloAtencionCausal(
            inicializador=inicializador,
            longitud_contexto=32,
            semilla=17,
        )
        modelo.entrenar_map(
            corpus, epocas=5, tasa_aprendizaje=0.02, tamano_lote=128
        )
        return modelo

    def agregar_pdf(self, datos: bytes) -> str:
        try:
            lector = PdfReader(BytesIO(datos), strict=True)
        except (PdfReadError, OSError, ValueError) as error:
            raise ValueError(f"No se pudo leer el PDF: {error}") from error
        if len(lector.pages) > 30:
            raise ValueError("El PDF supera el limite de 30 paginas.")
        texto = "\n\n".join(pagina.extract_text() or "" for pagina in lector.pages)
        if not texto.strip():
            raise ValueError(
                "El PDF no contiene texto seleccionable. Los PDF escaneados requieren OCR."
            )
        texto = sanitizar_texto(texto, self.tokenizador)
        if len(texto) > 10_000:
            raise ValueError(
                "El PDF supera el limite de 10000 caracteres de texto extraido."
            )
        corpus_actualizado = f"{self.corpus}\n\n{texto}"
        with self.bloqueo:
            modelo_actualizado = self._entrenar(corpus_actualizado)
            RUTA_CORPUS.write_text(corpus_actualizado + "\n", encoding="utf-8")
            self.corpus = corpus_actualizado
            self.modelo = modelo_actualizado
        return texto

    def responder(
        self,
        mensaje: str,
        historial: list[dict[str, str]],
        temperatura: float,
        top_k: int,
    ) -> str:
        if not 0.1 <= temperatura <= 2.0:
            raise ValueError("La temperatura debe estar entre 0.1 y 2.")
        if not 1 <= top_k <= 20:
            raise ValueError("Top-k debe estar entre 1 y 20.")
        partes: list[str] = []
        for turno in historial[-6:]:
            if turno.get("role") in {"user", "assistant"} and isinstance(
                turno.get("content"), str
            ):
                etiqueta = "Usuario" if turno["role"] == "user" else "Asistente"
                contenido = sanitizar_texto(turno["content"], self.tokenizador)
                partes.append(f"{etiqueta}: {contenido}\n")
        contenido_actual = sanitizar_texto(mensaje, self.tokenizador)
        partes.append(f"Usuario: {contenido_actual}\nAsistente:")
        inicio = "".join(partes)
        with self.bloqueo:
            respuesta = self.modelo.generar(
                inicio,
                cantidad=120,
                semilla=int(np.random.SeedSequence().generate_state(1)[0]),
                temperatura=temperatura,
                top_k=top_k,
                penalizacion_repeticion=1.15,
            )
        respuesta = respuesta[len(inicio):].split("\n", 1)[0].strip()
        return respuesta or "No pude completar una respuesta. Prueba con otra pregunta."


def crear_pdf_ejemplo(texto: str) -> bytes:
    """Crea un PDF pequeño con texto seleccionable para la galeria de ejemplos."""
    lineas = [texto[i:i + 82] for i in range(0, len(texto), 82)]
    comandos = ["BT", "/F1 11 Tf", "72 740 Td", "15 TL"]
    for linea in lineas:
        segura = linea.encode("ascii", errors="replace").decode("ascii")
        segura = segura.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        comandos.append(f"({segura}) Tj")
        comandos.append("T*")
    comandos.append("ET")

    writer = PdfWriter()
    pagina = writer.add_blank_page(width=612, height=792)
    fuente = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    pagina[NameObject("/Resources")] = DictionaryObject({
        NameObject("/Font"): DictionaryObject({NameObject("/F1"): fuente})
    })
    contenido = DecodedStreamObject()
    contenido.set_data("\n".join(comandos).encode("ascii"))
    pagina[NameObject("/Contents")] = writer._add_object(contenido)
    salida = BytesIO()
    writer.write(salida)
    return salida.getvalue()


def crear_manejador(modelo: ModeloWeb) -> type[BaseHTTPRequestHandler]:
    class ManejadorHTTP(BaseHTTPRequestHandler):
        server_version = "DemoLLMLocal/1.0"

        def log_message(self, formato: str, *argumentos: object) -> None:
            logging.info("%s - %s", self.address_string(), formato % argumentos)

        def _responder(self, estado: int, contenido: bytes, tipo: str) -> None:
            self.send_response(estado)
            self.send_header("Content-Type", tipo)
            self.send_header("Content-Length", str(len(contenido)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(contenido)

        def _json(self, estado: int, datos: dict[str, object]) -> None:
            contenido = json.dumps(datos, ensure_ascii=False).encode("utf-8")
            self._responder(estado, contenido, "application/json; charset=utf-8")

        def do_GET(self) -> None:
            ruta = unquote(urlparse(self.path).path)
            if ruta == "/api/examples":
                self._json(200, {
                    "examples": [
                        {"id": clave, "title": valor["titulo"],
                         "description": valor["descripcion"]}
                        for clave, valor in EJEMPLOS.items()
                    ]
                })
                return
            if ruta.startswith("/api/examples/") and ruta.endswith(".pdf"):
                clave = ruta.removeprefix("/api/examples/").removesuffix(".pdf")
                ejemplo = EJEMPLOS.get(clave)
                if ejemplo is None:
                    self._json(404, {"error": "No se encontro ese PDF de ejemplo."})
                    return
                pdf = crear_pdf_ejemplo(ejemplo["texto"])
                self._responder(200, pdf, "application/pdf")
                return
            archivo = "index.html" if ruta in {"/", "/index.html"} else ruta.lstrip("/")
            if archivo not in {"index.html", "styles.css", "app.js"}:
                self._json(404, {"error": "No se encontro la ruta solicitada."})
                return
            ruta_archivo = DIRECTORIO_WEB / archivo
            if not ruta_archivo.is_file():
                self._json(404, {"error": "No se encontro el archivo web solicitado."})
                return
            tipos = {
                ".html": "text/html; charset=utf-8",
                ".css": "text/css; charset=utf-8",
                ".js": "text/javascript; charset=utf-8",
            }
            self._responder(200, ruta_archivo.read_bytes(), tipos[ruta_archivo.suffix])

        def do_POST(self) -> None:
            ruta = urlparse(self.path).path
            longitud = self.headers.get("Content-Length")
            if longitud is None or not longitud.isdecimal():
                self._json(411, {"error": "Se requiere Content-Length valido."})
                return
            tamano = int(longitud)
            limite = MAXIMO_PDF_BYTES if ruta == "/api/pdf" else 256_000
            if tamano > limite:
                self._json(413, {"error": "El contenido supera el limite permitido."})
                return
            cuerpo = self.rfile.read(tamano)
            try:
                if ruta == "/api/chat":
                    self._chat(cuerpo)
                elif ruta == "/api/pdf":
                    self._pdf(cuerpo)
                else:
                    self._json(404, {"error": "No se encontro la ruta solicitada."})
            except (ValueError, json.JSONDecodeError) as error:
                self._json(400, {"error": str(error)})
            except Exception:
                logging.exception("Error interno procesando una solicitud local.")
                self._json(500, {"error": "Error interno al procesar la solicitud."})

        def _chat(self, cuerpo: bytes) -> None:
            if self.headers.get_content_type() != "application/json":
                self._json(415, {"error": "El chat requiere una solicitud JSON."})
                return
            solicitud = json.loads(cuerpo.decode("utf-8"))
            if not isinstance(solicitud, dict):
                raise ValueError("La solicitud de chat debe ser un objeto JSON.")
            mensaje = solicitud.get("message")
            historial = solicitud.get("history", [])
            if not isinstance(mensaje, str) or not mensaje.strip():
                raise ValueError("Escribe un mensaje antes de enviarlo.")
            if len(mensaje) > MAXIMO_CARACTERES_CHAT:
                raise ValueError("El mensaje supera el limite de 1200 caracteres.")
            if not isinstance(historial, list) or any(
                not isinstance(turno, dict) for turno in historial
            ):
                raise ValueError("El historial de chat no es valido.")
            temperatura = solicitud.get("temperature", 0.8)
            top_k = solicitud.get("top_k", 8)
            if not isinstance(temperatura, (int, float)) or isinstance(temperatura, bool):
                raise ValueError("La temperatura debe ser numerica.")
            if not isinstance(top_k, int) or isinstance(top_k, bool):
                raise ValueError("Top-k debe ser un entero.")
            respuesta = modelo.responder(mensaje, historial, float(temperatura), top_k)
            self._json(200, {"reply": respuesta})

        def _pdf(self, cuerpo: bytes) -> None:
            if self.headers.get_content_type() != "application/pdf":
                self._json(415, {"error": "Sube un archivo PDF valido."})
                return
            if not cuerpo.startswith(b"%PDF-"):
                raise ValueError("El archivo enviado no parece ser un PDF valido.")
            texto = modelo.agregar_pdf(cuerpo)
            self._json(200, {
                "message": f"PDF incorporado al corpus ({len(texto)} caracteres).",
                "characters": len(texto),
            })

        def log_error(self, formato: str, *argumentos: object) -> None:
            logging.error(formato, *argumentos)

    return ManejadorHTTP


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    print("Preparando el modelo local. La primera carga puede tardar unos segundos.")
    modelo = ModeloWeb()
    servidor = ThreadingHTTPServer(
        ("127.0.0.1", 8765), crear_manejador(modelo)
    )
    print("Interfaz lista en http://127.0.0.1:8765")
    print("El servidor escucha solo en este equipo. Presiona Ctrl+C para detenerlo.")
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        print("\nCerrando servidor local.")
    finally:
        servidor.server_close()


if __name__ == "__main__":
    main()
