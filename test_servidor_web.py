from io import BytesIO
import json
import threading
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from http.server import ThreadingHTTPServer

import numpy as np
from pypdf import PdfReader

import servidor_web
from servidor_web import EJEMPLOS, ModeloWeb, crear_manejador, crear_pdf_ejemplo
from tokenizador_basico import TokenizadorBasico
from tokenizador_palabras import TokenizadorPalabras
from formato_conversacion import ASISTENTE, FIN, PENSAMIENTO, USUARIO


class PruebasServidorWeb(unittest.TestCase):
    def test_modelo_web_acepta_tokenizador_palabras_y_modo_cuantizado(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            directorio = Path(temporal)
            with patch.multiple(
                servidor_web,
                RUTA_CORPUS_MAXIMO=directorio / "corpus_maximo.txt",
                RUTA_CORPUS=directorio / "corpus.txt",
                RUTA_CORPUS_ANTIGUO=directorio / "legacy.txt",
                RUTA_PDFS_CORPUS=directorio / "pdfs",
                RUTA_MANIFIESTO_PDFS=directorio / "pdfs.json",
                RUTA_CONFIG_MODELO=directorio / "config.json",
                RUTA_FEEDBACK=directorio / "feedback.jsonl",
            ):
                modelo = ModeloWeb(tokenizador_tipo="palabras", cuantizado=True)

        self.assertIsInstance(modelo.tokenizador, servidor_web.TokenizadorPalabras)
        self.assertTrue(modelo.modelo.cuantizado)
        probabilidades = modelo.modelo.probabilidades_siguiente("Hola, ¿qué tal?")
        self.assertTrue(np.isfinite(probabilidades).all())
        self.assertAlmostEqual(float(np.sum(probabilidades)), 1.0, delta=1e-6)

    def test_responder_salta_fin_de_linea_inicial_y_conserva_ventana(self) -> None:
        tokenizador = TokenizadorPalabras("hola Respuesta útil Texto posterior")

        class ModeloGenerador:
            longitud_contexto = 8

            def __init__(self) -> None:
                self.argumentos_generacion: dict[str, object] = {}

            def comprimir_contexto(self, ids: np.ndarray) -> np.ndarray:
                return np.zeros(3)

            def generar(self, inicio: str, **opciones: object) -> str:
                self.argumentos_generacion = opciones
                self.inicio_generacion = inicio
                return (
                    inicio
                    + f" {ASISTENTE} Respuesta útil {FIN} Texto posterior"
                )

        modelo = ModeloWeb.__new__(ModeloWeb)
        modelo.tokenizador = tokenizador
        modelo.bloqueo = threading.Lock()
        modelo.modelo = ModeloGenerador()

        respuesta, pensamiento, ajuste_aplicado = modelo.responder(
            "hola", [], 0.8, 8
        )

        self.assertEqual(respuesta, "Respuesta útil")
        self.assertEqual(pensamiento, "")
        self.assertFalse(ajuste_aplicado)
        self.assertTrue(modelo.modelo.argumentos_generacion["evitar_salto_inicial"])
        self.assertLessEqual(
            len(tokenizador.codificar(modelo.modelo.inicio_generacion)) + 1,
            modelo.modelo.longitud_contexto,
        )
        with patch.object(
            modelo.modelo,
            "generar",
            side_effect=lambda inicio, **_: inicio + f" {FIN}",
        ):
            with self.assertRaisesRegex(RuntimeError, "sin producir texto"):
                modelo.responder("hola", [], 0.8, 8)

    def test_responder_detiene_y_limpia_turnos_posteriores(self) -> None:
        tokenizador = TokenizadorPalabras("hola Reviso los datos Respuesta limpia")

        class ModeloGenerador:
            longitud_contexto = 32

            def generar(self, inicio: str, **opciones: object) -> str:
                self.opciones = opciones
                return (
                    inicio
                    + f" Reviso los datos. {ASISTENTE} Respuesta limpia. "
                    + f"{FIN} {USUARIO} texto que no debe aparecer"
                )

        modelo = ModeloWeb.__new__(ModeloWeb)
        modelo.tokenizador = tokenizador
        modelo.bloqueo = threading.Lock()
        modelo.modelo = ModeloGenerador()

        respuesta, pensamiento, _ = modelo.responder(
            "hola", [], 0.8, 8
        )

        self.assertEqual(pensamiento, "Reviso los datos.")
        self.assertEqual(respuesta, "Respuesta limpia.")
        self.assertEqual(
            modelo.modelo.opciones["stop_sequences"],
            (FIN, USUARIO),
        )
        self.assertTrue(modelo.modelo.opciones["evitar_salto_inicial"])

    def test_error_de_generacion_registra_traceback_en_terminal(self) -> None:
        class ModeloConFallo(ModeloWeb):
            def __init__(self) -> None:
                pass

            def responder(
                self,
                mensaje: str,
                historial: list[dict[str, str]],
                temperatura: float,
                top_k: int,
                ajuste_online: bool = False,
            ) -> tuple[str, str, bool]:
                raise RuntimeError("fallo exacto de generación")

        servidor = ThreadingHTTPServer(
            ("127.0.0.1", 0), crear_manejador(ModeloConFallo())
        )
        hilo = threading.Thread(target=servidor.serve_forever, daemon=True)
        hilo.start()
        solicitud = Request(
            f"http://127.0.0.1:{servidor.server_port}/api/chat",
            data=json.dumps({"message": "hola"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with self.assertLogs(level="ERROR") as registros:
                with self.assertRaises(HTTPError) as error:
                    urlopen(solicitud)
                self.assertEqual(error.exception.code, 500)
                error.exception.read()
                error.exception.close()
            salida = "\n".join(registros.output)
            self.assertIn("Traceback (most recent call last)", salida)
            self.assertIn("RuntimeError: fallo exacto de generación", salida)
        finally:
            servidor.shutdown()
            servidor.server_close()
            hilo.join(timeout=5)

    def test_api_chat_deja_ajuste_online_apagado_si_no_se_indica(self) -> None:
        llamadas_ajuste: list[bool] = []

        class ModeloConversacion(ModeloWeb):
            tokenizador = TokenizadorBasico()
            longitud_contexto = 32

            def responder(
                self,
                mensaje: str,
                historial: list[dict[str, str]],
                temperatura: float,
                top_k: int,
                ajuste_online: bool = False,
            ) -> tuple[str, str, bool]:
                llamadas_ajuste.append(ajuste_online)
                return "respuesta", "", ajuste_online

        servidor = ThreadingHTTPServer(
            ("127.0.0.1", 0), crear_manejador(ModeloConversacion())
        )
        hilo = threading.Thread(target=servidor.serve_forever, daemon=True)
        hilo.start()
        base_url = f"http://127.0.0.1:{servidor.server_port}"
        try:
            for ajuste in (None, True):
                datos: dict[str, object] = {"message": "hola"}
                if ajuste is not None:
                    datos["online_fine_tuning"] = ajuste
                solicitud = Request(
                    f"{base_url}/api/chat",
                    data=json.dumps(datos).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                )
                with urlopen(solicitud) as respuesta:
                    resultado = json.load(respuesta)
                self.assertEqual(resultado["online_fine_tuned"], ajuste is True)
            self.assertEqual(llamadas_ajuste, [False, True])
        finally:
            servidor.shutdown()
            servidor.server_close()
            hilo.join(timeout=5)

    def test_pdf_de_ejemplo_contiene_texto_admitido_por_el_tokenizador(self) -> None:
        tokenizador = TokenizadorBasico()
        for ejemplo in EJEMPLOS.values():
            contenido = crear_pdf_ejemplo(ejemplo["texto"])
            pagina = PdfReader(BytesIO(contenido), strict=True).pages[0]
            texto = pagina.extract_text()

            self.assertTrue(texto)
            self.assertGreater(len(tokenizador.codificar(texto)), 0)

    def test_pdf_cargado_se_guarda_y_queda_disponible_para_previsualizar(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            directorio = Path(temporal)
            corpus = directorio / "corpus.txt"
            pdfs = directorio / "pdfs"
            manifiesto = directorio / "pdfs.json"
            corpus.write_text("Texto anterior del corpus.", encoding="utf-8")
            with patch.multiple(
                servidor_web,
                RUTA_CORPUS_MAXIMO=directorio / "corpus_maximo.txt",
                RUTA_CORPUS=corpus,
                RUTA_CORPUS_ANTIGUO=directorio / "legacy.txt",
                RUTA_PDFS_CORPUS=pdfs,
                RUTA_MANIFIESTO_PDFS=manifiesto,
                RUTA_CONFIG_MODELO=directorio / "config.json",
                RUTA_FEEDBACK=directorio / "feedback.jsonl",
            ):
                modelo = ModeloWeb()
                contenido = crear_pdf_ejemplo(EJEMPLOS["red-neuronal"]["texto"])
                documento = modelo.agregar_pdf(contenido, "../red neuronal.pdf")

                self.assertEqual(documento["filename"], "red neuronal.pdf")
                self.assertEqual(len(modelo.listar_pdfs()), 2)
                detalle = modelo.detalle_pdf(str(documento["id"]))
                self.assertIsNotNone(detalle)
                self.assertIn("red neuronal", str(detalle["text"]).lower())
                self.assertEqual(
                    modelo.obtener_pdf(str(documento["id"])), contenido
                )

                reiniciado = ModeloWeb()
                self.assertEqual(len(reiniciado.listar_pdfs()), 2)
                self.assertEqual(
                    reiniciado.detalle_pdf(str(documento["id"]))["filename"],
                    "red neuronal.pdf",
                )
                self.assertIn(
                    "Texto previo",
                    str(reiniciado.detalle_pdf("corpus-local")["filename"]),
                )

    def test_rangos_amplios_de_muestreo_se_aceptan(self) -> None:
        self.assertEqual(servidor_web.TEMPERATURA_MAXIMA, 3.0)
        self.assertGreater(TokenizadorBasico().tamano_vocabulario, 20)

    def test_api_lista_documentos_y_entrega_su_vista_previa_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            directorio = Path(temporal)
            with patch.multiple(
                servidor_web,
                RUTA_CORPUS_MAXIMO=directorio / "corpus_maximo.txt",
                RUTA_CORPUS=directorio / "corpus.txt",
                RUTA_CORPUS_ANTIGUO=directorio / "legacy.txt",
                RUTA_PDFS_CORPUS=directorio / "pdfs",
                RUTA_MANIFIESTO_PDFS=directorio / "pdfs.json",
                RUTA_CONFIG_MODELO=directorio / "config.json",
                RUTA_FEEDBACK=directorio / "feedback.jsonl",
            ):
                modelo = ModeloWeb()
                servidor = ThreadingHTTPServer(
                    ("127.0.0.1", 0), crear_manejador(modelo)
                )
                hilo = threading.Thread(target=servidor.serve_forever, daemon=True)
                hilo.start()
                base_url = f"http://127.0.0.1:{servidor.server_port}"
                try:
                    with urlopen(f"{base_url}/api/model-config") as respuesta:
                        configuracion = json.load(respuesta)
                    self.assertEqual(configuracion["temperature_max"], 3.0)
                    self.assertEqual(
                        configuracion["top_k_max"],
                        modelo.tokenizador.tamano_vocabulario,
                    )
                    self.assertEqual(configuracion["tokenizer_type"], "palabras")
                    self.assertEqual(configuracion["context_window"], 32)
                    self.assertEqual(configuracion["salience_threshold"], 0.75)
                    self.assertEqual(configuracion["d_model"], modelo.d_model)
                    self.assertEqual(configuracion["num_heads"], modelo.num_heads)
                    self.assertFalse(configuracion["quantized"])

                    solicitud_contexto = Request(
                        f"{base_url}/api/model-config",
                        data=json.dumps({"context_window": 8}).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urlopen(solicitud_contexto) as respuesta:
                        actualizado = json.load(respuesta)
                    self.assertEqual(actualizado["context_window"], 8)
                    self.assertEqual(modelo.modelo.longitud_contexto, 8)
                    self.assertEqual(
                        json.loads((directorio / "config.json").read_text())[
                            "context_window"
                        ],
                        8,
                    )
                    solicitud_saliencia = Request(
                        f"{base_url}/api/model-config",
                        data=json.dumps({"salience_threshold": 0.6}).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urlopen(solicitud_saliencia) as respuesta:
                        actualizado = json.load(respuesta)
                    self.assertEqual(actualizado["salience_threshold"], 0.6)
                    self.assertEqual(
                        json.loads((directorio / "config.json").read_text())[
                            "salience_threshold"
                        ],
                        0.6,
                    )

                    contenido_pdf = crear_pdf_ejemplo(
                        EJEMPLOS["atencion-causal"]["texto"]
                    )
                    solicitud = Request(
                        f"{base_url}/api/pdf",
                        data=contenido_pdf,
                        headers={
                            "Content-Type": "application/pdf",
                            "X-File-Name": "atencion%20causal.pdf",
                        },
                    )
                    with urlopen(solicitud) as respuesta:
                        carga = json.load(respuesta)
                    identificador = carga["document"]["id"]

                    with urlopen(f"{base_url}/api/pdfs") as respuesta:
                        lista = json.load(respuesta)["pdfs"]
                    self.assertEqual(lista[0]["filename"], "atencion causal.pdf")

                    with urlopen(
                        f"{base_url}/api/pdfs/{identificador}/preview"
                    ) as respuesta:
                        detalle = json.load(respuesta)
                    texto_normalizado = " ".join(detalle["text"].lower().split())
                    self.assertIn("atencion causal", texto_normalizado)

                    with urlopen(f"{base_url}/api/pdfs/{identificador}.pdf") as respuesta:
                        self.assertEqual(respuesta.headers.get_content_type(), "application/pdf")
                        self.assertEqual(respuesta.read(), contenido_pdf)
                finally:
                    servidor.shutdown()
                    servidor.server_close()
                    hilo.join(timeout=5)
