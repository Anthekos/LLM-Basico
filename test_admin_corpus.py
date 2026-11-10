"""Pruebas de autenticacion y edicion protegida del corpus máximo."""

from http.server import ThreadingHTTPServer
import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import patch

import servidor_web
from servidor_web import ModeloWeb, crear_manejador
from tokenizador_palabras import TokenizadorPalabras


class PruebasAdminCorpus(unittest.TestCase):
    def test_login_provisional_cambio_obligatorio_y_edicion_protegida(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            directorio = Path(temporal)
            archivo_maximo = directorio / "corpus_maximo.txt"
            archivo_maximo.write_text("Base previa.\n", encoding="utf-8")
            archivo_credenciales = directorio / "admin_credentials.json"
            modelo = ModeloWeb.__new__(ModeloWeb)
            modelo.corpus_maximo = "Base previa."
            modelo.core_fundacional = "Base previa."
            modelo.core_personalidad = "Personalidad previa."
            modelo.corpus_local = "Nota local."
            modelo.corpus = "Base previa.\n\nNota local."
            modelo.pdfs = []
            modelo.longitud_contexto = 8
            modelo.tokenizador = TokenizadorPalabras("base previa personalidad")
            modelo.bloqueo = threading.Lock()
            modelo.modelo = type("ModeloEntrenado", (), {
                "tokenizador": modelo.tokenizador,
            })()

            with (
                patch.object(servidor_web, "RUTA_CORPUS_MAXIMO", archivo_maximo),
                patch.object(
                    servidor_web,
                    "RUTA_CREDENCIALES_ADMIN",
                    archivo_credenciales,
                ),
                patch.object(
                    modelo,
                    "_entrenar",
                    return_value=modelo.modelo,
                ) as entrenar,
            ):
                servidor = ThreadingHTTPServer(
                    ("127.0.0.1", 0), crear_manejador(modelo)
                )
                hilo = threading.Thread(target=servidor.serve_forever, daemon=True)
                hilo.start()
                base_url = f"http://127.0.0.1:{servidor.server_port}"

                def solicitar(
                    ruta: str,
                    metodo: str = "GET",
                    datos: dict[str, str] | None = None,
                    token: str | None = None,
                ) -> tuple[int, dict[str, object]]:
                    cabeceras: dict[str, str] = {}
                    cuerpo = None
                    if datos is not None:
                        cabeceras["Content-Type"] = "application/json"
                        cuerpo = json.dumps(datos).encode("utf-8")
                    if token:
                        cabeceras["Authorization"] = f"Bearer {token}"
                    solicitud = Request(
                        f"{base_url}{ruta}",
                        data=cuerpo,
                        headers=cabeceras,
                        method=metodo,
                    )
                    try:
                        with urlopen(solicitud) as respuesta:
                            return respuesta.status, json.load(respuesta)
                    except HTTPError as error:
                        with error:
                            return error.code, json.load(error)

                try:
                    estado, _ = solicitar("/api/admin/corpus-maximo")
                    self.assertEqual(estado, 401)

                    estado, login = solicitar(
                        "/api/admin/login",
                        "POST",
                        {"username": "admin", "password": "admin"},
                    )
                    self.assertEqual(estado, 200)
                    self.assertTrue(login["must_change_password"])
                    token = str(login["token"])

                    estado, _ = solicitar(
                        "/api/admin/corpus-maximo",
                        token=token,
                    )
                    self.assertEqual(estado, 403)
                    estado, _ = solicitar(
                        "/api/admin/corpus-maximo",
                        "PUT",
                        {"text": "Cambio no autorizado."},
                        token,
                    )
                    self.assertEqual(estado, 403)

                    estado, _ = solicitar(
                        "/api/admin/password",
                        "POST",
                        {"new_password": "Nueva-clave-segura-2026"},
                        token,
                    )
                    self.assertEqual(estado, 200)

                    estado, corpus = solicitar(
                        "/api/admin/corpus-maximo",
                        token=token,
                    )
                    self.assertEqual(estado, 200)
                    self.assertEqual(corpus["core_fundacional"], "Base previa.")
                    self.assertEqual(corpus["core_personalidad"], "Personalidad previa.")
                    estado, _ = solicitar(
                        "/api/admin/corpus-maximo",
                        "PUT",
                        {"text": "Cambio no autorizado."},
                    )
                    self.assertEqual(estado, 401)

                    estado, _ = solicitar(
                        "/api/admin/corpus-maximo",
                        "PUT",
                        {
                            "core_fundacional": (
                                "<|usuario|> ¿Qué es una base? <|pensamiento|> "
                                "Defino el concepto. <|asistente|> Una base organiza "
                                "los ejemplos. <|fin|>"
                            ),
                            "core_personalidad": (
                                "<|usuario|> ¿Qué haces? <|pensamiento|> "
                                "Reviso los datos. <|asistente|> Respondo con cuidado. "
                                "<|fin|>"
                            ),
                        },
                        token,
                    )
                    self.assertEqual(estado, 200)
                    self.assertEqual(
                        archivo_maximo.read_text(encoding="utf-8").strip(),
                        "<|usuario|> ¿Qué es una base? <|pensamiento|> "
                        "Defino el concepto. <|asistente|> Una base organiza "
                        "los ejemplos. <|fin|>\n\n"
                        "<|usuario|> ¿Qué haces? <|pensamiento|> Reviso los datos. "
                        "<|asistente|> Respondo con cuidado. <|fin|>\n\n"
                        "<|usuario|> ¿Qué haces? <|pensamiento|> Reviso los datos. "
                        "<|asistente|> Respondo con cuidado. <|fin|>\n\n"
                        "<|usuario|> ¿Qué haces? <|pensamiento|> Reviso los datos. "
                        "<|asistente|> Respondo con cuidado. <|fin|>",
                    )
                    self.assertEqual(
                        (directorio / "core_fundacional.txt").read_text().strip(),
                        "<|usuario|> ¿Qué es una base? <|pensamiento|> "
                        "Defino el concepto. <|asistente|> Una base organiza "
                        "los ejemplos. <|fin|>",
                    )
                    self.assertEqual(
                        (directorio / "core_personalidad.txt").read_text().strip(),
                        "<|usuario|> ¿Qué haces? <|pensamiento|> Reviso los datos. "
                        "<|asistente|> Respondo con cuidado. <|fin|>",
                    )
                    self.assertEqual(modelo.corpus_local, "Nota local.")
                    entrenar.assert_called_once()

                    datos_credenciales = archivo_credenciales.read_text(
                        encoding="utf-8"
                    )
                    self.assertNotIn('"password":', datos_credenciales)
                    self.assertEqual(
                        json.loads(datos_credenciales)["must_change_password"],
                        False,
                    )
                    self.assertEqual(
                        archivo_credenciales.stat().st_mode & 0o077,
                        0,
                    )
                finally:
                    servidor.shutdown()
                    servidor.server_close()
                    hilo.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
