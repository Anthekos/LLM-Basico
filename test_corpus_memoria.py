"""Pruebas de persistencia separada y memorias documentales locales."""

from io import BytesIO
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import servidor_web
from generador_pdf import crear_pdf_memoria
from pdf_a_corpus import sanitizar_texto
from servidor_web import ModeloWeb
from tokenizador_basico import TokenizadorBasico
from pypdf import PdfReader


class PruebasCorpusMemoria(unittest.TestCase):
    def crear_modelo(self, directorio: Path) -> tuple[ModeloWeb, dict[str, object]]:
        rutas: dict[str, object] = {
            "RUTA_CORPUS_MAXIMO": directorio / "corpus_maximo.txt",
            "RUTA_CORPUS": directorio / "corpus_local.txt",
            "RUTA_CORPUS_ANTIGUO": directorio / "corpus.txt",
            "RUTA_PDFS_CORPUS": directorio / "pdfs_corpus",
            "RUTA_MANIFIESTO_PDFS": directorio / "pdfs_corpus.json",
            "RUTA_CONFIG_MODELO": directorio / "config.json",
            "RUTA_FEEDBACK": directorio / "feedback.jsonl",
        }
        Path(rutas["RUTA_CORPUS_MAXIMO"]).write_text(
            "Usuario: Saludo base.\nAsistente: Respuesta base.\n",
            encoding="utf-8",
        )
        parche = patch.multiple(servidor_web, **rutas)
        parche.start()
        self.addCleanup(parche.stop)
        modelo = ModeloWeb()
        return modelo, rutas

    def test_quitar_concepto_local_no_modifica_corpus_maximo(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            directorio = Path(temporal)
            modelo, rutas = self.crear_modelo(directorio)
            base_antes = Path(rutas["RUTA_CORPUS_MAXIMO"]).read_text(encoding="utf-8")
            modelo.agregar_concepto("Concepto de prueba para borrar.")
            with patch.object(modelo, "_entrenar", return_value=modelo.modelo):
                modelo.quitar_concepto("Concepto de prueba para borrar.")

            self.assertNotIn("Concepto de prueba para borrar.", modelo.corpus_local)
            self.assertEqual(
                Path(rutas["RUTA_CORPUS_MAXIMO"]).read_text(encoding="utf-8"),
                base_antes,
            )
            self.assertEqual(
                Path(rutas["RUTA_CORPUS"]).read_text(encoding="utf-8").strip(),
                modelo.corpus_local,
            )

    def test_pdf_de_memoria_se_reextrae_con_vocabulario_espanol_valido(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            ruta = crear_pdf_memoria(
                "Memoria de atención",
                "Usuario: ¿Qué hace la atención?\n"
                "Asistente: La niña describe una relación entre palabras.",
                "nota-prueba.pdf",
                temporal,
            )
            paginas = PdfReader(str(ruta), strict=True).pages
            extraido = "\n".join(pagina.extract_text() or "" for pagina in paginas)
            sanitizado = sanitizar_texto(extraido, TokenizadorBasico())

            self.assertIn("atención", sanitizado.lower())
            self.assertIn("niña", sanitizado.lower())
            TokenizadorBasico().codificar(sanitizado)

    def test_pdf_se_agrega_y_eliminarlo_quita_su_texto_e_indice(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            directorio = Path(temporal)
            modelo, rutas = self.crear_modelo(directorio)
            contenido = crear_pdf_memoria(
                "Concepto",
                "Usuario: ¿Qué es una neurona?\n"
                "Asistente: Una neurona procesa una señal.",
                "nota.pdf",
                directorio,
            ).read_bytes()
            documento = modelo.agregar_pdf(contenido, "nota.pdf")
            identificador = str(documento["id"])
            texto_documento = str(modelo.pdfs[0]["text"])
            base_antes = Path(rutas["RUTA_CORPUS_MAXIMO"]).read_text(encoding="utf-8")
            self.assertIn(texto_documento, modelo.corpus_local)

            modelo.eliminar_pdf(identificador)

            self.assertNotIn(texto_documento, modelo.corpus_local)
            self.assertFalse(
                (Path(rutas["RUTA_PDFS_CORPUS"]) / f"{identificador}.pdf").exists()
            )
            manifiesto = json.loads(
                Path(rutas["RUTA_MANIFIESTO_PDFS"]).read_text(encoding="utf-8")
            )
            self.assertEqual(manifiesto["pdfs"], [])
            self.assertEqual(
                Path(rutas["RUTA_CORPUS_MAXIMO"]).read_text(encoding="utf-8"),
                base_antes,
            )

    def test_feedback_guarda_recompensa_y_aplica_el_paso_online(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            directorio = Path(temporal)
            modelo, rutas = self.crear_modelo(directorio)
            with patch.object(modelo.modelo, "ajuste_online", return_value=1.0) as ajuste:
                modelo.registrar_recompensa("¿Qué es una neurona?", "Una célula.", 1)

            ajuste.assert_called_once()
            self.assertEqual(ajuste.call_args.kwargs["recompensa"], 1)
            registro = json.loads(
                Path(rutas["RUTA_FEEDBACK"]).read_text(encoding="utf-8").strip()
            )
            self.assertEqual(registro["reward"], 1)


if __name__ == "__main__":
    unittest.main()
