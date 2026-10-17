"""Pruebas de conversion de PDF y muestreo de caracteres."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from muestreo import muestrear_token
from pdf_a_corpus import convertir_pdf_a_corpus, sanitizar_texto
from tokenizador_basico import TokenizadorBasico


class PruebasPipelineYChat(unittest.TestCase):
    def test_sanitizacion_genera_texto_tokenizable(self) -> None:
        tokenizador = TokenizadorBasico()
        limpio = sanitizar_texto("  Espan\u0303a 😀\r\n  ¡Qué tal!\t  ", tokenizador)
        self.assertEqual(limpio, "España\n¡Qué tal!")
        tokenizador.codificar(limpio)

    def test_pdf_sin_texto_seleccionable_reporta_ocr_requerido(self) -> None:
        with tempfile.TemporaryDirectory() as directorio:
            ruta_pdf = Path(directorio) / "vacio.pdf"
            ruta_salida = Path(directorio) / "corpus.txt"
            escritor = PdfWriter()
            escritor.add_blank_page(width=72, height=72)
            with ruta_pdf.open("wb") as archivo:
                escritor.write(archivo)

            with self.assertRaisesRegex(ValueError, "OCR"):
                convertir_pdf_a_corpus(ruta_pdf, ruta_salida)
            self.assertFalse(ruta_salida.exists())

    def test_top_k_limita_candidatos_y_no_repite_ngram(self) -> None:
        probabilidades = np.array([0.01, 0.02, 0.90, 0.07])
        generador = np.random.default_rng(12)
        muestras = {
            muestrear_token(
                probabilidades,
                historial=[0, 1, 0, 1],
                generador=generador,
                temperatura=0.7,
                top_k=2,
                penalizacion_repeticion=1.0,
                tamano_ngram_sin_repetir=3,
            )
            for _ in range(40)
        }
        self.assertTrue(muestras <= {2, 3})
        self.assertNotIn(0, muestras)
        self.assertLessEqual(len(muestras), 2)

    def test_conversion_pdf_escribe_corpus_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as directorio:
            ruta_pdf = Path(directorio) / "texto.pdf"
            ruta_salida = Path(directorio) / "corpus" / "corpus.txt"
            escritor = PdfWriter()
            pagina = escritor.add_blank_page(width=612, height=792)
            fuente = escritor._add_object(DictionaryObject({
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }))
            pagina[NameObject("/Resources")] = DictionaryObject({
                NameObject("/Font"): DictionaryObject({NameObject("/F1"): fuente})
            })
            contenido = DecodedStreamObject()
            contenido.set_data(b"BT /F1 12 Tf 72 720 Td (Hola nino.) Tj ET")
            pagina[NameObject("/Contents")] = escritor._add_object(contenido)
            with ruta_pdf.open("wb") as archivo:
                escritor.write(archivo)

            salida = convertir_pdf_a_corpus(ruta_pdf, ruta_salida)
            corpus = salida.read_text(encoding="utf-8")
            TokenizadorBasico().codificar(corpus)
            self.assertIn("Hola nino.", corpus)

    def test_muestreador_valida_parametros(self) -> None:
        with self.assertRaisesRegex(ValueError, "temperatura"):
            muestrear_token(np.array([0.4, 0.6]), [], np.random.default_rng(1), temperatura=0)
        with self.assertRaisesRegex(ValueError, "top_k"):
            muestrear_token(np.array([0.4, 0.6]), [], np.random.default_rng(1), top_k=0)


if __name__ == "__main__":
    unittest.main()
