from io import BytesIO
import unittest

from pypdf import PdfReader

from servidor_web import EJEMPLOS, crear_pdf_ejemplo
from tokenizador_basico import TokenizadorBasico


class PruebasServidorWeb(unittest.TestCase):
    def test_pdf_de_ejemplo_contiene_texto_admitido_por_el_tokenizador(self) -> None:
        tokenizador = TokenizadorBasico()
        for ejemplo in EJEMPLOS.values():
            contenido = crear_pdf_ejemplo(ejemplo["texto"])
            pagina = PdfReader(BytesIO(contenido), strict=True).pages[0]
            texto = pagina.extract_text()

            self.assertTrue(texto)
            self.assertGreater(len(tokenizador.codificar(texto)), 0)
