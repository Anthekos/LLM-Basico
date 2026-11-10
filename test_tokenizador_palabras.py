"""Pruebas del tokenizador por palabras frecuentes."""

import unittest

from tokenizador_palabras import TokenizadorPalabras


class PruebasTokenizadorPalabras(unittest.TestCase):
    def test_vocabulario_limitado_incluye_desconocido_pad_y_puntuacion(self) -> None:
        texto = ("hola niña. ¿qué tal? " * 20) + "palabraúnica"
        texto_estructural = (
            "<|usuario|> Hola. <|pensamiento|> Reviso el contexto. "
            "<|asistente|> Respondo. <|fin|>"
        )
        tokenizador = TokenizadorPalabras(texto + texto_estructural)

        self.assertLessEqual(tokenizador.tamano_vocabulario, 1500)
        self.assertIn("<UNK>", tokenizador.token_a_id)
        self.assertIn("<PAD>", tokenizador.token_a_id)
        self.assertIn("?", tokenizador.token_a_id)
        self.assertIn(":", tokenizador.token_a_id)
        self.assertEqual(
            [tokenizador.token_a_id[token] for token in tokenizador.TOKENS_ESPECIALES],
            [2, 3, 4, 5],
        )
        ids_estructurales = tokenizador.codificar(texto_estructural)
        self.assertNotIn(tokenizador.id_unk, ids_estructurales)
        self.assertEqual(tokenizador.decodificar(ids_estructurales), texto_estructural)
        ids = tokenizador.codificar("palabra-no-vista")
        self.assertIn(tokenizador.id_unk, ids)
        self.assertEqual(tokenizador.decodificar([tokenizador.id_pad]), "")

    def test_decode_conserva_espaciado_y_puntuacion_espanola(self) -> None:
        tokenizador = TokenizadorPalabras("¿Qué tal, niña? Sí.")
        ids = tokenizador.codificar("¿Qué tal, niña? Sí.")

        self.assertEqual(tokenizador.decodificar(ids), "¿Qué tal, niña? Sí.")


if __name__ == "__main__":
    unittest.main()
