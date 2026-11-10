"""Pruebas del núcleo modular de personalidad y el prompt de inferencia."""

from pathlib import Path
import threading
import unittest

from formato_conversacion import PENSAMIENTO
from servidor_web import ModeloWeb
from tokenizador_palabras import TokenizadorPalabras


DIRECTORIO = Path(__file__).parent


class PruebasPersonalidad(unittest.TestCase):
    def test_vocabulario_incluye_personalidad_y_tokens_estructurales(self) -> None:
        fundacional = (DIRECTORIO / "core_fundacional.txt").read_text(
            encoding="utf-8"
        )
        personalidad = (DIRECTORIO / "core_personalidad.txt").read_text(
            encoding="utf-8"
        )
        tokenizador = TokenizadorPalabras(
            f"{fundacional}\n\n{personalidad}",
            max_vocabulario=1500,
        )

        self.assertLessEqual(tokenizador.tamano_vocabulario, 1500)
        for token in (
            "<|usuario|>",
            "<|pensamiento|>",
            "<|asistente|>",
            "<|fin|>",
            ":",
            "de",
            "la",
            "que",
            "y",
            "en",
            "no",
            "con",
            "por",
            "para",
        ):
            self.assertIn(token, tokenizador.token_a_id)
        for termino in ("sobrio", "daño", "límite", "evidencia"):
            self.assertIn(termino, tokenizador.token_a_id)
        ids_estructura = tokenizador.codificar(
            "<|usuario|> No conozco el significado de esa palabra. "
            "<|pensamiento|> No debo presentar una suposición como un hecho. "
            "<|asistente|> No lo sé con certeza. <|fin|>"
        )
        self.assertNotIn(tokenizador.id_unk, ids_estructura)
        for delimitador in tokenizador.TOKENS_ESPECIALES:
            ids = tokenizador.codificar(delimitador)
            self.assertEqual(ids.tolist(), [tokenizador.token_a_id[delimitador]])
            self.assertNotEqual(tokenizador.token_a_id[delimitador], tokenizador.id_unk)

    def test_corpus_pondera_core_personalidad_sobre_la_base(self) -> None:
        fundacional = "Base sintáctica."
        personalidad = (
            "<|usuario|> Pregunta <|pensamiento|> Análisis. "
            "<|asistente|> Respuesta fluida. <|fin|>"
        )

        corpus = ModeloWeb._componer_corpus_maximo(fundacional, personalidad)

        self.assertEqual(corpus.count(personalidad), 3)
        self.assertEqual(corpus.count(fundacional), 1)
        self.assertLess(corpus.index(fundacional), corpus.index(personalidad))

    def test_respuestas_avanzadas_superan_los_ejemplos_cavernicolas_repetidos(self) -> None:
        corpus_maximo = (DIRECTORIO / "corpus_maximo.txt").read_text(
            encoding="utf-8"
        )
        corpus_local = (DIRECTORIO / "corpus_local.txt").read_text(
            encoding="utf-8"
        )

        self.assertGreater(
            corpus_maximo.count("<|asistente|> La idea principal es"),
            corpus_local.count("Asistente: si david."),
        )
        self.assertEqual(corpus_maximo.count("En la Fase 4"), 3)
        self.assertEqual(corpus_maximo.count("En la Fase 5"), 3)
        self.assertNotIn("xkr;;", corpus_maximo)
        self.assertNotIn("casa en cabeza", corpus_maximo)
        self.assertGreater(
            corpus_maximo.count("<|asistente|> La idea principal es"),
            corpus_local.count(
                "Asistente: letras hacen palabras y palabras hacen casa en cabeza."
            ),
        )
        modelo = ModeloWeb.__new__(ModeloWeb)
        modelo.corpus_maximo = corpus_maximo
        corpus_entrenamiento = modelo._corpus_efectivo(corpus_local)
        self.assertNotIn("xkr;;", corpus_entrenamiento)
        self.assertNotIn("si david.", corpus_entrenamiento)
        self.assertIn("xkr;;", corpus_local)

    def test_pipeline_prefija_pensamiento_y_separa_la_respuesta_en_ventana(self) -> None:
        prompt_ejemplo = (
            "<|usuario|> ¿Qué haces si no conoces una respuesta? "
            "<|pensamiento|>"
        )
        texto_prueba = (
            prompt_ejemplo
            + " Analizo la pregunta y reviso qué datos conozco. "
            "<|asistente|> Responderé con claridad y admitiré mis dudas. <|fin|>"
        )
        tokenizador = TokenizadorPalabras(texto_prueba)
        ventana = 128

        class GeneradorDeterminista:
            longitud_contexto = ventana

            def __init__(self) -> None:
                self.inicio = ""

            def generar(self, inicio: str, **_: object) -> str:
                self.inicio = inicio
                return (
                    inicio
                    + " Analizo la pregunta y reviso qué datos conozco. "
                    "<|asistente|> Responderé con claridad y admitiré mis dudas. <|fin|>"
                )

        modelo = ModeloWeb.__new__(ModeloWeb)
        modelo.tokenizador = tokenizador
        modelo.longitud_contexto = ventana
        modelo.bloqueo = threading.Lock()
        modelo.modelo = GeneradorDeterminista()

        respuesta, pensamiento, _ = modelo.responder(
            "¿Qué haces si no conoces una respuesta?",
            [],
            temperatura=0.8,
            top_k=8,
        )

        prefijo = (
            "<|usuario|> ¿Qué haces si no conoces una respuesta? "
            "<|pensamiento|>"
        )
        self.assertTrue(modelo.modelo.inicio.endswith(prefijo))
        self.assertEqual(modelo.modelo.inicio[-len(PENSAMIENTO):], PENSAMIENTO)
        self.assertEqual(
            len(tokenizador.codificar(modelo.modelo.inicio)),
            len(tokenizador.codificar(prefijo)),
        )
        self.assertLessEqual(
            len(tokenizador.codificar(modelo.modelo.inicio)),
            ventana - 1,
        )
        self.assertEqual(
            pensamiento,
            "Analizo la pregunta y reviso qué datos conozco.",
        )
        self.assertEqual(
            respuesta,
            "Responderé con claridad y admitiré mis dudas.",
        )


if __name__ == "__main__":
    unittest.main()
