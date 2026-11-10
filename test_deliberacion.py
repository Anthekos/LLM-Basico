"""Pruebas del formato de deliberacion y memoria por saliencia."""

import threading
import unittest
from unittest.mock import patch

import numpy as np

from memoria_rag import IndiceMemoriaRAG
from formato_conversacion import ASISTENTE, FIN, PENSAMIENTO, USUARIO
from servidor_web import ModeloWeb, separar_deliberacion
from tokenizador_palabras import TokenizadorPalabras


class PruebasDeliberacion(unittest.TestCase):
    def test_parser_separa_pensamiento_y_respuesta(self) -> None:
        pensamiento, respuesta = separar_deliberacion(
            f"{USUARIO} ¿Qué es un vector? {PENSAMIENTO} "
            f"Identifico que pide una definición breve. {ASISTENTE} "
            f"Un vector es una lista ordenada de valores. {FIN}"
        )

        self.assertEqual(pensamiento, "Identifico que pide una definición breve.")
        self.assertEqual(respuesta, "Un vector es una lista ordenada de valores.")

    def test_parser_conserva_respuesta_autorregresiva_sin_marcador(self) -> None:
        pensamiento, respuesta = separar_deliberacion(
            f"Una respuesta breve generada. {FIN} texto posterior"
        )

        self.assertEqual(pensamiento, "")
        self.assertEqual(respuesta, "Una respuesta breve generada.")

    def test_prompt_y_generacion_respetan_la_ventana_con_deliberacion(self) -> None:
        tokenizador = TokenizadorPalabras("mensaje anterior pregunta respuesta válida")

        class GeneradorFalso:
            longitud_contexto = 8

            def __init__(self) -> None:
                self.inicio = ""

            def comprimir_contexto(self, ids: np.ndarray) -> np.ndarray:
                return np.zeros(3)

            def generar(self, inicio: str, **_: object) -> str:
                self.inicio = inicio
                return (
                    inicio
                    + " Identifico la pregunta. "
                    f"{ASISTENTE} Respuesta válida. {FIN}"
                )

        modelo = ModeloWeb.__new__(ModeloWeb)
        modelo.tokenizador = tokenizador
        modelo.longitud_contexto = 8
        modelo.bloqueo = threading.Lock()
        modelo.modelo = GeneradorFalso()

        respuesta, pensamiento, _ = modelo.responder(
            "Una pregunta suficientemente larga para comprimir el historial",
            [{"role": "user", "content": "mensaje anterior " * 5}],
            0.8,
            8,
        )

        self.assertEqual(pensamiento, "Identifico la pregunta.")
        self.assertEqual(respuesta, "Respuesta válida.")
        self.assertLessEqual(
            len(tokenizador.codificar(modelo.modelo.inicio)),
            modelo.modelo.longitud_contexto - 1,
        )
        self.assertTrue(modelo.modelo.inicio.endswith(PENSAMIENTO))

    def test_novedad_lexica_y_guardado_sobre_umbral(self) -> None:
        indice = IndiceMemoriaRAG(
            "La neurona procesa señales y el modelo aprende patrones."
        )
        evaluacion = indice.evaluar(
            "Explica la topología de hiperparámetros",
            "La regularización bayesiana modifica la incertidumbre.",
        )
        self.assertGreater(evaluacion.puntuacion, 0.75)
        self.assertTrue(evaluacion.terminos_nuevos)

        modelo = ModeloWeb.__new__(ModeloWeb)
        modelo.corpus = "La neurona procesa señales y el modelo aprende patrones."
        modelo.umbral_saliencia = 0.75
        with patch.object(modelo, "agregar_concepto") as agregar:
            puntuacion, guardada = modelo.evaluar_y_guardar_memoria(
                "Explica la topología de hiperparámetros",
                "La regularización bayesiana modifica la incertidumbre.",
            )

        self.assertTrue(guardada)
        self.assertGreaterEqual(puntuacion, modelo.umbral_saliencia)
        agregar.assert_called_once()
        self.assertLessEqual(len(agregar.call_args.args[0]), 600)


if __name__ == "__main__":
    unittest.main()
