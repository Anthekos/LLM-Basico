"""Pruebas pequenas del bloque de atencion causal en NumPy."""

import unittest

import numpy as np

from atencion_causal import ModeloAtencionCausal
from embedding_basico import EmbeddingBasico
from tokenizador_basico import TokenizadorBasico


class PruebasAtencionCausal(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizador = TokenizadorBasico()
        self.inicializador = EmbeddingBasico(
            self.tokenizador, dimension=3, semilla=4
        )
        self.inicializador.entrenar_inicializacion_red(
            "hola niña. ¿qué tal?", epocas=3, tasa_aprendizaje=0.02
        )
        self.modelo = ModeloAtencionCausal(
            self.inicializador, longitud_contexto=4, semilla=5
        )

    def test_causalidad_no_usa_tokens_futuros(self) -> None:
        ids = self.tokenizador.codificar("niña")
        mascara = np.ones((1, 4), dtype=bool)
        logits_originales, _ = self.modelo._adelante(ids[None, :], mascara)

        modificados = ids.copy()
        modificados[3] = self.tokenizador.caracter_a_id["a"]
        logits_modificados, _ = self.modelo._adelante(modificados[None, :], mascara)
        np.testing.assert_allclose(
            logits_originales[:, :3], logits_modificados[:, :3], atol=1e-12
        )

    def test_gradientes_atencion_coinciden_con_diferencias_finitas(self) -> None:
        entradas, validas, objetivos = self.modelo._crear_ejemplos("hola")
        indices = np.array([1, 2])
        _, gradientes = self.modelo._gradientes_lote(
            entradas[indices], validas[indices], objetivos[indices]
        )
        parametros = self.modelo._parametros()
        pruebas = [
            (0, (self.tokenizador.caracter_a_id["h"], 0)),
            (1, (3, 1)),
            (2, (0, 0)),
            (3, (1, 2)),
            (4, (2, 1)),
            (5, (0, 1)),
            (6, (1,)),
            (7, (1, 0)),
            (8, (0,)),
            (9, (1,)),
            (10, (0,)),
            (11, (1,)),
            (12, (0,)),
            (13, (self.tokenizador.caracter_a_id["a"], 0)),
        ]
        epsilon = 1e-6
        for indice, posicion in pruebas:
            parametro = parametros[indice]
            original = parametro[posicion]
            parametro[posicion] = original + epsilon
            perdida_mas = self.modelo._gradientes_lote(
                entradas[indices], validas[indices], objetivos[indices]
            )[0]
            parametro[posicion] = original - epsilon
            perdida_menos = self.modelo._gradientes_lote(
                entradas[indices], validas[indices], objetivos[indices]
            )[0]
            parametro[posicion] = original
            numerico = (perdida_mas - perdida_menos) / (2 * epsilon)
            self.assertAlmostEqual(numerico, gradientes[indice][posicion], delta=2e-5)

    def test_inferencia_variacional_predice_y_genera(self) -> None:
        historial = self.modelo.inferencia_variacional(
            "hola niña. hola niña.", epocas=4, tamano_lote=8, muestras=2
        )
        self.assertEqual(len(historial), 4)
        self.assertTrue(np.isfinite(historial).all())
        probabilidades = self.modelo.probabilidades_siguiente(
            "hola", inferencia="variacional", muestras=8
        )
        self.assertAlmostEqual(float(np.sum(probabilidades)), 1.0)
        self.assertEqual(
            len(self.modelo.generar(
                "la", 5, semilla=2, inferencia="variacional",
                temperatura=0.7, top_k=5,
            )),
            7,
        )

    def test_entrenamiento_map_y_probabilidades(self) -> None:
        historial = self.modelo.entrenar_map(
            "hola niña. hola niña.", epocas=20, tamano_lote=8
        )
        self.assertLess(historial[-1], historial[0])
        probabilidades = self.modelo.probabilidades_siguiente("hola")
        self.assertEqual(probabilidades.shape, (self.tokenizador.tamano_vocabulario,))
        self.assertAlmostEqual(float(np.sum(probabilidades)), 1.0)
        self.assertEqual(
            len(self.modelo.generar(
                "la", 10, semilla=1, temperatura=0.9, top_k=6,
                penalizacion_repeticion=1.1, tamano_ngram_sin_repetir=3,
            )),
            12,
        )


if __name__ == "__main__":
    unittest.main()
