"""Pruebas pequenas del bloque de atencion causal en NumPy."""

import unittest
from unittest.mock import patch

import numpy as np

from atencion_causal import ModeloAtencionCausal
from embedding_basico import EmbeddingBasico
from cuantizacion import (
    decuantizar_tensor,
    multiplicar_cuantizado,
    cuantizar_tensor,
)
from tokenizador_basico import TokenizadorBasico
from tokenizador_palabras import TokenizadorPalabras


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
            self.inicializador, longitud_contexto=4, semilla=5, num_heads=3
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

    def test_atencion_multi_cabeza_conserva_forma_y_no_filtra_gradiente_futuro(self) -> None:
        ids = self.tokenizador.codificar("hola")
        logits, cache = self.modelo._adelante(
            np.stack((ids, ids)),
            np.ones((2, 4), dtype=bool),
        )
        self.assertEqual(logits.shape, (2, 4, self.tokenizador.tamano_vocabulario))
        self.assertEqual(cache["salida_atencion"].shape, (2, 4, self.modelo.d_model))

        id_futuro = self.tokenizador.caracter_a_id["ñ"]
        entrada = np.array(
            [[
                self.tokenizador.caracter_a_id["h"],
                self.tokenizador.caracter_a_id["o"],
                self.tokenizador.caracter_a_id["l"],
                id_futuro,
            ]]
        )
        validas = np.ones_like(entrada, dtype=bool)
        logits_base, _ = self.modelo._adelante(entrada, validas)
        embedding = self.modelo.embedding_contexto
        original = embedding[id_futuro, 0]
        epsilon = 1e-6
        embedding[id_futuro, 0] = original + epsilon
        logits_mas, _ = self.modelo._adelante(entrada, validas)
        embedding[id_futuro, 0] = original - epsilon
        logits_menos, _ = self.modelo._adelante(entrada, validas)
        embedding[id_futuro, 0] = original
        gradiente_futuro = (logits_mas[:, :3] - logits_menos[:, :3]) / (2 * epsilon)
        np.testing.assert_allclose(logits_base[:, :3], logits_mas[:, :3], atol=1e-12)
        np.testing.assert_allclose(gradiente_futuro, 0.0, atol=1e-10)

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
            (13, (0, 1)),
            (14, (self.tokenizador.caracter_a_id["a"], 0)),
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

    def test_generar_se_detiene_al_completar_cualquier_turno_de_usuario(self) -> None:
        tokenizador = TokenizadorPalabras("hola")
        tokenizador_original = self.modelo.tokenizador
        try:
            self.modelo.tokenizador = tokenizador
            for secuencia in ("<|fin|>", "<|usuario|>"):
                with self.subTest(secuencia=secuencia):
                    token_id = tokenizador.codificar(secuencia)[0]
                    probabilidades = np.zeros(tokenizador.tamano_vocabulario)
                    probabilidades[token_id] = 1.0
                    with patch.object(
                        self.modelo,
                        "probabilidades_siguiente",
                        return_value=probabilidades,
                    ) as predecir:
                        generado = self.modelo.generar(
                            "hola",
                            cantidad=4,
                            semilla=2,
                            stop_sequences=("<|fin|>", "<|usuario|>"),
                        )

                    self.assertEqual(generado, "hola")
                    self.assertEqual(predecir.call_count, 1)
        finally:
            self.modelo.tokenizador = tokenizador_original

    def test_memoria_comprimida_se_integra_en_inferencia_y_generacion(self) -> None:
        ids = self.tokenizador.codificar("hola niña. ¿qué tal? hola")
        memoria = self.modelo.comprimir_contexto(ids)

        self.assertEqual(memoria.shape, (self.modelo.dimension,))
        self.assertTrue(np.isfinite(memoria).all())
        probabilidades = self.modelo.probabilidades_siguiente(
            "tal", memoria_comprimida=memoria
        )
        self.assertTrue(np.isfinite(probabilidades).all())
        self.assertAlmostEqual(float(np.sum(probabilidades)), 1.0)
        with patch.object(self.modelo, "_adelante", wraps=self.modelo._adelante) as adelante:
            self.modelo.probabilidades_siguiente(
                "x" * 30, memoria_comprimida=memoria
            )
        self.assertLessEqual(
            adelante.call_args.args[0].shape[1],
            self.modelo.longitud_contexto,
        )
        self.assertEqual(
            len(self.modelo.generar(
                "tal", 3, semilla=7, memoria_comprimida=memoria
            )),
            6,
        )

    def test_ajuste_online_realiza_un_paso_finito(self) -> None:
        parametros = self.modelo._parametros()
        parametros_antes = [parametro.copy() for parametro in parametros]

        perdida = self.modelo.ajuste_online("Usuario: hola\nAsistente: niña")

        self.assertTrue(np.isfinite(perdida))
        self.assertTrue(all(np.isfinite(parametro).all() for parametro in parametros))
        self.assertTrue(all(
            np.isfinite(momento).all()
            for pares in self.modelo._momentos_ajuste_online
            for momento in pares
        ))
        self.assertTrue(any(
            not np.array_equal(antes, despues)
            for antes, despues in zip(parametros_antes, parametros)
        ))
        probabilidades = self.modelo.probabilidades_siguiente("Usuario: hola\n")
        self.assertTrue(np.isfinite(probabilidades).all())
        self.assertAlmostEqual(float(np.sum(probabilidades)), 1.0)
        generado = self.modelo.generar(
            "Usuario: hola\n",
            5,
            semilla=11,
            evitar_salto_inicial=True,
        )
        self.assertGreater(len(generado), len("Usuario: hola\n"))

    def test_ajuste_online_recorta_gradientes_extremos_sin_desbordar(self) -> None:
        gradientes = [np.zeros_like(parametro) for parametro in self.modelo._parametros()]
        gradientes[0][0, 0] = 1e308
        with patch.object(
            self.modelo,
            "_gradientes_lote",
            return_value=(1.0, gradientes),
        ):
            perdida = self.modelo.ajuste_online("hola niña")

        self.assertEqual(perdida, 1.0)
        self.assertTrue(all(
            np.isfinite(parametro).all() for parametro in self.modelo._parametros()
        ))

    def test_ajuste_online_rechaza_gradiente_nan_sin_modificar_pesos(self) -> None:
        parametros = self.modelo._parametros()
        parametros_antes = [parametro.copy() for parametro in parametros]
        gradientes = [np.zeros_like(parametro) for parametro in parametros]
        gradientes[0][0, 0] = np.nan
        with patch.object(
            self.modelo,
            "_gradientes_lote",
            return_value=(1.0, gradientes),
        ):
            with self.assertRaises(FloatingPointError):
                self.modelo.ajuste_online("hola niña")

        for antes, despues in zip(parametros_antes, parametros):
            np.testing.assert_array_equal(antes, despues)
        self.assertEqual(self.modelo._paso_ajuste_online, 0)

    def test_generacion_excluye_salto_de_linea_inicial(self) -> None:
        probabilidades = np.zeros(self.tokenizador.tamano_vocabulario)
        probabilidades[self.tokenizador.caracter_a_id["\n"]] = 0.99
        probabilidades[self.tokenizador.caracter_a_id["a"]] = 0.01
        with patch.object(
            self.modelo,
            "probabilidades_siguiente",
            return_value=probabilidades,
        ):
            resultado = self.modelo.generar(
                "h",
                1,
                semilla=1,
                top_k=None,
                penalizacion_repeticion=1.0,
                tamano_ngram_sin_repetir=0,
                evitar_salto_inicial=True,
            )

        self.assertEqual(resultado, "ha")

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

    def test_cuantizacion_int8_mantiene_error_menor_al_dos_por_ciento(self) -> None:
        generador = np.random.default_rng(17)
        entrada = generador.normal(0.0, 0.4, (32, 48)).astype(np.float32)
        pesos = generador.normal(0.0, 0.2, (48, 24)).astype(np.float32)
        pesos_int8, escala_pesos = cuantizar_tensor(pesos)
        aproximado = multiplicar_cuantizado(entrada, pesos_int8, escala_pesos)
        referencia = entrada @ pesos
        error_relativo = np.linalg.norm(aproximado - referencia) / np.linalg.norm(referencia)

        self.assertLess(error_relativo, 0.02)
        restaurado = decuantizar_tensor(pesos_int8, escala_pesos)
        self.assertEqual(restaurado.dtype, np.float32)

    def test_modo_cuantizado_usa_copias_y_no_altera_los_pesos(self) -> None:
        contexto = "hola"
        probabilidades_float = self.modelo.probabilidades_siguiente(contexto)
        parametros_antes = [parametro.copy() for parametro in self.modelo._parametros()]
        self.modelo.habilitar_cuantizacion()
        self.assertTrue(self.modelo.cuantizado)
        self.assertTrue(all(
            np.array_equal(antes, despues)
            for antes, despues in zip(parametros_antes, self.modelo._parametros())
        ))
        ids = self.tokenizador.codificar(contexto)[None, :]
        mascara = np.ones_like(ids, dtype=bool)
        cuantizado, _ = self.modelo._adelante(ids, mascara)
        cuantizado_referencia, _ = self.modelo._adelante(ids, mascara, self.modelo._parametros())
        self.assertEqual(cuantizado.shape, cuantizado_referencia.shape)
        self.assertTrue(np.isfinite(cuantizado).all())
        probabilidades_int8 = self.modelo.probabilidades_siguiente(contexto)
        error_inferencia = (
            np.linalg.norm(probabilidades_int8 - probabilidades_float)
            / np.linalg.norm(probabilidades_float)
        )
        self.assertLess(error_inferencia, 0.02)
        self.modelo.deshabilitar_cuantizacion()


if __name__ == "__main__":
    unittest.main()
