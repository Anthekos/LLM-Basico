"""Red neuronal densa pequena con entrenamiento binario y multiclase."""

import numpy as np

from funciones_de_activacion import FuncionActivacion, crear_activacion
from funciones_de_agregacion import SumaPonderada
from nodo import Nodo


class RedNeuronal:
    def __init__(
        self,
        tamanos: list[int],
        semilla: int = 7,
        activacion_salida: str = "sigmoide",
    ) -> None:
        if len(tamanos) < 2 or any(tamano < 1 for tamano in tamanos):
            raise ValueError("Se necesitan al menos dos capas con tamanos positivos.")
        if activacion_salida not in {"sigmoide", "lineal"}:
            raise ValueError("La activacion de salida debe ser 'sigmoide' o 'lineal'.")

        generador = np.random.default_rng(semilla)
        nombres_activacion = ["tanh"] * (len(tamanos) - 2) + [activacion_salida]
        self.capas = [
            Nodo(
                tamanos[indice],
                tamanos[indice + 1],
                crear_activacion(nombres_activacion[indice]),
                SumaPonderada(),
                generador,
            )
            for indice in range(len(tamanos) - 1)
        ]

    def predecir(self, entradas: np.ndarray) -> np.ndarray:
        salida = np.asarray(entradas, dtype=float)
        for capa in self.capas:
            salida = capa.adelante(salida)
        return salida

    def activaciones_por_capa(self, entradas: np.ndarray) -> list[np.ndarray]:
        activaciones = [np.asarray(entradas, dtype=float)]
        for capa in self.capas:
            activaciones.append(capa.adelante(activaciones[-1]))
        return activaciones

    def _validar_datos(
        self, entradas: np.ndarray, objetivos: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        entradas = np.asarray(entradas, dtype=float)
        objetivos = np.asarray(objetivos, dtype=float)
        if entradas.ndim != 2 or objetivos.ndim != 2 or len(entradas) != len(objetivos):
            raise ValueError("Entradas y objetivos deben ser matrices con igual cantidad de filas.")
        if entradas.shape[1] != self.capas[0].pesos.shape[0]:
            raise ValueError("La cantidad de columnas de entradas no coincide con la red.")
        if objetivos.shape[1] != self.capas[-1].pesos.shape[1]:
            raise ValueError("La cantidad de columnas de objetivos no coincide con la salida.")
        return entradas, objetivos

    def _retropropagar_ocultas(
        self, gradiente_entradas: np.ndarray, tasa_aprendizaje: float
    ) -> None:
        gradiente = gradiente_entradas
        for indice in range(len(self.capas) - 2, -1, -1):
            capa = self.capas[indice]
            if capa.pre_activacion is None:
                raise RuntimeError("Falta el estado de la capa oculta.")
            gradiente = gradiente * capa.activacion.derivada(capa.pre_activacion)
            gradiente = capa.retropropagar(gradiente, tasa_aprendizaje)

    def entrenar_epoca(
        self,
        entradas: np.ndarray,
        objetivos: np.ndarray,
        tasa_aprendizaje: float = 0.1,
    ) -> tuple[float, list[np.ndarray]]:
        entradas, objetivos = self._validar_datos(entradas, objetivos)
        if tasa_aprendizaje <= 0:
            raise ValueError("tasa_aprendizaje debe ser positiva.")

        activaciones = self.activaciones_por_capa(entradas)
        predicciones = activaciones[-1]
        predicciones_seguras = np.clip(predicciones, 1e-9, 1.0 - 1e-9)
        perdida = -np.mean(
            objetivos * np.log(predicciones_seguras)
            + (1.0 - objetivos) * np.log(1.0 - predicciones_seguras)
        )

        gradiente = (predicciones - objetivos) / objetivos.size
        gradiente_entradas = self.capas[-1].retropropagar(gradiente, tasa_aprendizaje)
        self._retropropagar_ocultas(gradiente_entradas, tasa_aprendizaje)
        return float(perdida), activaciones

    def entrenar_multiclase_epoca(
        self,
        entradas: np.ndarray,
        objetivos: np.ndarray,
        tasa_aprendizaje: float = 0.1,
    ) -> tuple[float, list[np.ndarray]]:
        """Entrena logits lineales con softmax y entropia cruzada categorica."""
        entradas, objetivos = self._validar_datos(entradas, objetivos)
        if tasa_aprendizaje <= 0:
            raise ValueError("tasa_aprendizaje debe ser positiva.")
        if self.capas[-1].activacion.nombre != "lineal":
            raise ValueError("El entrenamiento multiclase requiere salida lineal (logits).")
        if np.any(objetivos < 0) or not np.allclose(np.sum(objetivos, axis=1), 1.0):
            raise ValueError("Cada objetivo debe ser una distribucion categorica que sume 1.")

        activaciones = self.activaciones_por_capa(entradas)
        logits = activaciones[-1]
        logits_estables = logits - np.max(logits, axis=1, keepdims=True)
        exponenciales = np.exp(logits_estables)
        probabilidades = exponenciales / np.sum(exponenciales, axis=1, keepdims=True)
        log_normalizador = np.log(np.sum(exponenciales, axis=1, keepdims=True))
        perdida = -np.mean(np.sum(objetivos * (logits_estables - log_normalizador), axis=1))

        gradiente_logits = (probabilidades - objetivos) / len(entradas)
        gradiente_entradas = self.capas[-1].retropropagar(
            gradiente_logits, tasa_aprendizaje
        )
        self._retropropagar_ocultas(gradiente_entradas, tasa_aprendizaje)
        activaciones[-1] = probabilidades
        return float(perdida), activaciones

    def entrenar(
        self,
        entradas: np.ndarray,
        objetivos: np.ndarray,
        epocas: int = 5_000,
        tasa_aprendizaje: float = 0.1,
        verbose: bool = False,
    ) -> list[float]:
        entradas, objetivos = self._validar_datos(entradas, objetivos)
        if epocas < 1 or tasa_aprendizaje <= 0:
            raise ValueError("epocas y tasa_aprendizaje deben ser positivos.")

        historial: list[float] = []
        for etapa in range(1, epocas + 1):
            perdida, activaciones = self.entrenar_epoca(
                entradas, objetivos, tasa_aprendizaje
            )
            historial.append(perdida)
            if verbose:
                print(f"\nEtapa {etapa}/{epocas} | error: {perdida:.6f}")
                for indice, matriz in enumerate(activaciones):
                    print(f"Matriz de activacion {indice}:\n{np.array2string(matriz, precision=4)}")
                print("Pesos actualizados:")
                for indice, capa in enumerate(self.capas, start=1):
                    print(f"  Capa {indice}:\n{np.array2string(capa.pesos, precision=4)}")
        return historial

    def entrenar_multiclase(
        self,
        entradas: np.ndarray,
        objetivos: np.ndarray,
        epocas: int = 200,
        tasa_aprendizaje: float = 0.03,
    ) -> list[float]:
        entradas, objetivos = self._validar_datos(entradas, objetivos)
        if epocas < 1 or tasa_aprendizaje <= 0:
            raise ValueError("epocas y tasa_aprendizaje deben ser positivos.")

        historial: list[float] = []
        for _ in range(epocas):
            perdida, _ = self.entrenar_multiclase_epoca(
                entradas, objetivos, tasa_aprendizaje
            )
            historial.append(perdida)
        return historial
