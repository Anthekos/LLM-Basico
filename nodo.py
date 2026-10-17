"""Nodo/capa densa elemental implementada con arreglos de NumPy."""

import numpy as np

from funciones_de_activacion import FuncionActivacion
from funciones_de_agregacion import FuncionAgregacion


class Nodo:
    """Una capa densa: conecta todas sus entradas con todas sus salidas."""

    def __init__(
        self,
        cantidad_entradas: int,
        cantidad_salidas: int,
        activacion: FuncionActivacion,
        agregacion: FuncionAgregacion,
        generador: np.random.Generator,
    ) -> None:
        self.pesos = generador.normal(
            0.0, np.sqrt(2.0 / cantidad_entradas), (cantidad_entradas, cantidad_salidas)
        )
        self.sesgo = np.zeros((1, cantidad_salidas))
        self.activacion = activacion
        self.agregacion = agregacion
        self.entradas: np.ndarray | None = None
        self.pre_activacion: np.ndarray | None = None

    def adelante(self, entradas: np.ndarray) -> np.ndarray:
        self.entradas = entradas
        self.pre_activacion = self.agregacion(entradas, self.pesos, self.sesgo)
        return self.activacion(self.pre_activacion)

    def retropropagar(self, gradiente_pre_activacion: np.ndarray, tasa_aprendizaje: float) -> np.ndarray:
        if self.entradas is None:
            raise RuntimeError("Hay que ejecutar adelante() antes de retropropagar().")

        gradiente_entradas = gradiente_pre_activacion @ self.pesos.T
        gradiente_pesos = self.entradas.T @ gradiente_pre_activacion
        gradiente_sesgo = np.sum(gradiente_pre_activacion, axis=0, keepdims=True)
        self.pesos -= tasa_aprendizaje * gradiente_pesos
        self.sesgo -= tasa_aprendizaje * gradiente_sesgo
        return gradiente_entradas
