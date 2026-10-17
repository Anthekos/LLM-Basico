"""Funciones de activacion orientadas a objetos."""

from abc import ABC, abstractmethod

import numpy as np


class FuncionActivacion(ABC):
    nombre: str

    @abstractmethod
    def __call__(self, valores: np.ndarray) -> np.ndarray:
        """Aplica la funcion a los valores."""

    @abstractmethod
    def derivada(self, valores: np.ndarray) -> np.ndarray:
        """Calcula la derivada respecto a la entrada de la funcion."""


class Sigmoide(FuncionActivacion):
    nombre = "sigmoide"

    def __call__(self, valores: np.ndarray) -> np.ndarray:
        valores_seguros = np.clip(valores, -500, 500)
        return 1.0 / (1.0 + np.exp(-valores_seguros))

    def derivada(self, valores: np.ndarray) -> np.ndarray:
        salida = self(valores)
        return salida * (1.0 - salida)


class Tanh(FuncionActivacion):
    nombre = "tanh"

    def __call__(self, valores: np.ndarray) -> np.ndarray:
        return np.tanh(valores)

    def derivada(self, valores: np.ndarray) -> np.ndarray:
        salida = self(valores)
        return 1.0 - salida**2


class ReLU(FuncionActivacion):
    nombre = "relu"

    def __call__(self, valores: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, valores)

    def derivada(self, valores: np.ndarray) -> np.ndarray:
        return (valores > 0).astype(float)


class Lineal(FuncionActivacion):
    nombre = "lineal"

    def __call__(self, valores: np.ndarray) -> np.ndarray:
        return valores

    def derivada(self, valores: np.ndarray) -> np.ndarray:
        return np.ones_like(valores)


def crear_activacion(nombre: str) -> FuncionActivacion:
    activaciones: dict[str, type[FuncionActivacion]] = {
        "sigmoide": Sigmoide,
        "tanh": Tanh,
        "relu": ReLU,
        "lineal": Lineal,
    }
    try:
        return activaciones[nombre]()
    except KeyError as error:
        raise ValueError(f"Activacion no reconocida: {nombre}") from error
