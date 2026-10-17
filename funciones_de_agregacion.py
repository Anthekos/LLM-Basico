"""Operaciones de agregacion orientadas a objetos."""

from abc import ABC, abstractmethod

import numpy as np


class FuncionAgregacion(ABC):
    @abstractmethod
    def __call__(
        self, entradas: np.ndarray, pesos: np.ndarray, sesgo: np.ndarray
    ) -> np.ndarray:
        """Combina entradas, pesos y sesgo."""


class SumaPonderada(FuncionAgregacion):
    def __call__(
        self, entradas: np.ndarray, pesos: np.ndarray, sesgo: np.ndarray
    ) -> np.ndarray:
        return entradas @ pesos + sesgo
