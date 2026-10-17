"""Tokenizador de caracteres para texto espanol."""

import unicodedata

import numpy as np


class TokenizadorBasico:
    """Mapea cada caracter admitido a un ID estable, sin tokenizacion por palabras."""

    ALFABETO = "abcdefghijklmnñopqrstuvwxyzáéíóúü"
    PUNTUACION = " .,;:!?¿¡'\"()[]{}-–—…\n\t"

    def __init__(self) -> None:
        caracteres = self.ALFABETO + self.ALFABETO.upper() + self.PUNTUACION
        self.caracteres = tuple(dict.fromkeys(caracteres))
        self.caracter_a_id = {caracter: indice for indice, caracter in enumerate(self.caracteres)}
        self.id_a_caracter = dict(enumerate(self.caracteres))

    @property
    def tamano_vocabulario(self) -> int:
        return len(self.caracteres)

    def normalizar(self, texto: str) -> str:
        return unicodedata.normalize("NFC", texto)

    def codificar(self, texto: str) -> np.ndarray:
        texto = self.normalizar(texto)
        desconocidos = sorted(set(texto).difference(self.caracter_a_id))
        if desconocidos:
            lista = ", ".join(repr(caracter) for caracter in desconocidos)
            raise ValueError(f"Caracteres fuera del vocabulario espanol: {lista}")
        return np.fromiter(
            (self.caracter_a_id[caracter] for caracter in texto),
            dtype=np.int64,
            count=len(texto),
        )

    def decodificar(self, ids: np.ndarray | list[int]) -> str:
        resultado: list[str] = []
        for token_id in ids:
            indice = int(token_id)
            if indice not in self.id_a_caracter:
                raise ValueError(f"ID de caracter fuera del vocabulario: {indice}")
            resultado.append(self.id_a_caracter[indice])
        return "".join(resultado)

    def vectorizar(self, ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(ids, dtype=np.int64)
        if np.any(ids < 0) or np.any(ids >= self.tamano_vocabulario):
            raise ValueError("Hay IDs fuera del vocabulario.")
        return np.eye(self.tamano_vocabulario, dtype=float)[ids]
