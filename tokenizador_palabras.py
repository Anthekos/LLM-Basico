"""Tokenizador sencillo de palabras frecuentes y signos de puntuacion."""

from collections import Counter
import re
import unicodedata

import numpy as np


class TokenizadorPalabras:
    """Tokeniza palabras completas con vocabulario limitado y UNK/PAD."""

    TOKENS_ESPECIALES = (
        "<|usuario|>",
        "<|pensamiento|>",
        "<|asistente|>",
        "<|fin|>",
    )
    PATRON = re.compile(
        "|".join((
            *(re.escape(token) for token in TOKENS_ESPECIALES),
            r"\n",
            r"\t",
            r"[^\W\d_]+",
            r"\d+",
            r"[^\w\s]",
        )),
        re.UNICODE,
    )
    PUNTUACION = tuple(".,;:!?¿¡'\"()[]{}-–—…")
    CIERRES = set(".,;:!?%)]}»”’…")
    APERTURAS = set("¿¡([{«“‘")
    UNK = "<UNK>"
    PAD = "<PAD>"
    TOKENS_ESTRUCTURALES = (
        *TOKENS_ESPECIALES,
        "Usuario",
        "Humano",
        "Pensamiento",
        "Asistente",
        "usuario",
        "pensamiento",
        "asistente",
        "de",
        "la",
        "que",
        "y",
        "el",
        "en",
        "un",
        "una",
        "es",
        "no",
        "con",
        "por",
        "para",
        "si",
        "al",
        "del",
        "los",
        "las",
        "se",
        "a",
        "como",
        "pero",
        "porque",
        "puedo",
        "debo",
    )

    def __init__(self, texto: str, max_vocabulario: int = 1000) -> None:
        if not 500 <= max_vocabulario <= 1500:
            raise ValueError("max_vocabulario debe estar entre 500 y 1500.")
        self.max_vocabulario = max_vocabulario
        texto = self.normalizar(texto)
        tokens = self._tokenizar(texto)
        frecuencias = Counter(
            token for token in tokens
            if token not in self.PUNTUACION and token not in {"\n", "\t"}
        )
        tokens_fijos = [
            self.PAD,
            self.UNK,
            *self.TOKENS_ESPECIALES,
            *self.PUNTUACION,
            "\n",
            "\t",
            *self.TOKENS_ESTRUCTURALES,
        ]
        tokens_reservados = list(dict.fromkeys(tokens_fijos))
        espacio_restante = max_vocabulario - len(tokens_reservados)
        palabras = [
            palabra for palabra, _ in frecuencias.most_common()
            if palabra not in tokens_reservados
        ][:espacio_restante]
        self.tokens = tuple([*tokens_reservados, *palabras])
        self.token_a_id = {token: indice for indice, token in enumerate(self.tokens)}
        self.id_a_token = dict(enumerate(self.tokens))
        self.caracter_a_id = self.token_a_id
        self.id_pad = self.token_a_id[self.PAD]
        self.id_unk = self.token_a_id[self.UNK]

    @property
    def tamano_vocabulario(self) -> int:
        return len(self.tokens)

    def normalizar(self, texto: str) -> str:
        return unicodedata.normalize("NFC", texto)

    @classmethod
    def _tokenizar(cls, texto: str) -> list[str]:
        return cls.PATRON.findall(texto)

    def sanitizar(self, texto: str) -> str:
        texto = self.normalizar(texto).replace("\r\n", "\n").replace("\r", "\n")
        texto = re.sub(r"[ \f\v]+", " ", texto)
        texto = re.sub(r"\n{3,}", "\n\n", texto)
        if not self._tokenizar(texto):
            raise ValueError("El texto no contiene palabras ni signos admitidos.")
        return texto.strip()

    def codificar(self, texto: str) -> np.ndarray:
        tokens = self._tokenizar(self.normalizar(texto))
        return np.fromiter(
            (self.token_a_id.get(token, self.id_unk) for token in tokens),
            dtype=np.int64,
            count=len(tokens),
        )

    def _anexar(self, resultado: str, token: str) -> str:
        if token == self.PAD:
            return resultado
        if token == self.UNK:
            token = "�"
        if token in self.TOKENS_ESPECIALES:
            return resultado + ("" if not resultado or resultado[-1].isspace() else " ") + token
        if token == "\n":
            return resultado.rstrip() + "\n"
        if token == "\t":
            return resultado + "\t"
        if token in self.CIERRES:
            return resultado.rstrip() + token
        if token in self.APERTURAS:
            return resultado + ("" if not resultado or resultado[-1].isspace() else " ") + token
        return resultado + ("" if not resultado or resultado[-1].isspace() or resultado[-1] in self.APERTURAS else " ") + token

    def anexar_token(self, texto: str, token_id: int) -> str:
        if token_id not in self.id_a_token:
            raise ValueError(f"ID de token fuera del vocabulario: {token_id}")
        return self._anexar(texto, self.id_a_token[token_id])

    def decodificar(self, ids: np.ndarray | list[int]) -> str:
        resultado = ""
        for token_id in ids:
            indice = int(token_id)
            if indice not in self.id_a_token:
                raise ValueError(f"ID de token fuera del vocabulario: {indice}")
            resultado = self._anexar(resultado, self.id_a_token[indice])
        return resultado

    def vectorizar(self, ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(ids, dtype=np.int64)
        if np.any(ids < 0) or np.any(ids >= self.tamano_vocabulario):
            raise ValueError("Hay IDs fuera del vocabulario.")
        return np.eye(self.tamano_vocabulario, dtype=float)[ids]
