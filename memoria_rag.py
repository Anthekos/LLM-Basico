"""Indice léxico local para estimar novedad de interacciones."""

from collections import Counter
from dataclasses import dataclass
import re


_PATRON_TERMINO = re.compile(r"[a-záéíóúüñ]{3,}", re.IGNORECASE)
_PALABRAS_VACIAS = frozenset(
    {
        "para", "pero", "porque", "como", "cuando", "donde", "desde",
        "entre", "hasta", "sobre", "tambien", "tiene", "puede", "puedo",
        "quiero", "necesito", "esto", "esta", "estos", "estas", "ellos",
        "ellas", "algo", "cada", "todo", "toda", "todos", "todas",
        "hacer", "hace", "ser", "estar", "con", "sin", "una", "uno",
        "unos", "unas", "del", "las", "los", "por", "que", "cual",
        "cuales", "sus", "mis", "tus", "muy", "mas", "menos", "aqui",
        "ahi", "alli", "hay", "fue", "son", "sea", "han", "hemos",
    }
)


@dataclass(frozen=True)
class EvaluacionSaliencia:
    puntuacion: float
    terminos_nuevos: tuple[str, ...]


class IndiceMemoriaRAG:
    """Cuenta términos del corpus local para estimar novedad léxica."""

    def __init__(self, corpus: str) -> None:
        terminos = (
            termino.lower()
            for termino in _PATRON_TERMINO.findall(corpus)
            if termino.lower() not in _PALABRAS_VACIAS
        )
        self.frecuencias = Counter(terminos)

    def evaluar(self, prompt: str, respuesta: str) -> EvaluacionSaliencia:
        terminos = tuple(
            dict.fromkeys(
                termino.lower()
                for termino in _PATRON_TERMINO.findall(f"{prompt}\n{respuesta}")
                if termino.lower() not in _PALABRAS_VACIAS
            )
        )
        if not terminos:
            return EvaluacionSaliencia(0.0, ())

        nuevos = tuple(termino for termino in terminos if termino not in self.frecuencias)
        novedad = len(nuevos) / len(terminos)
        rareza_media = sum(
            1.0 / (1.0 + self.frecuencias[termino])
            for termino in terminos
        ) / len(terminos)
        puntuacion = 0.7 * novedad + 0.3 * rareza_media
        return EvaluacionSaliencia(float(puntuacion), nuevos)


def resumir_interaccion(prompt: str, respuesta: str, maximo: int = 600) -> str:
    """Crea una ficha extractiva breve, sin inventar detalles."""
    if maximo < 32:
        raise ValueError("maximo debe ser al menos 32 caracteres.")

    def compactar(texto: str, limite: int) -> str:
        limpio = " ".join(texto.split())
        if len(limpio) <= limite:
            return limpio
        truncado = limpio[:limite].rsplit(" ", 1)[0].rstrip(" ,;:")
        return f"{truncado}…"

    espacio = maximo - len("Usuario: \nAsistente: \n")
    limite_prompt = max(12, espacio // 3)
    limite_respuesta = max(12, espacio - limite_prompt)
    return (
        f"Usuario: {compactar(prompt, limite_prompt)}\n"
        f"Asistente: {compactar(respuesta, limite_respuesta)}"
    )
