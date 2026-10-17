"""Muestreo configurable para la generacion de caracteres."""

import numpy as np


def muestrear_token(
    probabilidades: np.ndarray,
    historial: list[int],
    generador: np.random.Generator,
    temperatura: float = 0.8,
    top_k: int | None = 10,
    penalizacion_repeticion: float = 1.05,
    tamano_ngram_sin_repetir: int = 4,
) -> int:
    """Muestrea con temperatura, top-k y penalizaciones opcionales anti-bucle."""
    probabilidades = np.asarray(probabilidades, dtype=float)
    if probabilidades.ndim != 1 or probabilidades.size == 0:
        raise ValueError("probabilidades debe ser un vector no vacio.")
    if not np.all(np.isfinite(probabilidades)) or np.any(probabilidades < 0):
        raise ValueError("probabilidades debe contener valores finitos no negativos.")
    if probabilidades.sum() <= 0:
        raise ValueError("La suma de probabilidades debe ser positiva.")
    if temperatura <= 0:
        raise ValueError("temperatura debe ser positiva.")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k debe ser positivo o None.")
    if penalizacion_repeticion < 1:
        raise ValueError("penalizacion_repeticion debe ser al menos 1.")
    if tamano_ngram_sin_repetir < 0:
        raise ValueError("tamano_ngram_sin_repetir no puede ser negativo.")

    logits = np.log(np.maximum(probabilidades, np.finfo(float).tiny))
    if penalizacion_repeticion > 1 and historial:
        vistos = set(historial[-64:])
        for token_id in vistos:
            if 0 <= token_id < len(logits):
                if logits[token_id] > 0:
                    logits[token_id] /= penalizacion_repeticion
                else:
                    logits[token_id] *= penalizacion_repeticion

    permitidos = np.ones(len(logits), dtype=bool)
    ngram = tamano_ngram_sin_repetir
    if ngram > 1 and len(historial) >= ngram - 1:
        prefijo = tuple(historial[-(ngram - 1):])
        for indice in range(len(historial) - ngram + 1):
            if tuple(historial[indice:indice + ngram - 1]) == prefijo:
                token_bloqueado = historial[indice + ngram - 1]
                if 0 <= token_bloqueado < len(logits):
                    permitidos[token_bloqueado] = False

    if not np.any(permitidos):
        permitidos[:] = True
    logits = np.where(permitidos, logits, -np.inf)
    logits /= temperatura

    if top_k is not None and top_k < len(logits):
        candidatos = np.flatnonzero(permitidos)
        cantidad = min(top_k, len(candidatos))
        mejores_locales = np.argpartition(logits[candidatos], -cantidad)[-cantidad:]
        conservar = candidatos[mejores_locales]
        mascara_top_k = np.zeros(len(logits), dtype=bool)
        mascara_top_k[conservar] = True
        logits = np.where(mascara_top_k, logits, -np.inf)

    logits -= np.max(logits)
    pesos = np.exp(logits)
    distribucion = pesos / pesos.sum()
    return int(generador.choice(len(distribucion), p=distribucion))
