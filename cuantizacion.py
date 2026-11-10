"""Cuantizacion simetrica int8 para inferencia con NumPy."""

import numpy as np


def cuantizar_tensor(pesos_float: np.ndarray) -> tuple[np.ndarray, float]:
    """Cuantiza un tensor al rango simetrico int8 y devuelve su escala."""
    pesos = np.asarray(pesos_float, dtype=np.float32)
    if not np.all(np.isfinite(pesos)):
        raise ValueError("El tensor a cuantizar debe contener valores finitos.")
    maximo = float(np.max(np.abs(pesos))) if pesos.size else 0.0
    escala = maximo / 127.0 if maximo else 1.0
    cuantizado = np.clip(np.rint(pesos / escala), -127, 127).astype(np.int8)
    return cuantizado, escala


def decuantizar_tensor(
    pesos_int8: np.ndarray, escala: float
) -> np.ndarray:
    """Devuelve una aproximacion float32 del tensor cuantizado."""
    pesos = np.asarray(pesos_int8)
    if pesos.dtype != np.int8:
        raise TypeError("pesos_int8 debe tener dtype int8.")
    if not np.isfinite(escala) or escala <= 0:
        raise ValueError("escala debe ser un numero finito positivo.")
    return pesos.astype(np.float32) * np.float32(escala)


def multiplicar_matrices_cuantizadas(
    entrada_int8: np.ndarray,
    pesos_int8: np.ndarray,
    escala_entrada: float,
    escala_pesos: float,
) -> np.ndarray:
    """Multiplica int8 con acumulacion int32 y devuelve el resultado float32."""
    entrada = np.asarray(entrada_int8)
    pesos = np.asarray(pesos_int8)
    if entrada.dtype != np.int8 or pesos.dtype != np.int8:
        raise TypeError("Las dos matrices deben tener dtype int8.")
    if entrada.ndim < 1 or pesos.ndim != 2 or entrada.shape[-1] != pesos.shape[0]:
        raise ValueError("Las dimensiones de entrada y pesos no son compatibles.")
    if (
        not np.isfinite(escala_entrada)
        or not np.isfinite(escala_pesos)
        or escala_entrada <= 0
        or escala_pesos <= 0
    ):
        raise ValueError("Las escalas deben ser numeros finitos positivos.")
    acumulado = np.matmul(
        entrada,
        pesos,
        dtype=np.int32,
        casting="unsafe",
    )
    return acumulado.astype(np.float32) * np.float32(
        escala_entrada * escala_pesos
    )


def multiplicar_cuantizado(
    entrada_float: np.ndarray,
    pesos_int8: np.ndarray,
    escala_pesos: float,
) -> np.ndarray:
    """Cuantiza activaciones, acumula en int32 y escala la salida."""
    entrada_int8, escala_entrada = cuantizar_tensor(entrada_float)
    return multiplicar_matrices_cuantizadas(
        entrada_int8, pesos_int8, escala_entrada, escala_pesos
    )
