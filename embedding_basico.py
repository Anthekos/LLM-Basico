"""Factorizacion bayesiana de transiciones entre embeddings de caracteres.

La probabilidad de transicion es softmax(u_caracter @ v_candidato.T). Los
embeddings reciben un prior normal isotropico independiente. La inferencia MAP
usa la moda posterior; la variacional aproxima el posterior con gaussianas
normales diagonales y maximiza una ELBO por gradiente reparametrizado.
"""

import numpy as np

from Red_neuronal import RedNeuronal
from tokenizador_basico import TokenizadorBasico
from muestreo import muestrear_token


class EmbeddingBasico:
    def __init__(
        self,
        tokenizador: TokenizadorBasico,
        dimension: int = 12,
        desviacion_prior: float = 1.0,
        semilla: int = 7,
    ) -> None:
        if dimension < 1 or desviacion_prior <= 0:
            raise ValueError("dimension y desviacion_prior deben ser positivos.")

        self.tokenizador = tokenizador
        self.dimension = dimension
        self.desviacion_prior = desviacion_prior
        self.generador = np.random.default_rng(semilla)
        escala = desviacion_prior * 0.1
        cantidad = tokenizador.tamano_vocabulario
        self.embedding_contexto = self.generador.normal(0.0, escala, (cantidad, dimension))
        self.embedding_siguiente = self.generador.normal(0.0, escala, (cantidad, dimension))
        self.media_contexto = self.embedding_contexto.copy()
        self.media_siguiente = self.embedding_siguiente.copy()
        self.log_desvio_contexto = np.full((cantidad, dimension), np.log(escala))
        self.log_desvio_siguiente = np.full((cantidad, dimension), np.log(escala))
        self._ids_contexto: np.ndarray | None = None
        self._ids_siguiente: np.ndarray | None = None
        self.historial_red: list[float] = []
        self.inicializado_por_red = False

    def entrenar_inicializacion_red(
        self,
        texto: str,
        epocas: int = 200,
        tasa_aprendizaje: float = 0.03,
        verbose: bool = False,
    ) -> list[float]:
        """Entrena la RedNeuronal siguiente-caracter y extrae sus embeddings."""
        ids = self.tokenizador.codificar(texto)
        if len(ids) < 2:
            raise ValueError("El texto debe contener al menos dos caracteres.")
        if epocas < 1 or tasa_aprendizaje <= 0:
            raise ValueError("epocas y tasa_aprendizaje deben ser positivos.")

        self._ids_contexto = ids[:-1]
        self._ids_siguiente = ids[1:]
        entradas = self.tokenizador.vectorizar(self._ids_contexto)
        objetivos = self.tokenizador.vectorizar(self._ids_siguiente)
        red = RedNeuronal(
            [self.tokenizador.tamano_vocabulario, self.dimension,
             self.tokenizador.tamano_vocabulario],
            semilla=int(self.generador.integers(0, np.iinfo(np.int32).max)),
            activacion_salida="lineal",
        )
        self.historial_red = red.entrenar_multiclase(
            entradas, objetivos, epocas=epocas, tasa_aprendizaje=tasa_aprendizaje
        )

        capa_oculta, capa_salida = red.capas
        self.embedding_contexto = capa_oculta.pesos.copy()
        self.embedding_siguiente = capa_salida.pesos.T.copy()
        self.media_contexto = self.embedding_contexto.copy()
        self.media_siguiente = self.embedding_siguiente.copy()
        escala = self.desviacion_prior * 0.1
        self.log_desvio_contexto.fill(np.log(escala))
        self.log_desvio_siguiente.fill(np.log(escala))
        self.inicializado_por_red = True
        if verbose:
            print(
                f"Inicializacion por RedNeuronal: error {self.historial_red[0]:.4f} "
                f"-> {self.historial_red[-1]:.4f} en {epocas} etapas"
            )
        return self.historial_red.copy()

    def _datos_entrenamiento(
        self, texto: str | None
    ) -> tuple[np.ndarray, np.ndarray]:
        if texto is not None:
            ids = self.tokenizador.codificar(texto)
            if len(ids) < 2:
                raise ValueError("El texto debe contener al menos dos caracteres.")
            return ids[:-1], ids[1:]
        if self._ids_contexto is None or self._ids_siguiente is None:
            raise RuntimeError(
                "Primero llama entrenar_inicializacion_red(texto) para entrenar "
                "la red inicial y preparar las transiciones."
            )
        return self._ids_contexto, self._ids_siguiente

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        logits_estables = logits - np.max(logits, axis=-1, keepdims=True)
        exponenciales = np.exp(logits_estables)
        return exponenciales / np.sum(exponenciales, axis=-1, keepdims=True)

    def _perdida_y_gradientes(
        self,
        contexto: np.ndarray,
        siguiente: np.ndarray,
        embedding_contexto: np.ndarray,
        embedding_siguiente: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        cantidad = len(contexto)
        vectores_contexto = embedding_contexto[contexto]
        logits = vectores_contexto @ embedding_siguiente.T
        probabilidades = self._softmax(logits)
        perdida = -np.mean(np.log(np.maximum(probabilidades[np.arange(cantidad), siguiente], 1e-12)))

        gradiente_logits = probabilidades
        gradiente_logits[np.arange(cantidad), siguiente] -= 1.0
        gradiente_logits /= cantidad
        gradiente_contexto = np.zeros_like(embedding_contexto)
        np.add.at(gradiente_contexto, contexto, gradiente_logits @ embedding_siguiente)
        gradiente_siguiente = gradiente_logits.T @ vectores_contexto
        return float(perdida), gradiente_contexto, gradiente_siguiente

    def _actualizar_adam(
        self,
        parametros: list[np.ndarray],
        gradientes: list[np.ndarray],
        momentos: list[tuple[np.ndarray, np.ndarray]],
        paso: int,
        tasa_aprendizaje: float,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        nuevos_momentos = []
        for indice, (parametro, gradiente) in enumerate(zip(parametros, gradientes)):
            primer_momento, segundo_momento = momentos[indice]
            primer_momento = beta1 * primer_momento + (1.0 - beta1) * gradiente
            segundo_momento = beta2 * segundo_momento + (1.0 - beta2) * gradiente**2
            m_corregido = primer_momento / (1.0 - beta1**paso)
            v_corregido = segundo_momento / (1.0 - beta2**paso)
            parametro -= tasa_aprendizaje * m_corregido / (np.sqrt(v_corregido) + epsilon)
            nuevos_momentos.append((primer_momento, segundo_momento))
        return nuevos_momentos

    def inferencia_map(
        self,
        texto: str | None = None,
        epocas: int = 500,
        tasa_aprendizaje: float = 0.03,
    ) -> list[float]:
        """Ajusta embeddings maximizando log-verosimilitud mas prior gaussiano."""
        contexto, siguiente = self._datos_entrenamiento(texto)
        if epocas < 1 or tasa_aprendizaje <= 0:
            raise ValueError("epocas y tasa_aprendizaje deben ser positivos.")

        total = len(contexto)
        varianza_prior = self.desviacion_prior**2
        parametros = [self.embedding_contexto, self.embedding_siguiente]
        momentos = [(np.zeros_like(p), np.zeros_like(p)) for p in parametros]
        historial: list[float] = []
        for paso in range(1, epocas + 1):
            perdida, grad_contexto, grad_siguiente = self._perdida_y_gradientes(
                contexto, siguiente, *parametros
            )
            regularizacion = sum(float(np.sum(p**2)) for p in parametros) / (
                2.0 * varianza_prior * total
            )
            gradientes = [
                grad_contexto + parametros[0] / (varianza_prior * total),
                grad_siguiente + parametros[1] / (varianza_prior * total),
            ]
            momentos = self._actualizar_adam(
                parametros, gradientes, momentos, paso, tasa_aprendizaje
            )
            historial.append(perdida + regularizacion)

        self.media_contexto = self.embedding_contexto.copy()
        self.media_siguiente = self.embedding_siguiente.copy()
        return historial

    def inferencia_variacional(
        self,
        texto: str | None = None,
        epocas: int = 500,
        tasa_aprendizaje: float = 0.01,
        muestras: int = 1,
    ) -> list[float]:
        """Ajusta un posterior gaussiano diagonal maximizando una ELBO."""
        contexto, siguiente = self._datos_entrenamiento(texto)
        if epocas < 1 or tasa_aprendizaje <= 0 or muestras < 1:
            raise ValueError("epocas, tasa_aprendizaje y muestras deben ser positivos.")

        total = len(contexto)
        varianza_prior = self.desviacion_prior**2
        parametros = [self.media_contexto, self.media_siguiente,
                      self.log_desvio_contexto, self.log_desvio_siguiente]
        momentos = [(np.zeros_like(p), np.zeros_like(p)) for p in parametros]
        historial: list[float] = []
        for paso in range(1, epocas + 1):
            gradientes_acumulados = [np.zeros_like(p) for p in parametros]
            perdida_media = 0.0
            for _ in range(muestras):
                ruido_contexto = self.generador.normal(size=self.media_contexto.shape)
                ruido_siguiente = self.generador.normal(size=self.media_siguiente.shape)
                desvio_contexto = np.exp(self.log_desvio_contexto)
                desvio_siguiente = np.exp(self.log_desvio_siguiente)
                muestra_contexto = self.media_contexto + desvio_contexto * ruido_contexto
                muestra_siguiente = self.media_siguiente + desvio_siguiente * ruido_siguiente
                perdida, grad_contexto, grad_siguiente = self._perdida_y_gradientes(
                    contexto, siguiente, muestra_contexto, muestra_siguiente
                )
                perdida_media += perdida / muestras
                gradientes_acumulados[0] += grad_contexto / muestras
                gradientes_acumulados[1] += grad_siguiente / muestras
                gradientes_acumulados[2] += (
                    grad_contexto * desvio_contexto * ruido_contexto / muestras
                )
                gradientes_acumulados[3] += (
                    grad_siguiente * desvio_siguiente * ruido_siguiente / muestras
                )

            # Tanto la verosimilitud como la KL se expresan por transicion:
            # la perdida empirica ya es media y la KL se divide entre N.
            kl = 0.0
            for indice in range(2):
                media = parametros[indice]
                log_desvio = parametros[indice + 2]
                desvio = np.exp(log_desvio)
                kl += float(np.sum(
                    np.log(self.desviacion_prior / desvio)
                    + (desvio**2 + media**2) / (2.0 * varianza_prior)
                    - 0.5
                ))
                gradientes_acumulados[indice] += media / (varianza_prior * total)
                gradientes_acumulados[indice + 2] += (
                    (desvio**2 / varianza_prior - 1.0) / total
                )

            momentos = self._actualizar_adam(
                parametros, gradientes_acumulados, momentos, paso, tasa_aprendizaje
            )
            self.log_desvio_contexto[:] = np.clip(self.log_desvio_contexto, -8.0, 2.0)
            self.log_desvio_siguiente[:] = np.clip(self.log_desvio_siguiente, -8.0, 2.0)
            historial.append(perdida_media + kl / total)

        self.embedding_contexto = self.media_contexto.copy()
        self.embedding_siguiente = self.media_siguiente.copy()
        return historial

    def probabilidades_siguiente(
        self,
        caracter: str,
        inferencia: str = "map",
        muestras: int = 100,
    ) -> np.ndarray:
        ids = self.tokenizador.codificar(caracter)
        if len(ids) != 1:
            raise ValueError("Se requiere exactamente un caracter de contexto.")
        indice = int(ids[0])
        if inferencia == "map":
            logits = self.embedding_contexto[indice] @ self.embedding_siguiente.T
            return self._softmax(logits)
        if inferencia != "variacional":
            raise ValueError("inferencia debe ser 'map' o 'variacional'.")
        if muestras < 1:
            raise ValueError("muestras debe ser positivo.")

        desviacion_contexto = np.exp(self.log_desvio_contexto[indice])
        desviacion_siguiente = np.exp(self.log_desvio_siguiente)
        probabilidades = np.zeros(self.tokenizador.tamano_vocabulario)
        for _ in range(muestras):
            vector_contexto = self.media_contexto[indice] + desviacion_contexto * self.generador.normal(
                size=self.dimension
            )
            vectores_siguiente = self.media_siguiente + desviacion_siguiente * self.generador.normal(
                size=self.media_siguiente.shape
            )
            probabilidades += self._softmax(vector_contexto @ vectores_siguiente.T)
        return probabilidades / muestras

    def generar(
        self,
        inicio: str,
        cantidad: int,
        inferencia: str = "map",
        semilla: int | None = None,
        temperatura: float = 0.8,
        top_k: int | None = 10,
        penalizacion_repeticion: float = 1.05,
        tamano_ngram_sin_repetir: int = 4,
    ) -> str:
        if cantidad < 0:
            raise ValueError("cantidad no puede ser negativa.")
        ids = self.tokenizador.codificar(inicio)
        inicio = self.tokenizador.decodificar(ids)
        if len(ids) == 0 and cantidad:
            raise ValueError("inicio debe contener al menos un caracter.")
        generador = np.random.default_rng(semilla)
        texto = inicio
        historial_ids = ids.tolist()
        for _ in range(cantidad):
            caracter = texto[-1:]
            probabilidades = self.probabilidades_siguiente(caracter, inferencia)
            siguiente = muestrear_token(
                probabilidades,
                historial_ids,
                generador,
                temperatura,
                top_k,
                penalizacion_repeticion,
                tamano_ngram_sin_repetir,
            )
            historial_ids.append(siguiente)
            texto += self.tokenizador.decodificar([siguiente])
        return texto
