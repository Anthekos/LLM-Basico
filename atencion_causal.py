"""Modelo de caracteres con una capa de auto-atencion causal en NumPy."""

import numpy as np

from embedding_basico import EmbeddingBasico
from muestreo import muestrear_token


class ModeloAtencionCausal:
    """Bloque causal pequeno que reutiliza las tablas de un EmbeddingBasico.

    Las consultas de entrenamiento predicen el caracter siguiente desde hasta
    ``longitud_contexto`` caracteres previos. Una mascara de claves ignora el
    padding izquierdo y la mascara triangular impide consultar el futuro.
    """

    def __init__(
        self,
        inicializador: EmbeddingBasico,
        longitud_contexto: int = 16,
        semilla: int = 19,
    ) -> None:
        if longitud_contexto < 1:
            raise ValueError("longitud_contexto debe ser positiva.")
        if not inicializador.inicializado_por_red:
            raise RuntimeError(
                "Inicializa primero los embeddings con entrenar_inicializacion_red()."
            )

        self.inicializador = inicializador
        self.tokenizador = inicializador.tokenizador
        self.longitud_contexto = longitud_contexto
        self.dimension = inicializador.dimension
        self.desviacion_prior = inicializador.desviacion_prior
        self.generador = np.random.default_rng(semilla)

        # Se reutilizan las mismas matrices: input embedding y LM head.
        self.embedding_contexto = inicializador.embedding_contexto
        self.embedding_siguiente = inicializador.embedding_siguiente
        escala = 1.0 / np.sqrt(self.dimension)
        self.embedding_posicion = self.generador.normal(
            0.0, 0.02, (longitud_contexto, self.dimension)
        )
        self.Wq = self.generador.normal(0.0, escala, (self.dimension, self.dimension))
        self.Wk = self.generador.normal(0.0, escala, (self.dimension, self.dimension))
        self.Wv = self.generador.normal(0.0, escala, (self.dimension, self.dimension))
        dimension_oculta = 4 * self.dimension
        self.W1 = self.generador.normal(
            0.0, np.sqrt(2.0 / self.dimension), (self.dimension, dimension_oculta)
        )
        self.b1 = np.zeros(dimension_oculta)
        self.W2 = self.generador.normal(
            0.0, np.sqrt(1.0 / dimension_oculta), (dimension_oculta, self.dimension)
        )
        self.b2 = np.zeros(self.dimension)
        self.gamma1 = np.ones(self.dimension)
        self.beta1 = np.zeros(self.dimension)
        self.gamma2 = np.ones(self.dimension)
        self.beta2 = np.zeros(self.dimension)
        self.epsilon_norm = 1e-5
        self.historial: list[float] = []

    def _parametros(self) -> list[np.ndarray]:
        return [
            self.embedding_contexto,
            self.embedding_posicion,
            self.Wq,
            self.Wk,
            self.Wv,
            self.W1,
            self.b1,
            self.W2,
            self.b2,
            self.gamma1,
            self.beta1,
            self.gamma2,
            self.beta2,
            self.embedding_siguiente,
        ]

    def _prior_medias(self) -> list[np.ndarray]:
        medias = [np.zeros_like(parametro) for parametro in self._parametros()]
        medias[9].fill(1.0)
        medias[11].fill(1.0)
        return medias

    @staticmethod
    def _softmax(logits: np.ndarray, mascara: np.ndarray | None = None) -> np.ndarray:
        if mascara is None:
            logits_estables = logits - np.max(logits, axis=-1, keepdims=True)
            exponenciales = np.exp(logits_estables)
            return exponenciales / np.sum(exponenciales, axis=-1, keepdims=True)

        logits_enmascarados = np.where(mascara, logits, -1e30)
        maximos = np.max(logits_enmascarados, axis=-1, keepdims=True)
        exponenciales = np.exp(logits_enmascarados - maximos) * mascara
        denominadores = np.sum(exponenciales, axis=-1, keepdims=True)
        return exponenciales / np.maximum(denominadores, 1.0)

    def _layer_norm(
        self, valores: np.ndarray, gamma: np.ndarray, beta: np.ndarray
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        media = np.mean(valores, axis=-1, keepdims=True)
        centrado = valores - media
        varianza = np.mean(centrado**2, axis=-1, keepdims=True)
        inversa_desviacion = 1.0 / np.sqrt(varianza + self.epsilon_norm)
        normalizado = centrado * inversa_desviacion
        return normalizado * gamma + beta, (normalizado, inversa_desviacion)

    def _layer_norm_backward(
        self,
        gradiente_salida: np.ndarray,
        cache: tuple[np.ndarray, np.ndarray],
        gamma: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        normalizado, inversa_desviacion = cache
        gradiente_gamma = np.sum(gradiente_salida * normalizado, axis=(0, 1))
        gradiente_beta = np.sum(gradiente_salida, axis=(0, 1))
        gradiente_normalizado = gradiente_salida * gamma
        dimension = gradiente_salida.shape[-1]
        gradiente_entrada = (inversa_desviacion / dimension) * (
            dimension * gradiente_normalizado
            - np.sum(gradiente_normalizado, axis=-1, keepdims=True)
            - normalizado
            * np.sum(gradiente_normalizado * normalizado, axis=-1, keepdims=True)
        )
        return gradiente_entrada, gradiente_gamma, gradiente_beta

    def _adelante(
        self,
        ids: np.ndarray,
        claves_validas: np.ndarray,
        parametros: list[np.ndarray] | None = None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        lote, longitud = ids.shape
        if longitud != self.longitud_contexto:
            raise ValueError("Los ejemplos deben tener longitud_contexto columnas.")
        if claves_validas.shape != ids.shape:
            raise ValueError("La mascara de claves debe tener la misma forma que los IDs.")
        if np.any(ids < 0) or np.any(ids >= self.tokenizador.tamano_vocabulario):
            raise ValueError("Hay IDs fuera del vocabulario.")

        if parametros is None:
            parametros = self._parametros()
        embedding_contexto = parametros[0]
        embedding_posicion = parametros[1]
        Wq, Wk, Wv = parametros[2:5]
        embedding_siguiente = parametros[13]
        estados = embedding_contexto[ids] + embedding_posicion[None, :, :]
        consultas = estados @ Wq
        llaves = estados @ Wk
        valores = estados @ Wv
        puntajes = consultas @ np.swapaxes(llaves, 1, 2) / np.sqrt(self.dimension)

        causal = np.tril(np.ones((longitud, longitud), dtype=bool))
        mascara = causal[None, :, :] & claves_validas[:, None, :]
        atencion = self._softmax(puntajes, mascara)
        contexto = atencion @ valores
        residual_atencion = estados + contexto
        normalizado_atencion, cache_norm1 = self._layer_norm(
            residual_atencion, parametros[9], parametros[10]
        )
        pre_ffn = normalizado_atencion @ parametros[5] + parametros[6]
        activacion_ffn = np.maximum(pre_ffn, 0.0)
        salida_ffn = activacion_ffn @ parametros[7] + parametros[8]
        residual_ffn = normalizado_atencion + salida_ffn
        salida, cache_norm2 = self._layer_norm(
            residual_ffn, parametros[11], parametros[12]
        )
        logits = salida @ embedding_siguiente.T
        cache = {
            "ids": ids,
            "estados": estados,
            "consultas": consultas,
            "llaves": llaves,
            "valores": valores,
            "atencion": atencion,
            "contexto": contexto,
            "residual_atencion": residual_atencion,
            "normalizado_atencion": normalizado_atencion,
            "cache_norm1": cache_norm1,
            "pre_ffn": pre_ffn,
            "activacion_ffn": activacion_ffn,
            "salida_ffn": salida_ffn,
            "residual_ffn": residual_ffn,
            "salida": salida,
            "cache_norm2": cache_norm2,
            "mascara": mascara,
        }
        return logits, cache

    def _gradientes_lote(
        self,
        ids: np.ndarray,
        claves_validas: np.ndarray,
        objetivos: np.ndarray,
        parametros: list[np.ndarray] | None = None,
    ) -> tuple[float, list[np.ndarray]]:
        lote = len(ids)
        if parametros is None:
            parametros = self._parametros()
        embedding_contexto = parametros[0]
        embedding_posicion = parametros[1]
        embedding_siguiente = parametros[13]
        logits, cache = self._adelante(ids, claves_validas, parametros)
        logits_finales = logits[:, -1, :]
        logits_estables = logits_finales - np.max(logits_finales, axis=1, keepdims=True)
        exp_logits = np.exp(logits_estables)
        probabilidades = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        perdida = -np.mean(
            np.log(np.maximum(probabilidades[np.arange(lote), objetivos], 1e-12))
        )

        gradiente_logits_finales = probabilidades
        gradiente_logits_finales[np.arange(lote), objetivos] -= 1.0
        gradiente_logits_finales /= lote
        gradiente_logits = np.zeros_like(logits)
        gradiente_logits[:, -1, :] = gradiente_logits_finales

        gradiente_salida = gradiente_logits @ embedding_siguiente
        gradiente_embedding_siguiente = np.einsum(
            "btv,btd->vd", gradiente_logits, cache["salida"]
        )

        gradiente_residual_ffn, gradiente_gamma2, gradiente_beta2 = (
            self._layer_norm_backward(
                gradiente_salida, cache["cache_norm2"], parametros[11]
            )
        )
        gradiente_normalizado_atencion = gradiente_residual_ffn.copy()
        gradiente_salida_ffn = gradiente_residual_ffn
        gradiente_W2 = np.einsum(
            "bti,btj->ij", cache["activacion_ffn"], gradiente_salida_ffn
        )
        gradiente_b2 = np.sum(gradiente_salida_ffn, axis=(0, 1))
        gradiente_activacion_ffn = gradiente_salida_ffn @ parametros[7].T
        gradiente_pre_ffn = gradiente_activacion_ffn * (cache["pre_ffn"] > 0.0)
        gradiente_W1 = np.einsum(
            "bti,btj->ij", cache["normalizado_atencion"], gradiente_pre_ffn
        )
        gradiente_b1 = np.sum(gradiente_pre_ffn, axis=(0, 1))
        gradiente_normalizado_atencion += gradiente_pre_ffn @ parametros[5].T
        gradiente_residual_atencion, gradiente_gamma1, gradiente_beta1 = (
            self._layer_norm_backward(
                gradiente_normalizado_atencion,
                cache["cache_norm1"],
                parametros[9],
            )
        )

        # Residual de atencion: residual_atencion = estados + atencion @ valores.
        gradiente_estados = gradiente_residual_atencion.copy()
        gradiente_contexto = gradiente_residual_atencion
        gradiente_atencion = gradiente_contexto @ np.swapaxes(cache["valores"], 1, 2)
        gradiente_valores = np.swapaxes(cache["atencion"], 1, 2) @ gradiente_contexto

        atencion = cache["atencion"]
        gradiente_puntajes = atencion * (
            gradiente_atencion
            - np.sum(gradiente_atencion * atencion, axis=-1, keepdims=True)
        )
        gradiente_puntajes = np.where(cache["mascara"], gradiente_puntajes, 0.0)
        escala = 1.0 / np.sqrt(self.dimension)
        gradiente_consultas = gradiente_puntajes @ cache["llaves"] * escala
        gradiente_llaves = np.swapaxes(gradiente_puntajes, 1, 2) @ cache["consultas"] * escala

        gradiente_Wq = np.einsum("bti,btj->ij", cache["estados"], gradiente_consultas)
        gradiente_Wk = np.einsum("bti,btj->ij", cache["estados"], gradiente_llaves)
        gradiente_Wv = np.einsum("bti,btj->ij", cache["estados"], gradiente_valores)
        gradiente_estados += (
            gradiente_consultas @ parametros[2].T
            + gradiente_llaves @ parametros[3].T
            + gradiente_valores @ parametros[4].T
        )

        gradiente_embedding_contexto = np.zeros_like(embedding_contexto)
        np.add.at(
            gradiente_embedding_contexto,
            cache["ids"],
            gradiente_estados * claves_validas[:, :, None],
        )
        gradiente_embedding_posicion = np.sum(
            gradiente_estados * claves_validas[:, :, None], axis=0
        )
        gradientes = [
            gradiente_embedding_contexto,
            gradiente_embedding_posicion,
            gradiente_Wq,
            gradiente_Wk,
            gradiente_Wv,
            gradiente_W1,
            gradiente_b1,
            gradiente_W2,
            gradiente_b2,
            gradiente_gamma1,
            gradiente_beta1,
            gradiente_gamma2,
            gradiente_beta2,
            gradiente_embedding_siguiente,
        ]
        return float(perdida), gradientes

    def _crear_ejemplos(self, texto: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ids = self.tokenizador.codificar(texto)
        if len(ids) < 2:
            raise ValueError("El texto debe contener al menos dos caracteres.")

        longitud = self.longitud_contexto
        relleno = 0
        entradas = np.full((len(ids) - 1, longitud), relleno, dtype=np.int64)
        validas = np.zeros_like(entradas, dtype=bool)
        objetivos = ids[1:].copy()
        for fila, objetivo_posicion in enumerate(range(1, len(ids))):
            contexto = ids[max(0, objetivo_posicion - longitud):objetivo_posicion]
            entradas[fila, -len(contexto):] = contexto
            validas[fila, -len(contexto):] = True
        return entradas, validas, objetivos

    def entrenar_map(
        self,
        texto: str,
        epocas: int = 100,
        tasa_aprendizaje: float = 0.01,
        tamano_lote: int = 128,
    ) -> list[float]:
        """Entrena el bloque con entropia cruzada y prior gaussiano (MAP)."""
        if epocas < 1 or tasa_aprendizaje <= 0 or tamano_lote < 1:
            raise ValueError("epocas, tasa_aprendizaje y tamano_lote deben ser positivos.")
        entradas, validas, objetivos = self._crear_ejemplos(texto)
        cantidad = len(objetivos)
        parametros = self._parametros()
        momentos = [(np.zeros_like(parametro), np.zeros_like(parametro)) for parametro in parametros]
        varianza_prior = self.desviacion_prior**2
        prior_medias = self._prior_medias()
        paso = 0
        historial: list[float] = []

        for _ in range(epocas):
            orden = self.generador.permutation(cantidad)
            perdida_epoca = 0.0
            for inicio in range(0, cantidad, tamano_lote):
                indices = orden[inicio:inicio + tamano_lote]
                perdida, gradientes = self._gradientes_lote(
                    entradas[indices], validas[indices], objetivos[indices]
                )
                gradientes = [
                    gradiente + (parametro - prior) / (varianza_prior * cantidad)
                    for parametro, prior, gradiente in zip(
                        parametros, prior_medias, gradientes
                    )
                ]
                paso += 1
                momentos = self.inicializador._actualizar_adam(
                    parametros, gradientes, momentos, paso, tasa_aprendizaje
                )
                perdida_epoca += perdida * len(indices) / cantidad

            regularizacion = sum(
                float(np.sum((parametro - prior)**2))
                for parametro, prior in zip(parametros, prior_medias)
            )
            regularizacion /= 2.0 * varianza_prior * cantidad
            historial.append(perdida_epoca + regularizacion)

        self.historial = historial
        return historial.copy()

    def inferencia_variacional(
        self,
        texto: str,
        epocas: int = 100,
        tasa_aprendizaje: float = 0.005,
        tamano_lote: int = 128,
        muestras: int = 1,
    ) -> list[float]:
        """Ajusta un posterior gaussiano diagonal sobre embeddings y Q/K/V."""
        if epocas < 1 or tasa_aprendizaje <= 0 or tamano_lote < 1 or muestras < 1:
            raise ValueError("epocas, tasa_aprendizaje, tamano_lote y muestras deben ser positivos.")
        entradas, validas, objetivos = self._crear_ejemplos(texto)
        cantidad = len(objetivos)
        medias = self._parametros()
        log_desvios = [
            np.full_like(media, np.log(self.desviacion_prior * 0.1))
            for media in medias
        ]
        parametros = medias + log_desvios
        momentos = [(np.zeros_like(p), np.zeros_like(p)) for p in parametros]
        varianza_prior = self.desviacion_prior**2
        prior_medias = self._prior_medias()
        paso = 0
        historial: list[float] = []

        for _ in range(epocas):
            orden = self.generador.permutation(cantidad)
            perdida_epoca = 0.0
            for inicio in range(0, cantidad, tamano_lote):
                indices = orden[inicio:inicio + tamano_lote]
                gradientes_medias = [np.zeros_like(p) for p in medias]
                gradientes_log_desvios = [np.zeros_like(p) for p in medias]
                perdida_lote = 0.0
                for _ in range(muestras):
                    ruidos = [self.generador.normal(size=media.shape) for media in medias]
                    desvios = [np.exp(log_desvio) for log_desvio in log_desvios]
                    muestra_parametros = [
                        media + desvio * ruido
                        for media, desvio, ruido in zip(medias, desvios, ruidos)
                    ]
                    perdida, gradientes = self._gradientes_lote(
                        entradas[indices], validas[indices], objetivos[indices],
                        muestra_parametros,
                    )
                    perdida_lote += perdida / muestras
                    for parametro_indice, (gradiente, desvio, ruido) in enumerate(
                        zip(gradientes, desvios, ruidos)
                    ):
                        gradientes_medias[parametro_indice] += gradiente / muestras
                        gradientes_log_desvios[parametro_indice] += (
                            gradiente * desvio * ruido / muestras
                        )

                for indice, (media, prior, log_desvio) in enumerate(
                    zip(medias, prior_medias, log_desvios)
                ):
                    desvio = np.exp(log_desvio)
                    diferencia_prior = media - prior
                    gradientes_medias[indice] += (
                        diferencia_prior / (varianza_prior * cantidad)
                    )
                    gradientes_log_desvios[indice] += (
                        (desvio**2 / varianza_prior - 1.0) / cantidad
                    )

                gradientes = gradientes_medias + gradientes_log_desvios
                paso += 1
                momentos = self.inicializador._actualizar_adam(
                    parametros, gradientes, momentos, paso, tasa_aprendizaje
                )
                perdida_epoca += perdida_lote * len(indices) / cantidad

            kl_total = 0.0
            for media, prior, log_desvio in zip(medias, prior_medias, log_desvios):
                desvio = np.exp(log_desvio)
                diferencia_prior = media - prior
                kl_total += float(np.sum(
                    np.log(self.desviacion_prior / desvio)
                    + (desvio**2 + diferencia_prior**2) / (2.0 * varianza_prior)
                    - 0.5
                ))
            historial.append(perdida_epoca + kl_total / cantidad)

        self.log_desvios_posteriores = log_desvios
        self.historial = historial
        return historial.copy()

    def probabilidades_siguiente(
        self,
        contexto: str,
        inferencia: str = "map",
        muestras: int = 100,
    ) -> np.ndarray:
        ids = self.tokenizador.codificar(contexto)
        if len(ids) == 0:
            raise ValueError("El contexto debe contener al menos un caracter.")
        ids = ids[-self.longitud_contexto:]
        entrada = np.zeros((1, self.longitud_contexto), dtype=np.int64)
        validas = np.zeros_like(entrada, dtype=bool)
        entrada[0, -len(ids):] = ids
        validas[0, -len(ids):] = True
        if inferencia == "map":
            logits, _ = self._adelante(entrada, validas)
            return EmbeddingBasico._softmax(logits[:, -1, :])[0]
        if inferencia != "variacional":
            raise ValueError("inferencia debe ser 'map' o 'variacional'.")
        if muestras < 1:
            raise ValueError("muestras debe ser positivo.")
        if not hasattr(self, "log_desvios_posteriores"):
            raise RuntimeError("Ejecuta inferencia_variacional() antes de usar el posterior.")

        medias = self._parametros()
        desviaciones = [np.exp(log_desvio) for log_desvio in self.log_desvios_posteriores]
        promedio = np.zeros(self.tokenizador.tamano_vocabulario)
        for _ in range(muestras):
            muestra = [
                media + desviacion * self.generador.normal(size=media.shape)
                for media, desviacion in zip(medias, desviaciones)
            ]
            logits, _ = self._adelante(entrada, validas, muestra)
            promedio += EmbeddingBasico._softmax(logits[:, -1, :])[0] / muestras
        return promedio

    def generar(
        self,
        inicio: str,
        cantidad: int,
        semilla: int | None = None,
        inferencia: str = "map",
        temperatura: float = 0.8,
        top_k: int | None = 10,
        penalizacion_repeticion: float = 1.05,
        tamano_ngram_sin_repetir: int = 4,
    ) -> str:
        if cantidad < 0:
            raise ValueError("cantidad no puede ser negativa.")
        if cantidad and not inicio:
            raise ValueError("inicio debe contener al menos un caracter.")
        ids = self.tokenizador.codificar(inicio)
        inicio = self.tokenizador.decodificar(ids)
        generador = np.random.default_rng(semilla)
        resultado = inicio
        historial_ids = ids.tolist()
        for _ in range(cantidad):
            probabilidades = self.probabilidades_siguiente(resultado, inferencia=inferencia)
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
            resultado += self.tokenizador.decodificar([siguiente])
        return resultado
