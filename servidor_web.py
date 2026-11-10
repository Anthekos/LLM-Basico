"""Servidor local para la interfaz web del modelo causal."""

from __future__ import annotations

import argparse
import hashlib
import hmac
from datetime import datetime
import json
import logging
import os
import re
import secrets
import threading
import tempfile
import textwrap
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import unquote, urlparse
from uuid import uuid4

import numpy as np
from pypdf import PdfReader, PdfWriter
from pypdf.generic import (
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
)
from pypdf.errors import PdfReadError

from atencion_causal import ModeloAtencionCausal
from embedding_basico import EmbeddingBasico
from formato_conversacion import (
    ASISTENTE,
    FIN,
    PENSAMIENTO,
    USUARIO,
    migrar_dialogos_legacy,
    separar_deliberacion,
    validar_corpus_estructurado,
)
from memoria_rag import IndiceMemoriaRAG, resumir_interaccion
from pdf_a_corpus import sanitizar_texto
from tokenizador_basico import TokenizadorBasico
from tokenizador_palabras import TokenizadorPalabras


DIRECTORIO_WEB = Path(__file__).parent / "web"
RUTA_CORPUS_MAXIMO = Path(__file__).parent / "corpus_maximo.txt"
RUTA_CORPUS_LOCAL = Path(__file__).parent / "corpus_local.txt"
RUTA_CORPUS = RUTA_CORPUS_LOCAL
RUTA_CORPUS_ANTIGUO = Path(__file__).parent / "corpus.txt"
RUTA_PDFS_CORPUS = Path(__file__).parent / "pdfs_corpus"
RUTA_MANIFIESTO_PDFS = Path(__file__).parent / "pdfs_corpus.json"
RUTA_FEEDBACK = Path(__file__).parent / "feedback_recompensa.jsonl"
RUTA_CONFIG_MODELO = Path(__file__).parent / "configuracion_modelo.json"
RUTA_CREDENCIALES_ADMIN = Path(__file__).parent / ".venv" / "admin_credentials.json"
MAXIMO_PDF_BYTES = 12 * 1024 * 1024
MAXIMO_CARACTERES_CHAT = 1200
MAXIMO_HISTORIAL_CHAT = 200
MAXIMO_CARACTERES_CORPUS_MAXIMO = 100_000
LONGITUD_MINIMA_PASSWORD_ADMIN = 12
TEMPERATURA_MAXIMA = 3.0
VENTANA_CONTEXTO_MINIMA = 4
VENTANA_CONTEXTO_MAXIMA = 128
VENTANA_CONTEXTO_PREDETERMINADA = 32
UMBRAL_SALIENCIA_MINIMO = 0.0
UMBRAL_SALIENCIA_MAXIMO = 1.0
UMBRAL_SALIENCIA_PREDETERMINADO = 0.75
PATRON_BITACORA_LEGADA = re.compile(
    r"(?ms)^Bitácora de Contacto: Registro de Intercambio\n.*?(?=^Memoria - |\Z)"
)
TEXTO_INICIAL = (
    "Una red neuronal aprende relaciones entre datos mediante ejemplos. "
    "Cada capa transforma una representacion y ajusta sus pesos para reducir "
    "el error. La atencion causal permite que cada caracter consulte el "
    "contexto anterior sin mirar el futuro. Un modelo de lenguaje pequeno "
    "predice el siguiente caracter usando probabilidades aprendidas. "
) * 3
EJEMPLOS = {
    "red-neuronal": {
        "titulo": "Introduccion a redes neuronales",
        "descripcion": "Un texto corto sobre capas, pesos y aprendizaje.",
        "texto": (
            "Una red neuronal es un modelo que aprende patrones a partir de "
            "ejemplos. Una capa recibe valores, los combina con pesos y aplica "
            "una funcion de activacion. Durante el entrenamiento, la red "
            "compara su respuesta con el objetivo y ajusta sus pesos para "
            "reducir el error. Las capas ocultas aprenden representaciones "
            "intermedias. Una red de caracteres puede aprender a predecir "
            "cual sera el siguiente caracter de una secuencia."
        ),
    },
    "atencion-causal": {
        "titulo": "Atencion causal y lenguaje",
        "descripcion": "Un ejemplo simple de contexto y prediccion.",
        "texto": (
            "Un modelo de lenguaje predice el siguiente elemento de una "
            "secuencia. La atencion causal compara cada posicion con las "
            "posiciones anteriores. Una mascara impide consultar el futuro. "
            "Las consultas, las llaves y los valores permiten construir "
            "representaciones del contexto. La salida se transforma en "
            "probabilidades y el muestreo selecciona el siguiente caracter."
        ),
    },
}


def cortar_en_turno_posterior(texto: str) -> str:
    """Descarta cualquier continuación que empiece un nuevo turno de usuario."""
    posiciones = [
        posicion
        for marcador in (FIN, USUARIO)
        if (posicion := texto.find(marcador)) >= 0
    ]
    return texto[:min(posiciones)] if posiciones else texto


class AutenticacionAdmin:
    """Gestiona credenciales y tokens efimeros para las operaciones admin."""

    ITERACIONES_HASH = 310_000
    DURACION_TOKEN_SEGUNDOS = 3600
    MAXIMO_INTENTOS = 5
    BLOQUEO_SEGUNDOS = 60

    def __init__(self, ruta_credenciales: Path = RUTA_CREDENCIALES_ADMIN) -> None:
        self.ruta_credenciales = ruta_credenciales
        self.bloqueo = threading.Lock()
        self.tokens: dict[str, tuple[str, float]] = {}
        self.intentos_fallidos = 0
        self.bloqueado_hasta = 0.0
        if not ruta_credenciales.is_file():
            self._guardar_credenciales(
                self._crear_credenciales("admin", "admin", debe_cambiar=True)
            )
            logging.warning(
                "Credenciales administrativas provisionales creadas en %s; "
                "usuario admin, contraseña admin. Cámbiala al iniciar sesión.",
                ruta_credenciales,
            )
        self.credenciales = self._leer_credenciales()

    @classmethod
    def _crear_credenciales(
        cls,
        usuario: str,
        contrasena: str,
        debe_cambiar: bool,
    ) -> dict[str, object]:
        sal = secrets.token_bytes(16)
        hash_contrasena = hashlib.pbkdf2_hmac(
            "sha256",
            contrasena.encode("utf-8"),
            sal,
            cls.ITERACIONES_HASH,
        )
        return {
            "username": usuario,
            "salt": sal.hex(),
            "password_hash": hash_contrasena.hex(),
            "iterations": cls.ITERACIONES_HASH,
            "must_change_password": debe_cambiar,
        }

    def _leer_credenciales(self) -> dict[str, object]:
        try:
            credenciales = json.loads(
                self.ruta_credenciales.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"No se pudieron leer las credenciales de administrador: {error}"
            ) from error
        if (
            not isinstance(credenciales, dict)
            or not isinstance(credenciales.get("username"), str)
            or not isinstance(credenciales.get("salt"), str)
            or not isinstance(credenciales.get("password_hash"), str)
            or not isinstance(credenciales.get("iterations"), int)
            or not isinstance(credenciales.get("must_change_password"), bool)
        ):
            raise RuntimeError(
                "El archivo local de credenciales administrativas no es válido."
            )
        try:
            bytes.fromhex(credenciales["salt"])
            bytes.fromhex(credenciales["password_hash"])
        except ValueError as error:
            raise RuntimeError(
                "El archivo local de credenciales administrativas está dañado."
            ) from error
        return credenciales

    def _guardar_credenciales(self, credenciales: dict[str, object]) -> None:
        self.ruta_credenciales.parent.mkdir(parents=True, exist_ok=True)
        ModeloWeb._escribir_atomico(
            self.ruta_credenciales,
            json.dumps(credenciales, indent=2) + "\n",
        )
        os.chmod(self.ruta_credenciales, 0o600)

    def iniciar_sesion(self, usuario: str, contrasena: str) -> tuple[str, bool]:
        with self.bloqueo:
            ahora = time.monotonic()
            if ahora < self.bloqueado_hasta:
                espera = max(1, int(self.bloqueado_hasta - ahora))
                raise PermissionError(
                    f"Demasiados intentos fallidos. Intenta de nuevo en {espera} segundos."
                )
            sal = bytes.fromhex(str(self.credenciales["salt"]))
            hash_proporcionado = hashlib.pbkdf2_hmac(
                "sha256",
                contrasena.encode("utf-8"),
                sal,
                int(self.credenciales["iterations"]),
            ).hex()
            valido = (
                hmac.compare_digest(
                    usuario.encode("utf-8"),
                    str(self.credenciales["username"]).encode("utf-8"),
                )
                and hmac.compare_digest(
                    hash_proporcionado,
                    str(self.credenciales["password_hash"]),
                )
            )
            if not valido:
                self.intentos_fallidos += 1
                if self.intentos_fallidos >= self.MAXIMO_INTENTOS:
                    self.bloqueado_hasta = ahora + self.BLOQUEO_SEGUNDOS
                    self.intentos_fallidos = 0
                raise PermissionError("Usuario o contraseña incorrectos.")
            self.intentos_fallidos = 0
            token = secrets.token_urlsafe(32)
            self.tokens[token] = (
                str(self.credenciales["username"]),
                ahora + self.DURACION_TOKEN_SEGUNDOS,
            )
            return token, bool(self.credenciales["must_change_password"])

    def validar_token(
        self,
        token: str,
        permitir_cambio_password: bool = False,
    ) -> bool:
        with self.bloqueo:
            sesion = self.tokens.get(token)
            if sesion is None:
                return False
            if sesion[1] <= time.monotonic():
                del self.tokens[token]
                return False
            if (
                self.credenciales["must_change_password"]
                and not permitir_cambio_password
            ):
                return False
            return True

    def cambiar_password(self, token: str, nueva: str) -> None:
        if not self.validar_token(token, permitir_cambio_password=True):
            raise PermissionError("La sesión administrativa no es válida o expiró.")
        if len(nueva) < LONGITUD_MINIMA_PASSWORD_ADMIN:
            raise ValueError(
                "La nueva contraseña debe tener al menos "
                f"{LONGITUD_MINIMA_PASSWORD_ADMIN} caracteres."
            )
        with self.bloqueo:
            credenciales = self._crear_credenciales(
                str(self.credenciales["username"]),
                nueva,
                debe_cambiar=False,
            )
            self._guardar_credenciales(credenciales)
            self.credenciales = credenciales
            sesion = self.tokens[token]
            self.tokens.clear()
            self.tokens[token] = sesion

    def cerrar_sesion(self, token: str) -> None:
        with self.bloqueo:
            self.tokens.pop(token, None)


class ModeloWeb:
    """Agrupa el corpus, el modelo y la sincronizacion de inferencia."""

    def __init__(
        self,
        tokenizador_tipo: str = "palabras",
        cuantizado: bool = False,
        d_model: int = 16,
        num_heads: int = 4,
        umbral_saliencia: float | None = None,
    ) -> None:
        if tokenizador_tipo not in {"caracteres", "palabras"}:
            raise ValueError("tokenizador_tipo debe ser 'caracteres' o 'palabras'.")
        self.tokenizador_tipo = tokenizador_tipo
        self.cuantizado = cuantizado
        self.d_model = d_model
        self.num_heads = num_heads
        if umbral_saliencia is not None:
            self._validar_umbral_saliencia(umbral_saliencia)
        self.bloqueo = threading.Lock()
        (
            self.core_fundacional,
            self.core_personalidad,
        ) = self._cargar_nucleos_corpus_maximo()
        self.corpus_maximo = self._componer_corpus_maximo(
            self.core_fundacional,
            self.core_personalidad,
        )
        self.pdfs, texto_legacy_indice = self._cargar_manifest_pdfs("")
        self.corpus_local = self._cargar_corpus_local(texto_legacy_indice)
        self.texto_corpus_previo = self.corpus_local
        self.tokenizador = self._crear_tokenizador(
            f"{self.corpus_maximo}\n{self.corpus_local}"
        )
        if not RUTA_CORPUS.is_file():
            self._escribir_atomico(RUTA_CORPUS, self.corpus_local)
        self.corpus = self._corpus_efectivo(self.corpus_local)
        self.longitud_contexto = self._cargar_ventana_contexto()
        self.umbral_saliencia = (
            umbral_saliencia
            if umbral_saliencia is not None
            else self._cargar_umbral_saliencia()
        )
        self.modelo = self._entrenar(self.corpus, self.longitud_contexto)

    @staticmethod
    def _escribir_atomico(ruta: Path, contenido: str) -> None:
        ruta.parent.mkdir(parents=True, exist_ok=True)
        temporal: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=ruta.parent,
                prefix=f".{ruta.name}.",
                delete=False,
            ) as archivo:
                temporal = Path(archivo.name)
                archivo.write(contenido)
                archivo.flush()
            temporal.replace(ruta)
        finally:
            if temporal is not None and temporal.exists():
                temporal.unlink()

    @staticmethod
    def _rutas_nucleos_corpus_maximo() -> tuple[Path, Path]:
        return (
            RUTA_CORPUS_MAXIMO.with_name("core_fundacional.txt"),
            RUTA_CORPUS_MAXIMO.with_name("core_personalidad.txt"),
        )

    @staticmethod
    def _componer_corpus_maximo(fundacional: str, personalidad: str) -> str:
        personalidad_ponderada = "\n\n".join([personalidad.strip()] * 3)
        return f"{fundacional.strip()}\n\n{personalidad_ponderada}".strip()

    @classmethod
    def _cargar_nucleos_corpus_maximo(cls) -> tuple[str, str]:
        ruta_fundacional, ruta_personalidad = cls._rutas_nucleos_corpus_maximo()
        if ruta_fundacional.is_file() != ruta_personalidad.is_file():
            raise RuntimeError(
                "La estructura del corpus máximo está incompleta: se necesitan "
                "core_fundacional.txt y core_personalidad.txt."
            )
        if ruta_fundacional.is_file():
            fundacional = ruta_fundacional.read_text(encoding="utf-8").strip()
            personalidad = ruta_personalidad.read_text(encoding="utf-8").strip()
        elif RUTA_CORPUS_MAXIMO.is_file():
            legado = RUTA_CORPUS_MAXIMO.read_text(encoding="utf-8").strip()
            bloques = re.split(r"(?m)(?=^Usuario:)", legado)
            fundacionales = [
                bloque.strip() for bloque in bloques
                if bloque.strip() and "Pensamiento:" not in bloque
            ]
            personalidades = [
                bloque.strip() for bloque in bloques
                if bloque.strip() and "Pensamiento:" in bloque
            ]
            fundacional = migrar_dialogos_legacy("\n\n".join(fundacionales))
            personalidad = (
                migrar_dialogos_legacy("\n\n".join(personalidades))
                if personalidades
                else ""
            )
            if not personalidad:
                personalidad = (
                    Path(__file__).parent / "core_personalidad.txt"
                ).read_text(encoding="utf-8").strip()
            if not fundacional:
                raise RuntimeError("El corpus máximo existente no tiene una base válida.")
        else:
            ruta_base = Path(__file__).parent
            fundacional = (
                ruta_base / "core_fundacional.txt"
            ).read_text(encoding="utf-8").strip()
            personalidad = (
                ruta_base / "core_personalidad.txt"
            ).read_text(encoding="utf-8").strip()

        corpus_compuesto = cls._componer_corpus_maximo(fundacional, personalidad)
        if not fundacional or not personalidad:
            raise RuntimeError("Los núcleos del corpus máximo no pueden estar vacíos.")
        try:
            validar_corpus_estructurado(fundacional)
            validar_corpus_estructurado(personalidad)
        except ValueError as error:
            raise RuntimeError(
                f"Los núcleos del corpus máximo no tienen delimitadores válidos: {error}"
            ) from error
        if len(corpus_compuesto) > MAXIMO_CARACTERES_CORPUS_MAXIMO:
            raise RuntimeError("Los núcleos del corpus máximo superan el límite permitido.")
        rutas = {
            ruta_fundacional: fundacional + "\n",
            ruta_personalidad: personalidad + "\n",
            RUTA_CORPUS_MAXIMO: corpus_compuesto + "\n",
        }
        anteriores = {
            ruta: ruta.read_text(encoding="utf-8") if ruta.is_file() else None
            for ruta in rutas
        }
        try:
            for ruta, contenido in rutas.items():
                actual = anteriores[ruta]
                if actual is None or actual.strip() != contenido.strip():
                    cls._escribir_atomico(ruta, contenido)
        except Exception:
            for ruta, contenido in anteriores.items():
                cls._restaurar_archivo(ruta, contenido)
            raise
        return fundacional, personalidad

    def _corpus_efectivo(self, corpus_local: str) -> str:
        corpus_local_limpio = PATRON_BITACORA_LEGADA.sub("", corpus_local).strip()
        return f"{self.corpus_maximo}\n\n{corpus_local_limpio}".strip()

    def actualizar_corpus_maximo(
        self,
        core_fundacional: str,
        core_personalidad: str | None = None,
    ) -> int:
        if core_personalidad is None:
            core_personalidad = self.core_personalidad
        if not core_fundacional.strip() or not core_personalidad.strip():
            raise ValueError("Los dos núcleos del corpus máximo son obligatorios.")
        fundacional = sanitizar_texto(
            core_fundacional,
            self.tokenizador,
        ).strip()
        personalidad = sanitizar_texto(
            core_personalidad,
            self.tokenizador,
        ).strip()
        validar_corpus_estructurado(fundacional)
        validar_corpus_estructurado(personalidad)
        nuevo_corpus = self._componer_corpus_maximo(fundacional, personalidad)
        if len(nuevo_corpus) > MAXIMO_CARACTERES_CORPUS_MAXIMO:
            raise ValueError(
                "El corpus máximo supera el límite de "
                f"{MAXIMO_CARACTERES_CORPUS_MAXIMO} caracteres."
            )
        with self.bloqueo:
            modelo_actualizado = self._entrenar(
                f"{nuevo_corpus}\n\n{self.corpus_local}".strip(),
                self.longitud_contexto,
            )
            ruta_fundacional, ruta_personalidad = self._rutas_nucleos_corpus_maximo()
            rutas = {
                ruta_fundacional: fundacional + "\n",
                ruta_personalidad: personalidad + "\n",
                RUTA_CORPUS_MAXIMO: nuevo_corpus + "\n",
            }
            anteriores = {
                ruta: ruta.read_text(encoding="utf-8") if ruta.is_file() else None
                for ruta in rutas
            }
            try:
                for ruta, contenido in rutas.items():
                    self._escribir_atomico(ruta, contenido)
            except Exception:
                for ruta, contenido in anteriores.items():
                    self._restaurar_archivo(ruta, contenido)
                raise
            self.core_fundacional = fundacional
            self.core_personalidad = personalidad
            self.corpus_maximo = nuevo_corpus
            self.corpus = self._corpus_efectivo(self.corpus_local)
            self.modelo = modelo_actualizado
            self.tokenizador = modelo_actualizado.tokenizador
        return len(nuevo_corpus)

    def _cargar_corpus_local(self, texto_legacy_indice: str) -> str:
        if RUTA_CORPUS.is_file():
            return RUTA_CORPUS.read_text(encoding="utf-8").strip()

        legado = ""
        if RUTA_CORPUS == RUTA_CORPUS_LOCAL and RUTA_CORPUS_ANTIGUO.is_file():
            legado = RUTA_CORPUS_ANTIGUO.read_text(encoding="utf-8").strip()
            if legado.startswith(TEXTO_INICIAL):
                legado = legado[len(TEXTO_INICIAL):].strip()
            for pdf in self.pdfs:
                texto_pdf = str(pdf["text"]).strip()
                posicion = legado.find(texto_pdf)
                if texto_pdf and posicion >= 0:
                    legado = (
                        legado[:posicion] + legado[posicion + len(texto_pdf):]
                    ).strip()

        texto_previo = legado or texto_legacy_indice
        bloques = [texto_previo] if texto_previo else []
        bloques.extend(str(pdf["text"]).strip() for pdf in self.pdfs)
        return "\n\n".join(bloque for bloque in bloques if bloque).strip()

    @staticmethod
    def _cargar_ventana_contexto() -> int:
        if not RUTA_CONFIG_MODELO.is_file():
            return VENTANA_CONTEXTO_PREDETERMINADA
        try:
            configuracion = json.loads(RUTA_CONFIG_MODELO.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"No se pudo leer la configuracion local del modelo: {error}") from error
        ventana = configuracion.get("context_window") if isinstance(configuracion, dict) else None
        if (
            not isinstance(ventana, int)
            or isinstance(ventana, bool)
            or not VENTANA_CONTEXTO_MINIMA <= ventana <= VENTANA_CONTEXTO_MAXIMA
        ):
            raise RuntimeError("La ventana de contexto guardada esta fuera del rango valido.")
        return ventana

    @staticmethod
    def _cargar_umbral_saliencia() -> float:
        if not RUTA_CONFIG_MODELO.is_file():
            return UMBRAL_SALIENCIA_PREDETERMINADO
        try:
            configuracion = json.loads(RUTA_CONFIG_MODELO.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"No se pudo leer la configuracion local del modelo: {error}"
            ) from error
        if not isinstance(configuracion, dict):
            raise RuntimeError("La configuracion local del modelo no es valida.")
        umbral = configuracion.get(
            "salience_threshold", UMBRAL_SALIENCIA_PREDETERMINADO
        )
        try:
            return ModeloWeb._validar_umbral_saliencia(umbral)
        except ValueError as error:
            raise RuntimeError(
                "El umbral de saliencia guardado esta fuera del rango valido."
            ) from error

    @staticmethod
    def _validar_umbral_saliencia(umbral: object) -> float:
        if (
            not isinstance(umbral, (int, float))
            or isinstance(umbral, bool)
            or not np.isfinite(umbral)
            or not UMBRAL_SALIENCIA_MINIMO
            <= umbral
            <= UMBRAL_SALIENCIA_MAXIMO
        ):
            raise ValueError(
                "El umbral de saliencia debe estar entre 0 y 1."
            )
        return float(umbral)

    @staticmethod
    def _cargar_manifest_pdfs(
        texto_corpus_previo: str,
    ) -> tuple[list[dict[str, object]], str]:
        if not RUTA_MANIFIESTO_PDFS.is_file():
            return [], texto_corpus_previo
        try:
            manifiesto = json.loads(RUTA_MANIFIESTO_PDFS.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"No se pudo leer el indice local de PDF: {error}") from error
        registros = manifiesto.get("pdfs") if isinstance(manifiesto, dict) else None
        if not isinstance(registros, list):
            raise RuntimeError("El indice local de PDF no tiene un formato valido.")
        texto_previo = manifiesto.get("legacy_text", texto_corpus_previo)
        if not isinstance(texto_previo, str):
            raise RuntimeError("El texto previo del corpus en el indice no es valido.")
        pdfs: list[dict[str, object]] = []
        for registro in registros:
            if (
                not isinstance(registro, dict)
                or not isinstance(registro.get("id"), str)
                or re.fullmatch(r"[0-9a-f]{32}", registro["id"]) is None
                or not isinstance(registro.get("filename"), str)
                or not isinstance(registro.get("text"), str)
                or not isinstance(registro.get("pages"), int)
                or not isinstance(registro.get("added_at"), str)
            ):
                raise RuntimeError("El indice local contiene un registro de PDF invalido.")
            if not (RUTA_PDFS_CORPUS / f"{registro['id']}.pdf").is_file():
                raise RuntimeError(
                    f"No se encontro el PDF local correspondiente a {registro['filename']}."
                )
            pdfs.append(registro)
        return pdfs, texto_previo

    def _crear_tokenizador(
        self, corpus: str
    ) -> TokenizadorBasico | TokenizadorPalabras:
        if self.tokenizador_tipo == "palabras":
            return TokenizadorPalabras(corpus)
        return TokenizadorBasico()

    def _entrenar(
        self,
        corpus: str,
        longitud_contexto: int = VENTANA_CONTEXTO_PREDETERMINADA,
    ) -> ModeloAtencionCausal:
        tokenizador = self._crear_tokenizador(corpus)
        inicializador = EmbeddingBasico(
            tokenizador=tokenizador,
            dimension=self.d_model,
            semilla=17,
        )
        inicializador.entrenar_inicializacion_red(
            corpus, epocas=8, tasa_aprendizaje=0.03
        )
        modelo = ModeloAtencionCausal(
            inicializador=inicializador,
            longitud_contexto=longitud_contexto,
            semilla=17,
            num_heads=self.num_heads,
        )
        modelo.entrenar_map(
            corpus, epocas=5, tasa_aprendizaje=0.02, tamano_lote=128
        )
        if self.cuantizado:
            modelo.habilitar_cuantizacion()
        return modelo

    def _persistir_estado_local(
        self,
        texto_local: str,
        pdfs: list[dict[str, object]],
    ) -> None:
        ruta_manifiesto = RUTA_MANIFIESTO_PDFS
        anterior_local = RUTA_CORPUS.read_text(encoding="utf-8") if RUTA_CORPUS.is_file() else None
        anterior_manifiesto = (
            ruta_manifiesto.read_text(encoding="utf-8")
            if ruta_manifiesto.is_file()
            else None
        )
        manifiesto = {
            "pdfs": pdfs,
            "legacy_text": texto_local,
        }
        try:
            self._escribir_atomico(RUTA_CORPUS, texto_local.rstrip() + "\n")
            self._escribir_atomico(
                ruta_manifiesto,
                json.dumps(manifiesto, ensure_ascii=False, indent=2) + "\n",
            )
        except Exception:
            self._restaurar_archivo(RUTA_CORPUS, anterior_local)
            self._restaurar_archivo(ruta_manifiesto, anterior_manifiesto)
            raise

    @classmethod
    def _restaurar_archivo(cls, ruta: Path, contenido: str | None) -> None:
        if contenido is None:
            if ruta.exists():
                ruta.unlink()
        else:
            cls._escribir_atomico(ruta, contenido)

    def agregar_concepto(self, texto: str) -> int:
        if not isinstance(texto, str) or not texto.strip():
            raise ValueError("El texto de memoria no puede estar vacio.")
        texto_sanitizado = sanitizar_texto(texto, self.tokenizador)
        if len(texto_sanitizado) > 10_000:
            raise ValueError("El texto de memoria supera el limite de 10000 caracteres.")
        with self.bloqueo:
            local_actualizado = "\n\n".join(
                parte for parte in (self.corpus_local, texto_sanitizado) if parte
            )
            modelo_actualizado = self._entrenar(
                self._corpus_efectivo(local_actualizado),
                self.longitud_contexto,
            )
            self._persistir_estado_local(local_actualizado, self.pdfs)
            self.corpus_local = local_actualizado
            self.texto_corpus_previo = local_actualizado
            self.corpus = self._corpus_efectivo(local_actualizado)
            self.modelo = modelo_actualizado
            self.tokenizador = modelo_actualizado.tokenizador
        return len(texto_sanitizado)

    def quitar_concepto(self, texto: str) -> int:
        if not isinstance(texto, str) or not texto.strip():
            raise ValueError("Indica el texto exacto que quieres quitar.")
        fragmento = sanitizar_texto(texto, self.tokenizador)
        with self.bloqueo:
            posicion = self.corpus_local.find(fragmento)
            if posicion < 0:
                raise ValueError("No se encontro ese texto en el corpus local.")
            local_actualizado = (
                self.corpus_local[:posicion]
                + self.corpus_local[posicion + len(fragmento):]
            )
            local_actualizado = re.sub(r"\n{3,}", "\n\n", local_actualizado).strip()
            modelo_actualizado = self._entrenar(
                self._corpus_efectivo(local_actualizado),
                self.longitud_contexto,
            )
            self._persistir_estado_local(local_actualizado, self.pdfs)
            self.corpus_local = local_actualizado
            self.texto_corpus_previo = local_actualizado
            self.corpus = self._corpus_efectivo(local_actualizado)
            self.modelo = modelo_actualizado
            self.tokenizador = modelo_actualizado.tokenizador
        return len(fragmento)

    def eliminar_pdf(self, identificador: str) -> None:
        if re.fullmatch(r"[0-9a-f]{32}", identificador) is None:
            raise ValueError("Identificador de PDF no valido.")
        with self.bloqueo:
            documento = next(
                (pdf for pdf in self.pdfs if pdf["id"] == identificador),
                None,
            )
            if documento is None:
                raise ValueError("No se encontro ese documento local.")
            ruta_pdf = RUTA_PDFS_CORPUS / f"{identificador}.pdf"
            if not ruta_pdf.is_file():
                raise FileNotFoundError(f"No se encontro el archivo {ruta_pdf.name}.")
            contenido_pdf = ruta_pdf.read_bytes()
            texto_documento = str(documento["text"])
            posicion = self.corpus_local.find(texto_documento)
            if posicion < 0:
                raise RuntimeError(
                    "El texto del documento no esta presente en corpus_local.txt."
                )
            local_actualizado = (
                self.corpus_local[:posicion]
                + self.corpus_local[posicion + len(texto_documento):]
            )
            local_actualizado = re.sub(r"\n{3,}", "\n\n", local_actualizado).strip()
            pdfs_actualizados = [
                pdf for pdf in self.pdfs if pdf["id"] != identificador
            ]
            modelo_actualizado = self._entrenar(
                self._corpus_efectivo(local_actualizado),
                self.longitud_contexto,
            )
            ruta_pdf.unlink()
            try:
                self._persistir_estado_local(local_actualizado, pdfs_actualizados)
            except Exception:
                self._escribir_bytes_atomico(ruta_pdf, contenido_pdf)
                raise
            self.pdfs = pdfs_actualizados
            self.corpus_local = local_actualizado
            self.texto_corpus_previo = local_actualizado
            self.corpus = self._corpus_efectivo(local_actualizado)
            self.modelo = modelo_actualizado
            self.tokenizador = modelo_actualizado.tokenizador

    @staticmethod
    def _escribir_bytes_atomico(ruta: Path, contenido: bytes) -> None:
        ruta.parent.mkdir(parents=True, exist_ok=True)
        temporal: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=ruta.parent,
                prefix=f".{ruta.name}.",
                delete=False,
            ) as archivo:
                temporal = Path(archivo.name)
                archivo.write(contenido)
                archivo.flush()
            temporal.replace(ruta)
        finally:
            if temporal is not None and temporal.exists():
                temporal.unlink()

    def registrar_recompensa(
        self,
        prompt: str,
        respuesta: str,
        recompensa: int,
    ) -> None:
        if recompensa not in {-1, 1}:
            raise ValueError("La recompensa debe ser -1 o 1.")
        texto = sanitizar_texto(
            f"{USUARIO} {prompt} {PENSAMIENTO} "
            f"Reviso la solicitud y la respuesta registrada. "
            f"{ASISTENTE} {respuesta} {FIN}",
            self.tokenizador,
        )
        if len(texto) < 2:
            raise ValueError("El intercambio necesita al menos dos caracteres.")
        with self.bloqueo:
            anterior = RUTA_FEEDBACK.read_text(encoding="utf-8") if RUTA_FEEDBACK.is_file() else ""
            registro = {
                "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                "reward": recompensa,
                "prompt": prompt,
                "response": respuesta,
            }
            self._escribir_atomico(
                RUTA_FEEDBACK,
                anterior + json.dumps(registro, ensure_ascii=False) + "\n",
            )
            try:
                self.modelo.ajuste_online(
                    texto,
                    tasa_aprendizaje=0.0001,
                    recompensa=recompensa,
                )
            except Exception:
                self._restaurar_archivo(RUTA_FEEDBACK, anterior or None)
                raise

    def cambiar_ventana_contexto(self, longitud_contexto: int) -> None:
        unidad = "tokens" if self.tokenizador_tipo == "palabras" else "caracteres"
        if (
            not isinstance(longitud_contexto, int)
            or isinstance(longitud_contexto, bool)
            or not VENTANA_CONTEXTO_MINIMA
            <= longitud_contexto
            <= VENTANA_CONTEXTO_MAXIMA
        ):
            raise ValueError(
                "La ventana de contexto debe estar entre "
                f"{VENTANA_CONTEXTO_MINIMA} y {VENTANA_CONTEXTO_MAXIMA} {unidad}."
            )
        self.configurar_modelo(longitud_contexto=longitud_contexto)

    def configurar_modelo(
        self,
        longitud_contexto: int | None = None,
        umbral_saliencia: float | None = None,
    ) -> None:
        if longitud_contexto is None and umbral_saliencia is None:
            raise ValueError("Indica al menos una opcion de configuracion.")
        if longitud_contexto is not None:
            unidad = "tokens" if self.tokenizador_tipo == "palabras" else "caracteres"
            if (
                not isinstance(longitud_contexto, int)
                or isinstance(longitud_contexto, bool)
                or not VENTANA_CONTEXTO_MINIMA
                <= longitud_contexto
                <= VENTANA_CONTEXTO_MAXIMA
            ):
                raise ValueError(
                    "La ventana de contexto debe estar entre "
                    f"{VENTANA_CONTEXTO_MINIMA} y {VENTANA_CONTEXTO_MAXIMA} {unidad}."
                )
        umbral_validado = (
            self._validar_umbral_saliencia(umbral_saliencia)
            if umbral_saliencia is not None
            else None
        )
        with self.bloqueo:
            contexto_actualizado = (
                longitud_contexto
                if longitud_contexto is not None
                else self.longitud_contexto
            )
            modelo_actualizado = (
                self._entrenar(self.corpus, longitud_contexto=contexto_actualizado)
                if contexto_actualizado != self.longitud_contexto
                else self.modelo
            )
            configuracion: dict[str, object] = {}
            if RUTA_CONFIG_MODELO.is_file():
                try:
                    contenido = json.loads(
                        RUTA_CONFIG_MODELO.read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError) as error:
                    raise RuntimeError(
                        f"No se pudo leer la configuracion local del modelo: {error}"
                    ) from error
                if not isinstance(contenido, dict):
                    raise RuntimeError(
                        "La configuracion local del modelo no es valida."
                    )
                configuracion = contenido
            configuracion["context_window"] = contexto_actualizado
            umbral_actualizado = (
                umbral_validado
                if umbral_validado is not None
                else self.umbral_saliencia
            )
            configuracion["salience_threshold"] = umbral_actualizado
            self._escribir_atomico(
                RUTA_CONFIG_MODELO,
                json.dumps(configuracion, indent=2) + "\n",
            )
            self.modelo = modelo_actualizado
            self.longitud_contexto = contexto_actualizado
            self.umbral_saliencia = umbral_actualizado

    def evaluar_y_guardar_memoria(
        self, prompt: str, respuesta: str
    ) -> tuple[float, bool]:
        if not hasattr(self, "umbral_saliencia") or not hasattr(self, "corpus"):
            return 0.0, False
        indice = IndiceMemoriaRAG(self.corpus)
        evaluacion = indice.evaluar(prompt, respuesta)
        if evaluacion.puntuacion < self.umbral_saliencia:
            return evaluacion.puntuacion, False
        ficha = resumir_interaccion(prompt, respuesta)
        self.agregar_concepto(ficha)
        logging.info(
            "Memoria autónoma guardada (saliencia %.3f, umbral %.3f).",
            evaluacion.puntuacion,
            self.umbral_saliencia,
        )
        return evaluacion.puntuacion, True

    def listar_pdfs(self) -> list[dict[str, object]]:
        registros = [
            {
                "id": pdf["id"],
                "filename": pdf["filename"],
                "pages": pdf["pages"],
                "characters": len(str(pdf["text"])),
                "added_at": pdf["added_at"],
                "kind": "pdf",
            }
            for pdf in self.pdfs
        ]
        if self.texto_corpus_previo:
            registros.append({
                "id": "corpus-local",
                "filename": "Texto previo de corpus_local.txt",
                "pages": 0,
                "characters": len(self.texto_corpus_previo),
                "added_at": "",
                "kind": "text",
            })
        return registros

    def detalle_pdf(self, identificador: str) -> dict[str, object] | None:
        if identificador == "corpus-local" and self.texto_corpus_previo:
            return {
                "filename": "Texto previo de corpus_local.txt",
                "text": self.texto_corpus_previo,
                "kind": "text",
            }
        for pdf in self.pdfs:
            if pdf["id"] == identificador:
                return {
                    "filename": pdf["filename"],
                    "text": pdf["text"],
                    "pages": pdf["pages"],
                    "characters": len(str(pdf["text"])),
                    "added_at": pdf["added_at"],
                    "kind": "pdf",
                }
        return None

    def obtener_pdf(self, identificador: str) -> bytes | None:
        if not any(pdf["id"] == identificador for pdf in self.pdfs):
            return None
        return (RUTA_PDFS_CORPUS / f"{identificador}.pdf").read_bytes()

    def resumen_corpus(self) -> dict[str, object]:
        return {
            "maximum_file": RUTA_CORPUS_MAXIMO.name,
            "maximum_characters": len(self.corpus_maximo),
            "local_file": RUTA_CORPUS.name,
            "local_characters": len(self.corpus_local),
            "pdf_count": len(self.pdfs),
        }

    def agregar_pdf(self, datos: bytes, nombre_archivo: str) -> dict[str, object]:
        try:
            lector = PdfReader(BytesIO(datos), strict=True)
        except (PdfReadError, OSError, ValueError) as error:
            raise ValueError(f"No se pudo leer el PDF: {error}") from error
        if len(lector.pages) > 30:
            raise ValueError("El PDF supera el limite de 30 paginas.")
        texto = "\n\n".join(pagina.extract_text() or "" for pagina in lector.pages)
        if not texto.strip():
            raise ValueError(
                "El PDF no contiene texto seleccionable. Los PDF escaneados requieren OCR."
            )
        texto = sanitizar_texto(texto, self.tokenizador)
        if len(texto) > 10_000:
            raise ValueError(
                "El PDF supera el limite de 10000 caracteres de texto extraido."
            )
        nombre_archivo = re.split(r"[\\/]", nombre_archivo)[-1].strip()
        nombre_archivo = re.sub(r"[\x00-\x1f\x7f]", "", nombre_archivo)
        if not nombre_archivo:
            nombre_archivo = "documento.pdf"
        if not nombre_archivo.lower().endswith(".pdf"):
            nombre_archivo += ".pdf"
        registro: dict[str, object] = {
            "id": uuid4().hex,
            "filename": nombre_archivo[:180],
            "text": texto,
            "pages": len(lector.pages),
            "added_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        with self.bloqueo:
            local_actualizado = "\n\n".join(
                bloque for bloque in (self.corpus_local, texto) if bloque
            )
            pdfs_actualizados = [*self.pdfs, registro]
            modelo_actualizado = self._entrenar(
                self._corpus_efectivo(local_actualizado),
                self.longitud_contexto,
            )
            RUTA_PDFS_CORPUS.mkdir(parents=True, exist_ok=True)
            ruta_pdf = RUTA_PDFS_CORPUS / f"{registro['id']}.pdf"
            self._escribir_bytes_atomico(ruta_pdf, datos)
            try:
                self._persistir_estado_local(local_actualizado, pdfs_actualizados)
            except Exception:
                ruta_pdf.unlink(missing_ok=True)
                raise
            self.corpus_local = local_actualizado
            self.texto_corpus_previo = local_actualizado
            self.corpus = self._corpus_efectivo(local_actualizado)
            self.pdfs = pdfs_actualizados
            self.modelo = modelo_actualizado
            self.tokenizador = modelo_actualizado.tokenizador
        return {
            "id": registro["id"],
            "filename": registro["filename"],
            "pages": registro["pages"],
            "characters": len(texto),
            "added_at": registro["added_at"],
            "kind": "pdf",
        }

    def responder(
        self,
        mensaje: str,
        historial: list[dict[str, str]],
        temperatura: float,
        top_k: int,
        ajuste_online: bool = False,
    ) -> tuple[str, str, bool]:
        if not 0.1 <= temperatura <= TEMPERATURA_MAXIMA:
            raise ValueError(
                f"La temperatura debe estar entre 0.1 y {TEMPERATURA_MAXIMA}."
            )
        if not 1 <= top_k <= self.tokenizador.tamano_vocabulario:
            raise ValueError(
                f"Top-k debe estar entre 1 y {self.tokenizador.tamano_vocabulario}."
            )
        partes: list[str] = []
        for turno in historial:
            if turno["role"] in {"user", "assistant"}:
                if turno["role"] == "user":
                    partes.append(
                        f"{USUARIO} {turno['content']} {PENSAMIENTO} "
                    )
                else:
                    pensamiento_previo = turno.get("thought", "")
                    partes.append(
                        f"{pensamiento_previo} {ASISTENTE} "
                        f"{turno['content']} {FIN} "
                    )
        contenido_actual = sanitizar_texto(mensaje, self.tokenizador)
        partes.append(f"{USUARIO} {contenido_actual} {PENSAMIENTO}")
        contexto_completo = sanitizar_texto("".join(partes), self.tokenizador)
        ids = self.tokenizador.codificar(contexto_completo)
        with self.bloqueo:
            memoria_comprimida = None
            longitud_reciente = self.modelo.longitud_contexto - 1
            if len(ids) > longitud_reciente:
                memoria_comprimida = self.modelo.comprimir_contexto(
                    ids[:-longitud_reciente]
                )
                ids = ids[-longitud_reciente:]
            inicio = self.tokenizador.decodificar(ids)
            generacion = self.modelo.generar(
                inicio,
                cantidad=120,
                semilla=int(np.random.SeedSequence().generate_state(1)[0]),
                temperatura=temperatura,
                top_k=top_k,
                penalizacion_repeticion=1.15,
                memoria_comprimida=memoria_comprimida,
                evitar_salto_inicial=True,
                stop_sequences=(FIN, USUARIO),
            )
            continuacion = cortar_en_turno_posterior(generacion[len(inicio):])
            pensamiento, respuesta = separar_deliberacion(continuacion)
            if not respuesta:
                raise RuntimeError(
                    "El modelo terminó la generación sin producir texto antes del salto de línea."
                )
            if ajuste_online:
                texto_ajuste = sanitizar_texto(
                    f"{USUARIO} {contenido_actual} {PENSAMIENTO} "
                    f"{pensamiento} {ASISTENTE} {respuesta} {FIN}",
                    self.tokenizador,
                )
                self.modelo.ajuste_online(texto_ajuste)
        saliencia, memoria_guardada = self.evaluar_y_guardar_memoria(
            contenido_actual,
            respuesta,
        )
        logging.info(
            "Interacción evaluada: saliencia %.3f; memoria autónoma %s.",
            saliencia,
            "guardada" if memoria_guardada else "no guardada",
        )
        return respuesta, pensamiento, ajuste_online


def crear_pdf_ejemplo(texto: str) -> bytes:
    """Crea un PDF pequeño con texto seleccionable para la galeria de ejemplos."""
    lineas = textwrap.wrap(texto, width=82, break_long_words=False)
    comandos = ["BT", "/F1 11 Tf", "72 740 Td", "15 TL"]
    for linea in lineas:
        segura = linea.encode("ascii", errors="replace").decode("ascii")
        segura = segura.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        comandos.append(f"({segura}) Tj")
        comandos.append("T*")
    comandos.append("ET")

    writer = PdfWriter()
    pagina = writer.add_blank_page(width=612, height=792)
    fuente = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    pagina[NameObject("/Resources")] = DictionaryObject({
        NameObject("/Font"): DictionaryObject({NameObject("/F1"): fuente})
    })
    contenido = DecodedStreamObject()
    contenido.set_data("\n".join(comandos).encode("ascii"))
    pagina[NameObject("/Contents")] = writer._add_object(contenido)
    salida = BytesIO()
    writer.write(salida)
    return salida.getvalue()


def crear_manejador(modelo: ModeloWeb) -> type[BaseHTTPRequestHandler]:
    autenticacion_admin = AutenticacionAdmin(RUTA_CREDENCIALES_ADMIN)

    class ManejadorHTTP(BaseHTTPRequestHandler):
        server_version = "DemoLLMLocal/1.0"

        def log_message(self, formato: str, *argumentos: object) -> None:
            logging.info("%s - %s", self.address_string(), formato % argumentos)

        def _responder(self, estado: int, contenido: bytes, tipo: str) -> None:
            self.send_response(estado)
            self.send_header("Content-Type", tipo)
            self.send_header("Content-Length", str(len(contenido)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(contenido)

        def _json(self, estado: int, datos: dict[str, object]) -> None:
            contenido = json.dumps(datos, ensure_ascii=False).encode("utf-8")
            self._responder(estado, contenido, "application/json; charset=utf-8")

        def _token_admin(self, permitir_cambio_password: bool = False) -> str | None:
            cabecera = self.headers.get("Authorization", "")
            esquema, separador, token = cabecera.partition(" ")
            if not separador or esquema.lower() != "bearer" or not token:
                self._json(401, {"error": "Inicia sesión como administrador."})
                return None
            if not autenticacion_admin.validar_token(
                token,
                permitir_cambio_password=True,
            ):
                self._json(401, {"error": "La sesión administrativa expiró. Inicia sesión otra vez."})
                return None
            if (
                autenticacion_admin.credenciales["must_change_password"]
                and not permitir_cambio_password
            ):
                self._json(
                    403,
                    {"error": "Debes cambiar la contraseña provisional antes de continuar."},
                )
                return None
            return token

        def do_GET(self) -> None:
            ruta = unquote(urlparse(self.path).path)
            if ruta == "/api/admin/corpus-maximo":
                if self._token_admin() is None:
                    return
                self._json(200, {
                    "text": modelo.corpus_maximo,
                    "core_fundacional": modelo.core_fundacional,
                    "core_personalidad": modelo.core_personalidad,
                })
                return
            if ruta == "/api/corpus":
                self._json(200, {"corpus": modelo.resumen_corpus()})
                return
            if ruta == "/api/pdfs":
                self._json(200, {"pdfs": modelo.listar_pdfs()})
                return
            if ruta == "/api/model-config":
                self._json(200, {
                    "temperature_min": 0.1,
                    "temperature_max": TEMPERATURA_MAXIMA,
                    "top_k_min": 1,
                    "top_k_max": modelo.tokenizador.tamano_vocabulario,
                    "temperature_default": 0.8,
                    "top_k_default": 8,
                    "context_window_min": VENTANA_CONTEXTO_MINIMA,
                    "context_window_max": VENTANA_CONTEXTO_MAXIMA,
                    "context_window_default": VENTANA_CONTEXTO_PREDETERMINADA,
                    "context_window": modelo.longitud_contexto,
                    "salience_threshold_min": UMBRAL_SALIENCIA_MINIMO,
                    "salience_threshold_max": UMBRAL_SALIENCIA_MAXIMO,
                    "salience_threshold_default": UMBRAL_SALIENCIA_PREDETERMINADO,
                    "salience_threshold": modelo.umbral_saliencia,
                    "context_window_unit": (
                        "tokens" if modelo.tokenizador_tipo == "palabras" else "caracteres"
                    ),
                    "tokenizer_type": modelo.tokenizador_tipo,
                    "d_model": modelo.d_model,
                    "num_heads": modelo.num_heads,
                    "quantized": modelo.cuantizado,
                })
                return
            if ruta.startswith("/api/pdfs/") and ruta.endswith("/preview"):
                identificador = ruta.removeprefix("/api/pdfs/").removesuffix("/preview")
                detalle = modelo.detalle_pdf(identificador)
                if detalle is None:
                    self._json(404, {"error": "No se encontro ese documento del corpus."})
                    return
                self._json(200, detalle)
                return
            if ruta.startswith("/api/pdfs/") and ruta.endswith(".pdf"):
                identificador = ruta.removeprefix("/api/pdfs/").removesuffix(".pdf")
                if re.fullmatch(r"[0-9a-f]{32}", identificador) is None:
                    self._json(404, {"error": "No se encontro ese PDF del corpus."})
                    return
                pdf = modelo.obtener_pdf(identificador)
                if pdf is None:
                    self._json(404, {"error": "No se encontro ese PDF del corpus."})
                    return
                self._responder(200, pdf, "application/pdf")
                return
            if ruta == "/api/examples":
                self._json(200, {
                    "examples": [
                        {"id": clave, "title": valor["titulo"],
                         "description": valor["descripcion"]}
                        for clave, valor in EJEMPLOS.items()
                    ]
                })
                return
            if ruta.startswith("/api/examples/") and ruta.endswith(".pdf"):
                clave = ruta.removeprefix("/api/examples/").removesuffix(".pdf")
                ejemplo = EJEMPLOS.get(clave)
                if ejemplo is None:
                    self._json(404, {"error": "No se encontro ese PDF de ejemplo."})
                    return
                pdf = crear_pdf_ejemplo(ejemplo["texto"])
                self._responder(200, pdf, "application/pdf")
                return
            archivo = "index.html" if ruta in {"/", "/index.html"} else ruta.lstrip("/")
            if archivo not in {"index.html", "styles.css", "app.js"}:
                self._json(404, {"error": "No se encontro la ruta solicitada."})
                return
            ruta_archivo = DIRECTORIO_WEB / archivo
            if not ruta_archivo.is_file():
                self._json(404, {"error": "No se encontro el archivo web solicitado."})
                return
            tipos = {
                ".html": "text/html; charset=utf-8",
                ".css": "text/css; charset=utf-8",
                ".js": "text/javascript; charset=utf-8",
            }
            self._responder(200, ruta_archivo.read_bytes(), tipos[ruta_archivo.suffix])

        def do_POST(self) -> None:
            ruta = urlparse(self.path).path
            longitud = self.headers.get("Content-Length")
            if longitud is None or not longitud.isdecimal():
                self._json(411, {"error": "Se requiere Content-Length valido."})
                return
            tamano = int(longitud)
            limite = MAXIMO_PDF_BYTES if ruta == "/api/pdf" else 1_000_000
            if tamano > limite:
                self._json(413, {"error": "El contenido supera el limite permitido."})
                return
            cuerpo = self.rfile.read(tamano)
            try:
                if ruta == "/api/admin/login":
                    self._admin_login(cuerpo)
                elif ruta == "/api/admin/password":
                    self._admin_cambiar_password(cuerpo)
                elif ruta == "/api/admin/logout":
                    self._admin_logout()
                elif ruta == "/api/chat":
                    self._chat(cuerpo)
                elif ruta == "/api/pdf":
                    self._pdf(cuerpo)
                elif ruta == "/corpus/agregar":
                    self._corpus_agregar(cuerpo)
                elif ruta == "/corpus/quitar":
                    self._corpus_quitar(cuerpo)
                elif ruta == "/api/memory":
                    self._memoria(cuerpo)
                elif ruta == "/api/feedback":
                    self._feedback(cuerpo)
                elif ruta == "/api/model-config":
                    self._configurar_modelo(cuerpo)
                else:
                    self._json(404, {"error": "No se encontro la ruta solicitada."})
            except PermissionError as error:
                self._json(401, {"error": str(error)})
            except (ValueError, json.JSONDecodeError) as error:
                self._json(400, {"error": str(error)})
            except Exception:
                logging.exception("Error interno procesando la ruta %s.", ruta)
                self._json(500, {"error": "Error interno al procesar la solicitud."})

        def do_PUT(self) -> None:
            ruta = urlparse(self.path).path
            if ruta != "/api/admin/corpus-maximo":
                self._json(404, {"error": "No se encontro la ruta solicitada."})
                return
            if self._token_admin() is None:
                return
            longitud = self.headers.get("Content-Length")
            if longitud is None or not longitud.isdecimal():
                self._json(411, {"error": "Se requiere Content-Length valido."})
                return
            tamano = int(longitud)
            if tamano > MAXIMO_CARACTERES_CORPUS_MAXIMO * 4:
                self._json(413, {"error": "El corpus supera el limite permitido."})
                return
            try:
                solicitud = self._leer_json(self.rfile.read(tamano), "administracion")
                texto = solicitud.get("text")
                fundacional = solicitud.get("core_fundacional")
                personalidad = solicitud.get("core_personalidad")
                if fundacional is not None or personalidad is not None:
                    if not isinstance(fundacional, str) or not isinstance(personalidad, str):
                        raise ValueError(
                            "Debes enviar ambos núcleos como cadenas de texto."
                        )
                    caracteres = modelo.actualizar_corpus_maximo(
                        fundacional,
                        personalidad,
                    )
                elif isinstance(texto, str):
                    caracteres = modelo.actualizar_corpus_maximo(texto)
                else:
                    raise ValueError(
                        "Envía los núcleos fundacional y de personalidad."
                    )
            except (ValueError, json.JSONDecodeError) as error:
                self._json(400, {"error": str(error)})
                return
            except Exception:
                logging.exception("Error actualizando el corpus máximo.")
                self._json(500, {"error": "No se pudo actualizar el corpus máximo."})
                return
            self._json(200, {
                "message": (
                    f"Corpus máximo actualizado ({caracteres} caracteres) y "
                    "modelo reentrenado."
                ),
                "corpus": modelo.resumen_corpus(),
            })

        def do_DELETE(self) -> None:
            ruta = unquote(urlparse(self.path).path)
            prefijo = "/api/pdfs/"
            if not ruta.startswith(prefijo):
                self._json(404, {"error": "No se encontro la ruta solicitada."})
                return
            identificador = ruta.removeprefix(prefijo)
            try:
                modelo.eliminar_pdf(identificador)
            except ValueError as error:
                self._json(404, {"error": str(error)})
                return
            except Exception:
                logging.exception("Error interno eliminando un PDF del corpus local.")
                self._json(500, {"error": "No se pudo eliminar el documento local."})
                return
            self._json(200, {"deleted": identificador, "corpus": modelo.resumen_corpus()})

        def _chat(self, cuerpo: bytes) -> None:
            if self.headers.get_content_type() != "application/json":
                self._json(415, {"error": "El chat requiere una solicitud JSON."})
                return
            solicitud = json.loads(cuerpo.decode("utf-8"))
            if not isinstance(solicitud, dict):
                raise ValueError("La solicitud de chat debe ser un objeto JSON.")
            mensaje = solicitud.get("message")
            historial = solicitud.get("history", [])
            ajuste_online = solicitud.get("online_fine_tuning", False)
            if not isinstance(mensaje, str) or not mensaje.strip():
                raise ValueError("Escribe un mensaje antes de enviarlo.")
            if len(mensaje) > MAXIMO_CARACTERES_CHAT:
                raise ValueError("El mensaje supera el limite de 1200 caracteres.")
            if not isinstance(historial, list) or len(historial) > MAXIMO_HISTORIAL_CHAT:
                raise ValueError("El historial de chat no es valido.")
            for turno in historial:
                if (
                    not isinstance(turno, dict)
                    or turno.get("role") not in {"user", "assistant"}
                    or not isinstance(turno.get("content"), str)
                    or len(turno["content"]) > MAXIMO_CARACTERES_CHAT
                    or (
                        "thought" in turno
                        and (
                            turno.get("role") != "assistant"
                            or not isinstance(turno.get("thought"), str)
                            or len(turno["thought"]) > MAXIMO_CARACTERES_CHAT
                        )
                    )
                ):
                    raise ValueError(
                        "Cada turno del historial debe tener rol y texto validos "
                        "de hasta 1200 caracteres."
                    )
            if not isinstance(ajuste_online, bool):
                raise ValueError("online_fine_tuning debe ser un valor booleano.")
            temperatura = solicitud.get("temperature", 0.8)
            top_k = solicitud.get("top_k", 8)
            if not isinstance(temperatura, (int, float)) or isinstance(temperatura, bool):
                raise ValueError("La temperatura debe ser numerica.")
            if not isinstance(top_k, int) or isinstance(top_k, bool):
                raise ValueError("Top-k debe ser un entero.")
            respuesta, pensamiento, ajuste_aplicado = modelo.responder(
                mensaje,
                historial,
                float(temperatura),
                top_k,
                ajuste_online=ajuste_online,
            )
            self._json(200, {
                "reply": respuesta,
                "thought": pensamiento,
                "online_fine_tuned": ajuste_aplicado,
            })

        def _configurar_modelo(self, cuerpo: bytes) -> None:
            if self.headers.get_content_type() != "application/json":
                self._json(415, {"error": "La configuracion requiere una solicitud JSON."})
                return
            solicitud = json.loads(cuerpo.decode("utf-8"))
            if not isinstance(solicitud, dict):
                raise ValueError("La configuracion debe ser un objeto JSON.")
            ventana = solicitud.get("context_window")
            umbral = solicitud.get("salience_threshold")
            if ventana is None and umbral is None:
                raise ValueError("Indica una ventana o un umbral de saliencia.")
            if ventana is not None and (
                not isinstance(ventana, int) or isinstance(ventana, bool)
            ):
                raise ValueError("La ventana de contexto debe ser un entero.")
            if umbral is not None:
                modelo._validar_umbral_saliencia(umbral)
            modelo.configurar_modelo(
                longitud_contexto=ventana,
                umbral_saliencia=umbral,
            )
            self._json(200, {
                "message": "Configuracion del modelo actualizada.",
                "context_window": modelo.longitud_contexto,
                "salience_threshold": modelo.umbral_saliencia,
            })

        def _pdf(self, cuerpo: bytes) -> None:
            if self.headers.get_content_type() != "application/pdf":
                self._json(415, {"error": "Sube un archivo PDF valido."})
                return
            if not cuerpo.startswith(b"%PDF-"):
                raise ValueError("El archivo enviado no parece ser un PDF valido.")
            nombre_archivo = unquote(
                self.headers.get("X-File-Name", "documento.pdf")
            )
            documento = modelo.agregar_pdf(cuerpo, nombre_archivo)
            self._json(200, {
                "message": (
                    f"{documento['filename']} añadido al corpus "
                    f"({documento['characters']} caracteres)."
                ),
                "document": documento,
            })

        def _corpus_agregar(self, cuerpo: bytes) -> None:
            solicitud = self._leer_json(cuerpo, "corpus")
            texto = solicitud.get("text")
            if not isinstance(texto, str):
                raise ValueError("El campo text debe ser una cadena.")
            caracteres = modelo.agregar_concepto(texto)
            self._json(200, {
                "message": f"Se añadieron {caracteres} caracteres al corpus local.",
                "corpus": modelo.resumen_corpus(),
            })

        def _corpus_quitar(self, cuerpo: bytes) -> None:
            solicitud = self._leer_json(cuerpo, "corpus")
            identificador = solicitud.get("pdf_id")
            if identificador is not None:
                if not isinstance(identificador, str):
                    raise ValueError("pdf_id debe ser una cadena.")
                modelo.eliminar_pdf(identificador)
                self._json(200, {
                    "message": "Documento eliminado del corpus local.",
                    "corpus": modelo.resumen_corpus(),
                })
                return
            texto = solicitud.get("text")
            if not isinstance(texto, str):
                raise ValueError("Indica pdf_id o el texto exacto que quieres quitar.")
            caracteres = modelo.quitar_concepto(texto)
            self._json(200, {
                "message": f"Se quitaron {caracteres} caracteres del corpus local.",
                "corpus": modelo.resumen_corpus(),
            })

        def _leer_json(self, cuerpo: bytes, nombre: str) -> dict[str, object]:
            if self.headers.get_content_type() != "application/json":
                raise ValueError(f"La ruta de {nombre} requiere una solicitud JSON.")
            solicitud = json.loads(cuerpo.decode("utf-8"))
            if not isinstance(solicitud, dict):
                raise ValueError(f"La solicitud de {nombre} debe ser un objeto JSON.")
            return solicitud

        def _admin_login(self, cuerpo: bytes) -> None:
            solicitud = self._leer_json(cuerpo, "administrador")
            usuario = solicitud.get("username")
            contrasena = solicitud.get("password")
            if not isinstance(usuario, str) or not isinstance(contrasena, str):
                raise ValueError("Indica el usuario y la contraseña.")
            token, debe_cambiar = autenticacion_admin.iniciar_sesion(
                usuario,
                contrasena,
            )
            self._json(200, {
                "token": token,
                "must_change_password": debe_cambiar,
            })

        def _admin_cambiar_password(self, cuerpo: bytes) -> None:
            token = self._token_admin(permitir_cambio_password=True)
            if token is None:
                return
            solicitud = self._leer_json(cuerpo, "administrador")
            nueva = solicitud.get("new_password")
            if not isinstance(nueva, str):
                raise ValueError("Indica la nueva contraseña.")
            autenticacion_admin.cambiar_password(token, nueva)
            self._json(200, {"message": "Contraseña administrativa actualizada."})

        def _admin_logout(self) -> None:
            token = self._token_admin(permitir_cambio_password=True)
            if token is None:
                return
            autenticacion_admin.cerrar_sesion(token)
            self._json(200, {"message": "Sesión administrativa cerrada."})

        def _memoria(self, cuerpo: bytes) -> None:
            solicitud = self._leer_json(cuerpo, "memoria")
            prompt = solicitud.get("prompt")
            respuesta = solicitud.get("response")
            titulo = solicitud.get("title", "Nota de memoria")
            if (
                not isinstance(prompt, str)
                or not prompt.strip()
                or not isinstance(respuesta, str)
                or not respuesta.strip()
            ):
                raise ValueError("La nota necesita el mensaje y la respuesta.")
            if len(prompt) > MAXIMO_CARACTERES_CHAT or len(respuesta) > MAXIMO_CARACTERES_CHAT:
                raise ValueError("El intercambio supera el limite de caracteres por mensaje.")
            if not isinstance(titulo, str) or not titulo.strip():
                raise ValueError("El titulo de memoria no puede estar vacio.")
            from generador_pdf import crear_pdf_memoria

            nombre = f"memoria-{uuid4().hex}.pdf"
            contenido = (
                f"{USUARIO} {prompt} {PENSAMIENTO} "
                f"Registro una respuesta previa con el contexto disponible. "
                f"{ASISTENTE} {respuesta} {FIN}"
            )
            with tempfile.TemporaryDirectory(prefix="memoria-staging-") as temporal:
                pdf_temporal = crear_pdf_memoria(
                    titulo,
                    contenido,
                    nombre,
                    temporal,
                )
                documento = modelo.agregar_pdf(
                    pdf_temporal.read_bytes(),
                    nombre,
                )
            self._json(200, {
                "message": "Intercambio guardado como memoria PDF y añadido al corpus local.",
                "document": documento,
                "download_url": f"/api/pdfs/{documento['id']}.pdf",
            })

        def _feedback(self, cuerpo: bytes) -> None:
            solicitud = self._leer_json(cuerpo, "feedback")
            prompt = solicitud.get("prompt")
            respuesta = solicitud.get("response")
            recompensa = solicitud.get("reward")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("El feedback necesita el mensaje del usuario.")
            if not isinstance(respuesta, str) or not respuesta.strip():
                raise ValueError("El feedback necesita la respuesta del modelo.")
            if len(prompt) > MAXIMO_CARACTERES_CHAT or len(respuesta) > MAXIMO_CARACTERES_CHAT:
                raise ValueError("El intercambio supera el limite de caracteres por mensaje.")
            if not isinstance(recompensa, int) or isinstance(recompensa, bool):
                raise ValueError("reward debe ser el entero 1 o -1.")
            if recompensa not in {-1, 1}:
                raise ValueError("reward debe ser el entero 1 o -1.")
            modelo.registrar_recompensa(prompt, respuesta, recompensa)
            etiqueta = "positiva" if recompensa > 0 else "negativa"
            self._json(200, {
                "message": f"Feedback {etiqueta} registrado y aplicado como recompensa.",
                "reward": recompensa,
            })

        def log_error(self, formato: str, *argumentos: object) -> None:
            logging.error(formato, *argumentos)

    return ManejadorHTTP


def main() -> None:
    parser = argparse.ArgumentParser(description="Servidor web local del modelo causal.")
    parser.add_argument(
        "--cuantizado",
        action="store_true",
        help="Usa pesos int8 para las matrices densas durante inferencia MAP.",
    )
    parser.add_argument(
        "--tokenizador",
        choices=("caracteres", "palabras"),
        default="palabras",
        help="Tokenizacion por palabras frecuentes (predeterminada) o caracteres.",
    )
    parser.add_argument(
        "--umbral-saliencia",
        type=float,
        default=None,
        help="Umbral de 0 a 1 para guardar interacciones novedosas en el corpus local.",
    )
    parser.add_argument("--d-model", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=4)
    argumentos = parser.parse_args()
    if argumentos.d_model < 1 or argumentos.num_heads < 1:
        parser.error("--d-model y --num-heads deben ser positivos.")
    if argumentos.d_model % argumentos.num_heads:
        parser.error("--d-model debe ser divisible por --num-heads.")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    print("Preparando el modelo local. La primera carga puede tardar unos segundos.")
    modelo = ModeloWeb(
        tokenizador_tipo=argumentos.tokenizador,
        cuantizado=argumentos.cuantizado,
        d_model=argumentos.d_model,
        num_heads=argumentos.num_heads,
        umbral_saliencia=argumentos.umbral_saliencia,
    )
    servidor = ThreadingHTTPServer(
        ("127.0.0.1", 8765), crear_manejador(modelo)
    )
    print("Interfaz lista en http://127.0.0.1:8765")
    print("El servidor escucha solo en este equipo. Presiona Ctrl+C para detenerlo.")
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        print("\nCerrando servidor local.")
    finally:
        servidor.server_close()


if __name__ == "__main__":
    main()
