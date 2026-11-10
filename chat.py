"""Chat de consola experimental con el modelo causal."""

import argparse
from pathlib import Path

from atencion_causal import ModeloAtencionCausal
from embedding_basico import EmbeddingBasico
from formato_conversacion import (
    ASISTENTE,
    FIN,
    PENSAMIENTO,
    USUARIO,
    separar_deliberacion,
)
from pdf_a_corpus import sanitizar_texto
from tokenizador_basico import TokenizadorBasico
from tokenizador_palabras import TokenizadorPalabras


def crear_modelo(
    ruta_corpus: Path,
    longitud_contexto: int,
    dimension: int,
    epocas_red: int,
    epocas_atencion: int,
    inferencia: str,
    semilla: int,
    tokenizador_tipo: str = "palabras",
    cuantizado: bool = False,
    num_heads: int = 4,
    ruta_corpus_maximo: Path = Path("corpus_maximo.txt"),
) -> tuple[ModeloAtencionCausal, TokenizadorBasico | TokenizadorPalabras]:
    if tokenizador_tipo not in {"caracteres", "palabras"}:
        raise ValueError("tokenizador_tipo debe ser 'caracteres' o 'palabras'.")
    if cuantizado and inferencia != "map":
        raise ValueError("La cuantizacion int8 requiere inferencia MAP.")
    if not ruta_corpus.is_file() and ruta_corpus == Path("corpus_local.txt"):
        ruta_legacy = Path("corpus.txt")
        if ruta_legacy.is_file():
            ruta_corpus = ruta_legacy
    if not ruta_corpus.is_file():
        raise FileNotFoundError(
            f"No existe {ruta_corpus}. Convierte primero un PDF con pdf_a_corpus.py."
        )
    texto_local = ruta_corpus.read_text(encoding="utf-8")
    if not ruta_corpus_maximo.is_file():
        raise FileNotFoundError(
            f"No existe el corpus máximo protegido: {ruta_corpus_maximo}."
        )
    texto = (
        ruta_corpus_maximo.read_text(encoding="utf-8").strip()
        + "\n\n"
        + texto_local.strip()
    )
    if tokenizador_tipo == "palabras":
        tokenizador = TokenizadorPalabras(texto)
        texto = sanitizar_texto(texto, tokenizador)
    else:
        tokenizador = TokenizadorBasico()
        texto = tokenizador.decodificar(tokenizador.codificar(texto))

    inicializador = EmbeddingBasico(
        tokenizador, dimension=dimension, semilla=semilla
    )
    print("Inicializando embeddings con la red de siguiente caracter...")
    perdida_red = inicializador.entrenar_inicializacion_red(
        texto, epocas=epocas_red, tasa_aprendizaje=0.03
    )
    print(f"Perdida de inicializacion: {perdida_red[0]:.4f} -> {perdida_red[-1]:.4f}")

    modelo = ModeloAtencionCausal(
        inicializador,
        longitud_contexto=longitud_contexto,
        semilla=semilla,
        num_heads=num_heads,
    )
    print(f"Entrenando bloque causal con inferencia {inferencia}...")
    if inferencia == "map":
        historial = modelo.entrenar_map(
            texto, epocas=epocas_atencion, tasa_aprendizaje=0.01
        )
    else:
        historial = modelo.inferencia_variacional(
            texto, epocas=epocas_atencion, tasa_aprendizaje=0.005, muestras=2
        )
    print(f"Objetivo final: {historial[-1]:.4f}")
    if cuantizado:
        modelo.habilitar_cuantizacion()
        print("Matrices densas cuantizadas a int8 para inferencia.")
    return modelo, tokenizador


def conversar(
    modelo: ModeloAtencionCausal,
    tokenizador: TokenizadorBasico | TokenizadorPalabras,
    temperatura: float,
    top_k: int,
    cantidad_maxima: int,
    penalizacion_repeticion: float,
    tamano_ngram_sin_repetir: int,
    inferencia: str,
    semilla: int,
) -> None:
    print("Chat experimental. Escribe /salir para terminar. Responde con /limpiar para vaciar el historial.")
    historial: list[str] = []
    turno = 0
    while True:
        try:
            mensaje = input("Tu: ").strip()
        except EOFError:
            print()
            return
        if not mensaje:
            continue
        mensaje = tokenizador.normalizar(mensaje)
        if mensaje == "/salir":
            return
        if mensaje == "/limpiar":
            historial.clear()
            print("Historial de chat vaciado.")
            continue
        try:
            tokenizador.codificar(mensaje)
        except ValueError as error:
            print(f"No puedo procesar ese texto: {error}")
            continue

        historial.append(f"{USUARIO} {mensaje} {PENSAMIENTO}")
        prompt = "\n".join(historial)
        contexto_ids = tokenizador.codificar(prompt)[-modelo.longitud_contexto:]
        contexto = tokenizador.decodificar(contexto_ids)
        try:
            respuesta_completa = modelo.generar(
                contexto,
                cantidad=cantidad_maxima,
                temperatura=temperatura,
                top_k=top_k,
                penalizacion_repeticion=penalizacion_repeticion,
                tamano_ngram_sin_repetir=tamano_ngram_sin_repetir,
                inferencia=inferencia,
                semilla=semilla + turno,
                evitar_salto_inicial=True,
                stop_sequences=(FIN, USUARIO),
            )
        except ValueError as error:
            print(f"No se pudo generar la respuesta: {error}")
            historial.pop()
            continue
        pensamiento, respuesta = separar_deliberacion(
            respuesta_completa[len(contexto):]
        )
        if not respuesta:
            respuesta = "(el modelo no genero texto; prueba otra temperatura o un corpus distinto)"
        print(f"Asistente: {respuesta}")
        historial[-1] = (
            f"{USUARIO} {mensaje} {PENSAMIENTO} {pensamiento} "
            f"{ASISTENTE} {respuesta} {FIN}"
        )
        turno += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat de consola del modelo causal (demo experimental).")
    parser.add_argument(
        "--corpus-local",
        type=Path,
        default=Path("corpus_local.txt"),
        help="Corpus editable local (si no existe se usa corpus.txt como legado).",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        help="Alias legado de --corpus-local.",
    )
    parser.add_argument(
        "--corpus-maximo",
        type=Path,
        default=Path("corpus_maximo.txt"),
        help="Corpus base protegido, concatenado antes del corpus local.",
    )
    parser.add_argument("--ventana", type=int, default=64)
    parser.add_argument("--dimension", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--epocas-red", type=int, default=150)
    parser.add_argument("--epocas-atencion", type=int, default=100)
    parser.add_argument("--inferencia", choices=("map", "variacional"), default="map")
    parser.add_argument(
        "--tokenizador", choices=("caracteres", "palabras"), default="palabras"
    )
    parser.add_argument(
        "--cuantizado",
        action="store_true",
        help="Cuantiza matrices densas int8 tras entrenar (solo inferencia MAP).",
    )
    parser.add_argument("--temperatura", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--penalizacion-repeticion", type=float, default=1.1)
    parser.add_argument("--ngram-sin-repetir", type=int, default=4)
    parser.add_argument("--max-caracteres", type=int, default=240)
    parser.add_argument("--semilla", type=int, default=7)
    argumentos = parser.parse_args()
    if argumentos.ventana < 1 or argumentos.dimension < 1 or argumentos.num_heads < 1:
        parser.error("--ventana, --dimension y --num-heads deben ser positivos.")
    if argumentos.dimension % argumentos.num_heads:
        parser.error("--dimension debe ser divisible por --num-heads.")
    if argumentos.max_caracteres < 1:
        parser.error("--max-caracteres debe ser positivo.")
    if argumentos.top_k < 1:
        parser.error("--top-k debe ser positivo.")

    modelo, tokenizador = crear_modelo(
        argumentos.corpus or argumentos.corpus_local,
        argumentos.ventana,
        argumentos.dimension,
        argumentos.epocas_red,
        argumentos.epocas_atencion,
        argumentos.inferencia,
        argumentos.semilla,
        argumentos.tokenizador,
        argumentos.cuantizado,
        argumentos.num_heads,
        argumentos.corpus_maximo,
    )
    conversar(
        modelo,
        tokenizador,
        argumentos.temperatura,
        argumentos.top_k,
        argumentos.max_caracteres,
        argumentos.penalizacion_repeticion,
        argumentos.ngram_sin_repetir,
        argumentos.inferencia,
        argumentos.semilla,
    )


if __name__ == "__main__":
    main()
