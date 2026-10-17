"""Chat de consola experimental con el modelo causal de caracteres."""

import argparse
from pathlib import Path

from atencion_causal import ModeloAtencionCausal
from embedding_basico import EmbeddingBasico
from tokenizador_basico import TokenizadorBasico


def crear_modelo(
    ruta_corpus: Path,
    longitud_contexto: int,
    dimension: int,
    epocas_red: int,
    epocas_atencion: int,
    inferencia: str,
    semilla: int,
) -> tuple[ModeloAtencionCausal, TokenizadorBasico]:
    if not ruta_corpus.is_file():
        raise FileNotFoundError(
            f"No existe {ruta_corpus}. Convierte primero un PDF con pdf_a_corpus.py."
        )
    texto = ruta_corpus.read_text(encoding="utf-8")
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
        inicializador, longitud_contexto=longitud_contexto, semilla=semilla
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
    return modelo, tokenizador


def conversar(
    modelo: ModeloAtencionCausal,
    tokenizador: TokenizadorBasico,
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

        historial.append(f"Usuario: {mensaje}\nAsistente:")
        prompt = "\n".join(historial)
        contexto = prompt[-modelo.longitud_contexto:]
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
            )
        except ValueError as error:
            print(f"No se pudo generar la respuesta: {error}")
            historial.pop()
            continue
        respuesta = respuesta_completa[len(contexto):].split("\n", maxsplit=1)[0].strip()
        if not respuesta:
            respuesta = "(el modelo no genero texto; prueba otra temperatura o un corpus distinto)"
        print(f"Asistente: {respuesta}")
        historial[-1] += f" {respuesta}"
        turno += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat de consola de caracteres (demo experimental).")
    parser.add_argument("--corpus", type=Path, default=Path("corpus.txt"))
    parser.add_argument("--ventana", type=int, default=64)
    parser.add_argument("--dimension", type=int, default=16)
    parser.add_argument("--epocas-red", type=int, default=150)
    parser.add_argument("--epocas-atencion", type=int, default=100)
    parser.add_argument("--inferencia", choices=("map", "variacional"), default="map")
    parser.add_argument("--temperatura", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--penalizacion-repeticion", type=float, default=1.1)
    parser.add_argument("--ngram-sin-repetir", type=int, default=4)
    parser.add_argument("--max-caracteres", type=int, default=240)
    parser.add_argument("--semilla", type=int, default=7)
    argumentos = parser.parse_args()
    if argumentos.ventana < 1 or argumentos.dimension < 1:
        parser.error("--ventana y --dimension deben ser positivos.")
    if argumentos.max_caracteres < 1:
        parser.error("--max-caracteres debe ser positivo.")
    if argumentos.top_k < 1:
        parser.error("--top-k debe ser positivo.")

    modelo, tokenizador = crear_modelo(
        argumentos.corpus,
        argumentos.ventana,
        argumentos.dimension,
        argumentos.epocas_red,
        argumentos.epocas_atencion,
        argumentos.inferencia,
        argumentos.semilla,
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
