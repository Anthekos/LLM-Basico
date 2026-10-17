"""Entrenamiento XOR o modelo bayesiano basico de caracteres."""

import argparse
import sys

import numpy as np

from Red_neuronal import RedNeuronal
from embedding_basico import EmbeddingBasico
from atencion_causal import ModeloAtencionCausal
from tokenizador_basico import TokenizadorBasico

EPOCAS_POR_DEFECTO = 5_000
TEXTO_DEMO = (
    "hola mundo. la red aprende caracteres en español. "
    "una neurona aprende patrones y predice el siguiente carácter. "
) * 8


def crear_datos() -> tuple[np.ndarray, np.ndarray]:
    entradas = np.array(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    )
    objetivos = np.array([[0.0], [1.0], [1.0], [0.0]])
    return entradas, objetivos


def imprimir_resultados(
    red: RedNeuronal, entradas: np.ndarray, perdidas: list[float]
) -> None:
    predicciones = red.predecir(entradas)
    print(f"Perdida inicial: {perdidas[0]:.4f}")
    print(f"Perdida final:   {perdidas[-1]:.4f}")
    print("\nXOR (entrada -> probabilidad -> clase):")
    for entrada, probabilidad in zip(entradas, predicciones):
        clase = int(probabilidad[0] >= 0.5)
        print(f"{entrada.astype(int)} -> {probabilidad[0]:.3f} -> {clase}")


def ejecutar_entrenamiento(epocas: int, verbose: bool = False) -> None:
    entradas, objetivos = crear_datos()
    red = RedNeuronal([2, 8, 1])
    perdidas = red.entrenar(entradas, objetivos, epocas=epocas, verbose=verbose)
    imprimir_resultados(red, entradas, perdidas)


def ejecutar_modelo_caracteres(
    texto: str,
    inferencia: str,
    epocas_red: int,
    epocas_bayes: int,
    verbose: bool = False,
) -> None:
    tokenizador = TokenizadorBasico()
    modelo = EmbeddingBasico(tokenizador, dimension=12, desviacion_prior=1.0)
    print("Entrenando red neuronal de siguiente caracter para inicializar embeddings...")
    perdida_red = modelo.entrenar_inicializacion_red(
        texto, epocas=epocas_red, tasa_aprendizaje=0.03
    )
    print(f"Error de la red: {perdida_red[0]:.4f} -> {perdida_red[-1]:.4f}")

    if inferencia == "map":
        historial = modelo.inferencia_map(epocas=epocas_bayes, tasa_aprendizaje=0.03)
    else:
        historial = modelo.inferencia_variacional(
            epocas=epocas_bayes, tasa_aprendizaje=0.01, muestras=2
        )
    nombre = "MAP" if inferencia == "map" else "variacional (ELBO)"
    print(f"Inferencia {nombre}: objetivo inicial {historial[0]:.4f}, "
          f"final {historial[-1]:.4f}")

    contexto = texto[-1]
    probabilidades = modelo.probabilidades_siguiente(contexto, inferencia=inferencia)
    mejores = np.argsort(probabilidades)[-5:][::-1]
    print(f"\nSiguientes caracteres mas probables despues de {contexto!r}:")
    for indice in mejores:
        caracter = tokenizador.decodificar([int(indice)])
        print(f"  {caracter!r}: {probabilidades[indice]:.3f}")
    print(f"\nMuestra generada: {modelo.generar('la', 80, inferencia=inferencia, semilla=7)!r}")
    if verbose:
        print(f"\nVocabulario ({tokenizador.tamano_vocabulario} caracteres):")
        print(tokenizador.caracteres)
        print("\nEmbedding de contexto:")
        print(np.array2string(modelo.embedding_contexto, precision=3))
        print("\nEmbedding de siguiente caracter:")
        print(np.array2string(modelo.embedding_siguiente, precision=3))



def ejecutar_modelo_atencion(
    texto: str,
    epocas_red: int,
    epocas_atencion: int,
    ventana: int,
    inferencia: str = "map",
    verbose: bool = False,
) -> None:
    tokenizador = TokenizadorBasico()
    inicializador = EmbeddingBasico(tokenizador, dimension=12, desviacion_prior=1.0)
    print("Entrenando la red neuronal para inicializar los embeddings...")
    perdida_red = inicializador.entrenar_inicializacion_red(
        texto, epocas=epocas_red, tasa_aprendizaje=0.03
    )
    print(f"Error de inicializacion: {perdida_red[0]:.4f} -> {perdida_red[-1]:.4f}")

    modelo = ModeloAtencionCausal(inicializador, longitud_contexto=ventana)
    print(f"Entrenando auto-atencion causal (ventana={ventana}, inferencia={inferencia})...")
    if inferencia == "map":
        historial = modelo.entrenar_map(
            texto, epocas=epocas_atencion, tasa_aprendizaje=0.01
        )
    else:
        historial = modelo.inferencia_variacional(
            texto, epocas=epocas_atencion, tasa_aprendizaje=0.005,
            muestras=2,
        )
    nombre_inferencia = "MAP" if inferencia == "map" else "variacional (ELBO)"
    print(f"Objetivo {nombre_inferencia}: {historial[0]:.4f} -> {historial[-1]:.4f}")

    contexto = texto[-ventana:]
    probabilidades = modelo.probabilidades_siguiente(
        contexto, inferencia=inferencia, muestras=40
    )
    mejores = np.argsort(probabilidades)[-5:][::-1]
    print(f"\nSiguientes caracteres despues de {contexto!r}:")
    for indice in mejores:
        caracter = tokenizador.decodificar([int(indice)])
        print(f"  {caracter!r}: {probabilidades[indice]:.3f}")
    print(
        f"\nMuestra generada: "
        f"{modelo.generar('la', 100, semilla=7, inferencia=inferencia)!r}"
    )
    if verbose:
        print("\nEmbedding posicional:")
        print(np.array2string(modelo.embedding_posicion, precision=3))
        for nombre, matriz in (
            ("Wq", modelo.Wq), ("Wk", modelo.Wk), ("Wv", modelo.Wv),
            ("W1 (FFN)", modelo.W1), ("W2 (FFN)", modelo.W2),
            ("gamma LayerNorm 1", modelo.gamma1),
            ("gamma LayerNorm 2", modelo.gamma2),
        ):
            print(f"\n{nombre}:")
            print(np.array2string(matriz, precision=3))


def ejecutar_visualizacion(epocas: int, parser: argparse.ArgumentParser) -> None:
    try:
        from visualizacion import VisualizadorRed
    except ImportError as error:
        if error.name == "tkinter" or error.name == "_tkinter":
            parser.error(
                "Tkinter no esta disponible en este Python. Instala una version "
                "de Python con soporte Tk y vuelve a intentar."
            )
        raise

    entradas, objetivos = crear_datos()
    red = RedNeuronal([2, 8, 1])
    VisualizadorRed(red, entradas, objetivos, epocas).ejecutar()


def mostrar_menu(parser: argparse.ArgumentParser) -> None:
    print("=== Red neuronal didactica ===")
    print("1. Entrenamiento XOR")
    print("2. Entrenamiento XOR verbose")
    print("3. Visualizacion Tkinter")
    print("4. Modelo de caracteres bayesiano (bigrama)")
    print("5. Modelo de caracteres con atencion causal")
    try:
        opcion = input("Elige una opcion [1]: ").strip() or "1"
    except EOFError:
        print("\nNo hay entrada interactiva; se ejecuta XOR normal.")
        opcion = "1"
    if opcion not in {"1", "2", "3", "4", "5"}:
        parser.error("Opcion no valida. Elige 1, 2, 3, 4 o 5.")

    if opcion == "3":
        ejecutar_visualizacion(EPOCAS_POR_DEFECTO, parser)
    elif opcion == "5":
        metodo = input("Inferencia [1=MAP, 2=variacional] [1]: ").strip() or "1"
        if metodo not in {"1", "2"}:
            parser.error("Elige 1 para MAP o 2 para variacional.")
        ejecutar_modelo_atencion(
            TEXTO_DEMO, epocas_red=150, epocas_atencion=100, ventana=16,
            inferencia="map" if metodo == "1" else "variacional",
        )
    elif opcion == "4":
        metodo = input("Inferencia [1=MAP, 2=variacional] [1]: ").strip() or "1"
        if metodo not in {"1", "2"}:
            parser.error("Elige 1 para MAP o 2 para variacional.")
        ejecutar_modelo_caracteres(
            TEXTO_DEMO,
            "map" if metodo == "1" else "variacional",
            epocas_red=150,
            epocas_bayes=250,
        )
    else:
        epocas = EPOCAS_POR_DEFECTO
        if opcion == "2":
            texto_epocas = input(f"Cantidad de etapas [{EPOCAS_POR_DEFECTO}]: ").strip()
            if texto_epocas:
                try:
                    epocas = int(texto_epocas)
                except ValueError:
                    parser.error("La cantidad de etapas debe ser un numero entero.")
                if epocas < 1:
                    parser.error("La cantidad de etapas debe ser positiva.")
        ejecutar_entrenamiento(epocas, verbose=opcion == "2")


def main() -> None:
    parser = argparse.ArgumentParser(description="Red neuronal y modelo bayesiano de caracteres.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="muestra pesos XOR o embeddings del modelo de caracteres")
    parser.add_argument("--gui", action="store_true",
                        help="abre la visualizacion interactiva de XOR con Tkinter")
    parser.add_argument("--epocas", type=int, default=EPOCAS_POR_DEFECTO,
                        help="etapas para el entrenamiento XOR")
    grupo_modelo = parser.add_mutually_exclusive_group()
    grupo_modelo.add_argument("--bayesiano", action="store_true",
                              help="entrena el modelo bayesiano de bigramas")
    grupo_modelo.add_argument("--atencion", action="store_true",
                              help="entrena el modelo de caracteres con atencion causal")
    parser.add_argument("--inferencia", choices=("map", "variacional"), default="map",
                        help="metodo posterior: MAP o aproximacion variacional")
    parser.add_argument("--texto", default=TEXTO_DEMO,
                        help="corpus de caracteres en espanol para el modelo bayesiano")
    parser.add_argument("--epocas-red", type=int, default=150,
                        help="etapas de la red que inicializa los embeddings")
    parser.add_argument("--epocas-bayes", type=int, default=250,
                        help="etapas de inferencia MAP o variacional del bigrama")
    parser.add_argument("--epocas-atencion", type=int, default=100,
                        help="etapas de entrenamiento MAP de la atencion causal")
    parser.add_argument("--ventana", type=int, default=16,
                        help="longitud maxima del contexto para la atencion causal")
    argumentos = parser.parse_args()

    if argumentos.atencion:
        ejecutar_modelo_atencion(
            argumentos.texto, argumentos.epocas_red, argumentos.epocas_atencion,
            argumentos.ventana, argumentos.inferencia, argumentos.verbose,
        )
        return
    if argumentos.bayesiano:
        ejecutar_modelo_caracteres(
            argumentos.texto,
            argumentos.inferencia,
            argumentos.epocas_red,
            argumentos.epocas_bayes,
            argumentos.verbose,
        )
        return
    if argumentos.gui:
        ejecutar_visualizacion(argumentos.epocas, parser)
        return
    if len(sys.argv) == 1:
        mostrar_menu(parser)
        return
    ejecutar_entrenamiento(argumentos.epocas, verbose=argumentos.verbose)


if __name__ == "__main__":
    main()
