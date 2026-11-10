"""Extrae y sanitiza texto PDF para usarlo con el tokenizador espanol."""

import argparse
import re
import unicodedata
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from tokenizador_basico import TokenizadorBasico
from tokenizador_palabras import TokenizadorPalabras


def sanitizar_texto(
    texto: str,
    tokenizador: TokenizadorBasico | TokenizadorPalabras | None = None,
) -> str:
    """Normaliza Unicode y elimina caracteres que el vocabulario no admite."""
    tokenizador = tokenizador or TokenizadorBasico()
    texto = unicodedata.normalize("NFC", texto).replace("\r\n", "\n").replace("\r", "\n")
    sanitizar = getattr(tokenizador, "sanitizar", None)
    if callable(sanitizar):
        return sanitizar(texto)
    lineas: list[str] = []
    for linea in texto.split("\n"):
        caracteres = "".join(
            caracter if caracter in tokenizador.caracter_a_id else " "
            for caracter in linea
        )
        lineas.append(re.sub(r"[ \t\f\v]+", " ", caracteres).strip())
    resultado = "\n".join(lineas)
    resultado = re.sub(r"\n{3,}", "\n\n", resultado).strip()
    if not resultado:
        raise ValueError("El texto no contiene caracteres admitidos por el tokenizador.")
    return resultado


def convertir_pdf_a_corpus(
    ruta_pdf: str | Path,
    ruta_salida: str | Path = "corpus.txt",
    tokenizador: TokenizadorBasico | TokenizadorPalabras | None = None,
) -> Path:
    """Extrae texto seleccionable de un PDF y escribe un corpus UTF-8 compatible."""
    ruta_pdf = Path(ruta_pdf)
    ruta_salida = Path(ruta_salida)
    try:
        lector = PdfReader(str(ruta_pdf), strict=True)
    except (OSError, PdfReadError) as error:
        raise ValueError(f"No se pudo leer el PDF {ruta_pdf}: {error}") from error

    paginas = [pagina.extract_text() or "" for pagina in lector.pages]
    texto_extraido = "\n\n".join(paginas)
    if not texto_extraido.strip():
        raise ValueError(
            "El PDF no contiene texto seleccionable. Si es un documento escaneado, "
            "se necesita OCR antes de convertirlo."
        )
    corpus = sanitizar_texto(texto_extraido, tokenizador)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    ruta_salida.write_text(corpus + "\n", encoding="utf-8")
    return ruta_salida


def main() -> None:
    parser = argparse.ArgumentParser(description="Convierte un PDF a corpus.txt en espanol.")
    parser.add_argument("pdf", type=Path, help="PDF con texto seleccionable")
    parser.add_argument("--salida", type=Path, default=Path("corpus.txt"))
    argumentos = parser.parse_args()
    salida = convertir_pdf_a_corpus(argumentos.pdf, argumentos.salida)
    print(f"Corpus sanitizado escrito en: {salida}")


if __name__ == "__main__":
    main()
