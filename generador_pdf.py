"""Genera PDF de memoria locales, con salida LaTeX o PDF seleccionable."""

from datetime import date
import logging
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import textwrap

from pypdf import PdfWriter
from pypdf.generic import (
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
)

from pdf_a_corpus import sanitizar_texto
from tokenizador_basico import TokenizadorBasico


DIRECTORIO_PDFS = Path(__file__).parent / "pdfs_corpus"


def _escapar_latex(texto: str) -> str:
    reemplazos = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(reemplazos.get(caracter, caracter) for caracter in texto)


def _plantilla_latex(titulo: str, contenido: str) -> str:
    parrafos = "\n\n".join(
        _escapar_latex(parrafo.strip())
        for parrafo in contenido.split("\n")
        if parrafo.strip()
    )
    return rf"""\documentclass[11pt]{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage[spanish]{{babel}}
\usepackage[margin=2.5cm]{{geometry}}
\usepackage{{parskip}}
\begin{{document}}
\begin{{center}}
{{\Large\bfseries {_escapar_latex(titulo)}}}\\[0.5em]
Memoria documental local\\
Fecha: {date.today().isoformat()}
\end{{center}}
\hrule
\vspace{{1em}}
\textbf{{Conceptos y diálogo}}

{parrafos}
\end{{document}}
"""


def _escapar_literal_pdf(texto: str) -> bytes:
    contenido = texto.encode("cp1252", errors="replace")
    return contenido.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)")


def _crear_pdf_seleccionable(titulo: str, contenido: str, destino: Path) -> None:
    texto = sanitizar_texto(
        f"{titulo}\nMemoria documental local\nFecha: {date.today().isoformat()}\n"
        f"Conceptos y diálogo\n{contenido}",
        TokenizadorBasico(),
    )
    lineas: list[str] = []
    for parrafo in texto.splitlines():
        if parrafo:
            lineas.extend(textwrap.wrap(parrafo, width=88, break_long_words=True))
        else:
            lineas.append("")
    writer = PdfWriter()
    fuente = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
        NameObject("/Encoding"): NameObject("/WinAnsiEncoding"),
    })
    lineas_por_pagina = 44
    for inicio in range(0, max(1, len(lineas)), lineas_por_pagina):
        pagina = writer.add_blank_page(width=612, height=792)
        pagina[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): fuente})
        })
        comandos = ["BT", "/F1 10 Tf", "72 740 Td", "14 TL"]
        for linea in lineas[inicio:inicio + lineas_por_pagina]:
            comandos.append(b"(" + _escapar_literal_pdf(linea) + b") Tj")
            comandos.append("T*")
        comandos.append("ET")
        contenido_pdf = DecodedStreamObject()
        contenido_pdf.set_data(
            b"\n".join(
                linea if isinstance(linea, bytes) else linea.encode("ascii")
                for linea in comandos
            )
        )
        pagina[NameObject("/Contents")] = writer._add_object(contenido_pdf)
    with destino.open("wb") as archivo:
        writer.write(archivo)


def crear_pdf_memoria(
    titulo: str,
    contenido_conceptos: str,
    nombre_archivo: str,
    directorio: str | Path = DIRECTORIO_PDFS,
) -> Path:
    """Crea un PDF de memoria y devuelve su ruta dentro del directorio indicado."""
    if not isinstance(titulo, str) or not titulo.strip():
        raise ValueError("El titulo de memoria no puede estar vacio.")
    if not isinstance(contenido_conceptos, str) or not contenido_conceptos.strip():
        raise ValueError("El contenido de memoria no puede estar vacio.")
    tokenizador = TokenizadorBasico()
    titulo_limpio = sanitizar_texto(titulo.strip(), tokenizador)
    contenido_limpio = sanitizar_texto(contenido_conceptos.strip(), tokenizador)
    if len(contenido_limpio) > 10_000:
        raise ValueError("El contenido de memoria supera 10000 caracteres.")

    nombre = Path(nombre_archivo).name
    nombre = re.sub(r"[^A-Za-z0-9._ -]", "_", nombre).strip(" .")
    if not nombre:
        nombre = "memoria"
    if not nombre.lower().endswith(".pdf"):
        nombre += ".pdf"
    destino = Path(directorio).resolve() / nombre
    destino.parent.mkdir(parents=True, exist_ok=True)
    if destino.exists():
        raise FileExistsError(f"Ya existe el archivo de memoria {nombre}.")

    pdflatex = shutil.which("pdflatex")
    if pdflatex is not None:
        try:
            with tempfile.TemporaryDirectory(prefix="memoria-pdf-") as temporal:
                directorio_temporal = Path(temporal)
                archivo_tex = directorio_temporal / "memoria.tex"
                archivo_tex.write_text(
                    _plantilla_latex(titulo_limpio, contenido_limpio),
                    encoding="utf-8",
                )
                subprocess.run(
                    [
                        pdflatex,
                        "-interaction=nonstopmode",
                        "-halt-on-error",
                        "-output-directory",
                        str(directorio_temporal),
                        str(archivo_tex),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                pdf_compilado = directorio_temporal / "memoria.pdf"
                if not pdf_compilado.is_file():
                    raise RuntimeError("pdflatex terminó sin crear el PDF.")
                destino.write_bytes(pdf_compilado.read_bytes())
                return destino
        except (OSError, subprocess.SubprocessError, RuntimeError) as error:
            logging.warning(
                "No se pudo compilar la memoria con pdflatex; se usará PDF local: %s",
                error,
            )

    _crear_pdf_seleccionable(titulo_limpio, contenido_limpio, destino)
    return destino
