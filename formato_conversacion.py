"""Delimitadores y parser del formato de turnos causales."""

import re

USUARIO = "<|usuario|>"
PENSAMIENTO = "<|pensamiento|>"
ASISTENTE = "<|asistente|>"
FIN = "<|fin|>"
DELIMITADORES = (USUARIO, PENSAMIENTO, ASISTENTE, FIN)


def validar_corpus_estructurado(texto: str) -> None:
    """Exige que cada bloque de entrenamiento tenga el formato causal completo."""
    bloques = [bloque.strip() for bloque in texto.strip().split("\n\n") if bloque.strip()]
    if not bloques:
        raise ValueError("El corpus debe contener al menos un turno estructurado.")
    for numero, bloque in enumerate(bloques, start=1):
        posiciones = [bloque.find(delimitador) for delimitador in DELIMITADORES]
        if (
            any(posicion < 0 for posicion in posiciones)
            or posiciones != sorted(posiciones)
            or any(bloque.count(delimitador) != 1 for delimitador in DELIMITADORES)
            or not bloque.startswith(USUARIO)
            or not bloque.endswith(FIN)
        ):
            raise ValueError(
                f"El turno {numero} debe seguir la secuencia "
                f"{USUARIO} ... {PENSAMIENTO} ... {ASISTENTE} ... {FIN}."
            )


def migrar_dialogos_legacy(texto: str) -> str:
    """Convierte pares legados con etiquetas en turnos con delimitadores reservados."""
    turnos: list[str] = []
    bloques = re.split(r"(?m)(?=^Usuario:\s*)", texto)
    for bloque in bloques:
        if not bloque.startswith("Usuario:"):
            continue
        contenido = bloque[len("Usuario:"):].strip()
        coincidencia_asistente = re.search(r"(?m)^Asistente:\s*", contenido)
        if coincidencia_asistente is None:
            continue
        antes_asistente = contenido[:coincidencia_asistente.start()].strip()
        respuesta = contenido[coincidencia_asistente.end():].strip()
        coincidencia_pensamiento = re.search(r"(?m)^Pensamiento:\s*", antes_asistente)
        if coincidencia_pensamiento is None:
            mensaje = antes_asistente
            pensamiento = "Respondo con la información disponible."
        else:
            mensaje = antes_asistente[:coincidencia_pensamiento.start()].strip()
            pensamiento = antes_asistente[coincidencia_pensamiento.end():].strip()
        campos = [
            re.sub(r"\s+", " ", campo).strip()
            for campo in (mensaje, pensamiento, respuesta)
        ]
        if all(campos):
            turnos.append(
                f"{USUARIO} {campos[0]} {PENSAMIENTO} {campos[1]} "
                f"{ASISTENTE} {campos[2]} {FIN}"
            )
    if not turnos:
        raise ValueError("El corpus legado no contiene pares Usuario/Asistente completos.")
    return "\n\n".join(turnos)


def separar_deliberacion(texto: str) -> tuple[str, str]:
    """Extrae pensamiento y respuesta de un turno generado o historico."""
    inicio_pensamiento = texto.rfind(PENSAMIENTO)
    inicio_contenido = (
        inicio_pensamiento + len(PENSAMIENTO)
        if inicio_pensamiento >= 0
        else 0
    )
    inicio_asistente = texto.find(ASISTENTE, inicio_contenido)

    if inicio_asistente >= 0:
        pensamiento = texto[inicio_contenido:inicio_asistente].strip()
        respuesta = texto[inicio_asistente + len(ASISTENTE):]
        return pensamiento, _limpiar_respuesta(respuesta)

    if inicio_pensamiento >= 0:
        return "", _limpiar_respuesta(texto[inicio_contenido:])

    if any(marcador in texto for marcador in (FIN, USUARIO)):
        return "", _limpiar_respuesta(texto)
    return _separar_formato_legacy(texto)


def _limpiar_respuesta(texto: str) -> str:
    posiciones = [
        posicion
        for marcador in (FIN, USUARIO)
        if (posicion := texto.find(marcador)) >= 0
    ]
    if posiciones:
        texto = texto[:min(posiciones)]
    return texto.strip()


def _separar_formato_legacy(texto: str) -> tuple[str, str]:
    marcador_pensamiento = "Pensamiento:"
    marcador_asistente = "Asistente:"
    inicio_pensamiento = texto.rfind(marcador_pensamiento)
    inicio_contenido = (
        inicio_pensamiento + len(marcador_pensamiento)
        if inicio_pensamiento >= 0
        else 0
    )
    inicio_asistente = texto.find(marcador_asistente, inicio_contenido)
    if inicio_asistente >= 0:
        pensamiento = texto[inicio_contenido:inicio_asistente].strip()
        respuesta = texto[inicio_asistente + len(marcador_asistente):]
        return pensamiento, _limpiar_respuesta_legacy(respuesta)
    if inicio_pensamiento >= 0:
        return "", _limpiar_respuesta_legacy(texto[inicio_contenido:])
    return "", _limpiar_respuesta_legacy(texto)


def _limpiar_respuesta_legacy(texto: str) -> str:
    posiciones = [
        posicion
        for marcador in ("\nUsuario:", "\nHumano:")
        if (posicion := texto.find(marcador)) >= 0
    ]
    if posiciones:
        texto = texto[:min(posiciones)]
    return texto.split("\n", 1)[0].strip()
