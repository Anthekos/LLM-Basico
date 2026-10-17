"""Panel Tkinter para observar matrices de activacion y error durante el entrenamiento."""

from __future__ import annotations

import time
import tkinter as tk
from tkinter import ttk

import numpy as np

from Red_neuronal import RedNeuronal


class VisualizadorRed:
    FONDO = "#1e1e1e"
    PANEL = "#252526"
    BORDE = "#3c3c3c"
    TEXTO = "#ffffff"
    SECUNDARIO = "#cccccc"
    ACENTO = "#007acc"
    VERDE = "#4ec9b0"

    def __init__(
        self,
        red: RedNeuronal,
        entradas: np.ndarray,
        objetivos: np.ndarray,
        epocas: int,
        tasa_aprendizaje: float = 0.1,
    ) -> None:
        if epocas < 1:
            raise ValueError("epocas debe ser positivo.")

        self.red = red
        self.entradas = entradas
        self.objetivos = objetivos
        self.epocas = epocas
        self.tasa_aprendizaje = tasa_aprendizaje
        self.etapa = 0
        self.historial: list[float] = []

        self.ventana = tk.Tk()
        self.velocidad_entrenamiento = tk.DoubleVar(value=0.05)
        self.entrenamiento_activo = False
        self._timer_id = None
        self.ventana.title("Red neuronal | visualizador")
        self.ventana.configure(bg=self.FONDO)
        self.ventana.geometry("1000x720")
        self.ventana.minsize(760, 560)

        estilo = ttk.Style(self.ventana)
        estilo.theme_use("clam")
        estilo.configure(
            "TCombobox",
            fieldbackground=self.PANEL,
            background=self.PANEL,
            foreground=self.TEXTO,
            arrowcolor=self.TEXTO,
        )
        estilo.map(
            "TCombobox",
            fieldbackground=[("readonly", self.PANEL)],
            foreground=[("readonly", self.TEXTO)],
        )

        self._construir_interfaz()

    def _construir_interfaz(self) -> None:
        encabezado = tk.Frame(self.ventana, bg=self.FONDO, padx=20, pady=16)
        encabezado.pack(fill="x")

        tk.Label(
            encabezado,
            text="RED NEURONAL",
            bg=self.FONDO,
            fg=self.TEXTO,
            font=("Helvetica", 16, "bold"),
        ).pack(side="left")
        self.etiqueta_etapa = tk.Label(
            encabezado,
            text=f"Etapa 0 / {self.epocas}",
            bg=self.FONDO,
            fg=self.ACENTO,
            font=("Helvetica", 12, "bold"),
        )
        self.etiqueta_etapa.pack(side="right")

        controles = tk.Frame(self.ventana, bg=self.PANEL, padx=16, pady=12)
        controles.pack(fill="x", padx=20, pady=(0, 14))
        tk.Label(
            controles,
            text="Matriz a visualizar:",
            bg=self.PANEL,
            fg=self.TEXTO,
            font=("Helvetica", 10),
        ).pack(side="left")

        self.selector_capa = ttk.Combobox(
            controles,
            state="readonly",
            values=self._nombres_matrices(),
            width=24,
        )
        self.selector_capa.current(0)
        self.selector_capa.pack(side="left", padx=10)
        self.selector_capa.bind("<<ComboboxSelected>>", self._redibujar_matriz)

        texto_velocidad = tk.Label(
            controles,
            text="Velocidad:",
            bg=self.PANEL,
            fg=self.TEXTO,
            font=("Helvetica", 10),
        )
        texto_velocidad.pack(side="left", padx=(18, 6))
        self.slider_velocidad = ttk.Scale(
            controles,
            from_=0.0,
            to=0.5,
            value=self.velocidad_entrenamiento.get(),
            orient="horizontal",
            length=180,
            command=self._actualizar_velocidad,
        )
        self.slider_velocidad.pack(side="left")
        self.etiqueta_velocidad = tk.Label(
            controles,
            text=f"{self.velocidad_entrenamiento.get():.2f}s",
            bg=self.PANEL,
            fg=self.SECUNDARIO,
            font=("Helvetica", 9),
        )
        self.etiqueta_velocidad.pack(side="left", padx=(8, 12))

        self.boton_info = tk.Button(
            controles,
            text="Info de capas",
            bg=self.ACENTO,
            fg=self.TEXTO,
            relief="flat",
            command=self._mostrar_explicacion_capas,
        )
        self.boton_info.pack(side="right", padx=(10, 0))

        tk.Label(
            controles,
            text="Color mas intenso = activacion mas cercana a 1",
            bg=self.PANEL,
            fg=self.SECUNDARIO,
            font=("Helvetica", 9),
        ).pack(side="right")

        capa_frame = tk.Frame(self.ventana, bg=self.FONDO)
        capa_frame.pack(fill="x", padx=20, pady=(0, 8))
        for indice in range(1, len(self.red.capas) + 1):
            boton = tk.Button(
                capa_frame,
                text=f"Capa {indice}",
                bg=self.PANEL,
                fg=self.TEXTO,
                relief="flat",
                command=lambda i=indice: self._mostrar_explicacion_capa(i),
            )
            boton.pack(side="left", padx=(0, 8))

        contenido = tk.Frame(self.ventana, bg=self.FONDO, padx=20)
        contenido.pack(fill="both", expand=True)
        panel_matriz = tk.Frame(contenido, bg=self.PANEL, padx=12, pady=12)
        panel_matriz.pack(side="left", fill="both", expand=True, padx=(0, 8))
        panel_error = tk.Frame(contenido, bg=self.PANEL, padx=12, pady=12)
        panel_error.pack(side="left", fill="both", expand=True, padx=(8, 0))

        self.etiqueta_matriz = tk.Label(
            panel_matriz,
            text="Matriz de activaciones",
            bg=self.PANEL,
            fg=self.TEXTO,
            font=("Helvetica", 11, "bold"),
        )
        self.etiqueta_matriz.pack(anchor="w", pady=(0, 8))
        self.canvas_matriz = tk.Canvas(
            panel_matriz, bg=self.PANEL, highlightthickness=0
        )
        self.canvas_matriz.pack(fill="both", expand=True)

        tk.Label(
            panel_error,
            text="Error por etapa",
            bg=self.PANEL,
            fg=self.TEXTO,
            font=("Helvetica", 11, "bold"),
        ).pack(anchor="w", pady=(0, 8))
        self.canvas_error = tk.Canvas(
            panel_error, bg=self.PANEL, highlightthickness=0
        )
        self.canvas_error.pack(fill="both", expand=True)

        controles_entrenamiento = tk.Frame(self.ventana, bg=self.FONDO)
        controles_entrenamiento.pack(fill="x", padx=20, pady=(0, 8))

        self.boton_iniciar = tk.Button(
            controles_entrenamiento,
            text="Iniciar entrenamiento",
            bg=self.ACENTO,
            fg=self.TEXTO,
            relief="flat",
            command=self.iniciar_entrenamiento,
        )
        self.boton_iniciar.pack(side="left", padx=(0, 8))

        self.boton_pausar = tk.Button(
            controles_entrenamiento,
            text="Pausar",
            bg=self.PANEL,
            fg=self.TEXTO,
            relief="flat",
            command=self.pausar_entrenamiento,
        )
        self.boton_pausar.pack(side="left")

        self.etiqueta_estado = tk.Label(
            self.ventana,
            text="Listo para iniciar",
            bg=self.FONDO,
            fg=self.SECUNDARIO,
            font=("Helvetica", 10),
            anchor="w",
            padx=20,
            pady=12,
        )
        self.etiqueta_estado.pack(fill="x")

    def _nombres_matrices(self) -> list[str]:
        nombres = ["Entrada"]
        nombres.extend(f"Capa {indice}" for indice in range(1, len(self.red.capas) + 1))
        return nombres

    def _actualizar_velocidad(self, valor: str) -> None:
        self.velocidad_entrenamiento.set(float(valor))
        self.etiqueta_velocidad.configure(text=f"{float(valor):.2f}s")

    def _mostrar_explicacion_capa(self, indice: int) -> None:
        descripcion = self._descripcion_capa(indice)
        self._mostrar_dialogo("Capa oculta", descripcion)

    def _mostrar_explicacion_capas(self) -> None:
        texto = "\n\n".join(
            [
                "Las capas de una red neuronal son bloques que transforman la informacion paso a paso.",
                *[
                    f"Capa {indice}: {self._descripcion_capa(indice)}"
                    for indice in range(1, len(self.red.capas) + 1)
                ],
                "La capa de entrada recibe los datos crudos; las capas ocultas extraen patrones; la capa final entrega la salida esperada.",
            ]
        )
        self._mostrar_dialogo("Explicacion de la red", texto)

    def _descripcion_capa(self, indice: int) -> str:
        total_capas = len(self.red.capas)
        if indice == 1 and total_capas > 1:
            return (
                "Es la primera capa oculta. Aprende combinaciones simples de las entradas. "
                "Cada neurona recibe varias entradas, multiplica por pesos y suma sesgos para detectar patrones."
            )
        if indice == total_capas:
            return (
                "Es la capa de salida. Convierte la representacion interna en la respuesta final. "
                "En este ejemplo produce la probabilidad de que la salida sea 1 para XOR."
            )
        return (
            f"Es una capa intermedia. Refina la representacion aprendiendo caracteristicas mas abstractas. "
            "Cuantas mas capas ocultas haya, mas capacidad tiene la red para separar patrones complejos."
        )

    def _mostrar_dialogo(self, titulo: str, texto: str) -> None:
        ventana = tk.Toplevel(self.ventana)
        ventana.title(titulo)
        ventana.configure(bg=self.FONDO)
        ventana.geometry("500x260")
        ventana.transient(self.ventana)
        ventana.grab_set()

        marco = tk.Frame(ventana, bg=self.FONDO, padx=16, pady=16)
        marco.pack(fill="both", expand=True)

        tk.Label(
            marco,
            text=titulo,
            bg=self.FONDO,
            fg=self.TEXTO,
            font=("Helvetica", 12, "bold"),
            anchor="w",
            justify="left",
        ).pack(fill="x", pady=(0, 10))

        texto_widget = tk.Text(
            marco,
            bg=self.PANEL,
            fg=self.TEXTO,
            wrap="word",
            height=10,
            padx=12,
            pady=10,
            insertbackground=self.TEXTO,
        )
        texto_widget.insert("1.0", texto)
        texto_widget.configure(state="disabled")
        texto_widget.pack(fill="both", expand=True)

        tk.Button(
            marco,
            text="Cerrar",
            bg=self.ACENTO,
            fg=self.TEXTO,
            relief="flat",
            command=ventana.destroy,
        ).pack(anchor="e", pady=(10, 0))

    def iniciar_entrenamiento(self) -> None:
        if self.entrenamiento_activo:
            return
        self.entrenamiento_activo = True
        self.etiqueta_estado.configure(text="Entrenando...")
        self._timer_id = self.ventana.after(50, self._entrenar_lote)

    def pausar_entrenamiento(self) -> None:
        self.entrenamiento_activo = False
        if self._timer_id is not None:
            self.ventana.after_cancel(self._timer_id)
            self._timer_id = None
        self.etiqueta_estado.configure(text="Entrenamiento pausado")

    def _entrenar_lote(self) -> None:
        if not self.entrenamiento_activo:
            return

        fin_lote = min(self.etapa + 10, self.epocas)
        activaciones: list[np.ndarray] | None = None
        while self.etapa < fin_lote:
            perdida, activaciones = self.red.entrenar_epoca(
                self.entradas, self.objetivos, self.tasa_aprendizaje
            )
            if self.velocidad_entrenamiento.get() > 0:
                time.sleep(self.velocidad_entrenamiento.get())
            self.historial.append(perdida)
            self.etapa += 1

        self.etiqueta_etapa.configure(text=f"Etapa {self.etapa} / {self.epocas}")
        self.etiqueta_estado.configure(
            text=f"Error: {self.historial[-1]:.6f}    |    "
            f"Tasa de aprendizaje: {self.tasa_aprendizaje:g}"
        )
        if activaciones is not None:
            self._dibujar_matriz(activaciones)
        self._dibujar_error()

        if self.etapa < self.epocas and self.entrenamiento_activo:
            self._timer_id = self.ventana.after(1, self._entrenar_lote)
        else:
            self.entrenamiento_activo = False
            self._timer_id = None
            self.etiqueta_estado.configure(
                text=f"Entrenamiento completado    |    Error final: {self.historial[-1]:.6f}"
            )

    def _redibujar_matriz(self, _evento: tk.Event[tk.Misc]) -> None:
        matrices = self.red.activaciones_por_capa(self.entradas)
        self._dibujar_matriz(matrices)

    def _dibujar_matriz(self, matrices: list[np.ndarray]) -> None:
        indice = self.selector_capa.current()
        matriz = matrices[indice]
        es_entrada = indice == 0
        if not es_entrada and self.red.capas[indice - 1].activacion.nombre == "tanh":
            intensidades = (np.clip(matriz, -1.0, 1.0) + 1.0) / 2.0
        else:
            intensidades = np.clip(matriz, 0.0, 1.0)

        self.etiqueta_matriz.configure(
            text=f"{self._nombres_matrices()[indice]}  |  {matriz.shape[0]} x {matriz.shape[1]}"
        )
        self.canvas_matriz.delete("all")
        ancho = max(self.canvas_matriz.winfo_width(), 280)
        alto = max(self.canvas_matriz.winfo_height(), 200)
        margen_x, margen_y = 42, 34
        ancho_celda = min(84, (ancho - margen_x - 12) / matriz.shape[1])
        alto_celda = min(58, (alto - margen_y - 12) / matriz.shape[0])

        for fila in range(matriz.shape[0]):
            centro_y = margen_y + fila * alto_celda + alto_celda / 2
            self.canvas_matriz.create_text(
                margen_x - 12,
                centro_y,
                text=f"x{fila + 1}",
                fill=self.SECUNDARIO,
                anchor="e",
                font=("Helvetica", 9),
            )
            for columna in range(matriz.shape[1]):
                x0 = margen_x + columna * ancho_celda
                y0 = margen_y + fila * alto_celda
                intensidad = float(intensidades[fila, columna])
                color = self._color_activacion(intensidad)
                self.canvas_matriz.create_rectangle(
                    x0 + 2,
                    y0 + 2,
                    x0 + ancho_celda - 3,
                    y0 + alto_celda - 3,
                    fill=color,
                    outline=self.BORDE,
                )
                self.canvas_matriz.create_text(
                    x0 + ancho_celda / 2,
                    y0 + alto_celda / 2,
                    text=f"{matriz[fila, columna]:.2f}",
                    fill=self.TEXTO,
                    font=("Helvetica", 9, "bold"),
                )
        for columna in range(matriz.shape[1]):
            centro_x = margen_x + columna * ancho_celda + ancho_celda / 2
            self.canvas_matriz.create_text(
                centro_x,
                margen_y - 14,
                text=f"n{columna + 1}",
                fill=self.SECUNDARIO,
                font=("Helvetica", 9),
            )

    def _color_activacion(self, intensidad: float) -> str:
        oscuro = (37, 37, 38)
        brillante = (0, 122, 204)
        mezcla = tuple(
            round(base + intensidad * (luz - base))
            for base, luz in zip(oscuro, brillante)
        )
        return f"#{mezcla[0]:02x}{mezcla[1]:02x}{mezcla[2]:02x}"

    def _dibujar_error(self) -> None:
        canvas = self.canvas_error
        canvas.delete("all")
        ancho = max(canvas.winfo_width(), 280)
        alto = max(canvas.winfo_height(), 200)
        margen = 38
        x_inicio, x_fin = margen, ancho - 12
        y_inicio, y_fin = 12, alto - margen

        canvas.create_line(x_inicio, y_inicio, x_inicio, y_fin, fill=self.BORDE)
        canvas.create_line(x_inicio, y_fin, x_fin, y_fin, fill=self.BORDE)
        if not self.historial:
            return

        maximo = max(self.historial)
        minimo = min(self.historial)
        rango = maximo - minimo or 1.0
        puntos: list[float] = []
        cantidad = len(self.historial)
        for indice, perdida in enumerate(self.historial):
            x = x_inicio + (x_fin - x_inicio) * indice / max(cantidad - 1, 1)
            y = y_inicio + (maximo - perdida) / rango * (y_fin - y_inicio)
            puntos.extend((x, y))
        if len(puntos) >= 4:
            canvas.create_line(*puntos, fill=self.VERDE, width=2, smooth=True)
        else:
            canvas.create_oval(
                puntos[0] - 2,
                puntos[1] - 2,
                puntos[0] + 2,
                puntos[1] + 2,
                fill=self.VERDE,
                outline=self.VERDE,
            )
        canvas.create_text(
            5,
            y_inicio,
            text=f"{maximo:.3f}",
            fill=self.SECUNDARIO,
            anchor="nw",
            font=("Helvetica", 8),
        )
        canvas.create_text(
            5,
            y_fin,
            text=f"{minimo:.3f}",
            fill=self.SECUNDARIO,
            anchor="sw",
            font=("Helvetica", 8),
        )

    def ejecutar(self) -> None:
        self.ventana.mainloop()
