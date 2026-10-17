# Red neuronal desde cero con NumPy

Proyecto didactico para estudiar redes neuronales y avanzar hacia un modelo de
lenguaje de caracteres. Incluye XOR, un modelo bayesiano de bigramas y un modelo
pequeno con auto-atencion causal. No es un LLM de produccion ni un Transformer
completo.

## Requisitos y entorno

- Python 3.10 o superior
- NumPy y pypdf (incluidos en `requirements.txt`)
- Tkinter solo para la visualizacion de XOR

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Ejecutar

Desde VS Code, ejecuta `main.py` con **Run Python File**. Sin argumentos se
abre un menu para XOR, visualizacion Tkinter, el bigrama bayesiano o el modelo
con atencion causal.

```bash
python main.py
python main.py --bayesiano --inferencia map
python main.py --bayesiano --inferencia variacional
python main.py --atencion --ventana 16 --epocas-atencion 100 --inferencia map
python main.py --atencion --ventana 16 --epocas-atencion 100 --inferencia variacional
python main.py --atencion --texto "hola niña. ¿qué tal?" --ventana 8
python main.py --gui
python pdf_a_corpus.py documento.pdf --salida corpus.txt
python chat.py --corpus corpus.txt --ventana 64 --temperatura 0.8 --top-k 8
python servidor_web.py
```

`--bayesiano` y `--atencion` ejecutan variantes separadas. Si no se especifica
`--texto`, se usa un corpus didactico en espanol incluido en `main.py`.

## Interfaz web local

Activa el entorno virtual e inicia el servidor local:

```bash
python servidor_web.py
```

Abre `http://127.0.0.1:8765` en el navegador. El servidor escucha unicamente
en el equipo local y utiliza el modelo de caracteres con atencion causal del
proyecto. La primera carga entrena el modelo y puede tardar unos segundos; cada
respuesta se genera caracter por caracter y puede ser incoherente, porque el
modelo no esta ajustado para seguir instrucciones.

La barra lateral permite abrir varias conversaciones, guardadas en el
almacenamiento local del navegador, y elegir el nombre de usuario. Desde ahi
tambien se pueden incorporar dos PDF de ejemplo o subir un PDF propio. Solo se
admiten PDF con texto seleccionable: no se ejecuta OCR. Se extraen y sanitizan
hasta 10000 caracteres por archivo, se agregan a `corpus.txt` y se vuelve a
entrenar el modelo. El archivo de corpus se mantiene en este equipo y esta
excluido de Git.

Los controles de temperatura y top-k ajustan el muestreo de la respuesta.
El limite de carga es 12 MB y 30 paginas; no se aceptan mas de 10000 caracteres
extraidos por PDF.

La interfaz web usa solamente la biblioteca estandar de Python, NumPy y
`pypdf`, ya incluidos en `requirements.txt`; no necesita instalar un framework
web ni cargar recursos externos.

## Tokenizador de caracteres

`tokenizador_basico.py` asigna un ID a cada letra espanola (incluidas `ñ`,
vocales acentuadas y `ü`), mayusculas, espacios, puntuacion y saltos de linea.
Normaliza Unicode a NFC y rechaza los caracteres que no pertenezcan a su
vocabulario.

## Modelo bayesiano de bigramas

`embedding_basico.py` representa cada caracter con un vector de contexto `u_i`
y uno de candidato `v_j`; la transicion es:

`P(j | i, U, V) = softmax_j(u_i · v_j)`

Los dos embeddings usan prior normal isotropico `N(0, sigma_prior^2 I)`.
La red densa de `Red_neuronal.py` aprende primero la tarea de siguiente
caracter. Su matriz de pesos de la capa oculta se copia directamente como tabla
de embeddings de entrada; la matriz de pesos de salida se copia como LM head.
Despues se puede aplicar:

- **MAP**: entropia cruzada mas penalizacion del prior gaussiano.
- **Variacional**: posterior gaussiano diagonal, optimizado con ELBO y
  reparametrizacion. Es una aproximacion de campo medio, no inferencia exacta.

Esta variante usa solo el caracter previo, por lo que no representa contexto
largo.

## Modelo con auto-atencion causal

`atencion_causal.py` reutiliza directamente las tablas de entrada y de salida
inicializadas por la misma red neuronal, y agrega:

1. Una tabla aprendible de posiciones `W_p` de forma `(T, d)`.
2. Proyecciones `W_q`, `W_k`, `W_v` de forma `(d, d)`.
3. Un bloque de atencion causal escalada:

   `Q = H W_q`, `K = H W_k`, `V = H W_v`

   `A = softmax(Q K^T / sqrt(d) + M)`

   `O = A V`, `R_1 = H + O`

   `H_1 = LayerNorm(R_1)`

   `F = ReLU(H_1 W_1 + b_1) W_2 + b_2`

   `R_2 = H_1 + F`, `Z = LayerNorm(R_2)`

   `logits = Z W_u^T`

   `M` es una mascara triangular: una posicion solo atiende a si misma y a
   posiciones anteriores. Otra mascara ignora el padding izquierdo usado en
   los contextos cortos. La FFN se aplica independientemente a cada posicion;
   LayerNorm estabiliza los estados en ambos caminos residuales.
4. Retropropagacion NumPy a traves de softmax, mascara, atencion, FFN y
   LayerNorm. Se usa Adam y un prior gaussiano para entrenar con MAP.

El entrenamiento crea ejemplos de siguiente caracter con hasta `T`
caracteres previos. `--ventana` controla `T` (16 por defecto). Esta version
permite aprender dependencias mas largas que un bigrama, pero sigue siendo un
unico bloque pequeno de una cabeza. La FFN expande cada estado a `4d`, aplica
ReLU y proyecta de vuelta a `d`; cada residual va seguido de LayerNorm con
parametros aprendibles. Admite MAP o un posterior variacional gaussiano diagonal
sobre embeddings, posiciones, proyecciones, FFN y parametros de normalizacion.
La inferencia variacional es una aproximacion de campo medio.

Ejemplo de entrenamiento mas largo:

```bash
python main.py --atencion --ventana 24 --epocas-red 150 --epocas-atencion 300
```

`--verbose` muestra las tablas de posiciones y proyecciones. Los corpus y
entrenamientos incluidos son pequenos y didacticos; una muestra de texto corta
no basta para aprender sintaxis fiable ni garantiza cerrar parentesis o comillas.


## Pipeline PDF, muestreo y chat

1. `pdf_a_corpus.py` extrae texto seleccionable con pypdf, normaliza Unicode a
   NFC, elimina caracteres fuera del vocabulario del tokenizador y escribe
   UTF-8 en `corpus.txt` (o en la ruta indicada con `--salida`). Para PDFs
   escaneados hace falta OCR externo; el convertidor lo reporta en vez de crear
   un corpus vacio.
2. Tanto el modelo de bigramas como el modelo causal aceptan temperatura,
   `top_k`, penalizacion de repeticion y bloqueo opcional de n-gramas durante
   `generar()`. Temperatura mas baja concentra la distribucion; `top_k` limita
   candidatos. Los defaults son 0.8, 10, 1.05 y n-gramas de 4 caracteres.
   Estas opciones reducen bucles, pero no garantizan texto coherente.
3. `chat.py` entrena el modelo causal en el corpus y abre un loop de consola con
   el formato `Usuario: ...` / `Asistente:`. Escribe `/salir` para cerrar o
   `/limpiar` para reiniciar el historial.

Ejemplo completo:

```bash
python pdf_a_corpus.py mi_documento.pdf --salida corpus.txt
python chat.py --corpus corpus.txt --ventana 64 --temperatura 0.8 --top-k 8
```

El loop es una demostracion de completado causal, no un asistente instruido: el
modelo continua el patron aprendido en el corpus, no comprende preguntas. El
vocabulario solo admite los caracteres declarados por el tokenizador; el pipeline
sustituye caracteres no admitidos por espacios.

## Visualizacion XOR

`python main.py --gui` abre la matriz de activaciones y la curva de error. La
ventana espera hasta que se presione **Iniciar entrenamiento**.

## Archivos

- `main.py`: menu y ejemplos ejecutables.
- `Red_neuronal.py`, `nodo.py`: red densa, feedforward y backpropagation.
- `funciones_de_activacion.py`, `funciones_de_agregacion.py`: componentes de capas.
- `tokenizador_basico.py`: vocabulario de caracteres espanoles.
- `embedding_basico.py`: inicializacion neuronal y bigrama bayesiano MAP/VI.
- `atencion_causal.py`: auto-atencion causal, FFN, LayerNorm e inferencia MAP/variacional.
- `muestreo.py`: temperatura, top-k y mitigacion simple de repeticion.
- `pdf_a_corpus.py`: extraccion y sanitizacion de PDF.
- `chat.py`: loop de interaccion de consola experimental.
- `test_atencion_causal.py`, `test_pipeline_chat.py`: pruebas de modelo, PDF y muestreo.
- `visualizacion.py`: interfaz Tkinter opcional.
- `requirements.txt`: dependencia NumPy.
