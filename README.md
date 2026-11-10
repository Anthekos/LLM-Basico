# Red neuronal desde cero con NumPy

Proyecto didactico para estudiar redes neuronales y avanzar hacia un modelo de
lenguaje causal de palabras o caracteres. Incluye XOR, un modelo bayesiano de
bigramas y un modelo pequeno con auto-atencion causal. No es un LLM de
produccion ni un Transformer completo.

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
python main.py --atencion --d-model 64 --num-heads 8 --ventana 16 --epocas-atencion 100 --inferencia map
python main.py --atencion --ventana 16 --epocas-atencion 100 --inferencia variacional
python main.py --atencion --texto "hola niña. ¿qué tal?" --ventana 8
python main.py --gui
python pdf_a_corpus.py documento.pdf --salida corpus.txt
python chat.py --corpus corpus.txt --ventana 64 --temperatura 0.8 --top-k 8
python chat.py --corpus corpus.txt --tokenizador palabras --ventana 64 --cuantizado
python chat.py --corpus corpus.txt --dimension 64 --num-heads 8 --cuantizado
python servidor_web.py
python servidor_web.py --tokenizador palabras --cuantizado
python servidor_web.py --d-model 64 --num-heads 8 --cuantizado
```

`--bayesiano` y `--atencion` ejecutan variantes separadas. Si no se especifica
`--texto`, se usa un corpus didactico en espanol incluido en `main.py`.

## Interfaz web local

Activa el entorno virtual e inicia el servidor local:

```bash
python servidor_web.py
```

Abre `http://127.0.0.1:8765` en el navegador. El servidor escucha unicamente
en el equipo local y utiliza por defecto el tokenizador de palabras frecuentes
con atencion causal; se puede elegir `--tokenizador caracteres` explícitamente.
La primera carga entrena el modelo y puede tardar unos segundos; cada respuesta
se genera token por token y puede ser incoherente, porque el modelo no esta
ajustado para seguir instrucciones.

La barra lateral permite abrir varias conversaciones, guardadas en el
almacenamiento local del navegador, y elegir el nombre de usuario. El panel
**Corpus local** distingue `corpus_maximo.txt`, el conjunto base versionado que
se ensambla en orden desde `core_fundacional.txt` y
`core_personalidad.txt`, de `corpus_local.txt`, donde se guardan PDFs y notas.
Al iniciar, el entrenamiento usa primero el núcleo fundacional, después
`core_personalidad.txt` repetido tres veces en `corpus_maximo.txt`, y por último
el corpus local. Los núcleos usan turnos estrictos con los delimitadores
`<|usuario|>`, `<|pensamiento|>`, `<|asistente|>` y `<|fin|>`. La ponderación
da mayor presencia a los ejemplos fluidos de las Fases 4 y 5. La transcripción
antigua de la bitácora cavernícola se conserva en `corpus_local.txt` para su
consulta, pero se excluye del entrenamiento. El servidor vuelve a inicializar
embeddings y reentrena el modelo al arrancar; no carga pesos entrenados de una
ejecución anterior. El panel
permite agregar texto, quitar una coincidencia exacta unicamente del corpus local,
previsualizar documentos y eliminar PDFs junto con el texto extraido. Desde la
galeria se pueden incorporar PDF de ejemplo o subir uno propio. Solo se admiten
PDF con texto seleccionable: no se ejecuta OCR. Se extraen y sanitizan hasta
10000 caracteres por archivo, se agregan al corpus local y se vuelve a entrenar
el modelo. Los PDF originales quedan en `pdfs_corpus/` y su indice en
`pdfs_corpus.json`; los datos locales estan excluidos de Git.

La seccion **Administracion del corpus maximo** permite editar y reentrenar la
base. El editor separa la base fundacional (ortografia, gramatica y
concordancia) del nucleo de personalidad (sobriedad, deliberacion, seguridad y
reconocimiento de limites); al guardar regenera `corpus_maximo.txt` en ese
orden antes de reentrenar con el corpus local. La edicion requiere iniciar
sesion. En la primera ejecucion se crea
`.venv/admin_credentials.json` con el usuario provisional `admin` y la
contraseña `admin`; se almacena un hash PBKDF2 con sal, no la contraseña en
texto plano. El primer inicio obliga a establecer una contraseña nueva de al
menos 12 caracteres. El archivo queda dentro del entorno virtual (ignorado por
Git), y el endpoint de escritura solo acepta sesiones administrativas activas.
La sesion expira al cabo de una hora. El servidor enlaza a `127.0.0.1`; no se
debe exponer este servidor de desarrollo a redes compartidas.

Cada respuesta del asistente ofrece **Guardar como nota de memoria**. La nota
incluye el intercambio, crea un PDF descargable y agrega su texto extraido al
corpus local. El boton **Reintentar** vuelve a generar la respuesta de la
ultima pregunta. Los botones de feedback util/no util aplican una recompensa
positiva o negativa con un paso de ajuste online pequeno y registran la
recompensa en `feedback_recompensa.jsonl`. El registro se conserva en el equipo,
pero no se reproduce al reiniciar: los pesos ajustados viven en memoria y se
reinicializan al reentrenar.

La inferencia termina el prompt con `<|pensamiento|>` y espera una continuación
con pensamiento, `<|asistente|>`, respuesta y `<|fin|>`. La generación se
detiene ante `<|fin|>` o un nuevo `<|usuario|>`. Si el modelo genera ambos
bloques, la interfaz presenta el
pensamiento en un desplegable colapsado y la respuesta por separado. Es una
salida experimental del modelo, no una explicacion fiable de sus procesos ni
una garantia de razonamiento correcto. Una puntuacion léxica local compara
terminos de cada intercambio con el corpus y guarda una ficha extractiva en
`corpus_local.txt` cuando supera el umbral. El umbral inicial es 0.75 y puede
ajustarse en **Configuracion modelo** o al iniciar con
`python servidor_web.py --umbral-saliencia 0.8`; el valor elegido se guarda en
`configuracion_modelo.json`.

El backend tambien expone `GET /api/corpus`, `POST /corpus/agregar`,
`POST /corpus/quitar`, `POST /api/memory`, `POST /api/feedback` y
`DELETE /api/pdfs/<id>` para integrar estos flujos desde la interfaz local. La
administracion usa `POST /api/admin/login`, `POST /api/admin/password`, `POST /api/admin/logout`,
`GET /api/admin/corpus-maximo` y `PUT /api/admin/corpus-maximo`; los dos ultimos
requieren un token Bearer obtenido en el inicio de sesion.

**Configuracion modelo** abre el panel lateral de muestreo. La temperatura se
puede ajustar de 0.1 a 3.0 y top-k desde 1 hasta el vocabulario completo del
tokenizador. Los valores se guardan en el navegador. Una temperatura mas alta
introduce mas variedad; top-k mas grande permite mas candidatos, pero ninguno
garantiza respuestas coherentes.

En el mismo panel se puede cambiar la ventana de contexto entre 4 y 128
caracteres. El cambio reconstruye y reentrena el bloque causal con el corpus
local, por lo que puede tardar; la ventana seleccionada queda guardada en
`configuracion_modelo.json` y se reutiliza al reiniciar el servidor. Aumentar
la ventana permite consultar mas caracteres previos, con mayor coste de
memoria y computo.

Cuando el historial supera el espacio de la ventana, el texto mas antiguo se
procesa en bloques y se resume en un vector de memoria contextual; los
caracteres mas recientes permanecen como entrada normal. Esta compresion es
con perdida y experimental: conserva una representacion numerica aproximada,
no el texto original ni una memoria exacta del dialogo.

El interruptor **Ajuste online** aparece bajo el perfil de usuario y esta
apagado por defecto, incluso al cargar preferencias antiguas del navegador.
Al activarlo, despues de cada respuesta se realiza un paso pequeno de
entrenamiento sobre el ultimo intercambio. Los pesos solo viven en memoria:
se pierden al reiniciar el servidor o reconstruir el modelo, y el ajuste puede
sobreajustar el modelo a una conversacion. No es persistente ni sustituye
el entrenamiento del corpus.

El limite de carga es 12 MB y 30 paginas; no se aceptan mas de 10000 caracteres
extraidos por PDF.

La interfaz web usa solamente la biblioteca estandar de Python, NumPy y
`pypdf`, ya incluidos en `requirements.txt`; no necesita instalar un framework
web ni cargar recursos externos. Para compilar las notas con LaTeX se usa
`pdflatex` si esta instalado; en su ausencia el generador crea un PDF local
ligero con texto seleccionable.

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
2. Proyecciones `W_q`, `W_k`, `W_v` y `W_out` de forma `(d, d)`.
3. Atencion causal multi-cabeza: `d_model` se divide equitativamente en
   `num_heads`, con `head_dim = d_model // num_heads`.

   `Q, K, V: (B, H, T, head_dim)`

   `A = softmax(Q K^T / sqrt(head_dim) + M)`

   `O = concat_heads(A V) W_out`, `R_1 = H + O`

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

El entrenamiento crea ejemplos del siguiente token con hasta `T` tokens
previos. `--ventana` controla `T` (16 por defecto). El numero de cabezas se
puede pasar como `num_heads` al constructor y debe dividir a `d_model`; el
default elige el mayor divisor de hasta cuatro. Sigue siendo un unico bloque
didactico. La FFN expande cada estado a `4d`, aplica
ReLU y proyecta de vuelta a `d`; cada residual va seguido de LayerNorm con
parametros aprendibles. Admite MAP o un posterior variacional gaussiano diagonal
sobre embeddings, posiciones, proyecciones, FFN y parametros de normalizacion.
La inferencia variacional es una aproximacion de campo medio.

El entrenamiento y el backward de la MHA estan implementados en NumPy; los
gradientes de `W_q`, `W_k`, `W_v` y `W_out` se verifican con diferencias finitas.
La mascara causal bloquea tanto la atencion como la influencia de tokens futuros.

## Inferencia int8 y tokenizador por palabras

`cuantizacion.py` implementa cuantizacion simetrica por tensor `float32`→`int8`
y multiplicacion de matrices con acumuladores `int32`. El modo cuantizado
prepara copias de `W_q`, `W_k`, `W_v`, `W_out`, las dos matrices FFN y la LM
head al finalizar el entrenamiento; las actualizaciones online vuelven a
cuantizar las copias. Se ejecuta con `--cuantizado` en `chat.py` o
`servidor_web.py` y requiere inferencia MAP. NumPy puede aprovechar kernels del
procesador disponible, aunque la ganancia concreta depende de la version de
NumPy y no se promete una aceleracion ARM fija.

`tokenizador_palabras.py` crea vocabulario de hasta 1500 palabras frecuentes,
signos de puntuacion, `<UNK>` y `<PAD>`. En corpus pequenos el vocabulario puede
ser menor; el limite evita que crezca sin control. El servidor y el chat usan
`--tokenizador palabras` por defecto; `--tokenizador caracteres` conserva el
modo de caracteres para comparaciones.
Con palabras, la ventana cuenta tokens en lugar de caracteres. Los tokens fuera
del vocabulario se mapean a `<UNK>`.

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
   UTF-8 en `corpus.txt` (o en la ruta indicada con `--salida`). Al incorporar
   documentos desde el servidor, su texto se guarda en `corpus_local.txt`.
   Para PDFs escaneados hace falta OCR externo; el convertidor lo reporta en
   vez de crear un corpus vacio.
2. Tanto el modelo de bigramas como el modelo causal aceptan temperatura,
   `top_k`, penalizacion de repeticion y bloqueo opcional de n-gramas durante
   `generar()`. Temperatura mas baja concentra la distribucion; `top_k` limita
   candidatos. Los defaults son 0.8, 10, 1.05 y n-gramas de 4 elementos.
   Estas opciones reducen bucles, pero no garantizan texto coherente.
3. `chat.py` entrena el modelo causal en el corpus y abre un loop de consola con
   el formato `Usuario: ...` / `Asistente:`. Escribe `/salir` para cerrar o
   `/limpiar` para reiniciar el historial.

Ejemplo completo:

```bash
python pdf_a_corpus.py mi_documento.pdf --salida corpus.txt
python chat.py --corpus corpus.txt --ventana 64 --temperatura 0.8 --top-k 8
python chat.py --corpus corpus.txt --tokenizador palabras --cuantizado
```

El loop es una demostracion de completado causal, no un asistente instruido: el
modelo continua el patron aprendido en el corpus, no comprende preguntas. El
modo caracteres acepta el alfabeto definido por el tokenizador; el modo palabras
representa palabras no vistas con `<UNK>`.

## Visualizacion XOR

`python main.py --gui` abre la matriz de activaciones y la curva de error. La
ventana espera hasta que se presione **Iniciar entrenamiento**.

## Archivos

- `main.py`: menu y ejemplos ejecutables.
- `Red_neuronal.py`, `nodo.py`: red densa, feedforward y backpropagation.
- `funciones_de_activacion.py`, `funciones_de_agregacion.py`: componentes de capas.
- `tokenizador_basico.py`: vocabulario de caracteres espanoles.
- `tokenizador_palabras.py`: vocabulario acotado por frecuencia con `<UNK>` y `<PAD>`.
- `cuantizacion.py`: utilidades simetricas int8 e inferencia de matrices cuantizadas.
- `embedding_basico.py`: inicializacion neuronal y bigrama bayesiano MAP/VI.
- `atencion_causal.py`: MHA causal, FFN, LayerNorm, clipping e inferencia MAP/variacional.
- `memoria_rag.py`: indice léxico local, puntuacion de saliencia y fichas extractivas.
- `test_admin_corpus.py`: inicio de sesion y edicion autenticada del corpus maximo.
- `muestreo.py`: temperatura, top-k y mitigacion simple de repeticion.
- `pdf_a_corpus.py`: extraccion y sanitizacion de PDF.
- `generador_pdf.py`: creacion de notas de memoria PDF locales.
- `corpus_maximo.txt`: corpus base protegido y versionado.
- `core_fundacional.txt`, `core_personalidad.txt`: nucleos ordenados que se
  ensamblan en el corpus maximo y se incluyen antes del corpus local.
- `chat.py`: loop de interaccion de consola experimental.
- `test_deliberacion.py`: formato de deliberacion, ventana y memoria autonoma.
- `test_personalidad.py`: vocabulario reservado, prefijo del prompt y ventana.
- `test_atencion_causal.py`, `test_pipeline_chat.py`: pruebas de modelo, PDF y muestreo.
- `visualizacion.py`: interfaz Tkinter opcional.
- `requirements.txt`: dependencia NumPy.
