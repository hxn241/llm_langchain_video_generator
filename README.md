# 🎬 Datazo — generador de vídeos de ciencia y tecnología

Pipeline multiagente en Python (LangGraph) que busca un tema de **ciencia y tecnología**, lo investiga, escribe el guion, lo narra con una voz clonada, elige o genera las imágenes sincronizadas con cada palabra, monta el vídeo y lo deja listo para subir a YouTube. Se para en cada decisión importante para que la tome una persona.

Es el motor del canal **Datazo**: cada vídeo gira alrededor de un dato que sorprende y lo explica hasta que se entiende.

> **Filosofía:** la IA hace el trabajo pesado (buscar, documentarse, redactar, montar); el criterio es humano, en cada pausa. Eso mantiene el canal dentro de la política de "contenido no auténtico" de YouTube y es lo que da calidad.

**Dónde corre cada cosa** (una RTX 5070 de 12 GB):

| Local (gratis) | Nube |
|---|---|
| Voz (Qwen3-TTS), tiempos de cada palabra (Whisper), filtro de imágenes (SigLIP 2), imágenes generadas (Z-Image Turbo en ComfyUI), 2.5D (Depth Anything), montaje (ffmpeg + NVENC) | Texto: Claude Sonnet 5.5 en OpenRouter (de pago); extraer hechos, juez de imágenes y retratos: Claude Haiku 5.5. Búsquedas (Tavily o DuckDuckGo), Wikipedia, bancos de imágenes, YouTube |

Coste estimado de un vídeo largo (5 min): **~0,9–1,2 $** (todo el texto y el juez; la voz y las imágenes generadas en local no cuestan nada). El gasto real de cada run queda en `<run>/costes.md`, por nodo.

---

## Índice

1. [Formato del vídeo](#1-formato-del-vídeo)
2. [Stack](#2-stack)
3. [Arquitectura](#3-arquitectura)
4. [Nodos](#4-nodos)
5. [Lo que deja cada run](#5-lo-que-deja-cada-run)
6. [Estado del grafo](#6-estado-del-grafo)
7. [Estructura del repositorio](#7-estructura-del-repositorio)
8. [Configuración](#8-configuración)
9. [Instalación (Windows + RTX 5070)](#9-instalación-windows--rtx-5070)
10. [Uso](#10-uso)
11. [Calidad y políticas de YouTube](#11-calidad-y-políticas-de-youtube)
12. [Pendiente](#12-pendiente)

---

## 1. Formato del vídeo

- **Temática:** ciencia y tecnología (espacio, física, salud, biología, energía, clima, informática, IA, materiales, ingeniería, inventos). Se definen en `config.yaml` (`channel:`) y entran en todos los prompts.
- **Dos tipos de vídeo:**
  - **news:** algo que acaba de pasar. La investigación usa búsquedas de noticias.
  - **explainer:** algo que no caduca (cómo funciona algo, por qué pasa). La investigación usa la web general y 1-2 artículos de Wikipedia.
  - Se elige con `--type`; si no, lo decide el modelo a partir del tema.
- **Formatos:**

  | | Resolución | Duración | Escenas | Secciones |
  |---|---|---|---|---|
  | **Short** | 9:16, 1080×1920 | ~75 s (65–85), ~188 palabras | 10–16 | 3–4 |
  | **Largo** | 16:9, 1920×1080 | ~5,5 min (330 s; 300–360) | 24–40 | 4–5 |

- **Guion:** español de España, al espectador se le habla de **tú** (solo cuando el tema le toca). Gancho con la sorpresa primero, tensión hasta el final y un cierre que vuelve al gancho. La última frase **nunca** es una pregunta.
  - En los largos, justo después del gancho, una **presentación** de una o dos frases (ver `write_script`).
- **Estilo:** voz en off, imágenes y clips que cambian justo en la palabra que los nombra, movimiento de cámara en las fotos (2.5D opcional, `video.parallax`), **rótulos** (fuente, persona con su retrato, término clave) y subtítulos quemados. Opcionalmente, música propia.
  - Short: subtítulos de 3 palabras con la palabra que se dice resaltada.
  - Largo: frases clásicas en 1–2 líneas.

---

## 2. Stack

| Capa | Herramienta | Notas |
|---|---|---|
| Orquestación | **LangGraph** + `SqliteSaver` | Cada run es un *thread*: se para (*interrupt*) y se reanuda con `--resume` |
| Envoltorio LLM | `src/llmkit` + `services/llm_hub.py` (`LLMHub`) | Modelo por nodo, salida estructurada, pausas, checkpoints |
| Texto | **Claude Sonnet 5.5** vía **OpenRouter** | Todos los nodos de texto. Respaldo automático si falla: Nemotron 3 Super (gratis) |
| Extraer hechos | **Claude Haiku 5.5** vía OpenRouter (`llm.facts`) | Cada texto leído en `research` y `deep_research`; ~20 veces más barato que Sonnet. Respaldo: Sonnet |
| Juez de imágenes (visión) | **Claude Haiku 5.5** vía OpenRouter (`llm.visuals`) | Ve las fichas (título y descripción) de cada candidato. Respaldo: Gemma 4 31B `:free` |
| Rótulos y retratos | Sonnet (`llm.labels`) + **Wikidata** / Commons; Haiku confirma cada retrato (`llm.portraits`) | Sin foto si hay la menor duda |
| LLM local (opcional) | **LM Studio** (`localhost:1234/v1`) | Hoy no se usa en ningún nodo. `unload_llm` libera su VRAM antes de la fase de medios |
| Temas | `feedparser` + 17 feeds de ciencia y tecnología | `data/temas/registro.json`: ningún tema se repite |
| Investigación | `trafilatura`, Tavily o DuckDuckGo (`ddgs`), API de Wikipedia | Tiendas, anuncios, redes y foros nunca son fuente (`research.skip_domains`) |
| Voz | **Qwen3-TTS 1.7B** (clonación, local, Apache 2.0) | La voz de referencia sale de `scripts/design_voice.py` o de una grabación tuya. `edge-tts` solo para pruebas |
| Tiempos de cada palabra | **Whisper large-v3-turbo** (transformers, GPU) | Se alinea con las palabras del guion: `words[i]` es siempre la palabra i del guion |
| Fuentes de imagen | **Pexels**, **Pixabay** (vídeo y foto), **NASA** (dominio público), **Wikimedia Commons** (licencias libres, con créditos) | Sin clave en el `.env`, ese banco se salta |
| Filtro de imágenes | **SigLIP 2** (local) | Primer filtro texto-imagen; los 8 mejores pasan al juez |
| Imágenes generadas | **Z-Image Turbo** en **ComfyUI** (local) | Alternativa de pago: Nano Banana 2 en OpenRouter (`visuals.generator: openrouter`) |
| 2.5D | **Depth Anything V2 Small** (local) | El primer plano se mueve un poco más que el fondo |
| Montaje | **ffmpeg 7.1** de `imageio-ffmpeg` (dentro del venv) | `h264_nvenc`, `loudnorm`, `xfade`, subtítulos ASS |
| Subida | **YouTube Data API v3** (`google-api-python-client`, OAuth) | Vídeo, miniatura, subtítulos `.srt`, aviso de contenido sintético |

### Dos entornos de Python

- **`.venv`** (Python 3.13): el proyecto. No lleva torch.
- **`.venv-tts`** (Python 3.12): torch `cu128` + `qwen-tts` + `transformers`. Ahí corren los *workers* de GPU de `src/tts_worker/` (voz, Whisper, SigLIP 2, profundidad). El proyecto los lanza como subproceso.
  - Cada worker carga su modelo, hace todo el trabajo y termina, así que la VRAM se libera sola.
  - Las RTX 50 (Blackwell) necesitan PyTorch con **CUDA 12.8+**.
- **ComfyUI** va aparte, con su propio venv. El proyecto habla con él por HTTP (`services/comfy.py`) y libera su VRAM con `POST /free`.

---

## 3. Arquitectura

> El grafo detallado (qué tipo de pieza es cada nodo, cuántas llamadas hace, qué recibe y qué deja) está en **[ARQUITECTURA.md](ARQUITECTURA.md)**.

```
 run_pipeline.py
      │
      ├─ ─►sin --topic  find_topics 🤖 ─► ⏸ choose_topic ─┐
      │                                                    │
      └─ --topic "..." ───────────────────────────────────►┴─► research 🤖 (news | explainer)
                                                                   │
              ┌──────── [n] + motivo ───────┐                      ▼
              ▼                             │
          angles 🤖 ─────────────────► ⏸ choose_angle   (3 enfoques; [e] otro o el tuyo)
                                            │
              ┌──────── [n] + motivo ───────┤
              ▼                             ▼
          outline 🤖 ───────────────► ⏸ approve_outline
                                            │
                       secciones con preguntas (research.deep)
                                            ▼
                                  deep_research 🤖 ─► ⏸ review_research  ([n] tus enlaces por sección)
                                            │
          ┌─────────────────────────────────┤  (--manual: directo a manual_script)
          ▼                                 │
     write_script 🤖 ◄── algo grave (máx. review.max_revisions) ──┐
          │                                                       │
          ▼                                                       │
     review_script 🤖 ────────────────────────────────────────────┘
          │
          ▼
     ⏸ choose_script ── [n] + motivo ─► write_script
          │           └ [n] + Enter ──► ⏸ manual_script (tu guion y tus imágenes en <run>/manual/)
          ▼ [y]                                │
     unload_llm ◄──────────────────────────────┘
  ── fin de la fase de texto ──────────────────────────────────────────────
          │
          ▼
     voice (Qwen3-TTS) ─► align (Whisper) ─► brief 🤖 ─► beats 🤖 ─► labels 🤖 ─► visuals 🤖
                                                                                    │
                                                                                    ▼
                                                                 ⏸ review_visuals (<run>/montaje/)
                                                                                    │
                                                                                    ▼
                                                                               assemble ─► <run>/<fecha>_<tema>.mp4

  Fuera del grafo:  prepare_upload.py ─► data/listo_para_subir/<vídeo>/ ─► upload_youtube.py ─► YouTube

🤖 = nodo con LLM      ⏸ = pausa (decides tú; --yes lo aprueba todo, solo para pruebas)
```

### Principio de sincronización

- El **guion** solo tiene palabras, divididas en escenas.
- Cada escena tiene su audio. Whisper da el tiempo de **cada palabra**.
- Después, `beats` decide **qué se ve y desde qué palabra**. Una escena puede tener varios planos.
- Cada plano empieza exactamente cuando se dice lo que muestra.
- Las escenas duran su audio más `video.scene_gap`, redondeado a fotogramas enteros. La posición se acumula, así que no hay deriva entre imagen y voz.

### Fases y VRAM (12 GB)

- **Fase de texto:** todo en la nube. Nada ocupa la GPU, salvo un modelo de LM Studio si se configura uno local.
- **Fase de medios:** un modelo de GPU cada vez: Qwen3-TTS → Whisper → SigLIP 2 → Z-Image (ComfyUI) → Depth Anything → NVENC. Cada uno libera la VRAM al terminar.

---

## 4. Nodos

### Temas — `find_topics` 🤖 + `choose_topic` ⏸
Solo cuando el run empieza sin `--topic`.
- Lee los feeds de `topics.feeds`: entradas de los últimos `topics.days` días, sin las que ya están en el registro.
- El modelo agrupa las entradas que cuentan lo mismo y las puntúa para el canal:
  - ¿hay un dato que sorprende?
  - ¿se puede explicar?
  - ¿dura en el tiempo?
  - ¿hay algo que enseñar?
- Se suman los temas propuestos en los últimos `reserve_days` y no elegidos. Todo va a `data/temas/<fecha>.md`.
- Opciones en la pausa:
  - `[y]` el primero; `[e]` otro número.
  - `[n] x 3 5` descarta esos temas para siempre; `[n]` + Enter muestra otros.
- El tema elegido trae sus artículos (`--url`), su *datazo* (`--notes`), su tipo y su formato, y queda `en_curso` en `registro.json`.

### `research` 🤖
- Decide **news** o **explainer**, escribe las búsquedas y, si es explainer, elige los artículos de Wikipedia.
- Lee en este orden: tus `--url`, Wikipedia y los resultados de búsqueda.
  - Salta los muros de pago, las webs bloqueadas y `skip_domains`.
  - Salta también los textos en los que el modelo no encuentra nada.
- Extrae **hechos de cada texto por separado**, así cada hecho conserva su fuente exacta.
- Esos hechos son lo **único** que pueden usar el esquema y el guion.
- Los textos leídos quedan en `<run>/research/`.

### `angles` 🤖 + `choose_angle` ⏸
- Propone 3 enfoques distintos (pregunta central, gancho y por qué engancha) y recomienda uno.
- Busca el conflicto o la sorpresa, no el "resumen de la noticia".
- Opciones en la pausa:
  - `[y]` el recomendado.
  - `[e]` otro número o tu propio enfoque escrito.
  - `[n]` + motivo: tres nuevos.
- `angles.json` se puede editar antes de aprobar.

### `outline` 🤖 + `approve_outline` ⏸
- Escribe el título, la pregunta que guía el vídeo y las secciones.
- Cada sección lleva:
  - su papel: `hook`, `context`, `tension`, `development`, `reveal` o `conclusion`;
  - la emoción que busca;
  - sus hechos;
  - las **preguntas** que hay que averiguar para contarla.
- Las palabras de cada sección las fija el código (`lengths.py`), no el modelo.
- `outline.json` se puede editar. `[n]` + motivo lo rehace.

### `deep_research` 🤖 + `review_research` ⏸
Busca el contexto de cada sección que tenga preguntas. Corre en shorts y en largos (`research.deep`).
- Por cada sección:
  - búsquedas web y artículos de Wikipedia;
  - hasta `deep_sources_per_section` textos;
  - hechos nuevos asignados a esa sección.
- En la pausa ves cuántos hechos tiene cada sección y cuánto durará el vídeo.
  - Una sección no puede ser más larga de lo que dan sus hechos (`words_per_fact`). **Nunca se rellena.**
  - Una sección sin hechos se quita.
- `[n]` + `4 https://... 6 https://...` añade tus enlaces a esas secciones.

### `write_script` 🤖
- **Hasta `script.whole_max_words` (1200) palabras** —los shorts y los largos de 5–6 min—: el guion entero en una llamada, y las reescrituras también. Por secciones, cada una repetía las ideas de las otras.
- **Más largo:** una sección por llamada. Cada llamada ve:
  - el esquema entero;
  - sus hechos;
  - cómo acabó la sección anterior;
  - qué hechos ya se contaron.
  - La última sección, además, ve cómo empezó el vídeo.
- **Presentación (solo largos, `graph/intros.py`):** justo después del gancho, una o dos frases en las que el narrador presenta el vídeo. Hay 5 formas (`promesa`, `narrador`, `mapa`, `pregunta`, `viaje`); cada vídeo largo usa la siguiente a la del anterior, y un run conserva la suya en las reescrituras. El registro está en `data/presentaciones.json`.
- Antes de las escenas, el modelo escribe un **plan**: la pregunta que abre, dónde se complica, dónde va el dato potente y cómo cierra. Va a `<run>/plan_guion.md` y no se narra.
- Si no cumple la longitud o repite expresiones entre escenas, se le pide **una** vez más diciendo qué arreglar.
- Las fuentes del guion salen de la investigación, nunca del modelo.

### `review_script` 🤖
Un editor lee el guion entero con el esquema y los hechos delante (`prompts/review_script.md`) y devuelve:
- **issues**: qué arreglar y cómo, escena por escena.
  - Los **graves** vuelven a `write_script` como mucho `review.max_revisions` veces seguidas:
    - un dato que no sale de los hechos;
    - un tema sin responder;
    - algo clave que no se entiende;
    - un gancho que destripa el final;
    - el vídeo entero sin tensión;
    - una pregunta como cierre.
  - En un largo solo se reescriben las secciones con nota.
- **dubious**: hechos que parecen un error del artículo. No se corrigen solos: decides tú.

Comprueba también que se aplicó lo que se pidió en la reescritura anterior. Todo lo que queda va a `script.md` (sección "Revisión").

### `choose_script` ⏸ + `manual_script` ⏸
- `[y]`: se usa ese guion.
- `[n]` + motivo: se reescribe y se revisa otra vez.
- `[n]` + Enter: **guion propio** en `<run>/manual/guion.md` (ver `src/graph/manual.py`):
  - un párrafo por escena;
  - `## ` empieza una sección;
  - `[archivo.jpg]` delante de una palabra hace que ese archivo de `manual/visuales/` se vea desde ahí;
  - las escenas sin marcas se ilustran solas.
- Con `--manual` se llega aquí directamente desde el esquema.
- Lo que dejes antes en `data/manual/` se copia al run.

### `unload_llm`
Fin de la fase de texto: `lms unload --all` si `llm.unload_after_text_phase`.

### `voice`
- `voice.mode: clone`: Qwen3-TTS con la voz de `assets/voice_reference.wav` y su transcripción exacta en `voice_reference.txt`.
  - Todas las escenas en un solo trabajo.
  - Semilla fija: el mismo texto da el mismo audio.
- Se corta el silencio del final de cada escena y se normaliza el volumen.
- `edge` (voces de Microsoft) solo sirve para pruebas: no cumple la política de YouTube.
- `record` (grabarlo tú) está pendiente.

### `align`
- Whisper transcribe cada escena con tiempos por palabra.
- Sus palabras se emparejan con las del guion (`difflib`). Si Whisper escribe una palabra de otra forma ("1.144" por "mil ciento cuarenta y cuatro"), su tiempo se interpola entre sus vecinas.

### `brief` 🤖
Una llamada por vídeo con el guion final. Define el universo visual:
- los objetos, lugares y acciones filmables del tema;
- lo que **nunca** debe verse (`avoid`);
- cómo se enseña cada metáfora del guion (sin tomarla al pie de la letra);
- el estilo común de las imágenes generadas.

Se guarda en `brief.json`. Si lo editas, `--redo brief` lo vuelve a leer.

### `beats` 🤖
Planifica **por idea**: 1–2 ideas por escena. Cada idea lleva:
- su tipo: `entity`, `generic`, `data`, `source` o `abstract`;
- lo que no debe verse;
- sus **planos distintos**, cada uno con lo que tiene que verse, 1–3 búsquedas en inglés (de la más concreta a la más genérica) y un prompt por si hay que generarlo.

Después, el código coloca cada idea en su palabra:
- une los planos más cortos que `min_beat_seconds`;
- parte los más largos que `beat_seconds` (short 2–4 s, largo 4–6 s), y cada parte enseña otro plano de la idea.

Se guarda en `beats.json`, `manual/prompts_imagenes.md` y `plan_visual.json` (los prompts, para generar imágenes a mano con otra IA).

### `labels` 🤖
Una llamada por vídeo con el guion narrado y los hechos (`prompts/labels.md`). Decide dónde aparece cada rótulo, como mucho una vez cada uno:
- **fuente:** el estudio, la revista o la institución ("Estudio publicado en Nature · 2026");
- **persona:** alguien con nombre, y su cargo;
- **término:** un concepto clave, la primera vez que se explica.

Cada rótulo empieza en la palabra en la que se dice y `assemble` lo dibuja encima del vídeo (`graph/onscreen.py`).
- **Retratos** (`video.labels.portraits`): un rótulo de persona lleva su foto si Wikidata tiene a alguien con ese nombre exacto y una foto libre en Commons, **y** Haiku confirma que esa ficha es la persona del vídeo (`prompts/portraits.md`). Ante cualquier duda, solo el texto.
- Se guarda en `rotulos.json` (y las fotos en `retratos/`). Si existe, se usa tal cual: puedes cambiar un texto, quitar un rótulo o borrar una `"foto"`. Bórralo para que se generen otros (`--redo labels`).
- `video.labels.enabled: false`: sin rótulos.

### `visuals` 🤖
Por cada plano:
1. **Candidatos** de las fuentes de su tipo (`visuals.sources`):

   | Tipo | Fuentes |
   |---|---|
   | `entity` | stock, Wikimedia, NASA |
   | `generic` | stock, NASA |
   | `abstract` | stock, NASA |
   | `data`, `source` | stock |

   Todas se reparten `max_candidates` y las búsquedas se guardan en caché.
2. **SigLIP 2** se queda con los `top_k` más parecidos al texto.
3. El **juez con visión** los ve en una sola cuadrícula, recortados como saldrán en pantalla y con las palabras que se dicen, y elige uno. Por debajo de `min_score` no se usa ninguno.
4. Lo que no pasa el juez se **genera** con Z-Image en ComfyUI (o con Nano Banana si `generator: openrouter`). Con el mismo prompt se reutiliza la imagen, no se regenera.

Más reglas:
- En vídeos largos, una foto no dura más de `photo_max_seconds` (7,5 s).
  - Primero se busca un **vídeo** para ese plano.
  - Si no hay, el plano se reparte entre varias imágenes (`05.jpg` + `05b.jpg`).
- Cada clip o foto se usa una sola vez por vídeo.
- Al final se copia todo en orden a `<run>/montaje/`.

### `review_visuals` ⏸
`<run>/montaje/` tiene un archivo por plano (`01.mp4`, `02.jpg`...) y `lista.md` (tiempo, texto y origen de cada uno). Ahí puedes:
- **sustituir** un plano: borra `05.mp4` y deja el tuyo como `05.jpg`;
- **quitarlo**: el plano anterior sigue durante ese tiempo;
- **añadir** uno: `05b.mp4` comparte el tiempo del plano 5;
- poner **música**: `musica.mp3`.

`[y]` monta exactamente lo que hay, sin juez. `[n]` para sin montar; luego `--redo assemble` vuelve a leer la carpeta.

### `assemble`
- **Clips de vídeo:** se quitan las bandas negras y se recortan para llenar el cuadro. Si un clip es corto, se ralentiza un poco y se congela el último fotograma.
- **Fotos:** movimiento lento y distinto en cada una (zoom, paneo, inclinación), con **2.5D** (Depth Anything) si `video.parallax.enabled` (desactivado por defecto).
- **Rótulos:** los de `rotulos.json`, con su retrato si lo tienen, encima del vídeo durante `video.labels.seconds`.
- **Transiciones:** fundido cruzado solo entre escenas. Dentro de una escena, corte seco en la palabra clave.
- **Audio:**
  - la voz se normaliza a −14 LUFS;
  - la música va a un nivel fijo por debajo de la voz (`video.music.db`), con un hueco en 1–4 kHz y fundidos.
- **Final:** subtítulos ASS quemados (`video.subtitles`) y `h264_nvenc` → `<run>/<yyyymmdd>_<tema>.mp4`.
- Los segmentos quedan en `<run>/segments/`.

### Publicación (fuera del grafo)
- **`scripts/prepare_upload.py --run <id>`** crea `data/listo_para_subir/<fecha_tema>/` con:
  - `video.mp4`;
  - `metadata.json` (título, tags, privacidad, contenido sintético...);
  - `descripcion.md` (resumen, capítulos, fuentes, **créditos** de las imágenes, hashtags);
  - `miniatura.jpg` (un plano del vídeo con 2–3 palabras grandes que no repiten el título; `--thumb N` elige el plano). En los largos, el texto va en el lado de la imagen con menos detalle, sobre un degradado oscuro, con la palabra clave en amarillo; en los shorts, centrado abajo;
  - `subtitulos.srt`.

  Solo el título, el resumen, las etiquetas y el texto de la miniatura salen del modelo (`prompts/publish.md`). Edita lo que quieras antes de subir.
- **`scripts/upload_youtube.py <carpeta>`** sube el vídeo, la miniatura y los subtítulos, y marca el contenido sintético. Después mueve la carpeta a `data/subidos/` con `youtube.json` y marca el tema como `subido` en el registro.
- **`scripts/retention.py`** escribe `data/retencion/<fecha>.md` con todos los vídeos del canal: vistas, duración media vista, % medio, cuánta gente sigue a los 3, 10 y 30 s y al final, y las caídas más fuertes con la frase que sonaba (de los `subtitulos.srt` de `data/subidos/`). Necesita la "YouTube Analytics API" activada en el mismo proyecto de Google Cloud. Las impresiones, el CTR y el "se quedaron viendo" de los shorts no están en la API: exporta el CSV en YouTube Studio (Modo avanzado → Contenido → descargar) y déjalo, `.csv` o `.zip`, en `data/retencion/studio/`.

---

## 5. Lo que deja cada run

`data/runs/<run_id>/` (`run_id` = `yyyymmdd-hhmmss`):

| Archivo | Qué es |
|---|---|
| `research/` | Cada texto leído (`s<sección>_<n>.txt` los de `deep_research`), para comprobar de dónde sale un hecho |
| `angles.md` / `angles.json` | Los 3 enfoques (el `.json` se edita) |
| `outline.md` / `outline.json` | El esquema (el `.json` se edita) |
| `plan_guion.md` | El plan de tensión del guionista |
| `script.md` / `script.json` | El guion con su revisión. Si cambias `script.json` y haces `--redo voice`, se vuelve a narrar |
| `manual/` | `guion.md` y `visuales/` (modo manual); `prompts_imagenes.md` y `plan_visual.json` (siempre) |
| `audio/` | Un WAV por escena |
| `brief.json` / `beats.json` | El plan visual (se editan; `--redo brief` o `--redo beats`) |
| `rotulos.json` / `retratos/` | Los rótulos en pantalla y las fotos de las personas (se edita; `--redo labels`) |
| `visuals/` | Lo descargado y generado; `thumbs/` (candidatos), `choices.json` (qué se eligió y por qué), `searches.json` (caché) |
| `montaje/` | Lo que entra en el vídeo, en orden, más `lista.md`, `.auto.json` y `musica.mp3` (opcional) |
| `creditos.txt` | Atribuciones de Wikimedia |
| `segments/` | Los trozos de vídeo de cada plano |
| `costes.md` | Lo que ha costado el run, por nodo |
| `<yyyymmdd>_<tema>.mp4` | El vídeo final |

Fuera de los runs:
- `data/checkpoints.sqlite`: el estado de LangGraph.
- `data/presentaciones.json`: la forma de presentación de cada vídeo largo.
- `data/temas/`: las búsquedas de temas y `registro.json`.
- `data/listo_para_subir/` y `data/subidos/`.

---

## 6. Estado del grafo

`src/graph/state.py`. El estado guarda **datos planos** (dicts y listas), que se leen con `Script.model_validate(state["script"])`. Hay dos tipos de esquema:

- **`*Draft`:** lo que rellena el LLM. Solo lleva campos editoriales: los ids, las secciones, las fuentes y las longitudes las pone el código, así el modelo no puede inventarlas.
- **`Outline`, `Script`, `Scene`:** lo que se guarda y edita una persona.

Claves principales:

| Clave | Contenido |
|---|---|
| `topic`, `notes`, `urls`, `format`, `video_type`, `manual` | Lo que se pide al empezar (o lo que trae el tema elegido) |
| `topic_candidates`, `topic_id` | Temas propuestos y el elegido (registro) |
| `sources`, `facts` | Artículos leídos y hechos (`{fact, source_url, section}`) |
| `angles`, `angle`, `outline` | Enfoques, el elegido y el esquema |
| `script`, `script_plan`, `script_review`, `script_notes`, `revisions` | Guion, plan, revisión y reescrituras |
| `audio` | `scene_id → {path, duration, words[{word, start, end}]}` |
| `brief`, `beats` | Plan visual: universo del vídeo y qué se ve desde qué palabra |
| `labels` | Rótulos en pantalla (tipo, texto, palabra en la que empiezan, foto) |
| `visuals`, `visual_extras`, `montage` | `beat_id →` visual elegido (+ los extra de planos largos) y lo que usa `assemble` |
| `video_path` | El `.mp4` final |

---

## 7. Estructura del repositorio

```
llm_video_generator/
├── README.md
├── ARQUITECTURA.md              # el grafo en detalle
├── pyproject.toml / uv.lock
├── config.yaml                  # todo lo que no es una clave
├── .env.example                 # OPENROUTER_API_KEY, PEXELS_API_KEY, PIXABAY_API_KEY, TAVILY_API_KEY...
├── assets/
│   ├── voice_reference.wav      # la voz del canal (no está en git)
│   ├── voice_reference.txt      # lo que se dice en ella, exacto
│   └── fonts/
├── data/                        # STATE_DIR: checkpoints, runs/, temas/, listo_para_subir/, subidos/, youtube/
├── scripts/
│   ├── run_pipeline.py          # lanza / reanuda / rehace un run
│   ├── find_topics.py           # solo la lista de temas, sin empezar un vídeo
│   ├── design_voice.py          # diseña la voz del canal (una vez)
│   ├── prepare_upload.py        # run terminado -> carpeta lista para subir
│   ├── upload_youtube.py        # carpeta -> YouTube
│   ├── retention.py             # retención de los vídeos del canal (YouTube Analytics + CSV de Studio)
│   └── compare_facts.py, compare_judge.py   # pruebas A/B de modelos (hechos, juez de imágenes)
└── src/
    ├── graph/
    │   ├── build.py             # el grafo
    │   ├── state.py             # VideoState y esquemas
    │   ├── lengths.py           # palabras, escenas y secciones según el formato
    │   ├── script_files.py      # script.json / script.md
    │   ├── manual.py            # guion.md y marcas [archivo]
    │   ├── intros.py            # las 5 formas de presentación de los vídeos largos
    │   ├── montage.py           # <run>/montaje/
    │   ├── onscreen.py          # dibuja los rótulos y retratos
    │   ├── subtitles.py         # subtítulos ASS y líneas del SRT
    │   └── nodes/               # un módulo por nodo (topics, research, angles, outline, deep_research,
    │                            #   write_script, review_script, manual_script, unload_llm, voice,
    │                            #   align, brief, beats, labels, visuals, review_visuals, assemble)
    ├── prompts/                 # un .md por nodo con LLM; {{canal}} y {{temas}} salen de config.yaml
    ├── services/
    │   ├── llm.py, llm_hub.py   # modelo por nodo (OpenRouter / LM Studio), respaldo, descarga de VRAM
    │   ├── costs.py             # <run>/costes.md
    │   ├── topics.py            # feeds + registro de temas
    │   ├── articles.py          # descarga y limpia artículos (trafilatura)
    │   ├── wikipedia.py
    │   ├── portraits.py         # fotos de personas (Wikidata + Commons)
    │   ├── stock.py             # Pexels, Pixabay, NASA, Wikimedia
    │   ├── contact_sheet.py     # cuadrícula de candidatos para el juez
    │   ├── comfy.py             # ComfyUI (Z-Image)
    │   ├── image_api.py         # generación de pago en OpenRouter
    │   ├── tts.py               # Qwen3-TTS / edge-tts
    │   ├── workers.py           # lanza los workers de GPU con .venv-tts
    │   ├── media.py, ass.py     # ffmpeg y subtítulos ASS
    │   ├── publish.py           # carpeta lista para subir
    │   └── youtube.py           # OAuth + subida
    ├── tts_worker/              # corren con .venv-tts: qwen, align (Whisper), embed (SigLIP 2), parallax
    ├── llmkit/                  # base reutilizable: modelo, agente, RAG, persistencia, aprobaciones...
    └── modules/pexels_scraper.py   # utilidad suelta (Playwright); el pipeline no la usa
```

---

## 8. Configuración

Todo está en `config.yaml`, comentado. Las claves van en el `.env`. Resumen de las secciones:

| Sección | Qué decide |
|---|---|
| `format` | `long` o `short` por defecto (`--format` lo cambia) |
| `channel` | Quién es el canal y de qué temas habla (rellena `{{canal}}` y `{{temas}}` en los prompts). Otro canal sería otro bloque, no una copia del código |
| `llm` | Proveedores (`local` = LM Studio, `openrouter`) y **modelo por nodo**: cada nodo hereda de `default` y puede tener `fallback` |
| `voice` | `mode` (`clone` / `edge`), referencia, modelo Qwen3-TTS, semilla |
| `workers` | Python de `.venv-tts` |
| `align` | Modelo de Whisper |
| `visuals` | Fuentes por tipo de plano, duración de los planos, umbral del juez, generador (`comfy` / `openrouter`), estilo |
| `comfy` | URL y modelos de Z-Image en ComfyUI |
| `video` | Resolución, duración, escenas y secciones por formato; subtítulos, transiciones, rótulos y retratos (`labels`), música, 2.5D, codificador |
| `research` | Búsquedas, artículos, `skip_domains`, investigación por secciones (`deep`) |
| `topics` | Feeds, días, candidatos, `reserve_days` |
| `script` | `whole_max_words`: hasta cuántas palabras se escribe el guion entero en una llamada |
| `review` | `max_revisions` del revisor del guion |
| `youtube` | Carpetas, credenciales, privacidad, categoría (28 = Ciencia y tecnología) |

En el `.env`:
- `OPENROUTER_API_KEY`: obligatoria y con saldo para Sonnet.
- `PEXELS_API_KEY` y `PIXABAY_API_KEY`.
- `SEARCH_PROVIDER` (`tavily` o `duckduckgo`) y `TAVILY_API_KEY`.
- `STATE_DIR=data`.

---

## 9. Instalación (Windows + RTX 5070)

1. **Proyecto principal** (Python 3.13):
   ```bash
   uv sync --extra youtube            # o: python -m venv .venv && .venv\Scripts\activate && pip install -e .[youtube]
   ```
   ffmpeg viene con `imageio-ffmpeg`, así que no hay que instalarlo en el sistema.
2. **`.venv-tts`** (Python 3.12) para los workers de GPU. Instala torch `cu128` **antes** que nada, para que PyPI no traiga el torch de CPU:
   ```bash
   uv venv .venv-tts --python 3.12
   .venv-tts\Scripts\python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
   .venv-tts\Scripts\python -m pip install qwen-tts transformers
   ```
   Los modelos (Qwen3-TTS, Whisper, SigLIP 2, Depth Anything) se descargan de Hugging Face la primera vez.
3. **Voz del canal:**
   - Diseñada: `python scripts/design_voice.py`, escucha los candidatos y luego `python scripts/design_voice.py --pick N`.
   - Tuya: graba ~10–15 s en `assets/voice_reference.wav` y escribe lo que dices, exacto, en `voice_reference.txt`.
4. **ComfyUI** en su propia carpeta y venv, con los modelos de Z-Image Turbo de `huggingface.co/Comfy-Org/z_image_turbo` (`split_files/`) en `models/diffusion_models`, `models/text_encoders` y `models/vae`. Tiene que estar **arrancado** antes de un run si `visuals.generator: comfy`:
   ```bash
   .venv\Scripts\python.exe main.py --listen 127.0.0.1 --port 8188
   ```
5. **`.env`:** copia `.env.example` y rellena las claves.
6. **YouTube**, una vez (detalle en `src/services/youtube.py`):
   1. Proyecto en Google Cloud con YouTube Data API v3.
   2. Pantalla de consentimiento *External*, con tu cuenta como usuario de prueba.
   3. Cliente OAuth "Desktop" con el JSON en `data/youtube/client_secret.json`.
   4. La primera subida abre el navegador para elegir el canal.
7. **LM Studio** (opcional, solo para modelos locales): servidor en `localhost:1234/v1` y CLI `lms`. Comprueba los ids con `lms ls`: con un id incorrecto usa sin avisar el modelo que tenga cargado.

> **Smart App Control (Windows 11)** puede bloquear DLL sin firmar ("Una directiva de Control de aplicaciones bloqueó este archivo"), en torch, scikit-learn o uuid_utils. Un "DLL load failed" en este PC no suele ser un fallo del código. Se desactiva en Seguridad de Windows → Control de aplicaciones y navegador, y puede volver a activarse con una actualización.

---

## 10. Uso

```bash
# Empezar un vídeo
python scripts/run_pipeline.py                                  # elige el tema entre los de los feeds
python scripts/run_pipeline.py --format short                   # ... y que sea un short
python scripts/run_pipeline.py --topic "Cómo funciona un reloj nuclear" --type explainer
python scripts/run_pipeline.py --topic "..." --notes "el datazo: ..." --url https://... --url https://...
python scripts/run_pipeline.py --topic "..." --manual           # del esquema directo a tu guion

# Reanudar un run parado (Ctrl+C en una pausa, un error, sin saldo...)
python scripts/run_pipeline.py --resume 20261013-025807

# Rehacer un nodo de medios y los siguientes (también en un run terminado)
python scripts/run_pipeline.py --resume <run_id> --redo voice       # narración cambiada en script.json
python scripts/run_pipeline.py --resume <run_id> --redo brief       # brief.json editado
python scripts/run_pipeline.py --resume <run_id> --redo beats       # beats.json editado (o borrado: nuevos)
python scripts/run_pipeline.py --resume <run_id> --redo labels      # rotulos.json editado (o borrado: nuevos)
python scripts/run_pipeline.py --resume <run_id> --redo visuals     # elegir otra vez (respeta tus archivos de montaje/)
python scripts/run_pipeline.py --resume <run_id> --redo assemble    # volver a montar con lo que hay en montaje/

# Temas sin empezar un vídeo
python scripts/find_topics.py                  # --days 3 --top 5 · --no-llm (gratis) · --discard <id>

# Publicar
python scripts/publish_run.py --run <run_id>                 # prepara y sube en un paso (pregunta antes de subir)
python scripts/prepare_upload.py --run <run_id>              # --thumb 3 · --force
python scripts/upload_youtube.py data/listo_para_subir/<carpeta>   # --privacy unlisted
python scripts/retention.py                                  # -> data/retencion/<fecha>.md
```

`--yes` aprueba todas las pausas sin preguntar. Úsalo solo para pruebas.

**Las pausas, en orden:**
1. Tema (sin `--topic`).
2. Enfoque.
3. Esquema.
4. Investigación por secciones (si alguna sección tiene preguntas).
5. Guion.
6. Montaje.

En cada una, el archivo que se puede editar está en la carpeta del run y se vuelve a leer al aprobar.

---

## 11. Calidad y políticas de YouTube

**Política de "contenido no auténtico"** (YPP, desde julio de 2025): YouTube no monetiza contenido producido en masa o con plantilla.

- [x] Revisión humana en cada paso (tema, enfoque, esquema, guion, montaje). No hay modo automático para publicar.
- [x] Cada dato del guion sale de un hecho con fuente; un revisor marca lo que no, y los hechos dudosos los decides tú.
- [x] Fuentes en la descripción y créditos de Wikimedia. Nunca fotos de prensa.
- [x] Contenido sintético **marcado** al subir si hay imágenes generadas o archivos que no son de stock.
- [ ] Voz: hoy es una voz **diseñada** con Qwen3-TTS, siempre la misma en todos los vídeos. Para más seguridad: tu voz grabada como referencia, o el modo `record` (pendiente).
- [ ] Material propio: gráficos de datos y capturas de las fuentes (pendiente; hoy los planos `data` y `source` buscan en stock).
- [ ] Ritmo humano: pocos vídeos por semana, todos revisados.

**Nota sobre la API:** hasta que Google audita el proyecto de API, los vídeos se suben **privados** y se publican a mano en YouTube Studio. Cuota: 10.000 unidades/día (una subida ~1.600, miniatura ~50, subtítulos ~400). La miniatura necesita un canal verificado por teléfono.

**Checklist por vídeo:**
- La sorpresa en los primeros 2–3 s, sin destripar el final.
- Ningún plano que no encaje con lo que se dice (revísalo en `montaje/lista.md`).
- Volumen estable (−14 LUFS) y música por debajo de la voz.
- Subtítulos sin errores (salen del guion, no de la transcripción).
- Hechos dudosos de la revisión comprobados.

---

## 12. Pendiente

- **Fase 2 (imágenes):**
  - `brief` y `beats` pensados para generar (enfoque según el tipo de plano, estilo fijo de Datazo, sin sujetos repetidos);
  - el `brief` no debe prohibir las imágenes habituales del tema;
  - cada plano con una búsqueda genérica de respaldo;
  - juez sobre las imágenes generadas y una regeneración;
  - imágenes con licencia de las fuentes (NASA, ESA, papers CC BY) con su crédito en pantalla.
- **Fase 3:** el *datazo* en pantalla, una tarjeta animada hecha por código con la cifra que sorprende.
- **Gráficos y capturas** para los planos `data` y `source`.
- **Paquete antes que el guion:** título y texto de la miniatura en la pausa del enfoque, texto en pantalla los primeros 2–3 s, 3 opciones de título y miniatura vertical para shorts.
- **Señales de demanda** en el buscador de temas (autocompletado y vistas de YouTube).
- **Generación:** si una imagen se rechaza en OpenRouter, reintentar o saltar solo esa (hoy pasa todas las que quedan a ComfyUI).
- **Modelo local de respaldo** (Gemma 4 en LM Studio) en vez de Nemotron, que a veces no cierra el JSON.
- **Modo `record`** de la voz.
- **Retención:** tras 5–10 vídeos, leer las curvas de YouTube Studio y relacionar las caídas con las escenas (se conocen sus tiempos).
