# 🎬 AI News Video Generator

Pipeline multiagente en Python (LangGraph) que selecciona noticias de **tecnología e IA**, genera un vídeo narrado con imágenes y clips sincronizados con la voz, y lo sube a YouTube tras una revisión humana.

Todo corre **en local** sobre una **RTX 5070 (12 GB)**, incluido el LLM (LM Studio). Solo salen a internet las APIs de stock, la subida a YouTube y, opcionalmente, OpenRouter para algún nodo concreto.

> **Filosofía:** la IA hace el trabajo pesado (buscar, documentarse, redactar, montar); el criterio y la voz son humanos. Esto es lo que mantiene el canal dentro de la política de "contenido no auténtico" de YouTube y lo que da calidad.

---

## Índice

1. [Formato del vídeo](#1-formato-del-vídeo)
2. [Stack](#2-stack)
3. [Arquitectura](#3-arquitectura)
4. [Estado del grafo](#4-estado-del-grafo)
5. [Nodos](#5-nodos)
6. [Estructura del repositorio](#6-estructura-del-repositorio)
7. [Configuración](#7-configuración)
8. [Instalación (Windows + RTX 5070)](#8-instalación-windows--rtx-5070)
9. [Uso](#9-uso)
10. [Calidad y políticas de YouTube](#10-calidad-y-políticas-de-youtube)
11. [Roadmap](#11-roadmap)

---

## 1. Formato del vídeo

- **Temática:** noticias de tecnología e IA *explicadas*, con contexto y opinión propia (no leer titulares).
- **Estilo:** voz en off + imágenes/clips que cambian según lo que se narra + subtítulos palabra a palabra + música de fondo suave.
- **Formatos:**
  - **Short** — 9:16, 1080×1920, 45–60 s, 6–10 escenas.
  - **Largo** — 16:9, 1920×1080, 8–12 min, 40–80 escenas (mejor RPM).
- **Ritmo:** 3–5 vídeos/semana, todos revisados.

---

## 2. Stack

| Capa | Herramienta | Notas |
|---|---|---|
| Orquestación | **LangGraph** + `SqliteSaver` | Persistencia, reanudación e *interrupt* para revisión humana |
| LLM (local) | **LM Studio** con **Qwen3.5 9B (GGUF Q6_K, ~7 GB)** | Servidor compatible con OpenAI en `localhost:1234/v1`. Muy bueno en español y con salida estructurada |
| LLM (alternativa local) | **Gemma 4 12B (QAT Q4_0, ~6,5 GB)** | Buena redacción; si la versión descargada tiene visión, sirve para el nodo `visuals` |
| LLM (opcional, nube) | **OpenRouter** | Solo para nodos donde el modelo local se quede corto (p. ej. `write_script`). El modelo se configura por nodo |
| Base de datos | **SQLite** | Noticias, estado, histórico de vídeos |
| Ingesta | `feedparser`, `trafilatura`, `httpx` | RSS de medios tech + extracción del artículo completo |
| Búsqueda extra | Tavily o DuckDuckGo | Contexto adicional para el investigador |
| Voz | **Tu voz grabada** (preferente) o **Chatterbox Multilingual** clonando tu voz | Chatterbox: licencia MIT, español, clonación con ~10 s de referencia |
| Imágenes generadas | **Flux.1-dev GGUF (Q5/Q8)** en **ComfyUI** | Fallback cuando no hay stock adecuado |
| Clips generados (opcional) | **Wan 2.2** en ComfyUI | Solo planos clave; lento en 12 GB |
| Stock | **Pexels API** + **Pixabay API** | Gratis, metraje real con licencia de uso |
| Material propio | `playwright` (capturas), `matplotlib`/`plotly` (gráficos) | Capturas de webs/repos/papers y gráficos con datos de la noticia |
| Subtítulos | **faster-whisper** (GPU) | Timestamps por palabra → `.ass` estilo karaoke |
| Montaje | **moviepy 2.x** + **ffmpeg** (`h264_nvenc`) | Codificación por GPU |
| Música | YouTube Audio Library / Pixabay Music | *Ducking* con `sidechaincompress` de ffmpeg |
| Miniaturas | Flux + `Pillow` (texto) | Solo vídeos largos |
| Subida | **YouTube Data API v3** (`google-api-python-client`, OAuth) | Marca de contenido sintético cuando aplique |
| Observabilidad | **LangFuse** | Trazas, costes y prompts por nodo |

**Requisito de GPU:** las RTX 50 (Blackwell) necesitan **PyTorch con CUDA 12.8+** (`cu128`). ComfyUI portable ya lo incluye; en el resto de entornos instalar torch desde el índice `cu128`.

### Gestión de la VRAM (12 GB)

El LLM (~7 GB) no cabe a la vez que Flux, Chatterbox o Wan. Como el grafo es secuencial, se reparte por fases:

| Fase | Nodos | Qué ocupa la GPU |
|---|---|---|
| 1. Texto | `research`, `write_script`, `review`, `metadata` | LLM en LM Studio |
| 2. Medios | `voice`, `visuals`, `subtitles`, `assemble` | Chatterbox, Flux (ComfyUI), faster-whisper, NVENC |

- Al terminar la fase de texto se descarga el LLM con `lms unload --all` (el nodo `human_script_edit` es un buen punto, porque el grafo se para ahí de todas formas).
- `metadata` se ejecuta en la fase de texto, justo después de aprobar el guion, para no volver a cargar el LLM al final.
- Si se usa el LLM con visión en `visuals`, se carga solo para esa escena y se descarga antes de lanzar Flux. Alternativa: activar en LM Studio la descarga automática por inactividad (TTL).
- ComfyUI libera memoria entre trabajos con `POST /free`.

---

## 3. Arquitectura

```
                ┌──────────────────────────────────────┐
  cron / tarea  │  ingest.py  (fuera del grafo)        │
  programada ──►│  RSS → trafilatura → SQLite (news)   │
                └──────────────────┬───────────────────┘
                                   │
┌──────────────────────────────────▼──────────────────────────────────────┐
│                           GRAFO LANGGRAPH                               │
│                                                                         │
│  select_news ──► research 🤖 ──► write_script 🤖 ──► review 🤖           │
│                                        ▲                │               │
│                                        └── feedback ────┤ (máx. 2)      │
│                                                         ▼ OK            │
│                                              ⏸ human_script_edit        │
│                                                         │               │
│                                    metadata 🤖 ──► unload_llm           │
│  ── fin fase de texto (LLM) ─────────────────────────── │ ───────────── │
│                    ┌────────────────────────────────────┤               │
│                    ▼                                    ▼               │
│               voice (por escena)               visuals 🤖 (por escena)  │
│                    │                                    │               │
│                    └──────────────┬─────────────────────┘               │
│                                   ▼                                     │
│                      subtitles ──► assemble ──► thumbnail               │
│                                                              │          │
│                                                  ⏸ human_final_review   │
│                                                              │          │
│                                                           upload        │
└─────────────────────────────────────────────────────────────────────────┘

🤖 = nodo con LLM      ⏸ = interrupt (revisión humana)
```

`voice` y `visuals` son independientes entre sí. Se pueden ejecutar **en paralelo** (fan-out/fan-in de LangGraph) solo si la VRAM lo permite; con 12 GB, lo seguro es ejecutarlos en secuencia (ver *Gestión de la VRAM*).

El LLM local (LM Studio) se usa en la fase de texto y se descarga antes de la fase de medios. Por eso `metadata` se genera justo después de aprobar el guion.

### Principio de sincronización

El guion se divide en **escenas**. Cada escena tiene su narración y su visual. El audio de cada escena se genera (o graba) por separado y **la duración del visual = duración de su audio**. Así la imagen cambia exactamente cuando cambia la idea. Dentro de una escena, los timestamps por palabra de faster-whisper permiten cortes más finos y subtítulos animados.

---

## 4. Estado del grafo

```python
from typing import TypedDict, Literal
from pydantic import BaseModel, Field

class Scene(BaseModel):
    id: int
    narration: str = Field(description="1-2 frases que se narran")
    visual_type: Literal["stock", "generated", "screenshot", "chart"]
    visual_query: str = Field(description="Búsqueda de stock en inglés")
    visual_prompt: str = Field(description="Prompt para Flux si no hay stock")
    on_screen_text: str | None = Field(default=None, description="Texto/cifra destacada")
    # se rellenan después
    audio_path: str | None = None
    audio_duration: float | None = None
    visual_path: str | None = None

class Script(BaseModel):
    title: str
    hook: str                 # primera escena, 2-3 s, debe enganchar
    scenes: list[Scene]
    sources: list[str]

class VideoState(TypedDict, total=False):
    run_id: str
    format: Literal["short", "long"]
    news: dict                # noticia seleccionada
    facts: list[dict]         # {fact, source_url}
    script: Script
    review_ok: bool
    review_feedback: str
    revisions: int
    subtitles_path: str
    video_path: str
    metadata: dict            # title, description, tags, synthetic_media
    thumbnail_path: str
    approved: bool
    youtube_id: str
```

---

## 5. Nodos

### Ingesta (fuera del grafo) — `ingest/`
- Lee feeds RSS configurados (medios de tecnología/IA, blogs oficiales, arXiv).
- Extrae el texto completo con `trafilatura`.
- Deduplica por URL y por similitud de título.
- Guarda en `news` con `status = 'new'`, categoría y fecha.
- Se ejecuta cada pocas horas (cron en el servidor o Programador de tareas de Windows).

### `select_news`
- Elige al azar una noticia `new` de los últimos N días, con pesos por categoría para que haya variedad.
- Excluye temas ya cubiertos (similitud con títulos de vídeos anteriores).
- Marca la noticia como `in_progress`.

### `research` 🤖
- Parte del artículo completo y añade 2–3 búsquedas para contexto (qué es, por qué importa, precedentes).
- Devuelve una lista de **hechos con su fuente**. Nada sin fuente.

### `write_script` 🤖
- Salida estructurada → `Script`.
- Reglas del prompt: gancho en los primeros 2–3 s, una idea por escena, explicar *por qué importa*, incluir una opinión o conclusión, lenguaje hablado (no escrito), duración objetivo según `format`.
- Si viene `review_feedback`, reescribe aplicándolo.

### `review` 🤖
- Comprueba cada afirmación del guion contra `facts` (detecta inventos).
- Evalúa el gancho, el ritmo y la claridad.
- Devuelve `review_ok` + `review_feedback`. Vuelve a `write_script` como máximo 2 veces.

### `human_script_edit` ⏸
- *Interrupt*: el guion se exporta a un `.md`/`.json` editable.
- Aquí se añade la opinión propia, se corrigen frases o se reordenan escenas.
- Al reanudar se vuelve a validar el esquema `Script`.

### `voice`
- **Modo `record`** (preferente): genera un guion de lectura por escenas; se graba cada escena (o una toma completa que se corta por silencios / alineación con whisper).
- **Modo `clone`**: Chatterbox Multilingual con un audio de referencia de tu voz → un `.wav` por escena.
- Normaliza volumen (`ffmpeg loudnorm`, −14 LUFS) y guarda `audio_duration` por escena.

### `visuals` 🤖
Para cada escena, según `visual_type`:
1. **stock** → busca en Pexels/Pixabay (vídeo primero, en la orientación del formato). Un LLM con visión puntúa las miniaturas de los 5–8 primeros resultados contra la narración y elige la mejor. Si ninguna supera el umbral → fallback a `generated`.
2. **generated** → Flux vía API de ComfyUI con `visual_prompt` (estilo visual común del canal en el prompt).
3. **screenshot** → `playwright` captura la web/repositorio/paper de la fuente.
4. **chart** → gráfico generado con los datos de la noticia (estilo del canal).

Evita repetir el mismo clip dentro del vídeo y entre vídeos recientes.

### `subtitles`
- `faster-whisper` sobre el audio completo → timestamps por palabra.
- Genera un `.ass` con estilo del canal (2–4 palabras por línea, palabra activa resaltada).

### `assemble`
- Por escena: visual ajustado al encuadre (crop/scale), duración = audio + 0,15 s; imágenes con zoom lento (Ken Burns).
- Transiciones cortas entre escenas, `on_screen_text` superpuesto cuando exista.
- Música de fondo con *ducking* bajo la voz.
- Subtítulos quemados con ffmpeg.
- Exporta con `h264_nvenc`, 30 fps, AAC, −14 LUFS.

### `metadata` 🤖
- Título (≤ 70 caracteres), descripción con resumen + **fuentes enlazadas**, tags, capítulos (en vídeos largos).
- Decide `synthetic_media = True` si hay imágenes/vídeos generados realistas o voz clonada.

### `thumbnail` (solo vídeos largos)
- Imagen Flux + texto grande con `Pillow` siguiendo la plantilla del canal.

### `human_final_review` ⏸
- *Interrupt*: se revisa el vídeo final, el título y la miniatura.
- Opciones: aprobar, rehacer un nodo concreto (p. ej. `visuals` de la escena 4) o descartar.

### `upload`
- YouTube Data API v3 (`videos.insert`, subida reanudable) + `thumbnails.set`.
- Sube como **privado o programado**; marca contenido sintético si aplica.
- Actualiza la noticia a `used` y registra el vídeo en `videos`.

---

## 6. Estructura del repositorio

```
ai-news-video/
├── README.md
├── pyproject.toml
├── config.yaml
├── .env.example                 # OPENROUTER_API_KEY, PEXELS_API_KEY, ...
├── data/
│   ├── app.db                   # SQLite: news, videos
│   ├── checkpoints.db           # SqliteSaver de LangGraph
│   └── runs/<run_id>/           # audio/, visuals/, subs/, final.mp4
├── assets/
│   ├── voice_reference.wav      # tu voz para clonación
│   ├── music/
│   ├── fonts/
│   └── comfy_workflows/         # flux_image.json, wan_clip.json
├── src/
│   ├── ingest/
│   │   ├── feeds.py
│   │   └── run_ingest.py
│   ├── graph/
│   │   ├── state.py             # VideoState, Scene, Script
│   │   ├── build.py             # construcción del grafo
│   │   └── nodes/
│   │       ├── select_news.py
│   │       ├── research.py
│   │       ├── write_script.py
│   │       ├── review.py
│   │       ├── voice.py
│   │       ├── visuals.py
│   │       ├── subtitles.py
│   │       ├── assemble.py
│   │       ├── metadata.py
│   │       ├── thumbnail.py
│   │       └── upload.py
│   ├── prompts/                 # un .md por nodo con LLM
│   ├── services/
│   │   ├── llm.py               # cliente LM Studio / OpenRouter (modelo por nodo) + unload
│   │   ├── comfy.py             # cliente HTTP de ComfyUI (/prompt, /history)
│   │   ├── tts.py               # Chatterbox
│   │   ├── stock.py             # Pexels / Pixabay
│   │   ├── capture.py           # playwright
│   │   ├── media.py             # helpers ffmpeg / moviepy
│   │   └── youtube.py           # OAuth + subida
│   └── db.py
├── scripts/
│   ├── run_pipeline.py          # lanza / reanuda un run
│   └── auth_youtube.py          # primer login OAuth
└── tests/
```

**Entornos separados:** el proyecto principal, Chatterbox y ComfyUI van en entornos distintos para no pisar dependencias de torch. El proyecto habla con ComfyUI por HTTP y con Chatterbox por un pequeño servidor local o `subprocess`.

---

## 7. Configuración

`config.yaml` (ejemplo):

```yaml
format: short                 # short | long

llm:
  providers:
    local:
      base_url: http://localhost:1234/v1     # LM Studio
      api_key: lm-studio                     # no se usa, pero es obligatorio
    openrouter:
      base_url: https://openrouter.ai/api/v1
      api_key_env: OPENROUTER_API_KEY
  default:      {provider: local, model: qwen3.5-9b, temperature: 0.3}
  write_script: {provider: local, model: qwen3.5-9b, temperature: 0.8}
  review:       {provider: local, model: qwen3.5-9b, temperature: 0.0}
  visuals:      {provider: local, model: gemma-4-12b}   # requiere versión con visión
  # Si el guion no convence en local, cambiar solo este nodo:
  # write_script: {provider: openrouter, model: <modelo-fuerte>}
  unload_after_text_phase: true             # lms unload --all

voice:
  mode: clone                 # record | clone
  reference: assets/voice_reference.wav
  language: es

visuals:
  stock_providers: [pexels, pixabay]
  min_match_score: 0.7
  style_prompt: "cinematic, soft light, tech aesthetic, no text"

video:
  short:  {width: 1080, height: 1920, target_seconds: 55}
  long:   {width: 1920, height: 1080, target_seconds: 600}
  fps: 30
  loudness_lufs: -14

review:
  max_revisions: 2

youtube:
  privacy: private            # private | unlisted | scheduled
  category_id: 28             # Science & Technology
  default_language: es
```

---

## 8. Instalación (Windows + RTX 5070)

1. **Proyecto principal**
   ```bash
   python -m venv .venv && .venv\Scripts\activate
   pip install -e .
   playwright install chromium
   ```
2. **PyTorch para Blackwell** (en cada entorno que use GPU)
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
   ```
3. **LM Studio**
   - Descargar **Qwen3.5 9B** (GGUF Q6_K) desde *Discover*; opcionalmente **Gemma 4 12B** (QAT Q4_0).
   - Pestaña *Developer* → **Start Server** (`http://localhost:1234/v1`).
   - Contexto de 8k–16k tokens es suficiente; más contexto consume más VRAM.
   - Instalar el CLI `lms` (desde LM Studio) para poder cargar/descargar modelos desde el pipeline.
   - Comprobar: `curl http://localhost:1234/v1/models`
4. **ComfyUI portable** (build NVIDIA) + ComfyUI-Manager → instalar `ComfyUI-GGUF` (y `ComfyUI-WanVideoWrapper` si se usa Wan). Descargar Flux.1-dev GGUF, encoders y VAE. Arrancar con `--listen 127.0.0.1`.
5. **Chatterbox** en un venv aparte (instalar torch `cu128` antes que el resto).
6. **faster-whisper** con soporte CUDA.
7. **ffmpeg** en el `PATH`, con NVENC (`ffmpeg -encoders | findstr nvenc`).
8. **YouTube:** crear proyecto en Google Cloud Console → habilitar YouTube Data API v3 → credenciales OAuth (app de escritorio) → `python scripts/auth_youtube.py`.
9. Copiar `.env.example` a `.env` y rellenar las claves.

---

## 9. Uso

```bash
# 1. Ingesta de noticias (o programarla)
python -m src.ingest.run_ingest

# 2. Lanzar un vídeo
python scripts/run_pipeline.py --format short

# 3. Se detiene en human_script_edit → editar data/runs/<run_id>/script.md
python scripts/run_pipeline.py --resume <run_id>

# 4. Se detiene en human_final_review → revisar final.mp4
python scripts/run_pipeline.py --resume <run_id> --approve
#   o rehacer un nodo:  --redo visuals --scene 4
```

---

## 10. Calidad y políticas de YouTube

**Política de "contenido no auténtico"** (YPP, desde julio de 2025): YouTube no monetiza contenido producido en masa o con plantilla. Para estar del lado bueno:

- [ ] Voz propia o clonada de tu voz (no voces genéricas).
- [ ] Cada vídeo incluye explicación y opinión propias, añadidas en `human_script_edit`.
- [ ] Mezcla de material propio (capturas, gráficos) con stock y generado.
- [ ] Estructura variada entre vídeos; nada de plantilla idéntica.
- [ ] Ritmo humano: 3–5 vídeos/semana, todos revisados.
- [ ] Contenido sintético realista **marcado** en la subida.
- [ ] Fuentes citadas en la descripción; sin imágenes de prensa con copyright.
- [ ] Evitar temas que reciben anuncios limitados (tragedias, violencia, política).

**Nota sobre la API:** los proyectos de API no verificados por Google suben los vídeos como **privados**. Para publicar directamente hay que pasar la auditoría de la API; mientras tanto, se cambia a público a mano tras la revisión. Cuota por defecto: ~10.000 unidades/día (~6 subidas).

**Checklist de calidad por vídeo:**
- Gancho en los primeros 2–3 s.
- Ninguna escena con visual que no encaje con lo que se dice.
- Sin cortes de audio, volumen estable (−14 LUFS).
- Subtítulos legibles y sin errores.
- Datos verificados contra las fuentes.

---

## 11. Roadmap

1. **Núcleo:** `write_script` → `voice` → `visuals` (solo stock) → `assemble`, con una noticia fija.
2. Subtítulos palabra a palabra y música con *ducking*.
3. Fallback a Flux vía ComfyUI y selector de stock con visión.
4. `research` + `review` con bucle de revisión.
5. Interrupts de revisión humana y reanudación con `SqliteSaver`.
6. Ingesta RSS + `select_news`.
7. `metadata` + `upload` a YouTube.
8. Formato largo, miniaturas y capítulos.
9. Capturas y gráficos propios como tipos de visual.
10. LangFuse y métricas (retención, CTR) para mejorar los prompts.
