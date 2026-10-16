"""The graph's state and the schemas the LLM fills.

Two kinds of model:
- *Draft: what the LLM writes (only editorial fields; ids, sections, sources,
  word targets and media are NOT asked from the model, so it cannot make them up).
- Outline / Script / Scene: as stored and edited by a human (code adds the rest).

The state stores PLAIN DATA (dicts, lists, str): LangGraph's checkpoint will stop
loading Pydantic objects in a future version, and a dict is what a human edits.
Read them with Outline.model_validate(state["outline"]), Script.model_validate(...).

Media goes in separate keys, NOT inside the script: audio (by scene id), beats and
visuals (by beat id). Two branches writing the same key in the same step is an error
in LangGraph.

The script is only words: what is SHOWN is decided later, by the beats node, once the
audio and its word times exist (a scene can have several visuals).
"""
from typing import Literal, TypedDict

from pydantic import BaseModel, Field, field_validator

Format = Literal["short", "long"]
# news: something that has just happened; explainer: something that does not expire (how it works,
# why it is so). It decides where the research looks (graph/nodes/research.py)
VideoType = Literal["news", "explainer"]
# What a beat needs to be shown (graph/nodes/beats.py, prompts/beats.md)
BeatKind = Literal["entity", "generic", "data", "source", "abstract"]
# What a section does in the story (prompts/outline.md); a long video repeats tension-development-reveal
SectionRole = Literal["hook", "context", "tension", "development", "reveal", "conclusion"]


# ---------------------------------------------------------------- research
class SearchPlan(BaseModel):
    video_type: VideoType = Field(description="news = cuenta algo que acaba de pasar (un lanzamiento, un descubrimiento, "
                                              "un anuncio con fecha); explainer = cuenta algo que no caduca (cómo "
                                              "funciona algo, por qué pasa, ventajas e inconvenientes de una técnica, "
                                              "la historia de un invento)")
    queries: list[str] = Field(min_length=1, max_length=10,   # a cap: small local models loop and never close the list
                               description="Búsquedas en español, distintas entre sí: para un buscador de noticias "
                                           "si es news; para un buscador general si es explainer")
    wikipedia: list[str] = Field(default_factory=list, max_length=3,
                                 description="explainer: 1-2 artículos de Wikipedia que lo explican, como 'es:Título' "
                                             "o 'en:Title' (el nombre del artículo, no una frase); news: []")


class FactDraft(BaseModel):
    fact: str = Field(description="Un dato concreto y verificable, en una frase que se entienda sola "
                                  "(quién, qué, cuándo, cifra). Opiniones solo atribuidas: 'Según X, ...'")


MAX_FACTS = 30                # per text: more are cut, not rejected (a lost call loses the whole text)


class FactsDraft(BaseModel):
    facts: list[FactDraft] = Field(description="Los datos del artículo relevantes para el tema; [] si no hay")

    @field_validator("facts", mode="before")
    @classmethod
    def at_most(cls, value):
        return value[:MAX_FACTS] if isinstance(value, list) else value


class Fact(TypedDict, total=False):
    fact: str
    source_url: str
    section: int              # deep_research: the outline section it was looked for (1-based)


class SectionSearches(BaseModel):
    web: list[str] = Field(min_length=1, max_length=3,
                           description="1-3 búsquedas web en español (3-7 palabras), una por cosa que averiguar")
    wikipedia: list[str] = Field(default_factory=list, max_length=3,
                                 description="0-3 artículos de Wikipedia que lo explican, como 'es:Título' o "
                                             "'en:Title' (el nombre del artículo, no una frase)")


class Source(TypedDict, total=False):
    url: str
    title: str
    outlet: str               # the medium (El País, Reuters...)
    date: str                 # ISO date, if known


# ------------------------------------------------------------------ angles
class Angle(BaseModel):
    title: str = Field(description="El enfoque en una frase: desde dónde se cuenta el tema")
    question: str = Field(description="La pregunta central que el vídeo responde al final y que hace seguir viéndolo")
    why: str = Field(description="Por qué engancharía a alguien que no sabe nada del tema")
    hook: str = Field(description="Primera frase del vídeo, máx. 15 palabras, basada en un hecho")
    facts: list[int] = Field(default_factory=list, max_length=15,
                             description="Números de los hechos que sostienen este enfoque")


class AnglesDraft(BaseModel):
    angles: list[Angle] = Field(min_length=3, max_length=3, description="Tres enfoques DISTINTOS entre sí")
    reason: str = Field(description="Cuál de los tres es el más fuerte y por qué, en 1-2 frases")
    best: int = Field(description="Número del enfoque más fuerte: 1, 2 o 3 (se cuenta desde 1, no desde 0)")

    def recommended(self):
        """The recommended angle (an out-of-range best, from the model or a hand edit, means the first)."""
        return self.angles[self.best - 1] if 1 <= self.best <= len(self.angles) else self.angles[0]


# ----------------------------------------------------------------- outline
class OutlineSectionDraft(BaseModel):
    title: str = Field(description="Título corto de la sección (para el esquema, no se narra)")
    role: SectionRole = Field(description="Función en la historia: hook = engancha y plantea la pregunta; "
                                          "context = lo mínimo para entenderla; tension = el problema o lo que no "
                                          "cuadra; development = cómo funciona, causas, consecuencias, comparaciones; "
                                          "reveal = el dato o giro que cambia la mirada (el 'ah, por eso'); "
                                          "conclusion = responde la pregunta y vuelve al gancho")
    goal: str = Field(description="Qué cuenta esta sección y por qué importa, en 1-2 frases")
    emotion: str = Field(description="Qué debe sentir el espectador aquí, en 1-3 palabras (curiosidad, "
                                     "sorpresa, indignación, alivio...)")
    facts: list[int] = Field(default_factory=list, max_length=60,     # deep_research adds the ones it finds
                             description="Números de los hechos que usa esta sección")
    questions: list[str] = Field(default_factory=list, max_length=3,
                                 description="Cosas concretas que hay que AVERIGUAR para contar esta sección y que "
                                             "los hechos no dicen (cómo funciona, antecedentes, quién es quién, cifras "
                                             "de referencia): 1-3 en vídeos largos, 0-2 en shorts; [] si no hace falta")


class OutlineDraft(BaseModel):
    title: str = Field(description="Título del vídeo: claro, atractivo, sin clickbait engañoso")
    angle: str = Field(description="El hilo conductor: la pregunta o idea central que recorre todo el vídeo")
    sections: list[OutlineSectionDraft] = Field(min_length=2, max_length=12, description="Secciones en orden: la primera "
                                                "engancha y presenta el tema; la última concluye")


class OutlineSection(OutlineSectionDraft):
    words: int                # narration words for this section (set by code from the format)
    facts: list[int] = Field(default_factory=list)   # no cap: deep_research adds up to 15 per text read


class Outline(BaseModel):
    title: str
    angle: str
    sections: list[OutlineSection] = Field(min_length=1)


# ------------------------------------------------------------------ script
class SceneDraft(BaseModel):
    narration: str = Field(description="Lo que se narra en esta escena: 1-2 frases en español hablado")


class SectionDraft(BaseModel):
    scenes: list[SceneDraft] = Field(min_length=1, max_length=20, description="Escenas de esta sección, en orden")


PLAN_HELP = ("El plan ANTES de escribir, en pocas líneas: la pregunta que abre, dónde se complica y dónde se "
             "resuelve, los nuevos ganchos, dónde va el dato potente y cómo cierra. No se narra")


class SectionPlanDraft(BaseModel):
    """One section of a long video in one call: its plan first (written before the scenes), then its scenes."""
    plan: str = Field(description=PLAN_HELP + " (el de esta sección, con su puente a la siguiente)")
    scenes: list[SceneDraft] = Field(min_length=1, max_length=20, description="Escenas de esta sección, en orden")


class ScriptDraft(BaseModel):
    """A whole script in one call (shorts, and long videos up to script.whole_max_words): the plan
    first (it is written before the scenes), then one entry per section of the outline."""
    plan: str = Field(description=PLAN_HELP + " (en un short, con el bucle: cómo enlaza la última frase con la "
                                              "primera; en un vídeo largo, una línea por sección con su puente)")
    sections: list[SectionDraft] = Field(min_length=1, max_length=12,
                                         description="Las secciones del esquema, en su orden, cada una con sus escenas")


class Scene(SceneDraft):
    id: int
    section: int = 1          # 1-based section of the outline it belongs to


class Script(BaseModel):
    title: str
    scenes: list[Scene] = Field(min_length=1)
    sources: list[str] = Field(default_factory=list)

    @classmethod
    def from_sections(cls, title, sections, sources):
        """[[SceneDraft, ...] per section] -> Script with scenes numbered 1..n.
        The sources come from the research, not from the LLM."""
        scenes = [s for i, section in enumerate(sections, start=1) for s in
                  [Scene(id=0, section=i, **d.model_dump()) for d in section]]
        for n, scene in enumerate(scenes, start=1):
            scene.id = n
        return cls(title=title, scenes=scenes, sources=list(dict.fromkeys(sources)))

    def section_scenes(self, section):
        return [s for s in self.scenes if s.section == section]

    @property
    def hook(self):
        """The hook is the first scene (2-3 s that decide whether people keep watching)."""
        return self.scenes[0].narration

    def word_count(self):
        return sum(len(s.narration.split()) for s in self.scenes)


# ------------------------------------------------------------------ review
class ReviewIssue(BaseModel):
    scene: int = Field(description="Número de la escena; 0 si es del vídeo entero")
    problem: str = Field(description="Qué falla, en una frase")
    fix: str = Field(description="Cómo arreglarlo, concreto (la frase nueva o qué cambiar), solo con los hechos")
    serious: bool = Field(description="true si hay que arreglarlo antes de grabar la voz: un dato que no sale de los "
                                      "hechos o dice más que ellos, un vídeo que no responde a su tema, algo clave "
                                      "que no se entiende, un gancho o un cierre que no funcionan")


class DubiousFact(BaseModel):
    fact: int = Field(description="Número del hecho")
    why: str = Field(description="Por qué parece un error, en una frase (lo comprobará el autor)")


class ScriptReview(BaseModel):
    summary: str = Field(description="Lo más importante de la revisión, en 1-2 frases, sin elogios")
    issues: list[ReviewIssue] = Field(default_factory=list, max_length=12,
                                      description="Lo que hay que arreglar, de más a menos importante (como mucho 8); "
                                                  "[] si el guion está bien")
    dubious: list[DubiousFact] = Field(default_factory=list, max_length=10,
                                       description="Hechos que usa el guion y parecen un error del artículo; [] si no hay")


# ------------------------------------------------------------ brief/beats/visuals
class MetaphorDraft(BaseModel):
    phrase: str = Field(description="La expresión figurada, copiada del guion (p. ej. 'la burbuja está a punto de estallar')")
    show: str = Field(description="Qué se ve en su lugar, en español: cosas reales del universo visual, nunca la "
                                  "metáfora literal (p. ej. 'grúas paradas sobre pisos a medio construir')")


class BriefDraft(BaseModel):
    universe: list[str] = Field(min_length=4, max_length=20,
                                description="Objetos, lugares, personas y acciones FILMABLES de ESTE vídeo, incluido el "
                                            "problema y las consecuencias de las que habla el guion, en INGLÉS y como se "
                                            "buscarían en un banco de stock (1-3 sustantivos cada uno): 'for rent sign', "
                                            "'hospital corridor', 'dry reservoir'")
    avoid: list[str] = Field(max_length=10,
                             description="Lo que NUNCA debe verse en este vídeo, en ESPAÑOL: lo que llevaría a otro tema "
                                         "o a una lectura equivocada (p. ej. 'pompas de jabón', 'ciudades de otro país', "
                                         "'gente sonriendo'). Nunca algo de lo que habla el guion")
    metaphors: list[MetaphorDraft] = Field(default_factory=list, max_length=12,
                                           description="Cada expresión figurada del guion y cómo se enseña; [] si no hay")
    style: str = Field(description="Estilo común de las imágenes, en INGLÉS, 10-25 palabras: SOLO tipo de foto, luz, "
                                   "paleta y encuadre, sin ningún objeto, lugar ni fondo (va al final de cada prompt)")

    @field_validator("universe", "avoid", "metaphors", mode="before")
    @classmethod
    def at_most(cls, value, info):
        """More than the field's max_length is cut, not rejected (2026-10-15: 11 avoid lost Sonnet's whole brief)."""
        limit = next(m.max_length for m in cls.model_fields[info.field_name].metadata if hasattr(m, "max_length"))
        return value[:limit] if isinstance(value, list) else value


ShotLook = Literal["photo", "visualization"]


class ShotDraft(BaseModel):
    look: ShotLook = Field(default="photo",
                           description="photo = algo que una cámara puede grabar; visualization = algo invisible por su "
                                       "escala o su naturaleza (un átomo, un núcleo, una molécula, una célula por dentro, "
                                       "un campo magnético, un agujero negro): se enseña con una visualización científica "
                                       "fotorrealista, como en un documental")
    subject: str = Field(description="Qué se ve, en español, concreto (p. ej. 'sede del Banco de España en Madrid', "
                                     "'pareja joven revisando facturas en la cocina')")
    must_show: list[str] = Field(min_length=1, max_length=2,
                                 description="1-2 cosas fotografiables que TIENEN que verse en ESTE plano, en español "
                                             "(no las de los otros planos de la idea). Nunca cifras, textos ni ausencias")
    queries: list[str] = Field(min_length=1, max_length=3,
                               description="1-3 búsquedas en INGLÉS de ESTE plano para bancos de vídeo y fotos, de la "
                                           "más concreta a la más genérica: 1-3 sustantivos de lo que se VE")
    prompt: str = Field(description="Prompt en INGLÉS para generar la imagen si no se encuentra una buena: "
                                    "40-70 palabras, sujeto, acción, lugar, luz, encuadre, terminado con el estilo")
    generate: bool = Field(default=False,
                           description="true = se genera directamente, sin buscar en los bancos: un objeto "
                                       "protagonista sobre fondo de estudio, una escena de época, una "
                                       "visualización o algo técnico tan concreto que ningún banco lo tendrá. "
                                       "false = se busca primero (el mundo real corriente: lugares, gente, "
                                       "industria, naturaleza, el espacio)")


class IdeaDraft(BaseModel):
    start: str = Field(description="Las primeras 3-5 palabras del tramo, copiadas EXACTAMENTE de la narración")
    idea: str = Field(description="Qué cuenta este tramo, en una frase en español: la idea, no las palabras")
    kind: BeatKind = Field(description="entity = algo real y concreto con nombre propio (persona pública, "
                                       "institución, empresa, edificio, lugar, producto, evento); generic = algo "
                                       "real pero genérico que se puede filmar (gente, calles, objetos, acciones); "
                                       "data = una cifra o comparación; source = la noticia o el documento que se "
                                       "cita; abstract = una idea que no se puede filmar")
    must_not_show: list[str] = Field(default_factory=list, max_length=4,
                                     description="0-4 cosas que en esta idea llevarían a una lectura equivocada, "
                                                 "en español (además de lo prohibido en todo el vídeo)")
    shots: list[ShotDraft] = Field(min_length=1, max_length=6,
                                   description="Planos DISTINTOS de la misma idea (otra cosa, otro encuadre, nunca el "
                                               "mismo con otras palabras), en el orden en que la narración dice las "
                                               "cosas: tantos como necesita su tramo (los planos de la escena). Cada "
                                               "uno con lo suyo que tiene que verse y sus búsquedas")


class SceneIdeasDraft(BaseModel):
    scene: int = Field(description="Número de la escena")
    ideas: list[IdeaDraft] = Field(min_length=1, max_length=3, description="Las ideas de la escena (1-2), en orden")


class BeatsDraft(BaseModel):
    scenes: list[SceneIdeasDraft] = Field(description="Una entrada por cada escena pedida, en orden")


class Beat(TypedDict, total=False):
    id: int                   # 1..n over the whole video (visuals are keyed by it)
    scene: int
    word: int                 # index of its first word in the scene's narration
    start: float              # seconds from the start of the scene's audio
    seconds: float            # how long it is on screen
    text: str                 # the words it covers
    kind: str                 # BeatKind
    idea: str                 # what this stretch tells (the parts of a split idea share it)
    must_show: list[str]      # this shot's (old beats.json: the idea's)
    must_not_show: list[str]  # the idea's; the brief's `avoid` counts for every beat
    subject: str              # this shot of the idea (a split idea: another shot per part)
    look: str                 # ShotLook: photo | visualization (old beats.json: photo)
    generate: bool            # straight to the image generator, not searched (old beats.json: false)
    queries: list[str]        # this shot's stock searches, most concrete first (old beats.json: the idea's;
                              # older ones: "query", one string)
    prompt: str


# ------------------------------------------------------------------ on-screen labels (rótulos)
LabelKind = Literal["fuente", "persona", "termino"]


class LabelDraft(BaseModel):
    scene: int = Field(description="Número de la escena donde aparece")
    start: str = Field(description="Las primeras 3-5 palabras desde donde aparece, copiadas EXACTAMENTE de la "
                                   "narración de esa escena (donde se nombra a la persona, el estudio o el término)")
    kind: LabelKind = Field(description="fuente = el estudio, la revista o la institución de donde sale lo que se "
                                        "cuenta; persona = alguien con nombre que habla o hizo algo; termino = un "
                                        "concepto clave del vídeo, la primera vez que se explica")
    title: str = Field(max_length=48, description="La línea principal: el nombre de la persona, la institución o "
                                                  "la revista, o el término (en mayúsculas no: se pone solo)")
    subtitle: str = Field(default="", max_length=70,
                          description="Segunda línea (fuente y persona): qué es o quién es ('Ecólogo visual · "
                                      "Universidad de Exeter', 'Estudio publicado en Nature · 2026'). Vacía en "
                                      "un término")


class LabelsDraft(BaseModel):
    labels: list[LabelDraft] = Field(default_factory=list, max_length=16,
                                     description="Los rótulos del vídeo, en el orden en que aparecen")


class PortraitChoice(BaseModel):
    reason: str = Field(description="Por qué es o no es la persona del vídeo: sus ocupaciones, dónde trabaja y "
                                    "sus años frente al rótulo y la narración (1-2 frases)")
    id: str = Field(default="", description="El id de la ficha (Q...) que es SEGURO la persona del vídeo; vacío "
                                            "si ninguna lo es o hay duda")


class Label(TypedDict, total=False):
    scene: int
    word: int                 # index of the word it appears on, in the scene's narration
    start: float              # seconds from the start of the scene's audio
    text: str                 # the words it was placed on (to find it again with a new voice)
    kind: str                 # LabelKind
    title: str
    subtitle: str
    foto: str                 # persona: their portrait, relative to the run (retratos/...jpg); optional
    foto_ficha: str           # the Wikidata page it was taken from (to check it is them)
    foto_quien: str           # what that page says they are
    foto_credito: str         # author · licence · Commons page (goes to creditos.txt)


class VisualChoice(BaseModel):
    seen: list[str] = Field(default_factory=list, max_length=16,
                            description="PRIMERO, qué es cada imagen de verdad, en el orden de la cuadrícula, 3-10 "
                                        "palabras cada una, sin interpretarla para que encaje: lo que reconocería "
                                        "cualquiera ('1: satélite en una sala blanca', '2: gota de agua flotando', "
                                        "'3: mesa óptica con un láser azul')")
    reason: str = Field(description="En una frase: qué muestra la imagen elegida y por qué encaja con lo que se "
                                    "dice (o por qué ninguna encaja)")
    best: int = Field(description="Número de la imagen que mejor muestra lo que se dice; 0 si ninguna")
    score: int = Field(ge=0, le=10, description="0-10: cuánto encaja la elegida (10 = muestra exactamente eso)")


# ------------------------------------------------------------------- media
class Word(TypedDict):
    word: str
    start: float              # seconds from the start of the scene's audio
    end: float


class SceneAudio(TypedDict, total=False):
    path: str
    duration: float           # seconds; the scene's visual lasts the same
    words: list[Word]         # one per word of the narration, with its time (edge-tts, or the align node)


class SceneVisual(TypedDict, total=False):
    path: str                 # "" when kind is none
    kind: str                 # video | photo | generated | none (later: screenshot, chart)
    source: str               # the source that gave it: stock | generate
    provider: str             # pexels | pixabay | nasa | wikimedia | manual | z-image-turbo
    prompt: str               # generated: the image prompt (English)
    query: str                # found: the search that found it
    score: int                # found: the vision model's 0-10
    source_url: str           # where it comes from (stock page), for credits/licences
    author: str
    ai_generated: bool        # made with AI: counts for YouTube's synthetic content label
    duration: float           # videos: seconds of the clip (as the provider says)


# ------------------------------------------------------------------- state
class VideoState(TypedDict, total=False):
    run_id: str
    format: Format
    topic: str                            # what the video is about (--topic, or chosen at choose_topic)
    topic_candidates: list[dict]          # find_topics: the topics to choose from (data/temas/registro.json items)
    topics_seen: list[str]                # their ids, so "other topics" shows new ones
    topic_id: str                         # the chosen one's id in the record
    notes: str                            # your angle / opinion / what to stress (optional)
    urls: list[str]                       # articles you want used (optional)
    manual: bool                          # --manual: you write the script and bring the visuals
    video_type: VideoType                 # news | explainer (--type, or the research LLM decides it)
    manual_visuals: dict[int, list[dict]] # manual: scene id -> your marks [{"word", "path"}] (graph/manual.py)
    sources: list[Source]                 # articles the research read
    facts: list[Fact]                     # the ONLY data the script may use
    angles: dict                          # AnglesDraft.model_dump(): the 3 proposed angles
    angle_feedback: str                   # why you rejected the angles (they are proposed again with it)
    angle: dict                           # Angle.model_dump(): the one you chose (or wrote)
    outline: dict                         # Outline.model_dump()
    outline_feedback: str                 # why you rejected the outline (it is redone with it)
    deep_researched: bool                 # long videos: deep_research already looked up every section
    extra_urls: dict[int, list[str]]      # review_research: section -> your links, to be read
    script: dict                          # Script.model_dump()
    script_warnings: list[str]            # what write_script could not fix (e.g. length)
    script_plan: str                      # write_script's plan (the tension before the scenes): <run>/plan_guion.md
    script_requests: str                  # what the last rewrite was asked (yours, the reviewer's): the reviewer
                                          # checks it was done
    script_review: dict                   # review_script's last review: {summary, issues, dubious}
    script_notes: list[dict]              # review_script's serious issues: write_script fixes them
    review_feedback: str                  # your reason at the choose_script pause: write_script rewrites with it
    revisions: int                        # review_script's automatic rewrites (review.max_revisions)
    keep_visuals: bool                    # run_pipeline --keep-visuals: visuals reuses its choice (new voice only)
    audio: dict[int, SceneAudio]          # scene id -> audio (+ word times, from align)
    brief: dict                           # BriefDraft.model_dump(): the visual universe of the video
    beats: list[Beat]                     # what is shown and from which word
    labels: list[Label]                   # on-screen labels (rótulos): source, person, key term
    visuals: dict[int, SceneVisual]       # beat id -> visual
    visual_extras: dict[int, list[SceneVisual]]   # beat id -> more visuals that share its time (long
                                          # videos: no photo longer than visuals.photo_max_seconds)
    visual_warnings: list[str]
    montage: dict[int, list[dict]]        # beat id -> what assemble uses [{"path", "kind"}] (<run>/montaje/)
    subtitles_path: str
    video_path: str
    metadata: dict                        # title, description, tags, synthetic_media
    thumbnail_path: str
    approved: bool
    youtube_id: str
