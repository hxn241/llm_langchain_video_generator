"""The presentation of a LONG video: right after the hook (1-2 scenes), one or two sentences where the
narrator presents what the video is about, so it does not start "de sopetón" (user, 2026-10-16).
The hook stays first: an opening "Hoy vamos a hablar de..." loses people in the first seconds.

Five forms, chosen by code so they do not repeat: each new long video takes the next one in FORMS
after the last video's (the record: <STATE_DIR>/presentaciones.json, run id -> form). A run keeps
its form in every rewrite. The model writes it for its topic: the examples are from other topics.
"""
import json

from llmkit import config

RECORD = config.STATE_DIR / "presentaciones.json"
FORMS = [
    {"name": "promesa", "how": "di qué va a ver el espectador en este vídeo, de tú y en concreto (no \"un tema "
                               "fascinante\")",
     "example": "En este vídeo vas a ver cómo un hongo es capaz de controlar el cuerpo de una hormiga."},
    {"name": "narrador", "how": "el narrador cuenta en primera persona por qué quiso contar esto (qué le "
                                "sorprendió al descubrirlo) y que ahora te lo cuenta",
     "example": "Cuando leí cómo funciona, tuve que leerlo dos veces. Así que hoy te lo cuento desde el principio."},
    {"name": "mapa", "how": "anuncia el recorrido del vídeo: las dos o tres cosas por las que hay que pasar para "
                            "entenderlo, en el orden de las secciones y sin destripar el giro",
     "example": "Para entenderlo hay que pasar por tres cosas: qué es un agujero negro, por qué no deja escapar "
                "la luz y cómo, aun así, conseguimos fotografiar uno."},
    {"name": "pregunta", "how": "formula la pregunta que guía el vídeo y avisa de que vais a ir paso a paso",
     "example": "La pregunta es sencilla: ¿cómo sabe un ave migratoria hacia dónde volar? La respuesta no tanto, "
                "así que vamos por partes."},
    {"name": "viaje", "how": "propone empezar por el origen: el momento, el lugar o la persona donde empezó la "
                             "historia que cuenta el vídeo (solo con lo que dicen los hechos)",
     "example": "Para entenderlo hay que volver a un laboratorio de los años cincuenta, donde alguien encontró "
                "algo que no estaba buscando."},
]


def _record():
    try:
        return json.loads(RECORD.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def form_for(run_id):
    """The presentation form of this run: the one it already has, or the next after the last video's."""
    record, by_name = _record(), {f["name"]: f for f in FORMS}
    if str(run_id) in record and record[str(run_id)] in by_name:
        return by_name[record[str(run_id)]]
    names = [f["name"] for f in FORMS]
    last = next((name for name in reversed(list(record.values())) if name in names), None)
    form = FORMS[(names.index(last) + 1) % len(FORMS)] if last else FORMS[0]
    record[str(run_id)] = form["name"]
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    return form


def request_lines(form):
    """For the writer's request: where the presentation goes and the form it takes this time."""
    return ["", "# La presentación (vídeo largo)",
            "Justo después del gancho (en la escena 2, o en la 3 si el gancho necesita dos), una o dos frases en "
            f"las que el narrador presenta el vídeo. Esta vez, con esta forma: {form['how']}.",
            f"Ejemplo de la forma, de OTRO tema (no copies la frase, escríbela para este vídeo): \"{form['example']}\""]
