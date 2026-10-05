"""How long things must be, from config.yaml (video.<format>): seconds x words_per_second.

    video     length_targets()    total words and scenes, sections of the outline
              duration_text()     the length as text, for a prompt
    sections  assign_words()      the words of each section (the ends are shorter)
              section_targets()   words and scenes one section must have
    checks    length_problems()   what is wrong, in Spanish: it goes back to the model
"""
SECTION_TOLERANCE = 0.2       # ±20 % of a section's words: the whole video stays inside its range
END_WEIGHT = 0.7              # the first (hook) and last (conclusion) sections are shorter


def length_targets(video_format, settings):
    """Words and scenes of the whole video, and the number of sections of its outline."""
    video, pace = settings["video"][video_format], settings["video"]["words_per_second"]
    min_seconds, max_seconds = video["seconds"]
    min_scenes, max_scenes = video["scenes"]
    min_sections, max_sections = video["sections"]
    return {"seconds": video["target_seconds"], "words": round(video["target_seconds"] * pace),
            "min_words": round(min_seconds * pace), "max_words": round(max_seconds * pace),
            "min_scenes": min_scenes, "max_scenes": max_scenes,
            "min_sections": min_sections, "max_sections": max_sections,
            "words_per_scene": round(video["target_seconds"] * pace) / ((min_scenes + max_scenes) / 2)}


def duration_text(seconds):
    """For the prompts: 'unos 55 segundos' / 'unos 11 minutos'."""
    return f"unos {seconds} segundos" if seconds < 120 else f"unos {round(seconds / 60)} minutos"


def assign_words(n_sections, total_words):
    """Words per section adding up to total_words; the first and last weigh END_WEIGHT."""
    weights = [1.0] * n_sections
    if n_sections >= 3:
        weights[0] = weights[-1] = END_WEIGHT
    unit = total_words / sum(weights)
    return [round(w * unit) for w in weights]


def section_targets(words, words_per_scene):
    scenes = max(1, round(words / words_per_scene))
    return {"words": words, "min_words": round(words * (1 - SECTION_TOLERANCE)),
            "max_words": round(words * (1 + SECTION_TOLERANCE)),
            "min_scenes": max(1, scenes - 1), "max_scenes": scenes + 1}


def length_problems(words, scenes, targets, where="La narración"):
    """[] = OK. Otherwise what to fix, in Spanish (it is sent back to the model)."""
    problems = []
    expected = f"entre {targets['min_words']} y {targets['max_words']} (objetivo: {targets['words']})"
    if words < targets["min_words"]:
        problems.append(f"{where} tiene {words} palabras y debe tener {expected}. "
                        "Añade contenido de los hechos (contexto, por qué importa), no palabras de relleno.")
    elif words > targets["max_words"]:
        problems.append(f"{where} tiene {words} palabras y debe tener {expected}. "
                        "Quita la idea menos importante.")
    if not targets["min_scenes"] <= scenes <= targets["max_scenes"]:
        problems.append(f"{where} tiene {scenes} escenas y deben ser entre "
                        f"{targets['min_scenes']} y {targets['max_scenes']}.")
    return problems
