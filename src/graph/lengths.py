"""How long things must be, from config.yaml (video.<format>): seconds x words_per_second.

    video     length_targets()    total words and scenes, sections of the outline
              duration_text()     the length as text, for a prompt
    sections  assign_words()      the words of each section (the ends are shorter)
              words_by_facts()    the same, but no section longer than its facts allow (long videos
                                  after deep_research): a section without facts gets 0 words
              section_targets()   words and scenes one section must have
              planned_targets()   the video's targets for the words the outline really has
    checks    length_problems()   what is wrong, in Spanish: it goes back to the model
"""
SECTION_TOLERANCE = 0.2       # ±20 % of a section's words: the whole video stays inside its range
END_WEIGHT = 0.7              # the first (hook) and last (conclusion) sections are shorter
MIN_END_SHARE = 0.3           # words_by_facts: the hook and conclusion keep at least 30 % of their share


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
    """For the prompts: 'unos 55 segundos' / 'unos 5 minutos'."""
    return f"unos {seconds} segundos" if seconds < 120 else f"unos {round(seconds / 60)} minutos"


def assign_words(n_sections, total_words):
    """Words per section adding up to total_words; the first and last weigh END_WEIGHT."""
    weights = [1.0] * n_sections
    if n_sections >= 3:
        weights[0] = weights[-1] = END_WEIGHT
    unit = total_words / sum(weights)
    return [round(w * unit) for w in weights]


def words_by_facts(fact_counts, total_words, words_per_fact):
    """Words per section from assign_words, but a middle section gets at most its facts x
    words_per_fact: what one section cannot tell goes to those with facts to spare, and what
    nobody can tell is left out (the video is shorter, never padded). The first (hook) and last
    (conclusion) sections are not limited by their facts (they ask the video's question and answer
    it), but they shrink with the middle: a short video does not get a full-length hook."""
    n = len(fact_counts)
    words = assign_words(n, total_words)
    base = list(words)
    caps = [None if i in (0, n - 1) else c * words_per_fact for i, c in enumerate(fact_counts)]
    surplus = 0
    for i, cap in enumerate(caps):
        if cap is not None and words[i] > cap:
            surplus += words[i] - cap
            words[i] = cap
    while surplus >= 1:                                      # to those with room left, by their room
        room = {i: cap - words[i] for i, cap in enumerate(caps) if cap is not None and cap > words[i]}
        if not room:
            break
        total_room, given = sum(room.values()), 0
        for i, r in room.items():
            add = min(r, round(surplus * r / total_room))
            words[i] += add
            given += add
        if not given:
            break
        surplus -= given
    middle = [i for i, cap in enumerate(caps) if cap is not None]
    if middle:
        ratio = sum(words[i] for i in middle) / max(1, sum(base[i] for i in middle))
        for i in (0, n - 1):
            words[i] = round(base[i] * max(ratio, MIN_END_SHARE))
    return words


def planned_targets(targets, planned_words):
    """The whole video's targets scaled to the words its outline really plans (a long video with
    less material than its target length is shorter): same tolerance and words per scene."""
    if not planned_words or abs(planned_words - targets["words"]) <= targets["words"] * 0.05:
        return targets
    k = planned_words / targets["words"]
    return {**targets, "words": planned_words, "min_words": round(targets["min_words"] * k),
            "max_words": round(targets["max_words"] * k), "seconds": round(targets["seconds"] * k),
            "min_scenes": max(1, round(targets["min_scenes"] * k)), "max_scenes": max(1, round(targets["max_scenes"] * k))}


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
