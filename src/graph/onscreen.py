"""The on-screen labels (rótulos, graph/nodes/labels.py) as an ASS document, burned in by assemble
over the video and under nothing else (the subtitles go at the bottom, the labels at the top left):

    fuente / persona   two lines in dark boxes: the name in white, what it is smaller and grey
                       ("Jolyon Troscianko" / "Ecólogo visual · Universidad de Exeter")
    termino            one line in a red box, in capitals ("CAMUFLAJE DE MOVIMIENTO")

Each one slides in a little and fades, stays video.labels.seconds[kind] and never over another:
one that would start while the previous is on screen waits for it (or is dropped if it would wait
more than MAX_WAIT, it would be shown far from its words).
"""
from services import ass

DARK = "0E1218"                  # the box of fuente / persona
ACCENT = "C8102E"                # the box of a termino
GREY = "D6D9DE"                  # the second line
SECONDS = {"fuente": 4.5, "persona": 4.5, "termino": 3.0}
GAP = 0.3                        # between one label leaving and the next arriving
MAX_WAIT = 2.0                   # a label later than this after its words is not shown
SLIDE = 40                       # pixels it slides in from the left
TOP_LEFT = 7                     # ASS alignment (numpad)


def timed(labels, starts, lengths, seconds):
    """[(start, end, label)] on the video's clock, in order, never two at once."""
    out, free_at = [], 0.0
    for label in sorted(labels, key=lambda l: starts.get(l["scene"], 0) + l.get("start", 0)):
        if label["scene"] not in starts:
            continue
        wanted = starts[label["scene"]] + label.get("start", 0.0)
        start = max(wanted, free_at)
        if start - wanted > MAX_WAIT:
            continue
        end = start + seconds.get(label["kind"], 4.0)
        out.append((start, end, label))
        free_at = end + GAP
    return out


def document(labels, starts, lengths, width, height, font, options=None):
    """ASS with the labels (options: video.labels in config.yaml); None if there are none."""
    options = options or {}
    events = timed(labels or [], starts, lengths, {**SECONDS, **(options.get("seconds") or {})})
    if not events:
        return None
    portrait = height > width
    title_size = round((width if portrait else height) * (0.045 if portrait else 0.034))
    sub_size = round(title_size * 0.72)
    x, y = round(width * 0.05), round(height * (0.09 if portrait else 0.07))
    pad = round(title_size * 0.35)
    styles = [ass.style("Rotulo", font, title_size, outline_colour=ass.colour(DARK, 40), box=True, outline=pad,
                        alignment=TOP_LEFT, margin_h=x, margin_v=y),
              ass.style("Termino", font, round(title_size * 1.05), outline_colour=ass.colour(ACCENT), box=True,
                        outline=pad, alignment=TOP_LEFT, margin_h=x, margin_v=y)]
    lines = []
    for start, end, label in events:
        move = f"{{\\move({x - SLIDE},{y},{x},{y},0,300)\\fad(250,300)}}"
        if label["kind"] == "termino":
            text, style = move + ass.escape(label["title"].upper()), "Termino"
        else:
            text, style = move + ass.escape(label["title"]), "Rotulo"
            if label.get("subtitle"):
                text += (f"\\N{{\\fs{sub_size}\\b0\\1c{ass.inline_colour(GREY)}}}" + ass.escape(label["subtitle"]))
        lines.append(ass.event(start, end, style, text, fade_ms=None, raw=True))
    return ass.document(width, height, styles, lines)
