"""The on-screen labels (rótulos, graph/nodes/labels.py) as an ASS document, burned in by assemble
over the video and under nothing else (the subtitles go at the bottom, the labels at the top left):

    fuente / persona   two lines in dark boxes: the name in white, what it is smaller and grey
                       ("Jolyon Troscianko" / "Ecólogo visual · Universidad de Exeter")
    termino            one line in a red box, in capitals ("CAMUFLAJE DE MOVIMIENTO")
    persona + "foto"   the portrait in a white frame on the left, the two lines beside it and
                       "Foto: Wikimedia Commons" under it. ASS cannot draw images: photos() gives
                       where and when, and assemble overlays the photo with ffmpeg, sliding and
                       fading like the text.

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
SLIDE_SECONDS = 0.3
FADE_IN, FADE_OUT = 0.25, 0.3
TOP_LEFT = 7                     # ASS alignment (numpad)
PHOTO_HEIGHT = 0.2               # of the frame's shorter side
PHOTO_RATIO = 0.8                # width / height (a 4:5 portrait)
PHOTO_CREDIT = "Foto: Wikimedia Commons"


def geometry(width, height):
    """Sizes and positions (pixels) shared by the text and the photos."""
    portrait = height > width
    title = round((width if portrait else height) * (0.045 if portrait else 0.034))
    x, y = round(width * 0.05), round(height * (0.09 if portrait else 0.07))
    pad = round(title * 0.35)
    photo_h = round(min(width, height) * PHOTO_HEIGHT)
    photo_w = round(photo_h * PHOTO_RATIO)
    return {"title": title, "sub": round(title * 0.72), "x": x, "y": y, "pad": pad,
            # the photo's top-left corner: where the text box's would be; the text goes on its right
            "photo_x": x - pad, "photo_y": y - pad, "photo_w": photo_w, "photo_h": photo_h,
            "text_x": x + photo_w + pad}


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


def _events(labels, starts, lengths, options):
    return timed(labels or [], starts, lengths, {**SECONDS, **((options or {}).get("seconds") or {})})


def photos(labels, starts, lengths, width, height, options=None):
    """The portraits to overlay: [{"path" (the label's "foto"), "start", "end", "x", "y", "w", "h"}],
    on the same clock as document()'s labels. x, y: where it ends its slide."""
    g = geometry(width, height)
    return [{"path": label["foto"], "start": start, "end": end, "x": g["photo_x"], "y": g["photo_y"],
             "w": g["photo_w"], "h": g["photo_h"]}
            for start, end, label in _events(labels, starts, lengths, options) if label.get("foto")]


def _move(x, y):
    return (f"{{\\move({x - SLIDE},{y},{x},{y},0,{round(SLIDE_SECONDS * 1000)})"
            f"\\fad({round(FADE_IN * 1000)},{round(FADE_OUT * 1000)})}}")


def document(labels, starts, lengths, width, height, font, options=None):
    """ASS with the labels (options: video.labels in config.yaml); None if there are none."""
    events = _events(labels, starts, lengths, options)
    if not events:
        return None
    g = geometry(width, height)
    x, y, pad = g["x"], g["y"], g["pad"]
    styles = [ass.style("Rotulo", font, g["title"], outline_colour=ass.colour(DARK, 40), box=True, outline=pad,
                        alignment=TOP_LEFT, margin_h=x, margin_v=y),
              ass.style("Termino", font, round(g["title"] * 1.05), outline_colour=ass.colour(ACCENT), box=True,
                        outline=pad, alignment=TOP_LEFT, margin_h=x, margin_v=y),
              ass.style("Credito", font, round(g["sub"] * 0.7), outline=2, bold=False, alignment=TOP_LEFT,
                        margin_h=x, margin_v=y)]
    lines = []
    for start, end, label in events:
        photo = bool(label.get("foto")) and label["kind"] != "termino"
        move = _move(g["text_x"] if photo else x, y)
        if label["kind"] == "termino":
            text, style = move + ass.escape(label["title"].upper()), "Termino"
        else:
            text, style = move + ass.escape(label["title"]), "Rotulo"
            if label.get("subtitle"):
                text += (f"\\N{{\\fs{g['sub']}\\b0\\1c{ass.inline_colour(GREY)}}}" + ass.escape(label["subtitle"]))
        lines.append(ass.event(start, end, style, text, fade_ms=None, raw=True))
        if photo:
            credit_y = g["photo_y"] + g["photo_h"] + round(pad * 0.6)
            lines.append(ass.event(start, end, "Credito", _move(g["photo_x"], credit_y) + ass.escape(PHOTO_CREDIT),
                                   fade_ms=None, raw=True))
    return ass.document(width, height, styles, lines)
