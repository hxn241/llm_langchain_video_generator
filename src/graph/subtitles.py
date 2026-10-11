"""Subtitles from the align node's word times (state["audio"][scene]["words"]), never across scenes.

    lines(words, max_words, max_seconds, ...)   the words of one scene in subtitles, cut at punctuation
    burned_document(...)                         ASS burned in by assemble (video.subtitles in config.yaml):
        highlight  a few words at a time, the one being said in another colour (Shorts)
        classic    whole phrases in 1-2 lines, white with a black outline (long videos)

The SRT for YouTube (services/publish.py) uses lines() too.
"""
import re

from services import ass

HIGHLIGHT = "FFD200"             # the word being said
LINGER = 0.4                     # a subtitle stays this long after its last word (if the next one is later)
END_PUNCTUATION = re.compile(r"[.,;:]+$")       # dropped on screen in highlight mode; ? and ! are kept
SENTENCE_END = re.compile(r"[.?!…]$")
PAUSE = re.compile(r"[,;:]$")
SHORT_WORD = 3                   # "el", "de", "por", "una"...: a subtitle does not end on them if it can help it


def lines(words, max_words, max_seconds, comma_after=1):
    """[[word, ...], ...]: at most max_words and max_seconds each, cut after the end of a sentence,
    and after a comma (; :) once the subtitle has comma_after words. One that is too long is cut at
    its last comma (in its second half) or, without one, before the short words it ends with
    ("... por el mal" -> "por el mal" goes to the next one)."""
    out, line = [], []
    for i, w in enumerate(words):
        line.append(w)
        if i + 1 == len(words) or SENTENCE_END.search(w["word"]) or (PAUSE.search(w["word"]) and len(line) >= comma_after):
            out.append(line)
            line = []
        elif len(line) >= max_words or w["end"] - line[0]["start"] >= max_seconds:
            cut = break_point(line)
            out.append(line[:cut])
            line = line[cut:]
    return out


def break_point(line):
    """Where to cut a subtitle that is too long: after its last comma in the second half, otherwise
    before the short words at its end (at most 3, keeping at least half of the subtitle)."""
    half = (len(line) + 1) // 2
    for j in range(len(line) - 1, half - 1, -1):
        if PAUSE.search(line[j]["word"]):
            return j + 1
    cut = len(line)
    while cut > max(half, len(line) - 3) and len(line[cut - 1]["word"]) <= SHORT_WORD:
        cut -= 1
    return cut


def burned_document(audio, starts, lengths, width, height, font, options):
    """ASS with the narration (options: video.subtitles.<format>: words, seconds, highlight).
    None if there are no word times or words is 0."""
    max_words, highlight = options.get("words", 0), options.get("highlight", False)
    if not max_words:
        return None
    timed = []                                   # (start, end, [(word start, text)])
    for sid, start in starts.items():
        scene_end = start + lengths[sid]
        for line in lines(audio[sid].get("words") or [], max_words, options.get("seconds", 3),
                          comma_after=1 if highlight else max_words // 2):
            words = [(start + w["start"], ass.escape(END_PUNCTUATION.sub("", w["word"]) if highlight else w["word"]))
                     for w in line]
            timed.append((words[0][0], min(start + line[-1]["end"] + LINGER, scene_end), words))
    if not timed:
        return None
    events = []
    for n, (line_start, line_end, words) in enumerate(timed):
        end = min(line_end, timed[n + 1][0]) if n + 1 < len(timed) else line_end      # never over the next
        if not highlight:
            events.append(ass.event(line_start, end, "Subtitle", " ".join(t for _, t in words), fade_ms=None))
            continue
        for k, (word_start, _) in enumerate(words):
            a = line_start if k == 0 else word_start
            b = words[k + 1][0] if k + 1 < len(words) else end
            if b - a < 0.01:
                continue
            text = " ".join(f"{{\\1c{ass.inline_colour(HIGHLIGHT)}}}{t}{{\\1c{ass.inline_colour()}}}" if j == k
                            else t for j, (_, t) in enumerate(words))
            events.append(ass.event(a, b, "Subtitle", text, fade_ms=None, raw=True))
    portrait = height > width
    size = round(min(width, height) * (0.08 if highlight else 0.045))
    styles = [ass.style("Subtitle", font, size, outline=round(size * (0.12 if highlight else 0.08)),
                        shadow=round(size * 0.05), alignment=ass.BOTTOM,
                        margin_h=round(width * (0.08 if highlight else 0.12)),
                        margin_v=round(height * (0.25 if portrait else 0.06)))]
    return ass.document(width, height, styles, events)
