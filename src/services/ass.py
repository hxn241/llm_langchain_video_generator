"""ASS files (Advanced SubStation): text burned onto the video with ffmpeg's `ass` filter (libass).

Used for the on-screen text of each scene (assemble) and, later, the word-by-word subtitles.

    style(name, font, size, ...)  one line of [V4+ Styles]
    document(width, height, styles, events)
    event(start, end, style, text)
"""
FORMAT_STYLE = ("Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
                "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, "
                "Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding")
FORMAT_EVENT = "Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"

# Alignment (numpad): 2 bottom centre, 5 middle centre, 8 top centre
TOP, MIDDLE, BOTTOM = 8, 5, 2


def colour(rgb="FFFFFF", alpha=0):
    """'RRGGBB' + opacity loss (0 = opaque, 255 = invisible) -> ASS &HAABBGGRR."""
    return f"&H{alpha:02X}{rgb[4:6]}{rgb[2:4]}{rgb[0:2]}".upper()


def style(name, font, size, text=colour(), outline_colour=colour("000000"), back=colour("000000", 128),
          bold=True, box=False, outline=4, shadow=0, alignment=BOTTOM, margin_h=60, margin_v=60):
    """box=True: a box behind the text (libass draws it with outline_colour; outline = its padding)."""
    return ",".join(map(str, [name, font, size, text, text, outline_colour, back, -1 if bold else 0, 0, 0, 0,
                              100, 100, 0, 0, 3 if box else 1, outline, shadow, alignment,
                              margin_h, margin_h, margin_v, 1]))


def timestamp(seconds):
    """Seconds -> h:mm:ss.cc"""
    cs = round(max(0.0, seconds) * 100)
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def escape(text):
    """Plain text -> ASS text (braces start override tags; newlines are \\N)."""
    return text.replace("\\", "/").replace("{", "(").replace("}", ")").replace("\n", r"\N")


def event(start, end, style_name, text, fade_ms=(200, 200), raw=False):
    """raw=True: text already has ASS tags (karaoke, colours...)."""
    body = text if raw else escape(text)
    tags = f"{{\\fad({fade_ms[0]},{fade_ms[1]})}}" if fade_ms else ""
    return f"Dialogue: 0,{timestamp(start)},{timestamp(end)},{style_name},,0,0,0,,{tags}{body}"


def document(width, height, styles, events):
    return "\n".join(["[Script Info]", "ScriptType: v4.00+", f"PlayResX: {width}", f"PlayResY: {height}",
                      "WrapStyle: 0", "ScaledBorderAndShadow: yes", "",
                      "[V4+ Styles]", f"Format: {FORMAT_STYLE}", *[f"Style: {s}" for s in styles], "",
                      "[Events]", f"Format: {FORMAT_EVENT}", *events]) + "\n"
