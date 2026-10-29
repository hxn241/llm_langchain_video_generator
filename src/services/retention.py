"""The channel's videos -> how they are watched (scripts/retention.py).

    every video of the channel (its uploads playlist, YouTube Data API)
        -> views, average view duration, average % watched (YouTube Analytics API)
        -> the retention curve: 100 points, the share of the audience still watching (elapsedVideoTimeRatio)
        -> the steepest drops, with what the narration was saying then (subtitulos.srt of its folder in
           data/subidos, when there is one)
        + what only YouTube Studio has: thumbnail impressions, their click rate and, for shorts, the
          share that "stayed to watch" (vs swiped away), from the CSVs you export in Studio and leave in
          data/retencion/studio/ (the .csv, or the .zip as it downloads; the newest file wins per video)
    -> data/retencion/<yyyymmdd>.md (shorts and long videos apart, + the drops) and .json (raw, to compare)

YouTube gives the curve only once a video has enough views, and its data come 2-3 days late: a new
or barely watched video comes out without a curve. Impressions, click rate and "stayed to watch" are
not in the Analytics API (tried 2026-10-29: "query not supported"), hence the CSVs.
"""
import csv
import io
import json
import re
import unicodedata
import zipfile
from datetime import date, timedelta
from pathlib import Path

from llmkit import config
from services.settings import load_settings
from services.youtube import client

SHORT_MAX_SECONDS = 180
CHECKPOINTS = (3, 10, 30)         # seconds where the table says how many are still watching
DROPS = 3                         # drops listed per video
SKIP_START = 0.03                 # the first 3 % (the hook) goes in "a los 3 s", not among the drops
SKIP_END = 0.97                   # nor the last 3 %: everyone leaves when it ends
MIN_LOST = 0.01                   # a drop of less than 1 point of audience is noise
CONTEXT_SECONDS = 2.0             # the narration shown starts this long before the drop
VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
# Studio's CSV columns, recognised by words in the header (without accents, lower case), in Spanish
# or English: the first rule that matches a header names it
STUDIO_COLUMNS = (
    ("ctr", ("clic", "click")),
    ("stayed", ("quedaron", "stayed")),
    ("impressions", ("impresiones", "impressions")),
)


def _options():
    return load_settings().get("youtube", {})


def _folder():
    return config.BASE / _options().get("retention_dir", "data/retencion")


def folders():
    """{video id: its folder in uploaded_dir} (the ones prepared and uploaded by the pipeline)."""
    done = config.BASE / _options().get("uploaded_dir", "data/subidos")
    return {json.loads(info.read_text(encoding="utf-8"))["id"]: info.parent for info in done.glob("*/youtube.json")}


def channel_videos(youtube):
    """The ids of every video of the channel, newest first (its uploads playlist)."""
    channel = youtube.channels().list(part="contentDetails", mine=True).execute()["items"][0]
    playlist, ids, token = channel["contentDetails"]["relatedPlaylists"]["uploads"], [], None
    while True:
        page = youtube.playlistItems().list(part="contentDetails", playlistId=playlist, maxResults=50,
                                            pageToken=token).execute()
        ids += [i["contentDetails"]["videoId"] for i in page.get("items", [])]
        token = page.get("nextPageToken")
        if not token:
            return ids


def seconds(iso):
    """ISO 8601 duration (PT4M19S) -> seconds."""
    found = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso or "")
    h, m, s = found.groups(default="0") if found else ("0", "0", "0")
    return int(h) * 3600 + int(m) * 60 + int(s)


def srt(path):
    """[(start, end, text)] of a .srt; [] without one."""
    if not path or not path.exists():
        return []
    stamp = lambda t: sum(float(x) * f for x, f in zip(t.replace(",", ".").split(":"), (3600, 60, 1)))
    out = []
    for block in re.split(r"\n\s*\n", path.read_text(encoding="utf-8").strip()):
        lines = block.strip().splitlines()
        if len(lines) >= 3 and "-->" in lines[1]:
            a, b = (stamp(x.strip()) for x in lines[1].split("-->"))
            out.append((a, b, " ".join(lines[2:]).strip()))
    return out


def said(subtitles, start, end):
    """What the narration says between start and end."""
    return " ".join(t for a, b, t in subtitles if b > start and a < end)


def at(curve, ratio):
    """The curve's value at a point of the video (0-1), interpolated; None without a curve."""
    if not curve:
        return None
    for (x0, y0), (x1, y1) in zip(curve, curve[1:]):
        if x0 <= ratio <= x1:
            return y0 + (y1 - y0) * (ratio - x0) / ((x1 - x0) or 1)
    return curve[0][1] if ratio < curve[0][0] else curve[-1][1]


def drops(curve, duration, subtitles, n=DROPS):
    """The n steepest falls of the curve after the start and before the end, not next to each other:
    [{from, to, lost (points of audience), said}]."""
    steps = [(y0 - y1, x0, x1) for (x0, y0), (x1, y1) in zip(curve, curve[1:]) if SKIP_START <= x0 < SKIP_END]
    out = []
    for lost, x0, x1 in sorted(steps, reverse=True):
        if lost < MIN_LOST or len(out) == n:
            break
        if any(abs(x0 - d["x"]) < 0.025 for d in out):
            continue
        a, b = x0 * duration, x1 * duration
        out.append({"x": x0, "from": round(a, 1), "to": round(b, 1), "lost": round(lost * 100, 1),
                    "said": said(subtitles, a - CONTEXT_SECONDS, b)})
    return sorted(out, key=lambda d: d["x"])


# ------------------------------------------------------------------ YouTube Studio's CSVs
def _plain(text):
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()


def number(text):
    """"1,6 %" / "1.6" / "4.132" / "4,132" / "12.345,6" -> float; None if it is not a number."""
    s = (text or "").replace("%", "").replace(" ", "").replace(" ", "").strip()
    if not s or s in ("-", "—"):
        return None
    if "," in s and "." in s:                       # the last one is the decimal mark
        s = s.replace(".", "").replace(",", ".") if s.rfind(",") > s.rfind(".") else s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".") if len(s.split(",")[-1]) != 3 else s.replace(",", "")
    elif s.count(".") > 1 or (len(s.split(".")[-1]) == 3 and s.split(".")[0] not in ("", "0")):
        s = s.replace(".", "")                      # 4.132 = four thousand (Studio has no 3-decimal rates)
    try:
        return float(s)
    except ValueError:
        return None


def _studio_rows(path):
    """[(name, rows)] of a Studio export: a .csv, or every .csv inside a .zip."""
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            return [(f"{path.name}/{n}", list(csv.reader(io.StringIO(z.read(n).decode("utf-8-sig")))))
                    for n in z.namelist() if n.lower().endswith(".csv")]
    return [(path.name, list(csv.reader(io.StringIO(path.read_text(encoding="utf-8-sig")))))]


def studio(folder=None):
    """{video id: {impressions, ctr, stayed}} from the CSVs in data/retencion/studio/ (newest file
    last, so it wins), and the notes of what was read. A CSV without a video id column (the chart
    or the totals of an export) is skipped."""
    folder = Path(folder or _folder() / "studio")
    out, notes = {}, []
    files = sorted((p for p in folder.glob("**/*") if p.suffix.lower() in (".csv", ".zip")),
                   key=lambda p: p.stat().st_mtime) if folder.exists() else []
    for path in files:
        try:
            tables = _studio_rows(path)
        except (OSError, UnicodeDecodeError, zipfile.BadZipFile, csv.Error) as e:
            notes.append(f"{path.name}: no se puede leer ({e})")
            continue
        for name, rows in tables:
            if len(rows) < 2:
                continue
            header = [_plain(h) for h in rows[0]]
            id_col = next((i for i in range(len(header)) if sum(1 for r in rows[1:] if len(r) > i
                                                                  and VIDEO_ID.match(r[i].strip())) >= max(1, (len(rows) - 2) // 2)), None)
            if id_col is None:
                continue
            columns = {}
            for i, h in enumerate(header):
                key = next((k for k, words in STUDIO_COLUMNS if any(w in h for w in words)), None)
                if key and key not in columns:
                    columns[key] = i
            if not columns:
                notes.append(f"{name}: tiene vídeos pero ninguna columna de impresiones, clics ni «se quedaron viendo»")
                continue
            found = 0
            for r in rows[1:]:
                vid = r[id_col].strip() if len(r) > id_col else ""
                if VIDEO_ID.match(vid):
                    values = {k: number(r[i]) for k, i in columns.items() if i < len(r)}
                    out.setdefault(vid, {}).update({k: v for k, v in values.items() if v is not None})
                    found += 1
            notes.append(f"{name}: {found} vídeos · {', '.join(sorted(columns))}")
    return out, notes


# ------------------------------------------------------------------ collect
def _query(analytics, **kwargs):
    from googleapiclient.errors import HttpError
    try:
        return analytics.reports().query(ids="channel==MINE", **kwargs).execute()
    except HttpError as e:
        print(f"  ⚠️  {kwargs.get('filters')}: {e}")
        return {}


def collect():
    """[{id, title, published, duration, short, privacy, stats, studio, curve, drops, folder}] of every
    video of the channel, oldest first, and the notes about the Studio CSVs."""
    youtube, analytics = client(), client("youtubeAnalytics", "v2")
    local = folders()
    ids = list(dict.fromkeys(channel_videos(youtube) + list(local)))
    from_studio, notes = studio()
    today, out = date.today().isoformat(), []
    for i in range(0, len(ids), 50):
        for v in youtube.videos().list(part="snippet,contentDetails,status", id=",".join(ids[i:i + 50])).execute()["items"]:
            vid, duration = v["id"], seconds(v["contentDetails"]["duration"])
            published = v["snippet"]["publishedAt"][:10]
            # Analytics counts days in Pacific time: from the publication day in UTC, a video published
            # just after midnight UTC loses its first day (1534 views -> 338). A margin before costs nothing.
            start = (date.fromisoformat(published) - timedelta(days=2)).isoformat()
            filters = f"video=={v['id']}"
            summary = _query(analytics, startDate=start, endDate=today, filters=filters,
                             metrics="views,averageViewDuration,averageViewPercentage,likes,subscribersGained")
            row = (summary.get("rows") or [[0, 0, 0, 0, 0]])[0]
            stats = dict(zip(("views", "avg_seconds", "avg_percent", "likes", "subscribers"), row))
            curve = _query(analytics, startDate=start, endDate=today, filters=filters,
                           metrics="audienceWatchRatio", dimensions="elapsedVideoTimeRatio")
            curve = [(x, y) for x, y in curve.get("rows") or []]
            folder = local.get(vid)
            out.append({"id": vid, "title": v["snippet"]["title"], "published": published, "duration": duration,
                        "short": duration <= SHORT_MAX_SECONDS, "privacy": v["status"]["privacyStatus"],
                        "stats": stats, "studio": from_studio.get(vid, {}), "curve": curve,
                        "drops": drops(curve, duration, srt(folder / "subtitulos.srt" if folder else None)),
                        "folder": config.readable_path(folder) if folder else None})
    return sorted(out, key=lambda v: v["published"]), notes


# ------------------------------------------------------------------ report
def _pct(value):
    return "—" if value is None else f"{value * 100:.0f} %"


def _studio_pct(value):
    return "—" if value is None else f"{value:.1f} %".replace(".", ",")


def _table(videos, short):
    head = ["Vídeo", "Publicado", "Duración", "Vistas", "Vista media", "% medio"]
    head += ["Se quedaron viendo"] if short else ["Impresiones", "Clics (CTR)"]
    head += [f"a los {s} s" for s in CHECKPOINTS] + ["al final"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for v in videos:
        s, d, st = v["stats"], v["duration"] or 1, v["studio"]
        cells = [f"[{v['title']}](https://youtu.be/{v['id']})", v["published"], f"{d // 60}:{d % 60:02d}",
                 str(s["views"]), f"{s['avg_seconds']} s", f"{s['avg_percent']:.0f} %"]
        if short:
            cells.append(_studio_pct(st.get("stayed")))
        else:
            cells += [str(int(st["impressions"])) if st.get("impressions") is not None else "—", _studio_pct(st.get("ctr"))]
        cells += [_pct(at(v["curve"], c / d) if c < d else None) for c in CHECKPOINTS] + [_pct(at(v["curve"], 1.0))]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def markdown(videos, notes=()):
    lines = [f"# Retención · {date.today().isoformat()}", "",
             "Porcentaje de la audiencia que sigue viendo en cada momento (en shorts puede pasar del 100 % "
             "por quienes lo repiten). «—» = sin dato: YouTube aún no da la curva (pocas vistas o menos de "
             "2-3 días), o el dato es de YouTube Studio y no hay un CSV suyo en data/retencion/studio/.", ""]
    for short, name in ((True, "Shorts"), (False, "Vídeos largos")):
        group = [v for v in videos if v["short"] == short]
        if group:
            lines += [f"## {name}", "", *_table(group, short), ""]
    if notes:
        lines += ["CSV de YouTube Studio leídos:", *[f"- {n}" for n in notes], ""]
    lines += ["## Dónde se va la gente", ""]
    for v in videos:
        lines.append(f"### {v['title']}")
        if v["privacy"] != "public":
            lines.append(f"_({v['privacy']})_")
        if not v["curve"]:
            lines += ["Sin curva todavía.", ""]
            continue
        if not v["drops"]:
            lines.append("- Sin caídas bruscas.")
        for x in v["drops"]:
            lines.append(f"- **{x['from']:.0f}-{x['to']:.0f} s** · se va el {x['lost']} % de la audiencia"
                         + (f" · «{x['said']}»" if x["said"] else ""))
        if not v["folder"]:
            lines.append("- _(sin subtítulos guardados: solo los segundos)_")
        lines.append("")
    return "\n".join(lines)


def save(videos, notes=()):
    folder = _folder()
    folder.mkdir(parents=True, exist_ok=True)
    stem = date.today().strftime("%Y%m%d")
    (folder / f"{stem}.json").write_text(json.dumps(videos, ensure_ascii=False, indent=1), encoding="utf-8")
    (folder / f"{stem}.md").write_text(markdown(videos, notes) + "\n", encoding="utf-8")
    return folder / f"{stem}.md"
