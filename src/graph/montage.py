"""<run>/montaje/: what goes into the video, in order, with easy names, to check and change by hand.

visuals writes it when it finishes (write); the review_visuals pause reads it back (read) and
assemble uses exactly what is there, without any judge:
    01.mp4, 02.jpg, 03.mp4...   one file per beat, named by its number (zero-padded)
    lista.md                    each number with its time, its text and where the file comes from

Your changes (the folder is read again on every --redo assemble):
    replace   delete 05.mp4 and leave yours as 05.jpg / 05.mp4 (any video or jpg/png/webp)
    remove    delete 05.mp4: the previous shot goes on during beat 5
    add       05.jpg + 05b.mp4: beat 5's time is shared between them, in name order
    music     musica.mp3: the background music of the whole video (assemble; none = no music)

visuals may write more than one file per beat itself (long videos: a beat longer than
visuals.photo_max_seconds without a good video gets 05.jpg + 05b.jpg, so no photo stays too long).

--redo visuals chooses everything again but keeps your files: what visuals wrote is listed in
.auto.json (name, size, modification time); a file not in it, or changed, is yours. One of yours
named just by the number (05.jpg) replaces the chosen one; 05b.jpg is added next to it. A file
you removed comes back (it is a new choice).
"""
import json
import re
import shutil
from pathlib import Path

from graph import manual

FOLDER = "montaje"
LIST_FILE = "lista.md"
MANIFEST = ".auto.json"
MUSIC = "musica"                                  # musica.mp3 (or .wav, .m4a...): yours, the background music
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".opus"}
NUMBER = re.compile(r"^0*(\d+)(?!\d)")
PART = re.compile(r"^0*(\d+)([a-z]?)$")          # 05 -> beat 5, its first file; 05b -> its second...


def visual_of(name, visuals, extras=None):
    """The visual a montaje/ file name was written from: 05.jpg -> visuals[5], 05b.jpg -> extras[5][0].
    None if the name says nothing (one of yours named otherwise)."""
    m = PART.match(Path(name).stem)
    if not m:
        return None
    n, letter = int(m.group(1)), m.group(2)
    by_id = {int(k): v for k, v in (visuals or {}).items()}
    if not letter:
        return by_id.get(n)
    more = {int(k): v for k, v in (extras or {}).items()}.get(n) or []
    i = ord(letter) - ord("b")
    return more[i] if 0 <= i < len(more) else None


def folder_of(run_folder):
    return Path(run_folder) / FOLDER


def is_music(path):
    return Path(path).stem.lower() == MUSIC and Path(path).suffix.lower() in AUDIO_EXTENSIONS


def music_of(run_folder, default=None):
    """Your background music (montaje/musica.mp3...), or None: the video goes without music.
    Without one, `default` (video.music.default: the channel's track) is copied there first, so
    prepare_upload finds it and puts its credits (services/publish.py: music_credits)."""
    folder = folder_of(run_folder)
    found = next((p for p in sorted(folder.iterdir()) if p.is_file() and is_music(p)), None) \
        if folder.is_dir() else None
    if found or not default or not Path(default).is_file():
        return found
    folder.mkdir(parents=True, exist_ok=True)
    return Path(shutil.copy2(default, folder / f"{MUSIC}{Path(default).suffix.lower()}"))


def _stamp(path):
    stat = path.stat()
    return [stat.st_size, stat.st_mtime_ns]


def width_of(beats):
    return max(2, len(str(max((b["id"] for b in beats), default=1))))


def clock(seconds):
    return f"{int(seconds // 60)}:{seconds % 60:04.1f}"


def origin(visual):
    """Where a chosen visual comes from, for lista.md."""
    source = visual.get("source")
    if source == "generate":
        problems = f" · ⚠️ revisar: {'; '.join(visual['check'])}" if visual.get("check") else ""
        return f"generada ({visual.get('provider')}){problems}"
    if source == "manual":
        return f"tuya ({Path(visual['path']).name})"
    score = f" · {visual['score']}/10" if visual.get("score") is not None else ""
    return f"{visual.get('provider')} {visual.get('kind')}{score} · {visual.get('source_url', '')}"


def write(run_folder, beats, visuals, extras=None):
    """Copies each beat's chosen visual to montaje/NN.ext (and its extras, if any, to NNb.ext,
    NNc.ext...) and writes lista.md. Your files stay."""
    folder = folder_of(run_folder)
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / MANIFEST
    old = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    replaced = set()                               # beats where one of yours is named just by the number
    for p in sorted(folder.iterdir()):
        if not p.is_file() or p.name in (LIST_FILE, MANIFEST) or not manual.kind_of(p):
            continue
        if old.get(p.name) == _stamp(p):
            p.unlink()                             # ours and unchanged: chosen again below
        elif p.stem.isdigit():
            replaced.add(int(p.stem))              # 05.jpg replaces; 05b.jpg only adds

    digits, manifest = width_of(beats), {}
    extras = {int(k): v for k, v in (extras or {}).items()}
    for b in beats:
        visual = visuals.get(b["id"])
        if b["id"] in replaced or not visual or not visual.get("path"):
            continue                               # one of yours named 05.jpg takes the whole beat
        for letter, v in zip(["", *"bcdefgh"], [visual, *extras.get(b["id"], [])]):
            source = Path(v["path"])
            dst = folder / f"{b['id']:0{digits}d}{letter}{source.suffix.lower()}"
            shutil.copy2(source, dst)
            manifest[dst.name] = _stamp(dst)
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    write_list(run_folder, beats, visuals, manifest, extras)
    return folder


def write_list(run_folder, beats, visuals, manifest, extras=None):
    """lista.md, from what is in the folder now: ours with their origin, yours by name."""
    found, _ = read(run_folder, beats)
    digits, t = width_of(beats), 0.0
    lines = ["# Lo que va al vídeo, en orden", "",
             "Cambia lo que quieras en esta carpeta y responde [y] en la pausa (o `--redo assemble`):",
             "se monta exactamente lo que haya aquí, sin juez.", "",
             "- **Sustituir:** borra `05.mp4` y deja el tuyo como `05.jpg` o `05.mp4` (vídeo o jpg/png/webp).",
             "- **Quitar:** borra `05.mp4`: el plano anterior sigue durante ese tiempo.",
             "- **Añadir:** `05.jpg` + `05b.mp4`: el tiempo del fragmento 5 se reparte entre los dos.",
             "- **Música:** deja aquí `musica.mp3` (o .wav, .m4a...): suena de fondo todo el vídeo, por "
             "debajo de la voz. Sin ella, se usa la del canal (config.yaml: video.music.default).", "",
             "| Nº | Momento | Dura | Texto | Origen |", "|---|---|---|---|---|"]
    for b in beats:
        names = [Path(f["path"]).name for f in found[b["id"]]]
        source = " + ".join(f"`{n}` " + (origin(visual_of(n, visuals, extras) or {}) if n in manifest else "(tuyo)")
                            for n in names) or "sin archivo: sigue el plano anterior"
        lines.append(f"| {b['id']:0{digits}d} | {clock(t)} | {b.get('seconds', 0):.1f} s | {b['text']} | {source} |")
        t += b.get("seconds", 0)
    (folder_of(run_folder) / LIST_FILE).write_text("\n".join(lines) + "\n", encoding="utf-8")


def read(run_folder, beats):
    """montaje/ -> ({beat id: [{"path", "kind"}, ...]} (empty list: removed), problems)."""
    folder = folder_of(run_folder)
    ids = {b["id"] for b in beats}
    found, problems = {i: [] for i in ids}, []
    for p in sorted(folder.iterdir()) if folder.is_dir() else []:
        if not p.is_file() or p.name in (LIST_FILE, MANIFEST) or p.suffix.lower() == ".part" or is_music(p):
            continue
        m = NUMBER.match(p.stem)
        if not manual.kind_of(p) or not m:
            problems.append(f"{FOLDER}/{p.name}: no se usa (el nombre empieza por el número del fragmento "
                            "y es un vídeo o jpg/png/webp)")
        elif int(m.group(1)) not in ids:
            problems.append(f"{FOLDER}/{p.name}: no hay fragmento {int(m.group(1))} (van del 1 al {max(ids)})")
        else:
            found[int(m.group(1))].append({"path": str(p), "kind": manual.kind_of(p)})
    return found, problems
