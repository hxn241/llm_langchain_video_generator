"""A finished run -> prepared and uploaded to YouTube in one go: prepare_upload.py + upload_youtube.py.
Those two stay for doing it by hand (prepare, edit metadata.json / descripcion.md, then upload).

    python scripts/publish_run.py --run 20261023-165908
    python scripts/publish_run.py --run 20261023-165908 --thumb 3 --force    # thumbnail from montaje/03
    python scripts/publish_run.py --run 20261023-165908 --yes                # without the confirmation

Before uploading it shows the title and the description and asks [y/n] (the upload is public
once you publish it in YouTube Studio: without Google's audit the API only uploads private).
"""
import argparse

from llmkit import config
from services import LLMHub, costs, topics
from services.publish import prepare
from services.settings import runs_dir
from services.youtube import upload


def publish(run_id, thumb=None, force=False, privacy=None, confirm=True):
    """prepare + upload -> the video id, or None if you said no (the folder stays prepared)."""
    costs.start(runs_dir(run_id))
    folder, meta = prepare(run_id, LLMHub(), thumb_number=thumb, force=force)
    costs.summary(runs_dir(run_id))
    print(f"\nPreparado: {config.readable_path(folder)}")
    print(f"  título: {meta['title']}")
    print(f"  miniatura: {meta['thumbnail_text']!r} · IA: {'sí' if meta['synthetic_media'] else 'no'}\n")
    print((folder / "descripcion.md").read_text(encoding="utf-8"))
    if confirm and input("¿Lo subo a YouTube? [y]es / [n]o: ").strip().lower() not in ("y", "yes", "s", "si", "sí"):
        print(f"No subido. Puedes editarlo y subirlo con:\n  python scripts/upload_youtube.py "
              f"\"{config.readable_path(folder)}\"")
        return None
    video_id = upload(folder, privacy=privacy)
    if topics.mark_uploaded(run_id, f"https://youtu.be/{video_id}"):
        print("  · tema marcado como subido en data/temas/registro.json")
    return video_id


def main():
    parser = argparse.ArgumentParser(description="Prepara un vídeo terminado y lo sube a YouTube.")
    parser.add_argument("--run", required=True, metavar="RUN_ID", help="the run (data/runs/<RUN_ID>)")
    parser.add_argument("--thumb", type=int, metavar="N", help="thumbnail from montaje/N (default: the first)")
    parser.add_argument("--force", action="store_true", help="overwrite a folder already prepared")
    parser.add_argument("--privacy", choices=["private", "unlisted", "public"],
                        help="overrides metadata.json (without Google's audit only private works)")
    parser.add_argument("--yes", action="store_true", help="upload without asking")
    args = parser.parse_args()
    publish(args.run, args.thumb, args.force, args.privacy, confirm=not args.yes)


if __name__ == "__main__":
    main()
