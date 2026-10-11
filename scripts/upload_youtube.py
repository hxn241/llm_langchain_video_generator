"""A folder from scripts/prepare_upload.py -> YouTube (video, title, description, tags, AI notice,
privacy or scheduled date, thumbnail, subtitles). Then it moves to data/subidos/ with youtube.json.

    python scripts/upload_youtube.py data/listo_para_subir/20261011_tema
    python scripts/upload_youtube.py data/listo_para_subir/20261011_tema --privacy unlisted

The first time it opens the browser to authorize your channel. Setup: src/services/youtube.py.
"""
import argparse
from pathlib import Path

from services.youtube import upload


def main():
    parser = argparse.ArgumentParser(description="Sube a YouTube una carpeta preparada.")
    parser.add_argument("folder", help="data/listo_para_subir/<yyyymmdd_tema>")
    parser.add_argument("--privacy", choices=["private", "unlisted", "public"],
                        help="overrides metadata.json (without Google's audit only private works)")
    args = parser.parse_args()
    folder = Path(args.folder)
    if not (folder / "metadata.json").exists():
        raise SystemExit(f"{folder} has no metadata.json (prepare it with scripts/prepare_upload.py)")
    upload(folder, privacy=args.privacy)


if __name__ == "__main__":
    main()
