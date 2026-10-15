"""A finished run -> data/listo_para_subir/<yyyymmdd_tema>/, ready to check and upload.
Outside the pipeline: it reads the run's files and saved state, never runs the graph.

    python scripts/prepare_upload.py --run 20261010-223728
    python scripts/prepare_upload.py --run 20261010-223728 --thumb 3    # thumbnail from montaje/03
    python scripts/prepare_upload.py --run 20261010-223728 --force      # prepare it again

Then read and edit metadata.json and descripcion.md, and upload with scripts/upload_youtube.py.
See src/services/publish.py.
"""
import argparse

from llmkit import config
from services import LLMHub, costs
from services.publish import prepare
from services.settings import runs_dir


def main():
    parser = argparse.ArgumentParser(description="Prepara un vídeo terminado para subirlo a YouTube.")
    parser.add_argument("--run", required=True, metavar="RUN_ID", help="the run (data/runs/<RUN_ID>)")
    parser.add_argument("--thumb", type=int, metavar="N", help="thumbnail from montaje/N (default: the first)")
    parser.add_argument("--force", action="store_true", help="overwrite a folder already prepared")
    args = parser.parse_args()
    costs.start(runs_dir(args.run))                     # the publish call counts in the run's costes.md
    folder, meta = prepare(args.run, LLMHub(), thumb_number=args.thumb, force=args.force)
    costs.summary(runs_dir(args.run))
    print(f"\nListo para revisar: {config.readable_path(folder)}")
    print(f"  título: {meta['title']}")
    print(f"  miniatura: {meta['thumbnail_text']!r} · IA: {'sí' if meta['synthetic_media'] else 'no'} · "
          f"privacidad: {meta['privacy']}")
    print("  edita metadata.json y descripcion.md si quieres; luego:")
    print(f"  python scripts/upload_youtube.py \"{config.readable_path(folder)}\"")


if __name__ == "__main__":
    main()
