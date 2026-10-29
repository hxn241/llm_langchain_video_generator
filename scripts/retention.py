"""How the channel's videos are watched -> data/retencion/<yyyymmdd>.md: views, average view
duration and % watched, how many are left at 3/10/30 s and at the end, the steepest drops with what
the narration was saying then and, from the CSVs you export in YouTube Studio and leave in
data/retencion/studio/, impressions, click rate and "stayed to watch". See src/services/retention.py.

    python scripts/retention.py

Needs "YouTube Analytics API" enabled in the same Google Cloud project as the uploads.
"""
from llmkit import config
from services import retention


def main():
    videos, notes = retention.collect()
    if not videos:
        raise SystemExit("El canal no tiene vídeos.")
    path = retention.save(videos, notes)
    print(f"{len(videos)} vídeos · {sum(1 for v in videos if v['curve'])} con curva · "
          f"{sum(1 for v in videos if v['studio'])} con datos de Studio -> {config.readable_path(path)}")
    for n in notes:
        print(f"  · {n}")


if __name__ == "__main__":
    main()
