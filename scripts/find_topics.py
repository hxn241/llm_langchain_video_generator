"""Topic ideas from the feeds of config.yaml (topics.feeds) -> data/temas/<fecha>.md, without starting
a video (run_pipeline.py without --topic does the same and lets you choose one).

    python scripts/find_topics.py                       # last 7 days, the model scores the new entries
    python scripts/find_topics.py --days 3 --top 5
    python scripts/find_topics.py --feed https://...    # one more feed, only this time (repeatable)
    python scripts/find_topics.py --no-llm              # just the entries, without any model (free)
    python scripts/find_topics.py --discard 3f2a9c1d    # never propose that topic again (its id)

The topics go to data/temas/registro.json (see src/services/topics.py).
"""
import argparse

from llmkit import config
from services import LLMHub
from services import topics


def main():
    parser = argparse.ArgumentParser(description="Busca temas para el canal en los feeds de config.yaml.")
    parser.add_argument("--days", type=int, help="entries of the last N days (default: topics.days)")
    parser.add_argument("--top", type=int, help="how many candidates (default: topics.max_candidates)")
    parser.add_argument("--feed", action="append", default=[], help="one more feed or site (repeatable)")
    parser.add_argument("--no-llm", action="store_true", help="only list the entries, without scoring them")
    parser.add_argument("--discard", nargs="+", metavar="ID", help="discard these topics for good (their ids)")
    args = parser.parse_args()
    if args.discard:
        topics.discard(args.discard)
        print(f"Descartados: {', '.join(args.discard)}")
        return
    use_model = not args.no_llm
    path, items = topics.find(LLMHub() if use_model else None, days=args.days, how_many=args.top,
                              extra_feeds=args.feed, use_model=use_model)
    print(f"\n{config.readable_path(path)}")
    for n, item in enumerate(items, 1):
        print(f"  {topics.short_line(n, item)} · id {item['id']}")


if __name__ == "__main__":
    main()
