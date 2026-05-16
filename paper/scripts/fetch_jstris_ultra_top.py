"""Fetch the top-N Jstris Ultra leaderboard entries into a JSON file.

Uses the public, un-gated leaderboard endpoint:
    GET https://jstris.jezevec10.com/api/leaderboard/5?mode=<mode>

Game id 5 = Ultra. The default `mode=1` is the standard ranked mode.
The response is a list of {id, pos, game, ts, name}; `game` is the score for Ultra.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import structlog
from curl_cffi import requests
from simple_parsing import parse

from tetris_bot.constants import PROJECT_ROOT

logger = structlog.get_logger()


@dataclass
class FetchArgs:
    output_path: Path = (
        PROJECT_ROOT / "paper" / "results" / "jstris" / "ultra_leaderboard.json"
    )
    game: int = 5  # 5 = Ultra
    mode: int = 1
    top_n: int = 100
    timeout_sec: float = 30.0


def fetch_leaderboard(game: int, mode: int) -> list[dict]:
    url = f"https://jstris.jezevec10.com/api/leaderboard/{game}?mode={mode}"
    logger.info("Fetching Jstris leaderboard", url=url)
    response = requests.get(url, impersonate="chrome", timeout=30)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected response shape: {type(data).__name__}")
    return data


def main() -> None:
    args = parse(FetchArgs)
    entries = fetch_leaderboard(args.game, args.mode)
    top = entries[: args.top_n]
    payload = {
        "game": args.game,
        "mode": args.mode,
        "fetched_count": len(top),
        "entries": top,
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(payload, indent=2) + "\n")
    logger.info(
        "Saved top leaderboard slice",
        path=str(args.output_path),
        count=len(top),
        top_name=top[0]["name"] if top else None,
        top_score=top[0]["game"] if top else None,
    )


if __name__ == "__main__":
    main()
