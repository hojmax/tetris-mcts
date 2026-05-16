"""Download Jstris replay JSONs for the top-N leaderboard entries.

Uses curl_cffi with Chrome impersonation to bypass Cloudflare on the
`/replay/data?id=<id>&type=0` endpoint. The response is JSON with a config
section `c` (version, seed, mode, DAS, etc.) and a base64-encoded binary event
stream `d`.

Each replay is written as `paper/results/jstris/replays/<id>.json` so we can
parse them offline without re-fetching.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import structlog
from curl_cffi import requests
from simple_parsing import parse

from tetris_bot.constants import PROJECT_ROOT

logger = structlog.get_logger()


@dataclass
class DownloadArgs:
    leaderboard_path: Path = (
        PROJECT_ROOT / "paper" / "results" / "jstris" / "ultra_leaderboard.json"
    )
    output_dir: Path = PROJECT_ROOT / "paper" / "results" / "jstris" / "replays"
    sleep_between_requests: float = 0.3
    skip_existing: bool = True
    max_replays: int | None = None


def fetch_replay(replay_id: int) -> dict:
    url = f"https://jstris.jezevec10.com/replay/data?id={replay_id}&type=0"
    response = requests.get(
        url,
        impersonate="chrome",
        timeout=30,
        headers={"Referer": f"https://jstris.jezevec10.com/replay/{replay_id}"},
    )
    response.raise_for_status()
    return response.json()


def main() -> None:
    args = parse(DownloadArgs)
    leaderboard = json.loads(args.leaderboard_path.read_text())
    entries = leaderboard["entries"]
    if args.max_replays is not None:
        entries = entries[: args.max_replays]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    succeeded = 0
    failed = 0
    skipped = 0
    for entry in entries:
        replay_id = int(entry["id"])
        out_path = args.output_dir / f"{replay_id}.json"
        if args.skip_existing and out_path.exists():
            skipped += 1
            continue
        try:
            payload = fetch_replay(replay_id)
        except Exception as exc:
            failed += 1
            logger.error(
                "Failed to fetch replay",
                replay_id=replay_id,
                name=entry["name"],
                error=str(exc),
            )
            continue
        out_path.write_text(json.dumps(payload) + "\n")
        succeeded += 1
        logger.info(
            "Saved replay",
            replay_id=replay_id,
            name=entry["name"],
            pos=entry["pos"],
            d_bytes_b64=len(payload.get("d", "")),
        )
        time.sleep(args.sleep_between_requests)

    logger.info(
        "Done", succeeded=succeeded, failed=failed, skipped=skipped, total=len(entries)
    )


if __name__ == "__main__":
    main()
