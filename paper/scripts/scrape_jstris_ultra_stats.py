"""End-to-end Jstris Ultra leaderboard scrape: top-N → per-game attack at piece N.

This script is self-contained. Steps (each is skippable if outputs already exist):

  1. Fetch the top-N entries of the Ultra leaderboard via
     `GET https://jstris.jezevec10.com/api/leaderboard/5?mode=1` (open JSON API).
     Output: `paper/results/jstris/ultra_leaderboard.json`.

  2. For each entry, download the raw replay JSON via
     `GET /replay/data?id=<id>&type=0` using `curl_cffi` with Chrome impersonation
     (the endpoint is Cloudflare-protected; impersonate=chrome gets through).
     Output: `paper/results/jstris/replays/<id>.json`.

  3. Open the Jstris replay page ONCE in headed real Chrome (so all of Jstris's
     in-page JS — Replayer, ReplayController, Scoring, View — is loaded). For
     every downloaded replay JSON, call `replayController.loadReplay({0: data})`
     to swap the data into the *existing* Replayer; a 200 Hz JS-side observer
     records `replayer.gamedata.attack` and `replayer.placedBlocks` whenever
     placedBlocks changes; we read the snapshot at the target piece, then move
     on to the next replay. Only one navigation request to Jstris total — no
     Cloudflare rate limit possible.
     Output: `paper/results/jstris/ultra_stats/<id>.json`.

Run with `--max_replays 5` for a quick smoke test.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import structlog
from curl_cffi import requests
from playwright.sync_api import sync_playwright
from simple_parsing import parse

from tetris_bot.constants import PROJECT_ROOT

logger = structlog.get_logger()


@dataclass
class ScrapeArgs:
    output_dir: Path = PROJECT_ROOT / "paper" / "results" / "jstris"
    game: int = 5  # Ultra
    mode: int = 1
    top_n: int = 100
    target_piece: int = 50
    playback_speed: str = "2.0"
    per_replay_timeout_sec: float = 40.0
    bootstrap_replay_id: int | None = None  # Defaults to leaderboard rank 1
    sleep_between_downloads_sec: float = 0.25
    skip_existing: bool = True
    headless: bool = False
    max_replays: int | None = None


def fetch_leaderboard(args: ScrapeArgs, dest: Path) -> dict:
    """Fetch the full leaderboard (up to whatever the API returns, typically 500).

    We keep the *whole* pool so that download_all_replays() can fall through to
    lower-ranked entries when a top-N replay 404s (some old replays have been
    pruned from Jstris's servers).
    """
    if args.skip_existing and dest.exists():
        logger.info("Leaderboard cached; skipping fetch", path=str(dest))
        return json.loads(dest.read_text())
    url = f"https://jstris.jezevec10.com/api/leaderboard/{args.game}?mode={args.mode}"
    logger.info("Fetching leaderboard", url=url)
    r = requests.get(url, impersonate="chrome", timeout=30)
    r.raise_for_status()
    entries = r.json()
    if not isinstance(entries, list):
        raise RuntimeError(f"Unexpected leaderboard response: {type(entries).__name__}")
    payload = {
        "game": args.game,
        "mode": args.mode,
        "fetched_count": len(entries),
        "entries": entries,
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2) + "\n")
    logger.info("Saved leaderboard", path=str(dest), count=payload["fetched_count"])
    return payload


def download_replay(replay_id: int) -> dict | None:
    """Download a replay's raw JSON. Returns None on 404 (some old replays are gone)."""
    url = f"https://jstris.jezevec10.com/replay/data?id={replay_id}&type=0"
    r = requests.get(
        url,
        impersonate="chrome",
        timeout=30,
        headers={"Referer": f"https://jstris.jezevec10.com/replay/{replay_id}"},
    )
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


def download_all_replays(
    pool: list[dict], replays_dir: Path, target_n: int, args: ScrapeArgs
) -> list[dict]:
    """Walk down the leaderboard pool, downloading replays until we have `target_n`
    valid (non-404) replays. Returns the list of entries whose replays are on disk.
    """
    replays_dir.mkdir(parents=True, exist_ok=True)
    selected: list[dict] = []
    new = 0
    not_found = 0
    for entry in pool:
        if len(selected) >= target_n:
            break
        replay_id = int(entry["id"])
        out_path = replays_dir / f"{replay_id}.json"
        if out_path.exists():
            selected.append(entry)
            continue
        payload = download_replay(replay_id)
        if payload is None:
            not_found += 1
            logger.warning(
                "Replay not on server (404); will substitute next leaderboard entry",
                replay_id=replay_id,
                name=entry["name"],
                pos=entry["pos"],
            )
            continue
        out_path.write_text(json.dumps(payload) + "\n")
        new += 1
        selected.append(entry)
        logger.info(
            "Downloaded replay",
            pos=entry["pos"],
            name=entry["name"],
            replay_id=replay_id,
            d_b64_bytes=len(payload.get("d", "")),
        )
        time.sleep(args.sleep_between_downloads_sec)
    if len(selected) < target_n:
        logger.warning(
            "Exhausted leaderboard pool before reaching target_n",
            target_n=target_n,
            collected=len(selected),
            pool_size=len(pool),
        )
    logger.info(
        "Replay downloads done",
        target_n=target_n,
        selected=len(selected),
        newly_downloaded=new,
        skipped_404=not_found,
    )
    return selected


PROXY_AND_SAMPLER_INIT = """
    window.__perPiece = [];
    window.__samplerStarted = false;
    window.__lastPC = -1;
    window.__samplerId = null;

    // Patch ReplayController.prototype.update to capture `this` post-construction.
    // The page declares ReplayController with `var`, which bypasses any window-
    // level setter; but methods are looked up on the prototype at call time, so
    // patching the prototype always works regardless of when the page loads it.
    const tryPatch = () => {
        if (window.ReplayController && window.ReplayController.prototype && !window.__rcUpdatePatched) {
            const proto = window.ReplayController.prototype;
            if (proto.update) {
                window.__rcUpdatePatched = true;
                const origUpdate = proto.update;
                proto.update = function() {
                    window.__rc = this;
                    return origUpdate.apply(this, arguments);
                };
            }
        }
    };
    const patchTimer = setInterval(() => {
        tryPatch();
        if (window.__rcUpdatePatched) clearInterval(patchTimer);
    }, 5);

    // 200 Hz JS-side sampler. Records every change in placedBlocks. Reading from
    // Python at 20-50 Hz risks missing pieces at top-player PPS playback rates.
    const installSampler = () => {
        if (window.__rc && !window.__samplerStarted) {
            window.__samplerStarted = true;
            window.__samplerId = setInterval(() => {
                try {
                    const r = window.__rc && window.__rc.g && window.__rc.g[0];
                    if (!r || r.placedBlocks == null) return;
                    const pc = r.placedBlocks;
                    if (pc !== window.__lastPC) {
                        window.__lastPC = pc;
                        const g = r.gamedata || {};
                        window.__perPiece.push({
                            placedBlocks: pc,
                            timer_ms: r.timer,
                            reachedEnd: r.reachedEnd,
                            attack: g.attack,
                            linesSent: g.linesSent,
                            lines: g.lines,
                            singles: g.singles, doubles: g.doubles, triples: g.triples, tetrises: g.tetrises,
                            tspins: g.tspins,
                            PCs: g.PCs,
                            score: g.score,
                            B2B: g.B2B,
                            holds: g.holds,
                            maxCombo: g.maxCombo,
                        });
                    }
                } catch (e) {}
            }, 5);
        }
    };
    setInterval(installSampler, 10);

    // Helper called from Python between replays.
    window.__resetSampler = () => {
        window.__perPiece = [];
        window.__lastPC = -1;
    };
"""


def extract_one(page, replay_id: int, replay_json: dict, args: ScrapeArgs) -> dict:
    """Swap a pre-downloaded replay into the live Replayer and capture state at target piece.

    Mimics the user-clicks-Load path: lz-string-compress the replay JSON, write
    it to `textarea#rep0`, click Load. Passing the raw {c, d} dict directly to
    `rc.loadReplay(...)` doesn't work — the controller's parse path expects an
    already-normalized format, and on key-format mismatch falls back to reading
    the textarea, which means it would re-load whatever's left there.
    """
    load_result = page.evaluate(
        """(replayObj) => {
            window.__resetSampler();
            const rc = window.__rc;
            if (!rc) return {error: 'no rc'};
            if (!window.LZString || !window.LZString.compressToEncodedURIComponent) {
                return {error: 'no LZString on window'};
            }
            try {
                const compressed = window.LZString.compressToEncodedURIComponent(JSON.stringify(replayObj));
                const ta = document.getElementById('rep0');
                if (!ta) return {error: 'no textarea #rep0'};
                ta.value = compressed;
                document.getElementById('load').click();
            } catch (e) { return {error: e.toString().substring(0, 300)}; }
            return {ok: true};
        }""",
        replay_json,
    )
    if "error" in load_result:
        raise RuntimeError(f"loadReplay error: {load_result['error']}")

    target = args.target_piece
    started = time.perf_counter()
    snap_at_target: dict | None = None
    while time.perf_counter() - started < args.per_replay_timeout_sec:
        page.wait_for_timeout(150)
        latest = page.evaluate(
            """() => {
                const a = window.__perPiece;
                return a && a.length ? a[a.length - 1] : null;
            }"""
        )
        if not latest:
            continue
        pc = int(latest["placedBlocks"])
        if pc >= target:
            snap_at_target = page.evaluate(
                f"""() => {{
                    for (const s of window.__perPiece) {{
                        if (s.placedBlocks === {target}) return s;
                    }}
                    for (const s of window.__perPiece) {{
                        if (s.placedBlocks >= {target}) return s;
                    }}
                    return null;
                }}"""
            )
            page.evaluate(
                """() => {
                    if (window.__rc && window.__rc.pauseReplay) {
                        try { window.__rc.pauseReplay(); } catch(e) {}
                    }
                }"""
            )
            break
        if latest.get("reachedEnd"):
            break

    per_piece = page.evaluate("() => window.__perPiece || []")
    return {
        "replay_id": replay_id,
        "target_piece": target,
        "snapshot_at_target_piece": snap_at_target,
        "per_piece": per_piece,
        "last_piece_seen": per_piece[-1]["placedBlocks"] if per_piece else -1,
        "reached_end": per_piece[-1].get("reachedEnd") if per_piece else False,
    }


def extract_all(entries: list[dict], replays_dir: Path, stats_dir: Path, args: ScrapeArgs) -> None:
    stats_dir.mkdir(parents=True, exist_ok=True)
    bootstrap_id = args.bootstrap_replay_id or int(entries[0]["id"])

    succeeded = 0
    failed = 0
    skipped = 0
    missing = 0

    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=args.headless)
        ctx = browser.new_context(viewport={"width": 1400, "height": 1100})
        ctx.add_init_script(PROXY_AND_SAMPLER_INIT)
        page = ctx.new_page()

        boot_url = f"https://jstris.jezevec10.com/replay/{bootstrap_id}"
        logger.info("Bootstrap navigation", url=boot_url)
        page.goto(boot_url, wait_until="domcontentloaded", timeout=30000)
        page.wait_for_timeout(4000)

        # Set the playback speed once
        page.evaluate(
            f"""() => {{
                const sel = document.getElementById('speed');
                if (sel) {{ sel.value = '{args.playback_speed}'; sel.dispatchEvent(new Event('change')); }}
            }}"""
        )
        page.bring_to_front()

        # Confirm the controller was captured
        rc_ready = page.evaluate("() => !!window.__rc")
        if not rc_ready:
            raise RuntimeError(
                "ReplayController was never captured. The prototype patch may "
                "have missed its window — re-run; rerunning the bootstrap page "
                "load usually fixes this."
            )

        for entry in entries:
            replay_id = int(entry["id"])
            out_path = stats_dir / f"{replay_id}.json"
            if args.skip_existing and out_path.exists():
                skipped += 1
                continue

            replay_json_path = replays_dir / f"{replay_id}.json"
            if not replay_json_path.exists():
                missing += 1
                continue

            replay_json = json.loads(replay_json_path.read_text())
            try:
                result = extract_one(page, replay_id, replay_json, args)
            except Exception as exc:
                failed += 1
                logger.error(
                    "Extraction failed",
                    replay_id=replay_id,
                    name=entry["name"],
                    pos=entry["pos"],
                    error=str(exc)[:300],
                )
                continue

            attack_at_target = (
                result["snapshot_at_target_piece"]["attack"]
                if result["snapshot_at_target_piece"]
                else None
            )
            payload = {
                **result,
                "name": entry["name"],
                "pos": entry["pos"],
                "leaderboard_score": entry["game"],
                "attack_at_target_piece": attack_at_target,
            }
            out_path.write_text(json.dumps(payload) + "\n")
            succeeded += 1
            logger.info(
                "Scraped",
                pos=entry["pos"],
                name=entry["name"],
                replay_id=replay_id,
                last_piece=result["last_piece_seen"],
                attack_at_50=attack_at_target,
            )

        browser.close()

    logger.info(
        "Extraction done",
        succeeded=succeeded,
        failed=failed,
        skipped=skipped,
        missing=missing,
        total=len(entries),
    )


def main() -> None:
    args = parse(ScrapeArgs)

    leaderboard_path = args.output_dir / "ultra_leaderboard.json"
    replays_dir = args.output_dir / "replays"
    stats_dir = args.output_dir / "ultra_stats"

    # Step 1: leaderboard (full pool; we'll fall down it to substitute 404s)
    leaderboard = fetch_leaderboard(args, leaderboard_path)
    pool = leaderboard["entries"]
    target_n = args.top_n if args.max_replays is None else args.max_replays

    # Step 2: download replay JSONs, walking down the pool until we have target_n valid replays
    selected = download_all_replays(pool, replays_dir, target_n, args)

    # Step 3: drive the in-page Replayer to extract attack-at-piece-N
    extract_all(selected, replays_dir, stats_dir, args)


if __name__ == "__main__":
    main()
