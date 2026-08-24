# utils/retry.py
"""Utility to run a callable with exponential back‑off retries.

The callable should return an integer exit code (0 == success). The function
updates a JSON state file so that an orchestrator can resume or report progress.
"""
import json
import logging
import time
from pathlib import Path
from typing import Callable, Dict

log = logging.getLogger(__name__)


def _load_state(state_path: Path) -> Dict:
    if state_path.is_file():
        try:
            with state_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            log.error("Failed to load state file %s: %s", state_path, e)
    return {"stages": {}}


def _save_state(state_path: Path, state: Dict) -> None:
    tmp = state_path.with_suffix('.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(state, f, indent=2)
    tmp.replace(state_path)


def run_with_retry(
    func: Callable[[], int],
    stage_name: str,
    state_path: Path = Path("pipeline_state.json"),
    max_retries: int = 5,
    base_delay: float = 5.0,
    backoff_factor: float = 2.0,
    skip_if_complete: bool = False,
) -> int:
    """Execute *func* with exponential back‑off.

    Args:
        func: Callable that returns an exit code (0 on success).
        stage_name: Identifier for the pipeline stage.
        state_path: Path to the JSON file tracking retries.
        max_retries: Maximum number of attempts before raising.
        base_delay: Initial delay in seconds.
        backoff_factor: Multiplier for exponential back‑off.
        skip_if_complete: If True and state records stage as complete, skip execution.
    Returns:
        The exit code of the successful call (always 0).
    Raises:
        RuntimeError if the function fails after *max_retries* attempts.
    """
    state = _load_state(state_path)
    stage_info = state["stages"].setdefault(stage_name, {"last_retry": 0, "status": "pending"})

    if skip_if_complete and stage_info.get("status") == "complete":
        log.info("Stage '%s' already completed. Skipping.", stage_name)
        print(f"[*] Stage '{stage_name}' already completed in previous run. Skipping.")
        return 0

    retry = 0
    while retry <= max_retries:
        exit_code = func()
        if exit_code == 0:
            stage_info.update({"last_retry": retry, "status": "complete"})
            _save_state(state_path, state)
            return 0
        retry += 1
        stage_info.update({"last_retry": retry, "status": "failed"})
        _save_state(state_path, state)
        if retry > max_retries:
            break
        delay = base_delay * (backoff_factor ** (retry - 1))
        log.warning(
            "%s failed (attempt %d/%d). Retrying in %.1f seconds...",
            stage_name,
            retry,
            max_retries,
            delay,
        )
        print(f"[!] {stage_name} failed (attempt {retry}/{max_retries}). Retrying in {delay:.1f}s...")
        time.sleep(delay)
    raise RuntimeError(f"{stage_name} exceeded max retries ({max_retries})")
