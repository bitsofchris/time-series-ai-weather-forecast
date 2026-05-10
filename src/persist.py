"""Persist the forecast SQLite DB across HF Space rebuilds.

HF Spaces' free tier has ephemeral storage — every `git push` rebuilds the
container and wipes any local files. We back the forecast log with a
private HF Dataset:

  - On startup: pull the latest forecasts.db from the dataset (if any).
  - After every refresh: push the current forecasts.db back.

Environment:
  HF_TOKEN          must have write access to the dataset
  LOG_DATASET_REPO  override the default dataset repo id
"""

from __future__ import annotations

import os
import shutil
import threading
import time
import traceback

DEFAULT_REPO = "bitsofchris/toto-weather-forecast-log"
PATH_IN_REPO = "forecasts.db"
DEFAULT_LOCAL = "data/forecasts.db"

_push_lock = threading.Lock()
_last_push_at = 0.0
PUSH_MIN_INTERVAL = 60.0  # seconds — coalesce rapid pushes


def _repo_id() -> str:
    return os.environ.get("LOG_DATASET_REPO", DEFAULT_REPO)


def _token() -> str | None:
    return os.environ.get("HF_TOKEN")


def pull_db(local_path: str = DEFAULT_LOCAL) -> bool:
    """Download the latest DB from the dataset, overwriting any local copy.
    Returns True on success."""
    tok = _token()
    if not tok:
        print("[persist] HF_TOKEN not set — skipping pull")
        return False
    try:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415
        downloaded = hf_hub_download(
            repo_id=_repo_id(),
            repo_type="dataset",
            filename=PATH_IN_REPO,
            token=tok,
        )
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        shutil.copyfile(downloaded, local_path)
        print(f"[persist] pulled DB from {_repo_id()} ({os.path.getsize(local_path)} bytes)")
        return True
    except Exception:  # noqa: BLE001
        print(f"[persist] pull skipped (no remote DB or network error):")
        traceback.print_exc()
        return False


def push_db(local_path: str = DEFAULT_LOCAL) -> bool:
    """Upload the local DB to the dataset. Coalesced and lock-protected so
    overlapping refreshes don't issue redundant uploads."""
    global _last_push_at
    tok = _token()
    if not tok or not os.path.exists(local_path):
        return False

    # Coalesce: if we just pushed, skip.
    if time.time() - _last_push_at < PUSH_MIN_INTERVAL:
        return False
    if not _push_lock.acquire(blocking=False):
        return False
    try:
        from huggingface_hub import HfApi  # noqa: PLC0415
        api = HfApi(token=tok)
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=PATH_IN_REPO,
            repo_id=_repo_id(),
            repo_type="dataset",
            commit_message="forecast log update",
        )
        _last_push_at = time.time()
        print(f"[persist] pushed DB to {_repo_id()} ({os.path.getsize(local_path)} bytes)")
        return True
    except Exception:  # noqa: BLE001
        print("[persist] push failed:")
        traceback.print_exc()
        return False
    finally:
        _push_lock.release()


def push_db_async(local_path: str = DEFAULT_LOCAL) -> None:
    """Fire-and-forget push so refresh() returns to the user immediately."""
    threading.Thread(
        target=push_db, args=(local_path,), daemon=True, name="persist-push"
    ).start()


# --- multi-file push (forecast log + Ecowitt archive in one commit) ------
ARCHIVE_LOCAL = "data/ecowitt.db"
ARCHIVE_PATH_IN_REPO = "ecowitt.db"
_multi_lock = threading.Lock()
_multi_last = 0.0


def push_all(
    forecast_local: str = DEFAULT_LOCAL,
    archive_local: str = ARCHIVE_LOCAL,
) -> bool:
    """Upload both DBs in a single dataset commit."""
    global _multi_last
    tok = _token()
    if not tok:
        return False
    if time.time() - _multi_last < PUSH_MIN_INTERVAL:
        return False
    if not _multi_lock.acquire(blocking=False):
        return False
    try:
        from huggingface_hub import CommitOperationAdd, HfApi  # noqa: PLC0415
        api = HfApi(token=tok)
        ops = []
        for local, in_repo in (
            (forecast_local, PATH_IN_REPO),
            (archive_local, ARCHIVE_PATH_IN_REPO),
        ):
            if os.path.exists(local):
                ops.append(CommitOperationAdd(path_in_repo=in_repo, path_or_fileobj=local))
        if not ops:
            return False
        api.create_commit(
            repo_id=_repo_id(),
            repo_type="dataset",
            operations=ops,
            commit_message="forecast log + archive update",
        )
        _multi_last = time.time()
        sizes = ", ".join(f"{op.path_in_repo}={os.path.getsize(forecast_local if op.path_in_repo==PATH_IN_REPO else archive_local)}B" for op in ops)
        print(f"[persist] pushed multi to {_repo_id()} ({sizes})")
        return True
    except Exception:  # noqa: BLE001
        print("[persist] push_all failed:")
        traceback.print_exc()
        return False
    finally:
        _multi_lock.release()


def push_all_async(
    forecast_local: str = DEFAULT_LOCAL,
    archive_local: str = ARCHIVE_LOCAL,
) -> None:
    threading.Thread(
        target=push_all, args=(forecast_local, archive_local),
        daemon=True, name="persist-push-all",
    ).start()


def pull_all(
    forecast_local: str = DEFAULT_LOCAL,
    archive_local: str = ARCHIVE_LOCAL,
) -> None:
    """Pull both DBs from the dataset on startup. Each missing file is silently skipped."""
    pull_db(forecast_local)
    # Pull the archive too if it exists.
    tok = _token()
    if not tok:
        return
    try:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415
        downloaded = hf_hub_download(
            repo_id=_repo_id(),
            repo_type="dataset",
            filename=ARCHIVE_PATH_IN_REPO,
            token=tok,
        )
        os.makedirs(os.path.dirname(archive_local) or ".", exist_ok=True)
        shutil.copyfile(downloaded, archive_local)
        print(f"[persist] pulled archive ({os.path.getsize(archive_local)} bytes)")
    except Exception:
        # 404 on first run is expected.
        pass
