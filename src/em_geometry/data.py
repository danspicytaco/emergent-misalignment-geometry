"""Explicit dataset paths and opt-in HF transfer tools (stdlib-only import).

EM_DATA_DIR names the input data directory; EM_HF_CACHE defaults to
PROJECT/artifacts/hf. EM_OUTPUT_DIR defaults to artifacts/runs.
All relative overrides resolve against the project, never the shell cwd.
No authentication, torch, dotenv or network is touched on import.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from email.utils import parsedate_to_datetime

REPO_ID = "danspicytaco/emergent-misalignment-geometry"
PREFIX = "legacy/2026-11-16/tree"
PROFILES = ("legacy-analysis", "legacy-activations", "legacy-training", "released-14b", "all")
_RESOLVER_PACER = None
REQUIRED_ANALYSIS = {
    "data/analysis/gpt54_repeatability/distances_fullprecision.npy",
    "data/released_14b/chatbot_verified_prompts.json",
    "data/released_14b/chatbot_verified/qwen2.5-14b-em-medical-advice-vllm-hybrid/judge_gpt-5.4_chatbot_verified.json",
    *(f"data/released_14b/gpt54_repeats/pass_{i}.json" for i in (1, 2, 3)),
}


def analysis_missing(manifest):
    required = REQUIRED_ANALYSIS | {e["id"] for e in manifest["entries"]
                                    if Path(e["source"]).name.startswith("pairs_") and e["source"].endswith(".csv")}
    available = {e["id"] for e in manifest["entries"] if e["status"] == "include" and "legacy-analysis" in e["profiles"]}
    return sorted(required - available)



def project_root():
    return Path(__file__).resolve().parents[2]


def _root(value, env: str, default: str):
    p = Path(value if value is not None else os.environ.get(env, default)).expanduser()
    return (project_root() / p).resolve() if not p.is_absolute() else p.resolve()


def cache_root(value=None):
    return _root(value, "EM_HF_CACHE", "artifacts/hf")


def data_root(value=None):
    return _root(value, "EM_DATA_DIR", str(cache_root() / PREFIX / "data"))


def output_root(value=None):
    return _root(value, "EM_OUTPUT_DIR", "artifacts/runs")


def legacy_root():
    return data_root().parent


def safe_path(value):
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError(f"unsafe relative path: {value!r}")
    p = PurePosixPath(value)
    if p.is_absolute() or any(x in ("", ".", "..") for x in value.split("/")) or ":" in value:
        raise ValueError(f"unsafe relative path: {value!r}")
    return value


def _secret(value):
    return any(p.lower().startswith((".env", "credentials", "token", "secret")) or
               p.lower() in (".git", ".ssh", ".aws", ".venv", "venv", "__pycache__")
               for p in PurePosixPath(value).parts)


def _under(root, rel):
    rel = safe_path(rel)
    target = root / rel
    if not target.resolve().is_relative_to(root.resolve()) or target.is_symlink():
        raise ValueError(f"path escapes root or is a symlink: {rel}")
    return target


def digest(path):
    before = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError(f"file changed while hashing: {path}")
    return after.st_size, h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".metadata-")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _matches(path, patterns):
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


def _profiles(path):
    names = ["all"]
    if (path.startswith(("data/analysis/", "data/judges/", "data/centroids/", "data/arena_build/", "results/", "binning-sensitivity/results/"))
            or (path.startswith("data/chatbot") and Path(path).name.startswith("judge"))
            or path in REQUIRED_ANALYSIS or path.endswith("/notes.md")
            or path in ("data/chatbot_verified_prompts.json", "data/chatbot_arena_2k_random_prompts.json", "data/chatbot_verified.json")):
        names.append("legacy-analysis")
    if path.endswith((".pt", ".npy")):
        names.append("legacy-activations")
    if path.startswith("data/training/") or "/datasets/" in path:
        names.append("legacy-training")
    if "released_14b" in path:
        names.append("released-14b")
    return names


def inventory(root, includes=(), excludes=(), source_includes=()):
    """Hash known data roots; nothing is approved without --include.

    --include-source separately acknowledges source-input redistribution review.
    Excludes always win. Secrets/symlinks are never read; logs/usage stay local.
    Includes tracked, untracked and ignored evidence inside these explicit roots.
    """
    root = Path(root).resolve()
    tracked = set(subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0"))
    candidates = set()
    for base in ("data", "results", "binning-sensitivity/results", "artifacts/released_14b", "em-not-magical"):
        for folder, dirs, files in os.walk(root / base, followlinks=False):
            dirs[:] = [d for d in dirs if not _secret(d) and not (Path(folder) / d).is_symlink()]
            for name in files:
                path = (Path(folder) / name).relative_to(root).as_posix()
                if _secret(path) or name in (".DS_Store", "sources.json", "legacy-migration.json"):
                    continue
                if base == "em-not-magical" and not path.endswith((".json", ".jsonl", ".csv", ".npy", ".pt", ".bak")):
                    continue
                candidates.add(path)
    entries = []
    for path in sorted(candidates):
        f = _under(root, path)
        source = (path.startswith(("em-not-magical/", "data/training/", "data/arena_build/")) or
                  (path.startswith("data/") and len(PurePosixPath(path).parts) == 2) or
                  path == "data/released_14b/chatbot_verified_prompts.json")
        local = path.endswith((".log", ".tar.gz")) or "usage" in Path(path).name.lower() or "account" in Path(path).name.lower()
        status, reason = "withheld", "not approved by include allowlist"
        if local:
            status, reason = "local-only", "operational log/archive/usage/account metadata"
        elif _matches(path, excludes):
            reason = "explicit exclude"
        elif _matches(path, includes):
            if source and not _matches(path, source_includes):
                reason = "source redistribution review required (--include-source)"
            else:
                status, reason = "include", "explicitly approved allowlist"
        size, sha = (f.stat().st_size, None) if local else digest(f)
        entries.append(dict(id=path, source=path, destination=f"{PREFIX}/{path}", size=size,
                            sha256=sha, status=status, reason=reason,
                            classification="source-input" if source else "derived-evidence",
                            tracked=path in tracked, profiles=_profiles(path)))
    return dict(schema_version=1, repo_id=REPO_ID, prefix=PREFIX, entries=entries)


def validate_manifest(manifest):
    if manifest.get("repo_id") != REPO_ID or manifest.get("prefix") != PREFIX:
        raise ValueError("unexpected dataset or snapshot prefix")
    ids, destinations = set(), set()
    for e in manifest["entries"]:
        source, destination = safe_path(e["source"]), safe_path(e["destination"])
        if _secret(source) or e["id"] != source or destination != f"{PREFIX}/{source}":
            raise ValueError("invalid inventory identity/destination")
        if source in ids or destination in destinations:
            raise ValueError("duplicate inventory entry")
        ids.add(source)
        destinations.add(destination)
        if e["status"] not in ("include", "withheld", "local-only"):
            raise ValueError("invalid inventory status")
        if e["status"] == "include" and (source.endswith((".log", ".tar.gz")) or
                                          any(word in Path(source).name.lower() for word in ("usage", "account"))):
            raise ValueError("operational metadata cannot be included")
        if e["status"] == "include" and (type(e["size"]) is not int or e["size"] < 0 or
                                          not re.fullmatch(r"[0-9a-f]{64}", e["sha256"] or "")):
            raise ValueError("invalid size/hash")
    return [e for e in manifest["entries"] if e["status"] == "include"]


def _check(path, entry):
    if digest(Path(path)) != (entry["size"], entry["sha256"]):
        raise ValueError(f"size/SHA256 mismatch: {entry['id']}")


def _api():
    from huggingface_hub import HfApi
    return HfApi()


def publish(manifest_path, root, staging_parent, sources_path, upload=False, api=None):
    """Dry-run by default. Staging hardlinks persist for HF resume.

    Source MUST remain quiescent; remove the owned staging tree after verification.
    """
    manifest_path, root = Path(manifest_path), Path(root).resolve()
    sources_path = Path(sources_path)
    relative_manifest = os.path.relpath(manifest_path.resolve(), sources_path.parent.resolve())
    safe_path(relative_manifest)
    manifest_hash = digest(manifest_path)[1]
    manifest = read_json(manifest_path)
    entries = validate_manifest(manifest)
    if not entries:
        raise ValueError("empty publish allowlist")
    summary = dict(files=len(entries), bytes=sum(e["size"] for e in entries), upload=upload,
                   analysis_missing=analysis_missing(manifest))
    if not upload:
        return summary
    api = api if api is not None else _api()
    if api.whoami().get("name") != "danspicytaco":
        raise ValueError("HF account must be danspicytaco")
    Path(staging_parent).mkdir(parents=True, exist_ok=True)
    staging = Path(staging_parent) / ("em-publish-" + digest(manifest_path)[1])
    staging.mkdir(exist_ok=True)
    for e in entries:
        source = _under(root, e["source"])
        _check(source, e)
        target = _under(staging, e["destination"])
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            os.link(source, target)  # fail instead of duplicating 24 GB
        _check(target, e)
    write_json(staging / "legacy/2026-11-16/inventory.json", manifest)
    allowed = {e["destination"] for e in entries} | {"legacy/2026-11-16/inventory.json"}
    for target in staging.rglob("*"):
        rel = target.relative_to(staging).as_posix()
        if target.is_symlink() or (target.is_file() and not rel.startswith(".cache/") and rel not in allowed):
            raise ValueError(f"unexpected staging content: {rel}")
    api.create_repo(repo_id=REPO_ID, repo_type="dataset", private=False, exist_ok=True)
    if api.repo_info(repo_id=REPO_ID, repo_type="dataset").private is not False:
        raise ValueError("destination must be public before uploading")
    api.upload_large_folder(repo_id=REPO_ID, repo_type="dataset", folder_path=staging,
                            allow_patterns=["legacy/**"])
    for e in entries:
        _check(_under(root, e["source"]), e)
    info = api.repo_info(repo_id=REPO_ID, repo_type="dataset")
    revision = info.sha
    if info.private is not False or not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
        raise ValueError("read-back must confirm public dataset and full commit SHA")
    pinned = api.repo_info(repo_id=REPO_ID, repo_type="dataset", revision=revision)
    if pinned.sha != revision or pinned.private is not False:
        raise ValueError("pinned repo read-back failed")
    if digest(manifest_path)[1] != manifest_hash:
        raise ValueError("inventory changed during publication")
    sources = dict(schema_version=1, repo_id=REPO_ID, repo_type="dataset", revision=revision,
                   prefix=PREFIX, manifest=relative_manifest, manifest_sha256=manifest_hash,
                   manifest_remote="legacy/2026-11-16/inventory.json", manifest_bytes=manifest_path.stat().st_size,
                   analysis_missing=analysis_missing(manifest),
                   profile_counts={p: sum(p in e["profiles"] for e in entries) for p in PROFILES})
    write_json(sources_path, sources)
    return dict(summary, revision=revision, sources=str(sources_path))


def load_sources(path, profile="all", model=None):
    path = Path(path).resolve()
    sources = read_json(path)
    if (sources.get("repo_id") != REPO_ID or sources.get("repo_type") != "dataset" or
            sources.get("prefix") != PREFIX or not re.fullmatch(r"[0-9a-f]{40}", sources.get("revision", ""))):
        raise ValueError("sources must pin the public dataset at a full 40-hex commit")
    manifest_path = _under(path.parent, sources["manifest"])
    if digest(manifest_path)[1] != sources["manifest_sha256"]:
        raise ValueError("inventory hash mismatch")
    manifest = read_json(manifest_path)
    included = validate_manifest(manifest)

    for name in PROFILES:
        expected = {e["id"] for e in included if name in e["profiles"]}
        if "profiles" in sources:
            members = sources["profiles"][name]
            if len(set(members)) != len(members) or set(members) != expected:
                raise ValueError(f"inexact profile membership: {name}")
        elif sources.get("profile_counts", {}).get(name) != len(expected):
            raise ValueError(f"inexact profile count: {name}")
    if profile not in PROFILES:
        raise ValueError("unknown profile")
    if profile == "legacy-analysis" and analysis_missing(manifest):
        raise ValueError("incomplete required analysis inputs: " + ", ".join(analysis_missing(manifest)))
    selected = [e for e in included if profile in e["profiles"]]
    if model:
        selected = [e for e in selected if model.casefold() in e["source"].casefold()]
    if not selected:
        raise ValueError("profile/model selection is empty")
    return sources, selected


def _download(sources, entry, folder, downloader=None):
    if downloader is None:
        from huggingface_hub import hf_hub_download
        downloader = hf_hub_download
    return Path(downloader(repo_id=REPO_ID, repo_type="dataset", revision=sources["revision"],
                           filename=entry["destination"], local_dir=str(folder),
                           cache_dir=str(folder / ".hub-cache"), force_download=True))


def ensure_manifest(sources_path, downloader=None):
    """Explicit fetch/verify bootstrap; keep the full inventory off GitHub."""
    path = Path(sources_path).resolve()
    sources = read_json(path)
    if (sources.get("repo_id") != REPO_ID or sources.get("repo_type") != "dataset" or
            not re.fullmatch(r"[0-9a-f]{40}", sources.get("revision", ""))):
        raise ValueError("sources must pin the public dataset at a full 40-hex commit")
    target = _under(path.parent, sources["manifest"])
    if target.exists():
        return
    entry = {"id": sources["manifest"],
             "destination": safe_path(sources["manifest_remote"]),
             "size": sources["manifest_bytes"], "sha256": sources["manifest_sha256"]}
    if type(entry["size"]) is not int or entry["size"] < 0 or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
        raise ValueError("invalid inventory size/hash")
    scratch = cache_root()
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".manifest-", dir=scratch) as folder:
        downloaded = _download(sources, entry, Path(folder), downloader)
        _check(downloaded, entry)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(downloaded, target)


class _ResolverPacer:
    """Share HF resolver pacing and Retry-After cooldown across all workers."""

    def __init__(self):
        self.lock = threading.Lock()
        self.next_request = 0.0
        self.blocked_until = 0.0

    @staticmethod
    def applies(request):
        return request.url.host == "huggingface.co" and (
            "/resolve/" in request.url.path or request.url.path.startswith("/api/resolve-cache/"))

    def before_request(self, request):
        if not self.applies(request):
            return
        while True:
            with self.lock:
                now = time.monotonic()
                delay = max(self.next_request, self.blocked_until) - now
                if delay <= 0:
                    # 10 resolver requests/s leaves headroom below 5000/300s.
                    self.next_request = now + 0.1
                    return
            time.sleep(delay)

    def after_response(self, response):
        if response.status_code != 429 or not self.applies(response.request):
            return
        retry = response.headers.get("Retry-After", "")
        try:
            delay = float(retry)
        except ValueError:
            try:
                delay = parsedate_to_datetime(retry).timestamp() - time.time()
            except (ValueError, TypeError, OverflowError):
                delay = 300.0
        with self.lock:
            self.blocked_until = max(self.blocked_until, time.monotonic() + max(0.0, delay) + 1)
        print(f"HF resolver cooldown: respecting Retry-After ({delay:.0f}s)", flush=True)


def _pace_resolvers():
    global _RESOLVER_PACER
    if _RESOLVER_PACER is None:
        from huggingface_hub.utils import get_session
        _RESOLVER_PACER = _ResolverPacer()
        client = get_session()
        client.event_hooks["request"].append(_RESOLVER_PACER.before_request)
        client.event_hooks["response"].append(_RESOLVER_PACER.after_response)


def fetch(sources_path, profile, root=None, model=None, downloader=None):
    ensure_manifest(sources_path, downloader)
    sources, entries = load_sources(sources_path, profile, model)
    root = cache_root(root)
    root.mkdir(parents=True, exist_ok=True)
    for e in entries:
        target = _under(root, e["destination"])
        if target.exists():
            _check(target, e)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".fetch-", dir=root) as folder:
            downloaded = _download(sources, e, Path(folder), downloader)
            _check(downloaded, e)
            os.replace(downloaded, target)
    return dict(files=len(entries), root=str(root), revision=sources["revision"])


def verify(sources_path, scratch, report_path, max_batch_bytes=1024**3, resume=False, downloader=None, workers=8):
    """Independently download ALL included files, one bounded batch/file at a time.

    Scratch must be owned/disposable. A completed batch is removed immediately;
    report checkpoints permit explicit resume without redownloading prior batches.
    """
    if downloader is None:
        _pace_resolvers()
    ensure_manifest(sources_path, downloader)
    sources, entries = load_sources(sources_path)
    identity = {k: sources[k] for k in ("repo_id", "revision", "manifest_sha256")}
    report = dict(identity, verified={}, complete=False)
    if resume and Path(report_path).exists():
        report = read_json(report_path)
        if any(report.get(k) != v for k, v in identity.items()):
            raise ValueError("resume report identity mismatch")
    expected = {e["id"]: dict(size=e["size"], sha256=e["sha256"]) for e in entries}
    if any(expected.get(k) != v for k, v in report["verified"].items()):
        raise ValueError("resume report contains unexpected entries")
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    if max_batch_bytes < 1 or workers < 1:
        raise ValueError("scratch budget and workers must be positive")
    report["complete"] = False
    write_json(report_path, report)
    pending = [e for e in entries if e["id"] not in report["verified"]]

    def check_download(e):
        with tempfile.TemporaryDirectory(prefix="em-readback-", dir=scratch) as folder:
            _check(_download(sources, e, Path(folder), downloader), e)
        return e["id"]

    # Bound total batch bytes and concurrent files; checkpoint per batch, not
    # by rewriting the growing all-file report after every tiny tensor.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while pending:
            batch, size = [], 0
            while pending and len(batch) < 128:
                entry = pending[0]
                if entry["size"] > max_batch_bytes:
                    raise ValueError(f"insufficient bounded scratch budget for {entry['id']}")
                if batch and size + entry["size"] > max_batch_bytes:
                    break
                batch.append(pending.pop(0))
                size += entry["size"]
            if shutil.disk_usage(scratch).free < size + 64 * 1024**2:
                raise ValueError("insufficient free disk for bounded verification batch")
            for identity in pool.map(check_download, batch):
                report["verified"][identity] = expected[identity]
            write_json(report_path, report)
            print(f"verified {len(report['verified'])}/{len(entries)} files", flush=True)
    report["complete"] = report["verified"] == expected
    report["files"] = len(entries)
    report["bytes"] = sum(e["size"] for e in entries)
    report["scope"] = "all included inventory entries, not withheld/local-only evidence"
    report["analysis_missing"] = sources.get("analysis_missing", [])
    write_json(report_path, report)
    return report


def main(action, argv=None):
    parser = argparse.ArgumentParser(description=globals()[action].__doc__)
    if action == "inventory":
        parser.add_argument("--root", type=Path, default=project_root())
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--include", action="append", default=[])
        parser.add_argument("--exclude", action="append", default=[])
        parser.add_argument("--include-source", action="append", default=[])
    elif action == "publish":
        parser.add_argument("--manifest", required=True)
        parser.add_argument("--root", type=Path, default=project_root())
        parser.add_argument("--staging-parent", required=True)
        parser.add_argument("--sources", required=True)
        parser.add_argument("--upload", action="store_true")
    else:
        parser.add_argument("--sources", default=str(project_root() / "data/sources.json"))
        if action == "fetch":
            parser.add_argument("--profile", choices=PROFILES, required=True)
            parser.add_argument("--model", help="case-insensitive substring of original relative path")
            parser.add_argument("--cache-root", help="download root; distinct from the input EM_DATA_DIR")
        else:
            parser.add_argument("--scratch", required=True)
            parser.add_argument("--report", required=True)
            parser.add_argument("--max-batch-bytes", type=int, default=1024**3)
            parser.add_argument("--resume", action="store_true")
            parser.add_argument("--workers", type=int, default=8)
    a = parser.parse_args(argv)
    if action == "inventory":
        result = inventory(a.root, a.include, a.exclude, a.include_source)
        write_json(a.output, result)
        result = {s: sum(e["status"] == s for e in result["entries"]) for s in ("include", "withheld", "local-only")}
    elif action == "publish":
        result = publish(a.manifest, a.root, a.staging_parent, a.sources, a.upload)
    elif action == "fetch":
        result = fetch(a.sources, a.profile, a.cache_root, a.model)
    else:
        result = verify(a.sources, a.scratch, a.report, a.max_batch_bytes, a.resume, workers=a.workers)
        result = {k: v for k, v in result.items() if k != "verified"}
    print(json.dumps(result, indent=2))
