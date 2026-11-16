"""Offline transfer contracts; no credentials, models, or real HF traffic."""
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location("storage_data", Path(__file__).parents[1] / "src/em_geometry/data.py")
d = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d)


@pytest.fixture
def repo(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    names = [*sorted(d.REQUIRED_ANALYSIS), "data/analysis/m7/pairs_prompt_mean_cosine.csv",
             "data/analysis/m32/pairs_prompt_mean_cosine.csv", "data/training/input.jsonl",
             "data/analysis/old.json.bak", "artifacts/released_14b/m14/0.pt", "data/run.log",
             "data/gpu.tar.gz", "data/judge_usage.json", "data/.env", "data/credentials.json",
             "em-not-magical/sec4/datasets/raw.jsonl", "results/study/raw.json"]
    for n in names:
        p = tmp_path / n
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(("fixture:" + n).encode())
    (tmp_path / ".gitignore").write_text("*.bak\nartifacts/\n")
    return tmp_path


class API:
    def __init__(self, name="danspicytaco", private=False):
        self.name, self.private, self.calls = name, private, []

    def whoami(self):
        return {"name": self.name}

    def create_repo(self, **kw):
        self.calls.append(("create", kw))
        assert kw["private"] is False and kw["repo_type"] == "dataset"

    def upload_large_folder(self, **kw):
        self.calls.append(("upload", kw))
        assert kw["repo_type"] == "dataset"
        assert not list(Path(kw["folder_path"]).rglob(".env"))

    def repo_info(self, **kw):
        self.calls.append(("read", kw))
        return SimpleNamespace(sha="a" * 40, private=self.private)  # fixture only


def prepared(repo):
    m = d.inventory(repo, ["*"], source_includes=["*"])
    path = repo / "metadata/inventory.json"
    sources = repo / "metadata/sources.json"
    d.write_json(path, m)
    api = API()
    d.publish(path, repo, repo / "stage", sources, upload=True, api=api)
    return m, path, sources, api


def downloader(repo, calls, corrupt=False):
    def download(**kw):
        assert kw["repo_type"] == "dataset"
        assert kw["revision"] == "a" * 40
        assert kw["force_download"] is True
        calls.append(kw)
        out = Path(kw["local_dir"]) / kw["filename"]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"corrupt" if corrupt else (repo / kw["filename"].removeprefix(d.PREFIX + "/")).read_bytes())
        return str(out)
    return download


def test_roots_cwd_independent(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("EM_DATA_DIR", raising=False)
    monkeypatch.delenv("EM_HF_CACHE", raising=False)
    assert d.cache_root() == d.project_root() / "artifacts/hf"
    assert d.data_root() == d.cache_root() / d.PREFIX / "data"
    monkeypatch.setenv("EM_DATA_DIR", "relative-data")
    assert d.data_root() == d.project_root() / "relative-data"
    assert d.data_root("explicit") == d.project_root() / "explicit"
    assert d.legacy_root() == d.data_root().parent
    assert d.cache_root() == d.project_root() / "artifacts/hf"
    monkeypatch.setenv("EM_OUTPUT_DIR", "out")
    assert d.output_root() == d.project_root() / "out"


@pytest.mark.parametrize("value", ["../escape", "/abs", "a/../b", "a//b", "a\\b", "C:/x", "a/./b", ""])
def test_safe_path(value):
    with pytest.raises(ValueError):
        d.safe_path(value)


def test_inventory_allowlists_secrets_and_backups(repo):
    m = d.inventory(repo, ["*"], ["*m32*"], [])
    entries = {e["id"]: e for e in m["entries"]}
    assert "data/.env" not in entries and "data/credentials.json" not in entries
    assert entries["data/run.log"]["status"] == "local-only"
    assert entries["data/judge_usage.json"]["sha256"] is None
    assert entries["data/training/input.jsonl"]["status"] == "withheld"
    assert entries["data/analysis/m32/pairs_prompt_mean_cosine.csv"]["status"] == "withheld"
    assert entries["data/analysis/old.json.bak"]["status"] == "include"
    assert entries["artifacts/released_14b/m14/0.pt"]["status"] == "include"
    assert "data/released_14b/chatbot_verified_prompts.json" in d.analysis_missing(m)


def test_publish_dryrun_and_wrong_account(repo):
    p = repo / "metadata/inventory.json"
    d.write_json(p, d.inventory(repo, ["*"]))
    api = API("wrong")
    assert d.publish(p, repo, repo / "stage", repo / "metadata/sources.json", api=api)["upload"] is False
    assert api.calls == []
    with pytest.raises(ValueError, match="account"):
        d.publish(p, repo, repo / "stage", repo / "metadata/sources.json", True, api)
    assert api.calls == []


def test_publish_readback_and_private_rejected(repo):
    m, p, sources, api = prepared(repo)
    assert len([x for x in api.calls if x[0] == "read"]) == 3
    assert d.read_json(sources)["revision"] == "a" * 40
    sources.unlink()
    with pytest.raises(ValueError, match="public"):
        d.publish(p, repo, repo / "stage", sources, True, API(private=True))
    assert not sources.exists()


@pytest.mark.parametrize("revision", ["main", "a" * 39, "A" * 40, "../bad"])
def test_revision_validation(repo, revision):
    _, _, sources, _ = prepared(repo)
    s = d.read_json(sources)
    s["revision"] = revision
    d.write_json(sources, s)
    with pytest.raises(ValueError, match="40-hex"):
        d.load_sources(sources)


def test_selective_fetch_and_immutable_target(repo):
    _, _, sources, _ = prepared(repo)
    calls = []
    root = repo / "restored"
    result = d.fetch(sources, "legacy-analysis", root, "m7", downloader(repo, calls))
    assert result["files"] == len(calls) == 1
    dest = root / calls[0]["filename"]
    assert dest.read_bytes() == (repo / "data/analysis/m7/pairs_prompt_mean_cosine.csv").read_bytes()
    d.fetch(sources, "legacy-analysis", root, "m7", downloader(repo, calls))
    assert len(calls) == 1
    dest.write_bytes(b"changed")
    with pytest.raises(ValueError, match="mismatch"):
        d.fetch(sources, "legacy-analysis", root, "m7", downloader(repo, calls))


def test_fetch_hash_mismatch_no_partial_target(repo):
    _, _, sources, _ = prepared(repo)
    root = repo / "restored"
    with pytest.raises(ValueError, match="mismatch"):
        d.fetch(sources, "released-14b", root, "m14", downloader(repo, [], corrupt=True))
    assert not (root / d.PREFIX / "artifacts/released_14b/m14/0.pt").exists()


def test_full_readback_coverage_cleanup_resume(repo):
    m, _, sources, _ = prepared(repo)
    calls = []
    scratch, report = repo / "readback", repo / "report.json"
    result = d.verify(sources, scratch, report, downloader=downloader(repo, calls))
    included = d.validate_manifest(m)
    assert result["complete"] and result["files"] == len(included) == len(calls)
    assert set(result["verified"]) == {e["id"] for e in included}
    assert list(scratch.iterdir()) == []
    d.verify(sources, scratch, report, resume=True, downloader=downloader(repo, calls))
    assert len(calls) == len(included)


def test_verify_hash_mismatch_and_disk_bound(repo):
    _, _, sources, _ = prepared(repo)
    with pytest.raises(ValueError, match="mismatch"):
        d.verify(sources, repo / "fresh", repo / "report.json", downloader=downloader(repo, [], True))
    assert not d.read_json(repo / "report.json")["complete"]
    assert list((repo / "fresh").iterdir()) == []
    with pytest.raises(ValueError, match="budget"):
        d.verify(sources, repo / "fresh", repo / "report.json", max_batch_bytes=1, downloader=downloader(repo, []))


def test_import_is_offline_and_cli_dryrun(repo):
    import sys
    code = ("import sys; sys.path.insert(0, " + repr(str(Path(__file__).parents[1] / "src")) + "); "
            "import em_geometry.data; "
            "assert not any(n in sys.modules for n in ['torch','dotenv','huggingface_hub'])")
    subprocess.run([sys.executable, "-c", code], cwd=repo, check=True)
    p = repo / "metadata/inventory.json"
    d.write_json(p, d.inventory(repo, ["*"]))
    cli = Path(__file__).parents[1] / "scripts/data/publish.py"
    result = subprocess.run([sys.executable, str(cli), "--manifest", str(p), "--root", str(repo),
                             "--staging-parent", str(repo / "stage"), "--sources", str(repo / "metadata/sources.json")],
                            check=True, capture_output=True, text=True)
    assert json.loads(result.stdout)["upload"] is False
    assert not (repo / "stage").exists()


def test_hash_detects_changing_file(monkeypatch, tmp_path):
    p = tmp_path / "changing"
    p.write_bytes(b"before")
    sha = d.hashlib.sha256
    class ChangingHash:
        def __init__(self):
            self.h = sha()
        def update(self, block):
            self.h.update(block)
            p.write_bytes(b"after-and-different-size")
        def hexdigest(self):
            return self.h.hexdigest()
    monkeypatch.setattr(d.hashlib, "sha256", ChangingHash)
    with pytest.raises(ValueError, match="changed while hashing"):
        d.digest(p)


def test_traversal_and_incomplete_profile(repo):
    _, p, sources, _ = prepared(repo)
    s = d.read_json(sources)
    s["manifest"] = "../escape.json"
    d.write_json(sources, s)
    with pytest.raises(ValueError, match="unsafe"):
        d.load_sources(sources)
    m = d.read_json(p)
    m["entries"][0]["destination"] = "../escape"
    with pytest.raises(ValueError):
        d.validate_manifest(m)
    m = d.inventory(repo, ["*"])
    d.write_json(p, m)
    d.publish(p, repo, repo / "stage2", sources, True, API())
    with pytest.raises(ValueError, match="incomplete"):
        d.load_sources(sources, "legacy-analysis")


def test_compact_sources_bootstrap_pinned_inventory(repo, monkeypatch):
    manifest, path, sources, _ = prepared(repo)
    payload = path.read_bytes()
    locator = d.read_json(sources)
    assert "profiles" not in locator
    assert locator["manifest_bytes"] == len(payload)
    assert locator["profile_counts"]["all"] == len(d.validate_manifest(manifest))
    path.unlink()
    monkeypatch.setattr(d, "cache_root", lambda value=None: repo / "cache")
    calls = []
    def get_inventory(**kw):
        calls.append(kw)
        assert kw["revision"] == locator["revision"]
        assert kw["filename"] == locator["manifest_remote"]
        target = Path(kw["local_dir"]) / kw["filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        return target
    d.ensure_manifest(sources, get_inventory)
    assert path.read_bytes() == payload
    assert len(calls) == 1
    assert len(d.load_sources(sources)[1]) == locator["profile_counts"]["all"]


def test_resolver_pacing_and_shared_retry_after(monkeypatch):
    clock, waits = [0.0], []
    monkeypatch.setattr(d.time, "monotonic", lambda: clock[0])
    def sleep(seconds):
        waits.append(seconds)
        clock[0] += seconds
    monkeypatch.setattr(d.time, "sleep", sleep)
    gate = d._ResolverPacer()
    request = SimpleNamespace(url=SimpleNamespace(host="huggingface.co", path="/datasets/unit/resolve/revision/file"))
    gate.before_request(request)
    gate.before_request(request)
    assert clock[0] == pytest.approx(0.1)
    gate.after_response(SimpleNamespace(status_code=429, request=request, headers={"Retry-After": "7"}))
    gate.before_request(request)
    assert clock[0] >= 7
    assert waits
    before = clock[0]
    gate.before_request(SimpleNamespace(url=SimpleNamespace(host="huggingface.co", path="/api/whoami-v2")))
    assert clock[0] == before


def test_resolver_invalid_retry_after_is_conservative(monkeypatch):
    monkeypatch.setattr(d.time, "monotonic", lambda: 20.0)
    gate = d._ResolverPacer()
    request = SimpleNamespace(url=SimpleNamespace(host="huggingface.co", path="/api/resolve-cache/datasets/unit/file"))
    gate.after_response(SimpleNamespace(status_code=429, request=request, headers={"Retry-After": "invalid"}))
    assert gate.blocked_until >= 320
    gate.after_response(SimpleNamespace(status_code=200, request=request, headers={"Retry-After": "1000"}))
    assert gate.blocked_until < 1000
