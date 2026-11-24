"""Publish the fixed original coefficient/TCP artifact and a read-only audit.

Exact original bytes and all audit findings are preserved, including failures.
This helper never runs a model, a study grid, or a timing measurement.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import urllib.error
import urllib.request

REPOSITORY = "ConfusedDuckCollectibles/Research3"
REPOSITORY_ID = 1469572345
STUDY_COMMIT = "5e158fb8ecd9e4f70f532aeb93cc253bbea7741f"
DESIGN_COMMIT = "49cfba7261326d58e174e51c8ddf5abba6f4a966"
DESTINATION = "results/coefficient_packet"
MANIFEST_PATH = "scripts/coefficient_publication_manifest.json"
MANIFEST_SHA256 = "73c204d1444269e97f5ed03838b183aa84241d491ef36afe463dd87459a7a702"
AUDITOR_PATH = "scripts/audit_coefficient_outputs.py"
RESERVED_DESTINATIONS = {"audit_receipt.json", "publication_manifest.json", "publication_provenance.json"}
CLASSIFICATIONS = {"complete_study", "incomplete_study"}


class PublicationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PublicationError(message)


def fixed_child(root: Path, relative: str, *, create_parents: bool = False) -> Path:
    """Reject symlinks, traversal, unexpected parents and root escape."""
    root = root.absolute()
    require(root.is_dir(), "Root directory is unavailable")
    require(not any(path.is_symlink() for path in (root, *root.parents)),
            "Symlink in root path is forbidden")
    parts = PurePosixPath(relative)
    require(not parts.is_absolute() and relative and "\\" not in relative,
            "Relative POSIX path required")
    require(all(part not in ("", ".", "..") for part in relative.split("/")),
            "Traversal or empty path component is forbidden")
    target = root
    for part in parts.parts[:-1]:
        target = target / part
        require(not target.is_symlink(), "Symlink parent is forbidden")
        if create_parents and not target.exists():
            target.mkdir()
        require(target.is_dir(), "Expected parent directory is missing")
    target = target / parts.name
    require(not target.is_symlink(), "Symlink file is forbidden")
    require(target.resolve().is_relative_to(root.resolve()), "Path escaped root")
    return target



def metadata(path: Path) -> dict:
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            count += len(chunk)
    return {"sha256": digest.hexdigest(), "bytes": count}


def safe_relative(name: str) -> None:
    require(isinstance(name, str) and bool(re.fullmatch(r"[A-Za-z0-9_./-]+", name)),
            "Unexpected file-name characters")
    path = PurePosixPath(name)
    require(not path.is_absolute() and all(part not in ("", ".", "..", ".git")
            for part in name.split("/")), "Unsafe relative file path")


def valid_metadata(value: dict) -> None:
    require(isinstance(value, dict) and set(value) == {"sha256", "bytes"},
            "File metadata must contain exactly SHA256 and bytes")
    require(isinstance(value["sha256"], str) and
            re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is not None,
            "Invalid SHA256")
    require(type(value["bytes"]) is int and value["bytes"] >= 0, "Invalid byte count")


def checked_file(root: Path, name: str, expected: dict) -> Path:
    safe_relative(name)
    path = fixed_child(root, name)
    require(path.is_file() and metadata(path) == expected, "File byte identity mismatch: " + name)
    return path



def read_verified(root: Path, name: str, expected: dict) -> bytes:
    path = checked_file(root, name, expected)
    raw = path.read_bytes()
    require(len(raw) == expected["bytes"] and
            hashlib.sha256(raw).hexdigest() == expected["sha256"],
            "File changed while preparing publication bytes: " + name)
    return raw


def load_manifest(root: Path):
    require(re.fullmatch(r"[0-9a-f]{64}", MANIFEST_SHA256) is not None,
            "Publication template has no reviewed manifest digest")
    raw = fixed_child(root, MANIFEST_PATH).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == MANIFEST_SHA256,
            "Publication manifest differs from pinned bytes")
    value = json.loads(raw)
    require(value.get("schema") == "coefficient-publication-v2", "Unexpected publication schema")
    study = value.get("study", {})
    require(study.get("commit") == STUDY_COMMIT and
            study.get("frozen_design_commit") == DESIGN_COMMIT, "Different study/design")
    for name in ("run_id", "job_id"):
        require(type(study.get(name)) is int and study[name] > 0, "Missing original execution ID")
    require(study.get("run_attempt") == 1 and study.get("run_number") == 1,
            "Only the original execution may be packaged")
    require(study.get("classification") in CLASSIFICATIONS, "Unfrozen completeness classification")
    for name in ("workflow_conclusion", "job_conclusion"):
        require(study.get(name) in ("success", "failure", "cancelled", "timed_out"),
                "Original workflow/job conclusion is not fixed")
    artifact = value.get("artifact", {})
    require(type(artifact.get("id")) is int and artifact["id"] > 0, "Missing artifact ID")
    require(artifact.get("name") == f"coefficient-packet-{study['run_id']}-attempt-1",
            "Artifact name differs from original execution")
    require(type(artifact.get("size_in_bytes")) is int and artifact["size_in_bytes"] > 0,
            "Missing original artifact size")
    require(isinstance(artifact.get("digest"), str) and
            re.fullmatch(r"sha256:[0-9a-f]{64}", artifact["digest"]) is not None,
            "Missing original ZIP metadata digest")
    require(value.get("layout") == {"study_prefix": ""}, "Expected the original flat artifact")
    files, omitted = value.get("files", {}), value.get("omitted_raw_records", {})
    require(isinstance(files, dict) and files and "execution.json" in files,
            "Missing fixed original file inventory")
    for name, entry in files.items():
        safe_relative(name)
        valid_metadata(entry)
        require(name not in RESERVED_DESTINATIONS, "Original file collides with publication metadata")
    require(isinstance(omitted, dict) and not set(omitted).intersection(files),
            "Raw and published inventories overlap")
    for name, entry in omitted.items():
        require(name in ("diagnostics.jsonl", "trials.jsonl", "references.jsonl", "warmups.jsonl", "object_manifest.jsonl"), "Unapproved raw-file omission")
        require(set(entry) == {"raw", "gzip"} and entry["gzip"] == name + ".gz"
                and entry["gzip"] in files, "Omitted raw file lacks exact gzip")
        valid_metadata(entry["raw"])
    require(value.get("auditor", {}).get("path") == AUDITOR_PATH, "Unexpected auditor path")
    valid_metadata(value["auditor"]["file"])
    checked_file(root, AUDITOR_PATH, value["auditor"]["file"])
    require(value.get("execution_sha256") == files["execution.json"]["sha256"],
            "Recovered execution anchor differs from file inventory")
    return value, raw


def original_inventory(manifest):
    return {**manifest["files"],
            **{name: item["raw"] for name, item in manifest["omitted_raw_records"].items()}}


def verify_original(artifact_dir: Path, manifest: dict) -> dict:
    root = artifact_dir.absolute()
    require(root.is_dir() and not any(p.is_symlink() for p in (root, *root.parents)),
            "Original artifact directory is absent or symlinked")
    inventory = original_inventory(manifest)
    for name, entry in inventory.items():
        checked_file(root, name, entry)
    actual = set()
    for p in root.rglob("*"):
        require(not p.is_symlink(), "Symlink in original artifact")
        require(p.is_dir() or p.is_file(), "Nonregular original artifact entry")
        if p.is_file():
            actual.add(p.relative_to(root).as_posix())
    require(actual == set(inventory), "Original inventory is incomplete or changed")
    execution = json.loads((root / "execution.json").read_bytes())
    study = manifest["study"]
    require(execution.get("github_commit") == STUDY_COMMIT and
            str(execution.get("github_run_id")) == str(study["run_id"]) and
            execution.get("github_run_attempt") == 1 and execution.get("github_run_number") == 1,
            "Original execution identity mismatch")
    require(execution.get("frozen_design_commit") == DESIGN_COMMIT and
            execution.get("run_kind") == "preregistered_coefficient_packet_v2",
            "Original design/execution kind mismatch")
    require(execution.get("retained_files") ==
            {name: item for name, item in inventory.items() if name != "execution.json"},
            "Original wrapper inventory differs")
    for name, item in manifest["omitted_raw_records"].items():
        receipt = execution.get("compression_receipts", {}).get(name, {})
        require(receipt == {"raw": item["raw"], "gzip": manifest["files"][item["gzip"]],
                            "round_trip_verified": True}, "Raw/gzip receipt differs")
        h, count = hashlib.sha256(), 0
        with gzip.open(fixed_child(root, item["gzip"]), "rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                count += len(block)
                require(count <= item["raw"]["bytes"], "Gzip exceeds original raw size")
                h.update(block)
        require({"sha256": h.hexdigest(), "bytes": count} == item["raw"],
                "Gzip does not reproduce original raw bytes")
    return {"study_root": root, "execution": execution, "inventory": inventory}


def verify_source(root: Path, context: dict):
    launch = json.loads((context["study_root"] / "launch_source_manifest.json").read_bytes())
    require(launch.get("github_commit") == STUDY_COMMIT and
            launch.get("frozen_design_commit") == DESIGN_COMMIT,
            "Original launch source identity differs")
    expected_names = {
        "docs/coefficient_packet_design.md", "docs/alignment_geometry.md",
        "latent_bridge/__init__.py", "latent_bridge/alignment_core.py",
        "latent_bridge/alignment_packet.py", "scripts/run_alignment_numerical.py",
        "latent_bridge/coefficient_packet.py", "latent_bridge/coefficient_transport.py",
        "latent_bridge/coefficient_study.py", "scripts/run_coefficient_study.py",
        "tests/test_coefficient_packet.py", "tests/test_coefficient_transport.py",
        "tests/test_coefficient_execution.py", "tests/test_alignment_execution.py",
        ".github/workflows/coefficient-packet-v2.yml", "pyproject.toml",
    }
    require(set(launch.get("files", {})) == expected_names, "Original source file set differs")
    for name, item in launch["files"].items():
        valid_metadata(item)
        checked_file(root, name, item)
    return launch["files"]


def verify_audit(receipt_path: Path, manifest: dict, context: dict):
    require(receipt_path.is_file() and not receipt_path.is_symlink(), "Audit receipt is absent")
    require(not receipt_path.resolve().is_relative_to(context["study_root"].resolve()),
            "Audit receipt must stay outside original evidence")
    raw = receipt_path.read_bytes()
    receipt = json.loads(raw)
    require(receipt.get("audit_status") in ("completed", "completed_with_integrity_issues"),
            "Independent audit did not finish; retain its receipt without claiming completion")
    for key in ("new_fixture_generations", "model_imports_or_inference",
                "optimization_invocations", "timing_measurements"):
        require(receipt.get(key) == 0, "Audit performed unapproved fresh work")
    anchor = receipt.get("evidence", {}).get("execution", {})
    require(anchor.get("sha256") == manifest["execution_sha256"] and
            anchor.get("bytes") == manifest["files"]["execution.json"]["bytes"] and
            anchor.get("external_digest_supplied") is True and
            anchor.get("source_commit") == STUDY_COMMIT and
            str(anchor.get("run_id")) == str(manifest["study"]["run_id"]),
            "Audit inspected another or unanchored execution")
    runtime = receipt.get("audit_runtime", {})
    require(runtime.get("script_sha256") == manifest["auditor"]["file"]["sha256"],
            "Unexpected audit implementation")
    require(runtime.get("python") == "3.11.18" and runtime.get("numpy") == "2.4.6",
            "Unexpected independent audit runtime")
    require(Path(receipt.get("artifact_directory", "/missing")).resolve() ==
            context["study_root"].resolve(), "Audit inspected another directory")
    require(all(isinstance(receipt.get(key), dict) for key in
                ("data_integrity", "report_consistency", "scientific_validation")),
            "Audit outcome sections are missing")
    # An authentic negative mathematical result or consistency disagreement is
    # published as such, never filtered out by a success-only publication gate.
    return receipt, raw


def publication_names(manifest):
    return [DESTINATION + "/" + name for name in (
        *sorted(manifest["files"]), "audit_receipt.json",
        "publication_manifest.json", "publication_provenance.json")]


def verify_trackable(root: Path, manifest: dict) -> None:
    names = publication_names(manifest)
    result = subprocess.run(["git", "check-ignore", "--stdin"], cwd=root,
                            input=("\n".join(names) + "\n").encode(),
                            capture_output=True, timeout=30, check=False)
    require(result.returncode in (0, 1), "Git ignore preflight failed")
    require(not result.stdout, "One or more exact publication destinations remain ignored")
class NoRedirect(urllib.request.HTTPRedirectHandler):
    """Never forward the workflow token through a metadata redirect."""

    def redirect_request(self, request, file, code, message, headers, new_url):
        raise PublicationError("Unexpected GitHub API metadata redirect")


def github_json(path: str, token: str) -> dict:
    # ZIP download belongs exclusively to the official download-artifact action.
    # These metadata requests stay on the API host and never follow redirects.
    require(path.startswith("/actions/") and ".." not in path,
            "Unexpected GitHub API route")
    request = urllib.request.Request(
        "https://api.github.com/repos/" + REPOSITORY + path,
        headers={"Authorization": "Bearer " + token,
                 "Accept": "application/vnd.github+json",
                 "X-GitHub-Api-Version": "2022-11-28",
                 "User-Agent": "audited-readout-publication"},
    )
    try:
        with urllib.request.build_opener(NoRedirect()).open(request, timeout=30) as response:
            require(response.status == 200, "Unexpected GitHub metadata response")
            value = json.load(response)
    except urllib.error.HTTPError as error:
        raise PublicationError("GitHub metadata request failed: HTTP " + str(error.code)) from None
    except urllib.error.URLError:
        raise PublicationError("GitHub metadata request could not be completed") from None
    require(isinstance(value, dict), "Unexpected GitHub metadata structure")
    return value


def git(root: Path, *arguments: str, binary: bool = False):
    result = subprocess.run(
        ["git", *arguments], cwd=root, capture_output=True, timeout=120, check=False
    )
    if result.returncode != 0:
        # Do not print credentials, headers or remote error bodies.
        raise PublicationError("Git command failed without retry: " + arguments[0])
    return result.stdout if binary else result.stdout.decode("utf-8").strip()


def require_remote_unchanged(root: Path, expected: str) -> None:
    remote = git(root, "ls-remote", "--exit-code", "origin", "refs/heads/main")
    require(remote.splitlines() == [expected + "\trefs/heads/main"],
            "Main changed since this publication push; stop without merge, force or retry")




def verify_live_identity(token: str, manifest: dict) -> None:
    expected = manifest["artifact"]
    artifact = github_json("/actions/artifacts/" + str(expected["id"]), token)
    for name in ("id", "name", "size_in_bytes", "digest"):
        require(artifact.get(name) == expected[name], "Original artifact metadata mismatch: " + name)
    require(artifact.get("expired") is False, "Original artifact has expired")
    study = manifest["study"]
    parent = artifact.get("workflow_run", {})
    require(parent.get("id") == study["run_id"] and parent.get("head_sha") == STUDY_COMMIT and
            parent.get("repository_id") == REPOSITORY_ID and
            parent.get("head_repository_id") == REPOSITORY_ID and
            parent.get("head_branch") == "main", "Artifact points to another original execution")
    run = github_json("/actions/runs/" + str(study["run_id"]), token)
    require(run.get("id") == study["run_id"] and run.get("head_sha") == STUDY_COMMIT and
            run.get("repository", {}).get("id") == REPOSITORY_ID and run.get("head_branch") == "main" and
            run.get("status") == "completed" and run.get("run_attempt") == 1 and
            run.get("run_number") == 1 and run.get("conclusion") == study["workflow_conclusion"],
            "Original workflow identity or recorded conclusion differs")
    job = github_json("/actions/jobs/" + str(study["job_id"]), token)
    require(job.get("id") == study["job_id"] and job.get("run_id") == study["run_id"] and
            job.get("head_sha") == STUDY_COMMIT and job.get("status") == "completed" and
            job.get("conclusion") == study["job_conclusion"],
            "Original job identity or recorded conclusion differs")


def action_context(root: Path, artifact_dir: Path, receipt_path: Path):
    environment = os.environ
    require(environment.get("GITHUB_ACTIONS") == "true", "Publication requires GitHub Actions")
    require(environment.get("GITHUB_REPOSITORY") == REPOSITORY and
            environment.get("GITHUB_REPOSITORY_ID") == str(REPOSITORY_ID) and
            environment.get("GITHUB_SERVER_URL") == "https://github.com",
            "Unexpected publication repository/server")
    require(environment.get("GITHUB_EVENT_NAME") == "push" and
            environment.get("GITHUB_REF") == "refs/heads/main", "Publication requires a main push")
    expected = environment.get("GITHUB_SHA", "")
    require(re.fullmatch(r"[0-9a-f]{40}", expected) is not None, "Missing publication commit")
    require(Path(environment.get("GITHUB_WORKSPACE", "/missing")).absolute() == root,
            "Workspace does not match the publication repository")
    temporary = environment.get("RUNNER_TEMP", "")
    require(temporary and Path(temporary).is_absolute(), "Missing runner temporary directory")
    require(artifact_dir.absolute() == Path(temporary) / "coefficient-publication/artifact",
            "Unexpected extracted artifact path")
    expected_receipt = fixed_child(Path(temporary), "coefficient-publication/audit/coefficient_audit.json")
    require(receipt_path.absolute() == expected_receipt, "Unexpected independent audit output path")
    run_id = environment.get("GITHUB_RUN_ID", "")
    require(run_id.isdigit(), "Missing publication run ID")
    token = environment.get("GITHUB_TOKEN", "")
    require(bool(token), "Missing workflow token")
    require(Path(git(root, "rev-parse", "--show-toplevel")).absolute() == root and
            git(root, "rev-parse", "HEAD") == expected, "Checkout differs from publication GITHUB_SHA")
    urls = {"https://github.com/" + REPOSITORY, "https://github.com/" + REPOSITORY + ".git"}
    require(git(root, "remote", "get-url", "origin") in urls and
            git(root, "remote", "get-url", "--push", "origin") in urls, "Unexpected Git origin")
    require(not git(root, "status", "--porcelain=v1", "--untracked-files=all"),
            "Checkout must be clean before publishing results")
    return expected, run_id, token


def publish(root: Path, artifact_dir: Path, receipt_path: Path, manifest: dict, manifest_raw: bytes):
    expected, publication_run, token = action_context(root, artifact_dir, receipt_path)
    require(publication_run != str(manifest["study"]["run_id"]), "Original execution cannot package itself")
    verify_trackable(root, manifest)
    verify_live_identity(token, manifest)
    require_remote_unchanged(root, expected)
    context = verify_original(artifact_dir, manifest)
    source_files = verify_source(root, context)
    receipt, receipt_raw = verify_audit(receipt_path, manifest, context)
    payload = {name: read_verified(context["study_root"], name, item)
               for name, item in manifest["files"].items()}
    payload.update({"audit_receipt.json": receipt_raw, "publication_manifest.json": manifest_raw})
    provenance = {
        "kind": "packaging_original_coefficient_evidence_without_remeasurement",
        "study_commit": STUDY_COMMIT, "frozen_design_commit": DESIGN_COMMIT,
        "study_run_id": manifest["study"]["run_id"], "study_job_id": manifest["study"]["job_id"],
        "original_artifact": manifest["artifact"],
        "archive_zip_independently_rehashed_by_publisher": False,
        "published_study_files_independently_rehashed": True,
        "original_source_files_rehashed": source_files,
        "original_execution_sha256": manifest["execution_sha256"],
        "original_worker_returncode": context["execution"].get("worker_returncode"),
        "original_execution_failure": context["execution"].get("failure"),
        "classification": manifest["study"]["classification"],
        "audit_outcome": {key: receipt[key] for key in
                         ("audit_status", "data_integrity", "report_consistency", "scientific_validation")},
        "audit_receipt": {"sha256": hashlib.sha256(receipt_raw).hexdigest(), "bytes": len(receipt_raw)},
        "audit_script": manifest["auditor"],
        "publication_configuration_commit": expected, "publication_run_id": publication_run,
        "fixed_publication_manifest_sha256": MANIFEST_SHA256,
        "original_file_bytes_preserved": True,
        "omitted_raw_records": manifest["omitted_raw_records"],
        "omission_rule": "Only an independently rehashed exact gzip replaces a redundant raw file in Git; raw bytes remain in the original artifact.",
        "artifact_retention_days": context["execution"].get("artifact_retention_days"),
        "models_imported_or_executed": 0, "study_grid_invocations": 0, "timing_remeasurements": 0,
    }
    payload["publication_provenance.json"] = (
        json.dumps(provenance, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    destinations = []
    for name, content in payload.items():
        relative = DESTINATION + "/" + name
        target = fixed_child(root, relative, create_parents=True)
        if target.exists():
            require(target.is_file() and target.read_bytes() == content,
                    "Publication cannot overwrite different original evidence: " + relative)
        else:
            target.write_bytes(content)
        destinations.append(relative)
    require(set(destinations) == set(publication_names(manifest)), "Unexpected publication payload")
    git(root, "add", "--", *destinations)
    staged = git(root, "diff", "--cached", "--name-only", "-z", binary=True)
    names = {part.decode("utf-8") for part in staged.split(b"\0") if part}
    require(names.issubset(set(destinations)), "Unexpected staged path")
    for name, content in payload.items():
        indexed = git(root, "show", ":" + DESTINATION + "/" + name, binary=True)
        require(len(indexed) == len(content) and hashlib.sha256(indexed).digest() ==
                hashlib.sha256(content).digest(), "Git index altered verified bytes: " + name)
    require_remote_unchanged(root, expected)
    if not names:
        return {"status": "already_published", "commit": expected, "files": len(payload)}
    git(root, "-c", "user.name=github-actions[bot]",
        "-c", "user.email=41898282+github-actions[bot]@users.noreply.github.com",
        "commit", "-m", "Publish original coefficient packet evidence and read-only audit [skip ci]")
    commit = git(root, "rev-parse", "HEAD")
    require(git(root, "rev-parse", "HEAD^") == expected, "Unexpected publication parent")
    require_remote_unchanged(root, expected)
    git(root, "push", "origin", commit + ":refs/heads/main")
    return {"status": "published", "commit": commit, "parent": expected,
            "classification": manifest["study"]["classification"],
            "files": len(payload), "bytes": sum(map(len, payload.values())),
            "study_run": manifest["study"]["run_id"], "artifact_id": manifest["artifact"]["id"],
            "audit_status": receipt["audit_status"],
            "required_checks_failed":
                receipt["scientific_validation"]["required_checks_failed"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--metadata", action="store_true")
    modes.add_argument("--check-only", action="store_true")
    modes.add_argument("--publish", action="store_true")
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--audit-receipt", type=Path)
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        manifest, raw = load_manifest(root)
        if args.metadata:
            require(args.artifact_dir is None and args.audit_receipt is None,
                    "Metadata mode takes no artifact inputs")
            verify_trackable(root, manifest)
            result = {"study_run_id": manifest["study"]["run_id"],
                      "artifact_name": manifest["artifact"]["name"],
                      "artifact_id": manifest["artifact"]["id"],
                      "study_prefix": manifest["layout"]["study_prefix"],
                      "classification": manifest["study"]["classification"],
                      "execution_sha256": manifest["execution_sha256"],
                      "original_files": len(manifest["files"])}
            if args.github_output is not None:
                require(os.environ.get("GITHUB_ACTIONS") == "true" and
                        str(args.github_output) == os.environ.get("GITHUB_OUTPUT"),
                        "Metadata output must be the current action's output file")
                with args.github_output.open("a", encoding="utf-8") as stream:
                    for key, value in result.items():
                        text = str(value)
                        require("\n" not in text and "\r" not in text, "Unsafe action output")
                        stream.write(key + "=" + text + "\n")
        else:
            require(args.artifact_dir is not None and args.github_output is None,
                    "Artifact directory is required")
            if args.publish:
                require(args.audit_receipt is not None, "Publication requires an independent receipt")
                result = publish(root, args.artifact_dir.absolute(), args.audit_receipt.absolute(), manifest, raw)
            else:
                require(args.audit_receipt is None, "Check-only precedes the independent audit")
                context = verify_original(args.artifact_dir.absolute(), manifest)
                result = {"status": "original_bytes_verified",
                          "original_inventory_files": len(context["inventory"]),
                          "original_inventory_bytes": sum(entry["bytes"] for entry in context["inventory"].values()),
                          "study_run_id": manifest["study"]["run_id"],
                          "original_worker_returncode": context["execution"].get("worker_returncode")}
        print(json.dumps(result, sort_keys=True))
    except (PublicationError, OSError, ValueError, KeyError) as error:
        raise SystemExit("Coefficient publication stopped: " + str(error)) from None


if __name__ == "__main__":
    main()
