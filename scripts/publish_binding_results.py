"""Publish the fixed original binding artifact after an independent audit.

The immutable publication manifest pins original run identities and file bytes.
This helper never fits, generates with, or rescores a pretrained model.
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
STUDY_COMMIT = "3cfbf2b02f70235f089d86d429da33ee818818d8"
DESIGN_COMMIT = "dfce06b5dfb2860414af166ef4441eefd9548afc"
DESTINATION = "results/binding_study"
MANIFEST_PATH = "scripts/binding_publication_manifest.json"
MANIFEST_SHA256 = "f68e4e9d509c7eea65a1d4043d39cf83fd64de124f7cbe450d402ce8a6dc9b09"
AUDITOR_PATH = "scripts/audit_binding_outputs.py"
RESERVED_DESTINATIONS = {
    "audit_receipt.json", "publication_manifest.json", "publication_provenance.json",
    "binding-study-tests.log",
}
CLASSIFICATIONS = {"complete_study", "valid_receiver_limited_stop", "incomplete_study"}

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
            "Publication template is disabled: no reviewed manifest digest is pinned")
    path = fixed_child(root, MANIFEST_PATH)
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == MANIFEST_SHA256,
            "Publication manifest differs from its pinned bytes")
    value = json.loads(raw)
    require(value.get("schema") == "binding-publication-v1", "Unexpected publication schema")
    study = value.get("study", {})
    require(study.get("commit") == STUDY_COMMIT and study.get("frozen_design_commit") == DESIGN_COMMIT,
            "Publication targets a different study/design")
    for name in ("run_id", "job_id"):
        require(type(study.get(name)) is int and study[name] > 0, "Missing fixed study identifier")
    require(study.get("run_attempt") == 1 and study.get("run_number") == 1,
            "Publication must refer to the single original empirical execution")
    require(study.get("classification") in CLASSIFICATIONS, "Outcome classification is not frozen")
    for name in ("workflow_conclusion", "job_conclusion"):
        require(study.get(name) in ("success", "failure", "cancelled", "timed_out"),
                "Original workflow/job conclusion is not fixed")
    artifact = value.get("artifact", {})
    require(type(artifact.get("id")) is int and artifact["id"] > 0, "Missing fixed artifact ID")
    require(artifact.get("name") == f"binding-study-{study['run_id']}-attempt-1",
            "Artifact name does not identify the original study")
    require(type(artifact.get("size_in_bytes")) is int and artifact["size_in_bytes"] > 0,
            "Missing original artifact size")
    require(isinstance(artifact.get("digest"), str) and
            re.fullmatch(r"sha256:[0-9a-f]{64}", artifact["digest"]) is not None,
            "Missing original ZIP digest")
    layout = value.get("layout", {})
    require(layout.get("study_prefix") in ("", "checkpoints/binding-study"),
            "Unapproved artifact layout")
    require(layout.get("test_log") == "binding-study-tests.log", "Unexpected original test-log path")
    files = value.get("files", {})
    require(isinstance(files, dict) and files and "execution.json" in files,
            "Fixed original file allowlist is missing")
    for name, entry in files.items():
        safe_relative(name)
        valid_metadata(entry)
        require(name not in RESERVED_DESTINATIONS, "Original file collides with publication metadata")
    omitted = value.get("omitted_raw_predictions", {})
    require(isinstance(omitted, dict) and not set(omitted).intersection(files),
            "Omitted/raw allowlist overlaps")
    for name, entry in omitted.items():
        require(name in ("predictions.jsonl", "validation_predictions.jsonl"),
                "Only verified redundant raw predictions may be omitted")
        require(set(entry) == {"raw", "gzip"} and entry["gzip"] == name + ".gz" and
                entry["gzip"] in files, "Omitted raw file lacks its published gzip")
        valid_metadata(entry["raw"])
    require(value.get("test_log_policy") == "retained_original_archive_only",
            "The unhashed top-level test log must remain outside the Git payload")
    valid_metadata(value.get("auditor", {}).get("file", {}))
    require(value.get("auditor", {}).get("path") == AUDITOR_PATH, "Unexpected auditor path")
    require(value.get("execution_sha256") == files["execution.json"]["sha256"],
            "Independently recovered execution digest differs from allowlist")
    checked_file(root, AUDITOR_PATH, value["auditor"]["file"])
    return value, raw


def original_inventory(manifest):
    return {**manifest["files"],
            **{name: entry["raw"] for name, entry in manifest["omitted_raw_predictions"].items()}}


def artifact_study_root(artifact_dir: Path, manifest: dict) -> Path:
    prefix = manifest["layout"]["study_prefix"]
    if not prefix:
        require(artifact_dir.is_dir(), "Artifact directory is absent")
        require(not any(p.is_symlink() for p in (artifact_dir, *artifact_dir.parents)),
                "Symlink artifact directory")
        return artifact_dir
    path = fixed_child(artifact_dir, prefix)
    require(path.is_dir(), "Fixed study subdirectory is absent")
    return path


def verify_original(artifact_dir: Path, manifest: dict) -> dict:
    artifact_dir = artifact_dir.absolute()
    study_root = artifact_study_root(artifact_dir.absolute(), manifest)
    inventory = original_inventory(manifest)
    for name, entry in inventory.items():
        checked_file(study_root, name, entry)
    log = fixed_child(artifact_dir, manifest["layout"]["test_log"])
    require(log.is_file(), "Original top-level test log is absent")
    actual = set()
    for path in study_root.rglob("*"):
        require(not path.is_symlink(), "Symlink in original study tree")
        require(path.is_dir() or path.is_file(), "Nonregular artifact entry")
        if path.is_file():
            relative = path.relative_to(study_root).as_posix()
            if study_root == artifact_dir and path == log:
                continue
            actual.add(relative)
    require(actual == set(inventory), "Original artifact inventory is incomplete or changed")
    prefix = manifest["layout"]["study_prefix"]
    expected_paths = {(prefix + "/" if prefix else "") + name for name in inventory}
    expected_paths.add(manifest["layout"]["test_log"])
    expected_directories = set()
    for name in expected_paths:
        parts = PurePosixPath(name).parts
        for count in range(1, len(parts)):
            expected_directories.add("/".join(parts[:count]))
    for path in artifact_dir.rglob("*"):
        require(not path.is_symlink(), "Symlink in artifact root")
        relative = path.relative_to(artifact_dir).as_posix()
        if path.is_file():
            require(relative in expected_paths, "Unexpected file outside the exact original inventory")
        else:
            require(path.is_dir() and relative in expected_directories,
                    "Unexpected directory or special file in artifact root")
    execution = json.loads((study_root / "execution.json").read_bytes())
    study = manifest["study"]
    require(execution.get("github_commit") == STUDY_COMMIT and
            str(execution.get("github_run_id")) == str(study["run_id"]) and
            str(execution.get("github_run_attempt")) == "1" and
            str(execution.get("github_run_number")) == "1",
            "Original execution identity mismatch")
    require(execution.get("frozen_design_commit") == DESIGN_COMMIT and
            execution.get("run_kind") == "preregistered_binding_auxiliary_loss_study",
            "Original execution kind/design mismatch")
    require(execution.get("retained_files") ==
            {name: entry for name, entry in inventory.items() if name != "execution.json"},
            "Wrapper retained-file inventory differs from the fixed artifact")
    receipts = execution.get("prediction_receipts", {})
    for name, entry in manifest["omitted_raw_predictions"].items():
        expected_raw = entry["raw"]
        receipt = receipts.get(name, {})
        require(receipt.get("round_trip_verified") is True and
                receipt.get("raw") == expected_raw and
                receipt.get("gzip") == manifest["files"][entry["gzip"]],
                "Wrapper did not verify the omitted prediction representation")
        digest, count = hashlib.sha256(), 0
        compressed_path = fixed_child(study_root, entry["gzip"])
        with gzip.open(compressed_path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                count += len(chunk)
                require(count <= expected_raw["bytes"], "Compressed predictions exceed recorded raw size")
                digest.update(chunk)
        require(count == expected_raw["bytes"] and digest.hexdigest() == expected_raw["sha256"],
                "Published gzip does not reproduce the exact retained raw predictions")
    return {"study_root": study_root, "execution": execution, "inventory": inventory}


def verify_audit(receipt_path: Path, manifest: dict, context: dict):
    require(receipt_path.is_file() and not receipt_path.is_symlink(), "Audit receipt is missing")
    require(not receipt_path.resolve().is_relative_to(context["study_root"].resolve()),
            "Audit receipt must stay outside the immutable original artifact")
    raw = receipt_path.read_bytes()
    receipt = json.loads(raw)
    require(receipt.get("audit_status") == "passed" and
            type(receipt.get("error_count")) is int and receipt["error_count"] == 0,
            "Independent consistency audit did not pass")
    classification = receipt.get("classification")
    require(classification == manifest["study"]["classification"] and classification in CLASSIFICATIONS,
            "Audit classification differs from the reviewed publication manifest")
    require(receipt.get("primary_complete_study_claim_permitted") is (classification == "complete_study"),
            "Incomplete evidence was promoted to a complete-study claim")
    require(receipt.get("benchmark_invocations") == 0 and
            receipt.get("model_imports_or_inference") == 0,
            "Packaging audit must contain no model execution")
    provenance = receipt.get("evidence", {}).get("provenance", {})
    require(provenance.get("scientific_source_rehashed") is True and
            provenance.get("source_commit") == STUDY_COMMIT and
            str(provenance.get("run_id")) == str(manifest["study"]["run_id"]) and
            provenance.get("frozen_design_commit") == DESIGN_COMMIT,
            "Audit source/run verification differs")
    require(receipt.get("evidence", {}).get("retention", {}).get("inventory") == context["inventory"],
            "Independent audit did not bind every original study file")
    require(receipt.get("audit_runtime", {}).get("script_sha256") ==
            manifest["auditor"]["file"]["sha256"], "Unexpected independent audit implementation")
    require(receipt.get("audit_runtime", {}).get("python") == "3.11.18" and
            receipt.get("audit_runtime", {}).get("numpy") == "2.4.6", "Unexpected audit runtime")
    require(Path(receipt.get("artifact_directory", "/missing")).resolve() ==
            context["study_root"].resolve(), "Audit inspected another directory")
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
    require(artifact_dir.absolute() == Path(temporary) / "binding-publication/artifact",
            "Unexpected extracted artifact path")
    expected_receipt = fixed_child(Path(temporary), "binding-publication/audit/binding_audit.json")
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
    require(publication_run != str(manifest["study"]["run_id"]), "Packaging cannot be the original model run")
    verify_trackable(root, manifest)
    verify_live_identity(token, manifest)
    require_remote_unchanged(root, expected)
    context = verify_original(artifact_dir, manifest)
    receipt, receipt_raw = verify_audit(receipt_path, manifest, context)
    original = {}
    for name, entry in manifest["files"].items():
        original[name] = read_verified(context["study_root"], name, entry)
    payload = {
        **original, "audit_receipt.json": receipt_raw, "publication_manifest.json": manifest_raw,
    }
    provenance = {
        "kind": "packaging_existing_binding_evidence_without_model_execution",
        "study_commit": STUDY_COMMIT, "frozen_design_commit": DESIGN_COMMIT,
        "study_run_id": manifest["study"]["run_id"], "study_job_id": manifest["study"]["job_id"],
        "original_artifact": manifest["artifact"],
        "archive_zip_independently_rehashed_by_publisher": False,
        "published_study_files_independently_rehashed": True,
        "top_level_test_log": {
            "artifact_path": manifest["layout"]["test_log"], "published_to_git": False,
            "retained_in_original_fixed_id_archive": True,
            "independent_file_hash_claimed": False,
        },
        "original_execution_sha256": manifest["execution_sha256"],
        "original_execution_status": context["execution"].get("status"),
        "original_worker_returncode": context["execution"].get("worker_returncode"),
        "original_execution_failure": context["execution"].get("failure"),
        "original_scientific_status": receipt.get("declared_scientific_status"),
        "classification": receipt["classification"],
        "primary_complete_study_claim_permitted": receipt["primary_complete_study_claim_permitted"],
        "audit_receipt": {"sha256": hashlib.sha256(receipt_raw).hexdigest(), "bytes": len(receipt_raw)},
        "audit_script": manifest["auditor"],
        "publication_configuration_commit": expected, "publication_run_id": publication_run,
        "fixed_publication_manifest_sha256": MANIFEST_SHA256,
        "original_file_bytes_preserved": True,
        "omitted_raw_predictions": manifest["omitted_raw_predictions"],
        "omission_rule": "Only an independently rehashed gzip of identical raw bytes replaces a duplicate in Git; raw bytes remain in the original artifact.",
        "artifact_retention_days": context["execution"].get("artifact_retention_days"),
        "models_imported_or_executed": 0, "benchmark_invocations": 0,
    }
    payload["publication_provenance.json"] = (
        json.dumps(provenance, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    destinations = []
    for name, content in payload.items():
        relative = DESTINATION + "/" + name
        target = fixed_child(root, relative, create_parents=True)
        if target.exists():
            require(target.is_file() and target.read_bytes() == content,
                    "Different result already exists; publication never overwrites: " + relative)
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
        require(len(indexed) == len(content) and
                hashlib.sha256(indexed).digest() == hashlib.sha256(content).digest(),
                "Git index altered verified publication bytes: " + name)
    require_remote_unchanged(root, expected)
    if not names:
        return {"status": "already_published", "commit": expected,
                "classification": receipt["classification"], "files": len(payload)}
    git(root, "-c", "user.name=github-actions[bot]",
        "-c", "user.email=41898282+github-actions[bot]@users.noreply.github.com",
        "commit", "-m", "Publish audited binding-study evidence")
    commit = git(root, "rev-parse", "HEAD")
    require(git(root, "rev-parse", "HEAD^") == expected, "Unexpected publication parent")
    require_remote_unchanged(root, expected)
    # One normal push, without force, merge/rebase, fallback branches or retries.
    git(root, "push", "origin", commit + ":refs/heads/main")
    return {"status": "published", "commit": commit, "parent": expected,
            "classification": receipt["classification"],
            "primary_complete_study_claim_permitted": receipt["primary_complete_study_claim_permitted"],
            "files": len(payload), "bytes": sum(map(len, payload.values())),
            "study_run": manifest["study"]["run_id"], "artifact_id": manifest["artifact"]["id"]}


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
                          "original_execution_status": context["execution"].get("status")}
        print(json.dumps(result, sort_keys=True))
    except (PublicationError, OSError, ValueError, KeyError) as error:
        raise SystemExit("Binding publication stopped: " + str(error)) from None


if __name__ == "__main__":
    main()
