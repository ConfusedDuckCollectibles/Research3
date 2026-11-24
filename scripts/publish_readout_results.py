"""Publish fixed audited readout files without model execution.

--check-only verifies extracted bytes without Git or network changes.
--publish is restricted to the expected GitHub Actions push context.
"""
from __future__ import annotations
import argparse
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
STUDY_RUN = 45335536056
STUDY_JOB = 138246603169
STUDY_COMMIT = "9eb4478fae1ad373234eb2a725345b66c8c63c57"
HISTORICAL_COMMIT = "fb793ba31d7a10975a9e5c6032b44ecb5eb5774f"
ARTIFACT_ID = 14893873430
ARTIFACT_NAME = "readout-study-complete"
ARTIFACT_BYTES = 29234999
ARTIFACT_DIGEST = "sha256:4e0f4a6499563ad72875f4e0a50e58f54659b57a1d77e93202e3823bf0044b56"
DESIGN_SHA256 = "763f746970d7cca098ba468e7996517875698e5b8b2245e555159909c99e03e8"
DESTINATION = "results/readout_study"
EXPECTED_PACKAGES = {
    "torch": "2.6.0+cpu", "transformers": "4.48.2", "numpy": "2.4.6",
    "tokenizers": "0.21.4", "safetensors": "0.8.0", "huggingface-hub": "0.36.2",
}
# Values are [SHA256, byte count]. No artifact-provided path or wildcard
# determines publication. Raw predictions and every .pt file are excluded.
FILES = {
    "bridge_manifest.json": [
        "f749057fc899462745115030c0e98e2c464d5490a9f0fe621336e49cfdf180be",
        3258
    ],
    "dataset_manifest.json": [
        "f5f2bad7af515980cf822d5e0c57a29890de2136ccce42e04609b58cb2876f54",
        161798
    ],
    "execution.json": [
        "45f7bf568dbf63b7f6d2b5b059db0e039eb46f0a4c1c8072e875f0092f4f632a",
        22935
    ],
    "feature_manifest.json": [
        "c354e6482d9fdba24193a14f4afe5618f99fcf783dcf4ddacc8aa6955b828f79",
        740988
    ],
    "fit_records.json": [
        "ed29084b3586c4a12989d3715ae484ca1e212d8e78c254fb50cc2c1a90d0116f",
        689378
    ],
    "model_manifest.json": [
        "280de0ecf2d2f7cc7fe8e3fc231455b595cf4ad34b064b5d09eb7b642f95c465",
        2954
    ],
    "normalization.npz": [
        "b07b73f57bdf413fd5372935ec6f72dce1ac46cd5c4041ee53b38903d5fee773",
        392591
    ],
    "predictions.jsonl.gz": [
        "46b3e017ed973ea5f485980d62137d1d28313769dbca898891494073892747eb",
        2788581
    ],
    "reconstruction_gate.json": [
        "3c6a84c2d9c41f4665e3d8ebba90cdcb84fe7c2a3fa17073965849c112b5a1ec",
        2313
    ],
    "report.md": [
        "4e993d40aa75fad38e78cc2db68c4eef237a5475bd74d2ab14e8d46ef987e285",
        13565
    ],
    "results.json": [
        "f5902ae8ea3910cbace25e889a40e69f96e5e071e14047e6e0ce1bda79c39b1d",
        1265836
    ],
    "runtime_manifest.json": [
        "28e69fa4b1b99fff28f22c4b5e993f242e73576965600fe164a4ce9c59f3d312",
        607
    ],
    "secondary_trigger.json": [
        "9c659a9548df304c6ddb82510a86d92ba51b696e638870ee58efcfd4e10a14c4",
        271
    ],
    "selection_frozen.json": [
        "a48b3bcd97d69a7116066dcf4d7bc9853f5c8a4658a3d9c77ee5e7802d3e8f21",
        8485
    ],
    "source_manifest.json": [
        "fe559f3ceb4f8b4919a34fc15e9148c2a42c07f5e9d32b3ed0bb5e4247667cb9",
        2700
    ],
    "training_history.jsonl": [
        "7140afdf32bb6368e8a4760bcce3b7bc78cb8d51dc259772af13c0753ac7114e",
        470391
    ],
    "exports/bridge_weights.npz": [
        "b0cdf5a5b0bc66eb99711e3ffb6d79ef9c384fe436d816109637e323437c0b91",
        603752
    ],
    "exports/message_features.npz": [
        "5c8000c16d6845aa66620662282e4219e144a0901d441b2a63f69e160b8ac034",
        865126
    ],
    "exports/small_probe_weights.npz": [
        "b6e9935c64a4796e0a82616404bc13ef9064e5d49b26182d18f5e18c5a44067d",
        326575
    ]
}


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


def verified_bytes(root: Path, name: str) -> bytes:
    target = fixed_child(root, name)
    require(target.is_file(), "Missing regular file: " + name)
    digest, byte_count = FILES[name]
    require(target.stat().st_size == byte_count, "Byte count mismatch: " + name)
    content = target.read_bytes()
    require(len(content) == byte_count and hashlib.sha256(content).hexdigest() == digest,
            "SHA256 mismatch: " + name)
    return content


def verify_artifact(artifact_dir: Path) -> dict[str, bytes]:
    contents = {name: verified_bytes(artifact_dir, name) for name in FILES}
    parsed = {name: json.loads(value) for name, value in contents.items()
              if name.endswith(".json")}
    execution = parsed["execution.json"]
    runtime = parsed["runtime_manifest.json"]
    model = parsed["model_manifest.json"]
    result = parsed["results.json"]
    gate = parsed["reconstruction_gate.json"]
    for name, record in (("execution", execution), ("runtime", runtime), ("model", model)):
        require(record.get("github_commit") == STUDY_COMMIT,
                name + " source commit does not match the audited study")
        require(str(record.get("github_run_id")) == str(STUDY_RUN),
                name + " run does not match the audited study")
    require(execution.get("status") == "completed" and execution.get("failure") is None,
            "Study did not complete successfully")
    require(execution.get("historical_training_commit") == HISTORICAL_COMMIT,
            "Historical training source mismatch")
    require(execution.get("design_sha256") == DESIGN_SHA256 and
            model.get("design_sha256") == DESIGN_SHA256, "Frozen design mismatch")
    require(runtime.get("python") == "3.11.18" and
            runtime.get("expected_python") == "3.11.18", "Reconstruction Python mismatch")
    require(runtime.get("actual_packages") == EXPECTED_PACKAGES and
            runtime.get("expected_packages") == EXPECTED_PACKAGES,
            "Reconstruction package versions mismatch")
    source = parsed["source_manifest.json"]
    require(source.get("frozen_commit") == HISTORICAL_COMMIT and
            source.get("all_match") is True, "Historical source verification failed")
    require(gate.get("passed") is True and result.get("reconstruction_gate") == gate,
            "Reconstruction gate failed or differs from results")
    require(result.get("execution_status") == "executed_frozen_message_readout_diagnostic",
            "Unexpected result type")
    require(result.get("selection") == parsed["selection_frozen.json"] and
            result.get("fits") == parsed["fit_records.json"],
            "Selection or fit record mismatch")
    require(result.get("data_manifest") == parsed["dataset_manifest.json"] and
            result.get("download_manifest") == model, "Data or model manifest mismatch")
    integrity = result.get("prediction_integrity", {})
    require(integrity.get("records") == 170496 and
            integrity.get("raw_bytes") == 42545555 and
            integrity.get("raw_sha256") ==
            "3468fdb14ee9a6e25870bf398bfc17f49717f025bc297a9f7a988a4972bbf1f3",
            "Prediction identity mismatch")
    retained = execution.get("retained_files", {})
    for name, (digest, byte_count) in FILES.items():
        if name == "execution.json":
            continue  # The execution manifest excludes its own bytes.
        require(retained.get(name) == {"sha256": digest, "bytes": byte_count},
                "Execution file manifest mismatch: " + name)
    return contents


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


def verify_live_identity(token: str) -> None:
    artifact = github_json("/actions/artifacts/" + str(ARTIFACT_ID), token)
    for key, expected in (
        ("id", ARTIFACT_ID), ("name", ARTIFACT_NAME),
        ("size_in_bytes", ARTIFACT_BYTES), ("digest", ARTIFACT_DIGEST),
        ("expired", False),
    ):
        require(artifact.get(key) == expected, "Artifact metadata mismatch: " + key)
    parent = artifact.get("workflow_run", {})
    require(parent.get("id") == STUDY_RUN and parent.get("head_sha") == STUDY_COMMIT and
            parent.get("repository_id") == REPOSITORY_ID and
            parent.get("head_repository_id") == REPOSITORY_ID and
            parent.get("head_branch") == "main", "Artifact study identity mismatch")
    run = github_json("/actions/runs/" + str(STUDY_RUN), token)
    require(run.get("id") == STUDY_RUN and run.get("head_sha") == STUDY_COMMIT and
            run.get("head_branch") == "main" and
            run.get("repository", {}).get("id") == REPOSITORY_ID and
            run.get("status") == "completed" and run.get("conclusion") == "success",
            "Audited workflow identity/status mismatch")


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


def publish(root: Path, artifact_dir: Path, contents: dict[str, bytes]) -> dict:
    environment = os.environ
    require(environment.get("GITHUB_ACTIONS") == "true", "Publication requires GitHub Actions")
    require(environment.get("GITHUB_REPOSITORY") == REPOSITORY and
            environment.get("GITHUB_SERVER_URL") == "https://github.com",
            "Unexpected publication repository/server")
    require(environment.get("GITHUB_EVENT_NAME") == "push" and
            environment.get("GITHUB_REF") == "refs/heads/main",
            "Publication requires a main push")
    expected = environment.get("GITHUB_SHA", "")
    require(re.fullmatch(r"[0-9a-f]{40}", expected) is not None, "Missing publication commit")
    publication_run = environment.get("GITHUB_RUN_ID", "")
    require(publication_run.isdigit() and publication_run != str(STUDY_RUN),
            "Publication must be a distinct workflow run")
    require(Path(environment.get("GITHUB_WORKSPACE", "/missing")).absolute() == root,
            "Workspace does not match the script's repository root")
    runner_temp = environment.get("RUNNER_TEMP", "")
    require(runner_temp and Path(runner_temp).is_absolute(), "Missing runner temporary directory")
    expected_artifact = Path(runner_temp) / "readout-publication" / "artifact"
    require(artifact_dir.absolute() == expected_artifact.absolute(),
            "Publication input must be the action's fixed artifact directory")
    token = environment.get("GITHUB_TOKEN", "")
    require(bool(token), "Missing workflow token")
    verify_live_identity(token)
    require(Path(git(root, "rev-parse", "--show-toplevel")).absolute() == root and
            git(root, "rev-parse", "HEAD") == expected, "Checkout differs from GITHUB_SHA")
    expected_urls = {
        "https://github.com/" + REPOSITORY,
        "https://github.com/" + REPOSITORY + ".git",
    }
    require(git(root, "remote", "get-url", "origin") in expected_urls and
            git(root, "remote", "get-url", "--push", "origin") in expected_urls,
            "Unexpected Git origin")
    require(not git(root, "status", "--porcelain=v1", "--untracked-files=all"),
            "Checkout must be clean before publication")
    require_remote_unchanged(root, expected)
    destination_names = []
    for name in FILES:
        content = verified_bytes(artifact_dir, name)
        require(content == contents[name], "Artifact changed during verification")
        destination_name = DESTINATION + "/" + name
        target = fixed_child(root, destination_name, create_parents=True)
        if target.exists():
            require(target.is_file() and target.read_bytes() == content,
                    "Different destination already exists: " + destination_name)
        else:
            target.write_bytes(content)
        destination_names.append(destination_name)
    git(root, "add", "--", *destination_names)
    staged = git(root, "diff", "--cached", "--name-only", "-z", binary=True)
    staged_names = {part.decode("utf-8") for part in staged.split(b"\0") if part}
    require(staged_names.issubset(set(destination_names)), "Unexpected staged destination")
    for name, (digest, byte_count) in FILES.items():
        indexed = git(root, "show", ":" + DESTINATION + "/" + name, binary=True)
        require(len(indexed) == byte_count and hashlib.sha256(indexed).hexdigest() == digest,
                "Git index changed artifact bytes: " + name)
    require_remote_unchanged(root, expected)
    if not staged_names:
        return {"status": "already_published", "files": len(FILES), "commit": expected}
    git(root, "-c", "user.name=github-actions[bot]",
        "-c", "user.email=41898282+github-actions[bot]@users.noreply.github.com",
        "commit", "-m", "Publish audited frozen-message readout results")
    commit = git(root, "rev-parse", "HEAD")
    require(git(root, "rev-parse", "HEAD^") == expected, "Unexpected publication parent")
    require_remote_unchanged(root, expected)
    # One normal push: concurrent main advances are rejected. No force, merge,
    # rebase, branch fallback or automatic retry is permitted.
    git(root, "push", "origin", commit + ":refs/heads/main")
    return {"status": "published", "files": len(FILES),
            "bytes": sum(v[1] for v in FILES.values()), "commit": commit,
            "parent": expected, "study_run": STUDY_RUN}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check-only", action="store_true",
                       help="Verify extracted bytes; do not access Git or network")
    modes.add_argument("--publish", action="store_true",
                       help="Make one guarded Actions-only commit/push")
    args = parser.parse_args()
    try:
        contents = verify_artifact(args.artifact_dir)
        if args.publish:
            result = publish(Path(__file__).resolve().parents[1], args.artifact_dir, contents)
        else:
            result = {"status": "verified", "files": len(FILES),
                      "bytes": sum(v[1] for v in FILES.values()), "study_run": STUDY_RUN,
                      "study_commit": STUDY_COMMIT, "artifact_id": ARTIFACT_ID}
        print(json.dumps(result, sort_keys=True))
    except (PublicationError, OSError, ValueError, KeyError) as error:
        raise SystemExit("Publication stopped: " + str(error)) from None


if __name__ == "__main__":
    main()
