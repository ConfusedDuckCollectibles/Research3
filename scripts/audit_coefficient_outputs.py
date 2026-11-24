#!/usr/bin/env python3
"""Read-only independent audit of coefficient-packet v2 retained evidence.

Does not import coefficient_packet, coefficient_study, or coefficient_transport.
No fixtures, model inference, optimization, or latency measurements are run.
Only a SHA-pinned frozen dense numerical reference may be imported, for the nine
full-rank primary inputs already retained. Other replay is implemented below.
"""
from __future__ import annotations
import argparse
from collections import Counter
import gzip
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path, PurePosixPath
import platform
import statistics
import struct
import sys
import traceback

import numpy as np

SOURCE_COMMIT = "5e158fb8ecd9e4f70f532aeb93cc253bbea7741f"
DESIGN_COMMIT = "49cfba7261326d58e174e51c8ddf5abba6f4a966"
DESIGN_SHA256 = "b8307677f5320c3413b5a1007bdfe26230a844c57503d53b4443e5fc542647ba"
CORE_SHA256 = "de8909692a928266c03e22d12b13c800b14e2495a8acad39ae6b95dfc46bfa53"
ORIGINAL_PACKET_SHA256 = "6ae2c169b843007dd9793894676d8ce4a933c5f82fd2ce4063b83dea389c74fd"
ARMS = ("coefficient", "dual_direct", "thin_direct", "gram_guard", "gram_original", "token_lookup")
COMPARATORS = ("dual_direct", "thin_direct", "gram_guard")
ALL_COMPARATORS = COMPARATORS + ("gram_original",)
SHAPES = ((12, 192), (36, 512), (60, 960))
SEEDS = (17011, 17027, 17041)
PROFILE = {"loopback": (None, 0.0), "moderate": (10_000_000.0, .005),
           "constrained": (1_000_000.0, .020)}
JSONL_NAMES = ("diagnostics", "trials", "references", "warmups", "object_manifest")
WIRE = struct.Struct("<4sBBBBIIQ32s")
ACK = struct.Struct("<4sB3x32s")
MAX_FRAME = 16 * 1024 * 1024
CORE_COMMIT = "7aee330f7799751a6a9f4b2a8decb612abe9c266"
ATOL, RTOL = 1e-10, 1e-8


class EvidenceError(ValueError):
    pass


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": str(value)}
    return value


def close_scalar(a, b, atol=1e-10, rtol=1e-10):
    return (isinstance(a, (int, float)) and not isinstance(a, bool) and
            math.isfinite(a) and math.isfinite(b) and abs(a-b) <= atol+rtol*abs(b))


def metrics(actual, reference):
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        return {"passed": False, "error": "shape or finite-value mismatch"}
    absolute = float(np.linalg.norm(a-b))
    bn, an = float(np.linalg.norm(b)), float(np.linalg.norm(a))
    return {"absolute_frobenius": absolute,
            "relative_frobenius": absolute/bn if bn else (0.0 if absolute == 0 else None),
            "maximum_absolute_coordinate": float(np.abs(a-b).max()) if a.size else 0.0,
            "reference_norm": bn, "actual_norm": an, "norm_change": an-bn,
            "passed": absolute <= ATOL+RTOL*bn}


def helmert(k):
    h = np.zeros((k, k-1), dtype=np.float64)
    for j in range(1, k):
        h[:j, j-1] = 1 / math.sqrt(j*(j+1))
        h[j, j-1] = -j / math.sqrt(j*(j+1))
    return h


def postprocess(pre, vocab):
    target = float(np.sqrt(np.sum(vocab*vocab, axis=1)).mean())
    norms = np.linalg.norm(pre, axis=1)
    calibrated = pre * (target/np.maximum(norms, 1e-6))[:, None]
    qnorm = np.linalg.norm(calibrated, axis=1)
    vnorm = np.linalg.norm(vocab, axis=1)
    scores = (calibrated/np.maximum(qnorm, 1e-12)[:, None]) @ (
        vocab/np.maximum(vnorm, 1e-12)[:, None]).T
    ids = np.argmax(scores, axis=1)
    margins = scores[np.arange(len(pre)), ids] - np.partition(scores, -2, axis=1)[:, -2]
    final = .7*calibrated + .3*vocab[ids]
    stable = ((margins > 1e-6) & (norms > 1e-6) & (qnorm > 1e-12) & (vnorm[ids] > 1e-12))
    return {"calibrated": calibrated, "final": final, "ids": ids, "margins": margins,
            "norms": norms, "qnorm": qnorm, "vnorm": vnorm, "stable": stable, "scores": scores}


def gram_pre(ga, reference):
    """Independent positive-definite Bures reconstruction of actual Gram bytes."""
    k = len(reference)
    h = helmert(k)
    b = h.T @ reference
    def eig(a):
        values, vectors = np.linalg.eigh((a+a.T)*.5)
        if np.any(values <= 0) or not np.isfinite(values).all():
            raise EvidenceError("native Gram replay requires positive spectra")
        return values, vectors
    sv, su = eig(ga)
    rv, ru = eig(b @ b.T)
    xhalf = (su*np.sqrt(sv/np.maximum(sv/k+.001, 1e-6))) @ su.T
    y = (ru*(rv/np.maximum(rv/k+.001, 1e-6))) @ ru.T
    mid, basis = eig(xhalf @ y @ xhalf)
    m = xhalf @ ((basis*(1/np.sqrt(mid))) @ basis.T) @ xhalf
    return (h @ m) @ b + reference.mean(axis=0)


def session_manifest(vocab):
    return {
        "format": "SBG1", "version": 1, "algorithm_version": "coefficient_packet_v2",
        "rank_policy": {"threshold": 1e-10, "ambiguity": [1e-12,1e-8],
                        "scale": "max(1,largest_eigenvalue)"},
        "lambda": .001, "epsilon": 1e-6, "alpha": .3,
        "row_norm_floor": 1e-6, "cosine_norm_floor": 1e-12,
        "reference_source_sha256": "415957df427eb79a4212e760af4d133cb4ce10cc4619f11447082117922f366d",
        "vocabulary": {"sha256": sha(np.asarray(vocab,dtype="<f8").tobytes()),
                       "shape": list(vocab.shape), "dtype": "<f8"},
        "session_version": 2, "numerical_core_commit": CORE_COMMIT, "coefficient_mode": 6,
        "basis": "helmert_j0:first_j+1_positive;row_j+1_negative;denominator_sqrt((j+1)*(j+2))",
        "reconstruction_grouping": "(H @ M) @ (H.T @ R) + mean(R)",
        "coefficient_symmetrization": "(M + M.T) * 0.5 before lower_triangle",
        "triangle_order": "numpy.tril_indices(K-1),row_major",
        "wire_precisions": {"float64":"<f8","float32":"<f4","token_ids":"<u4","decoded_compute":"<f8"},
        "wire_modes": {"1":"dual_gram","3":"raw_fallback","4":"direct","5":"token_lookup","6":"coefficient"},
    }


def parse_packet(raw):
    if not isinstance(raw,bytes) or not WIRE.size <= len(raw) <= MAX_FRAME:
        raise EvidenceError("invalid packet length")
    magic,version,mode,code,flags,k,d,n,digest = WIRE.unpack_from(raw)
    if magic!=b"SBG1" or version!=1 or flags!=0 or mode not in {1,3,4,5,6}:
        raise EvidenceError("invalid packet header")
    if k<1 or d<1 or n!=len(raw)-WIRE.size:
        raise EvidenceError("packet dimension/size mismatch")
    payload=raw[WIRE.size:]
    out={"mode":mode,"code":code,"K":k,"d":d,"digest":digest,"payload":payload}
    if mode==5:
        if code!=0 or len(payload)!=4*k:
            raise EvidenceError("invalid token payload")
        out["ids"]=np.frombuffer(payload,dtype="<u4")
    else:
        if code not in {1,2}:
            raise EvidenceError("invalid numerical precision")
        dtype=np.dtype("<f8" if code==1 else "<f4")
        out["dtype"]=dtype
        if mode in {3,4}:
            if len(payload)!=k*d*dtype.itemsize:
                raise EvidenceError("invalid raw payload")
            out["raw"]=np.frombuffer(payload,dtype=dtype).reshape(k,d).astype(np.float64)
        else:
            if not 2<=k<d or len(payload)!=4*k+dtype.itemsize*k*(k-1)//2:
                raise EvidenceError("invalid triangular payload")
            out["ids"]=np.frombuffer(payload[:4*k],dtype="<u4")
            values=np.frombuffer(payload[4*k:],dtype=dtype).astype(np.float64)
            matrix=np.zeros((k-1,k-1))
            q=0
            for i in range(k-1):
                for j in range(i+1):
                    matrix[i,j]=matrix[j,i]=values[q]
                    q+=1
            out["matrix"]=matrix
        values=out.get("raw",out.get("matrix"))
        if not np.isfinite(values).all():
            raise EvidenceError("nonfinite numerical packet")
    return out


def split_frames(raw, allow_partial=False):
    frames=[]
    offset=0
    while offset<len(raw):
        if len(raw)-offset<4:
            if allow_partial: return frames,raw[offset:]
            raise EvidenceError("truncated length prefix")
        n=struct.unpack_from(">I",raw,offset)[0]
        if not 0<n<=MAX_FRAME:
            raise EvidenceError("invalid declared frame size")
        if len(raw)-offset<4+n:
            if allow_partial: return frames,raw[offset:]
            raise EvidenceError("truncated frame")
        frames.append(raw[offset+4:offset+4+n])
        offset+=4+n
    return frames,b""


def parse_ack(raw, magic):
    if len(raw)!=40:
        raise EvidenceError("ACK length")
    actual,status,digest=ACK.unpack(raw)
    if actual!=magic or raw[5:8]!=bytes(3) or status not in {0,1}:
        raise EvidenceError("ACK format")
    if status and digest!=bytes(32):
        raise EvidenceError("error ACK digest")
    return status,digest


def expected_plan():
    """Reconstruct the schedule only; never generate scientific input arrays."""
    def perm(seed,n):
        return np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed))).permutation(n).tolist()
    blocks=[]
    profiles=tuple(PROFILE)
    for k,d in SHAPES:
        for p,profile in enumerate(profiles):
            order=perm([23017,k,d,p,0],6)
            for i,seed in enumerate(SEEDS):
                for j in range(4):
                    rotation=(2*i+j)%6
                    block=f"warm-k{k}-d{d}-seed{seed}-{profile}-r{j}"
                    ids=order[rotation:]+order[:rotation]
                    blocks.append([{"trial_id":block+"-"+ARMS[a],"block_id":block,"mode":"warm",
                        "fixture_id":f"primary-k{k}-d{d}-seed{seed}","K":k,"d":d,"seed":seed,
                        "profile":profile,"repetition":j,"order_position":position,"arm":ARMS[a]}
                        for position,a in enumerate(ids)])
    order=perm([23017,0,0,2,1],6)
    for i,(k,d) in enumerate(SHAPES):
        for j in range(4):
            rotation=(2*i+j)%6
            block=f"cold-k{k}-d{d}-seed{SEEDS[0]}-constrained-r{j}"
            ids=order[rotation:]+order[:rotation]
            blocks.append([{"trial_id":block+"-"+ARMS[a],"block_id":block,"mode":"cold",
                "fixture_id":f"primary-k{k}-d{d}-seed{SEEDS[0]}","K":k,"d":d,"seed":SEEDS[0],
                "profile":"constrained","repetition":j,"order_position":position,"arm":ARMS[a]}
                for position,a in enumerate(ids)])
    rows=[row for i in perm([23029],len(blocks)) for row in blocks[i]]
    for i,row in enumerate(rows): row["sequence_index"]=i
    return rows


def geometric(values):
    return math.exp(sum(map(math.log,values))/len(values)) if values else None


def recompute_analysis(plan,trials):
    index={r["trial_id"]:r for r in trials}
    keys=sorted({(r["mode"],r["fixture_id"],r["profile"]) for r in plan})
    cases=[]
    for mode,fid,profile in keys:
        byarm={a:{} for a in ARMS}
        for p in plan:
            if (p["mode"],p["fixture_id"],p["profile"])!=(mode,fid,profile): continue
            row=index.get(p["trial_id"])
            if row and row.get("status")=="completed" and type(row.get("wall_ns")) is int and row["wall_ns"]>0:
                byarm[p["arm"]][p["repetition"]]=row
        case={"mode":mode,"fixture_id":fid,"profile":profile,"arms":{},"paired":{}}
        for a in ARMS:
            rows=[byarm[a][j] for j in sorted(byarm[a])]
            walls=[r["wall_ns"] for r in rows]
            case["arms"][a]={"completed_repetitions":len(rows),"wall_ns":walls,
                "median_wall_ns":statistics.median(walls) if walls else None,
                "logical_wire_bytes":[r.get("logical_wire_bytes") for r in rows]}
        for a in ALL_COMPARATORS:
            complete=all(j in byarm["coefficient"] and j in byarm[a] for j in range(4))
            ratios=[byarm["coefficient"][j]["wall_ns"]/byarm[a][j]["wall_ns"] for j in range(4)] if complete else []
            case["paired"][a]={"complete":complete,"coefficient_over_comparator":ratios,
                              "median_ratio":statistics.median(ratios) if complete else None}
        if all(len(byarm[a])==4 for a in COMPARATORS+("coefficient",)):
            choice=min(COMPARATORS,key=lambda a:case["arms"][a]["median_wall_ns"])
            case["hindsight_reference"]={"chosen_arm":choice,**case["paired"][choice],
                "selection":"smallest four-run median among fixed comparators; hindsight only"}
        else:
            case["hindsight_reference"]={"chosen_arm":None,"complete":False,
                                        "coefficient_over_comparator":[],"median_ratio":None}
        cases.append(case)
    aggregates={}
    for name,expected,predicate in (
        ("shaped_warm",18,lambda c:c["mode"]=="warm" and c["profile"]!="loopback"),
        ("loopback_warm",9,lambda c:c["mode"]=="warm" and c["profile"]=="loopback"),
        ("cold_constrained",3,lambda c:c["mode"]=="cold")):
        selected=[c for c in cases if predicate(c)]
        group={"expected_cases":expected,"observed_cases":len(selected),"comparisons":{}}
        for a in ALL_COMPARATORS+("hindsight_reference",):
            rows=[c[a] if a=="hindsight_reference" else c["paired"][a] for c in selected]
            values=[r["median_ratio"] for r in rows if r["complete"]]
            complete=len(selected)==expected and len(values)==expected
            group["comparisons"][a]={"complete":complete,"completed_cases":len(values),
                "geometric_mean_case_median":geometric(values) if complete else None,
                "available_only_descriptive_geometric_mean":geometric(values),
                "case_median_min":min(values) if values else None,
                "case_median_max":max(values) if values else None}
        aggregates[name]=group
    warm=[c["hindsight_reference"]["median_ratio"] for c in cases
          if c["mode"]=="warm" and c["hindsight_reference"]["complete"]]
    return {"cases":cases,"aggregates":aggregates,"all_warm_hindsight_complete":len(warm)==27,
            "worst_warm_case_paired_median_ratio":max(warm) if len(warm)==27 else None,
            "ratio_direction":"coefficient wall time divided by comparator wall time; smaller is faster"}


class Audit:
    def __init__(self,root,execution_digest):
        self.root=root
        self.expected_execution=execution_digest
        self.sections={k:{"checks":0,"issue_count":0,"issues":[]} for k in ("data_integrity","report_consistency")}
        self.science_checks=0
        self.science_failures=[]
        self.notes=[]
        self.raw_cache={}
        self.array_cache={}
        self.objects={}
        self.fixtures={}
        self.references={}
        self.packet_cache={}
        self.reference_calls=0
        self.numeric_replays=Counter()
        self.plan=expected_plan()
        self.partial=True
        self.execution={}
        self.results={}
        self.totals=Counter()
        self.forbidden={"new_fixture_generations":0,"model_imports_or_inference":0,
                        "optimization_invocations":0,"timing_measurements":0}

    def check(self,condition,label,section="data_integrity"):
        row=self.sections[section]
        row["checks"]+=1
        if not bool(condition):
            row["issue_count"]+=1
            if len(row["issues"])<200: row["issues"].append(label)
        return bool(condition)

    def science(self,condition,label):
        self.science_checks+=1
        if not bool(condition) and len(self.science_failures)<200:
            self.science_failures.append(label)
        return bool(condition)

    def same(self,actual,expected,label,section="report_consistency"):
        if isinstance(expected,dict):
            if not self.check(isinstance(actual,dict),label+": object",section): return
            for key,value in expected.items():
                self.check(key in actual,label+"/"+key+": present",section)
                if key in actual: self.same(actual[key],value,label+"/"+key,section)
        elif isinstance(expected,list):
            if not self.check(isinstance(actual,list) and len(actual)==len(expected),label+": list length",section): return
            for i,(a,b) in enumerate(zip(actual,expected)): self.same(a,b,f"{label}/{i}",section)
        elif type(expected) is float:
            self.check(close_scalar(actual,expected),label+": numeric value",section)
        elif type(expected) is bool:
            self.check(type(actual) is bool and actual==expected,label+": boolean",section)
        else:
            self.check(actual==expected and (expected is None or type(actual) is type(expected)),
                       label+": exact value",section)

    def path(self,relative):
        if not isinstance(relative,str): raise EvidenceError("nonstring artifact path")
        p=PurePosixPath(relative)
        if p.is_absolute() or ".." in p.parts or "\\" in relative or not p.parts:
            raise EvidenceError("unsafe artifact path")
        target=self.root.joinpath(*p.parts)
        if target.is_symlink() or not target.resolve().is_relative_to(self.root):
            raise EvidenceError("symlink/escaped artifact path")
        return target

    def raw(self,relative):
        if relative not in self.raw_cache:
            self.raw_cache[relative]=self.path(relative).read_bytes()
        return self.raw_cache[relative]

    def json(self,name,required=False):
        path=self.path(name)
        if not path.exists():
            if required: self.check(False,name+": required file absent")
            else: self.notes.append(name+": unavailable in partial evidence")
            return None
        try: return json.loads(self.raw(name))
        except Exception as exc:
            self.check(False,name+": invalid JSON: "+str(exc))
            return None

    def rows(self,name):
        filename=name+".jsonl"
        if self.path(filename).exists(): raw=self.raw(filename)
        elif self.path(filename+".gz").exists():
            raw=gzip.decompress(self.raw(filename+".gz"))
        else:
            self.notes.append(filename+": unavailable")
            return []
        rows=[]
        lines=raw.splitlines(keepends=True)
        for i,line in enumerate(lines):
            try:
                row=json.loads(line)
                if type(row) is not dict: raise ValueError("record is not object")
                rows.append(row)
            except Exception as exc:
                if self.partial and i==len(lines)-1 and not line.endswith(b"\n"):
                    self.notes.append(filename+": final incomplete record preserved, not interpreted")
                else: self.check(False,f"{filename}:{i+1}: invalid record: {exc}")
        return rows

    def reference_bytes(self,ref):
        if not isinstance(ref,dict) or "path" not in ref:
            raise EvidenceError("invalid retained object reference")
        raw=self.raw(ref["path"])
        self.check(sha(raw)==ref.get("sha256") and len(raw)==ref.get("bytes"),
                   ref["path"]+": reference file hash/bytes")
        entry=self.objects.get(ref["path"])
        if entry is not None: self.same(ref,entry,ref["path"]+": manifest identity","data_integrity")
        elif not self.partial: self.check(False,ref["path"]+": absent from object index")
        return raw

    def array(self,ref):
        raw=self.reference_bytes(ref)
        path=ref["path"]
        if path not in self.array_cache:
            a=np.load(io.BytesIO(raw),allow_pickle=False)
            if not isinstance(a,np.ndarray) or a.dtype.hasobject: raise EvidenceError("nonplain retained array")
            self.check(list(a.shape)==ref.get("shape") and a.dtype.str==ref.get("dtype"),path+": shape/dtype")
            self.check(a.nbytes==ref.get("data_bytes") and sha(a.tobytes(order="C"))==ref.get("data_sha256"),
                       path+": canonical data bytes")
            self.array_cache[path]=a
        return self.array_cache[path]

    def run_inventory(self):
        raw=self.raw("execution.json")
        self.check(sha(raw)==self.expected_execution,"externally anchored execution.json SHA256")
        self.execution=json.loads(raw)
        e=self.execution
        self.check(e.get("github_commit")==SOURCE_COMMIT,"execution source commit")
        self.check(e.get("frozen_design_commit")==DESIGN_COMMIT and e.get("design_sha256")==DESIGN_SHA256,
                   "execution frozen design identity")
        self.check(e.get("run_kind")=="preregistered_coefficient_packet_v2" and e.get("budget_seconds")==300,
                   "execution kind/budget")
        self.check(e.get("github_run_attempt")==1 and e.get("github_run_number")==1,
                   "single approved workflow attempt")
        self.check(str(e.get("github_run_id","")).isdigit(),"execution run ID")
        self.check(e.get("benchmark_retries")==0 and e.get("model_imports_or_inference")==0,
                   "execution no reruns/models")
        self.results=self.json("results.json") or {}
        self.partial=self.results.get("status")!="complete"
        inventory=e.get("retained_files",{})
        self.check(isinstance(inventory,dict),"retained file inventory")
        if not isinstance(inventory,dict): inventory={}
        verified=0
        for name,item in inventory.items():
            try:
                data=self.raw(name)
                self.check(len(data)==item["bytes"] and sha(data)==item["sha256"],name+": inventory hash/bytes")
                verified+=1
            except Exception as exc: self.check(False,name+": inventory read failed: "+str(exc))
        present={p.relative_to(self.root).as_posix() for p in self.root.rglob("*") if p.is_file()}
        self.check(present==set(inventory)|{"execution.json"},"inventory covers exact original artifact files")
        self.totals["retained_files_verified"]=verified
        for name,receipt in e.get("compression_receipts",{}).items():
            try:
                original=self.raw(name)
                packed=self.raw(name+".gz")
                uncompressed=gzip.decompress(packed)
                self.check(uncompressed==original,name+": gzip exact-byte roundtrip")
                self.same(receipt.get("raw"),{"sha256":sha(original),"bytes":len(original)},name+": raw compression receipt","data_integrity")
                self.same(receipt.get("gzip"),{"sha256":sha(packed),"bytes":len(packed)},name+": gzip receipt","data_integrity")
                self.check(receipt.get("round_trip_verified") is True,name+": declared roundtrip")
            except Exception as exc: self.check(False,name+": compression verification failed: "+str(exc))
        for stem in JSONL_NAMES:
            name=stem+".jsonl"
            if name in inventory:
                present_receipt=name in e.get("compression_receipts",{})
                if not self.partial:
                    self.check(present_receipt,name+": compression receipt present")
                elif not present_receipt:
                    self.notes.append(name+": compression not finalized in partial evidence")
                if not present_receipt and name+".gz" in present:
                    try:
                        self.check(gzip.decompress(self.raw(name+".gz"))==self.raw(name),
                                   name+": unreceipted gzip exact-byte roundtrip")
                    except Exception as exc:
                        self.check(False,name+": unreceipted gzip verification failed: "+str(exc))
        launch=self.json("launch_source_manifest.json") or {}
        source=self.json("source_manifest.json") or {}
        runtime=self.json("runtime_manifest.json") or {}
        for label,value in (("launch",launch),("source",source),("runtime",runtime)):
            if value:
                self.check(value.get("github_commit")==SOURCE_COMMIT,label+": source anchor")
                if "github_run_id" in value:
                    self.check(str(value["github_run_id"])==str(e.get("github_run_id")),label+": run anchor")
        sourcefiles=source.get("files",{})
        launchfiles=launch.get("files",{})
        for name,item in sourcefiles.items():
            if name in launchfiles: self.same(item,launchfiles[name],name+": launch/worker source identity","data_integrity")
        for name,expected in (("latent_bridge/alignment_core.py",CORE_SHA256),
                              ("latent_bridge/alignment_packet.py",ORIGINAL_PACKET_SHA256),
                              ("docs/coefficient_packet_design.md",DESIGN_SHA256)):
            item=sourcefiles.get(name) or launchfiles.get(name)
            if item: self.check(item.get("sha256")==expected,name+": fixed dependency hash")
            elif not self.partial: self.check(False,name+": missing source evidence")
        if runtime:
            self.check(runtime.get("python")=="3.11.18" and runtime.get("numpy")=="2.4.6","pinned study runtime")
            self.check(all(v=="1" for v in runtime.get("thread_environment",{}).values()) and
                       len(runtime.get("thread_environment",{}))==3,"study thread limits")
        for entry in self.rows("object_manifest"):
            name=entry.get("path")
            self.check(name not in self.objects,"object index unique path")
            self.objects[name]=entry
            try:
                data=self.reference_bytes(entry)
                if entry.get("kind")=="array": self.array(entry)
            except Exception as exc: self.check(False,str(name)+": invalid indexed object: "+str(exc))

    def load_fixtures(self):
        manifest=self.json("fixture_manifest.json") or {}
        entries=manifest.get("fixtures",{})
        expected={}
        for k,d in SHAPES:
            for seed in SEEDS: expected[f"primary-k{k}-d{d}-seed{seed}"]=(k,d,seed,None,"primary")
            expected[f"warmup-k{k}-d{d}-seed23041"]=(k,d,23041,None,"warmup")
        for seed in (19001,19013):
            for geometry in range(1,9): expected[f"stress-k6-d24-seed{seed}-g{geometry}"]=(6,24,seed,geometry,"stress")
        if entries: self.check(set(entries)==set(expected),"fixed 28 fixture identities")
        if not self.path("fixtures.npz").exists(): return
        raw=self.raw("fixtures.npz")
        self.same(manifest.get("archive"),{"sha256":sha(raw),"bytes":len(raw)},"fixture NPZ inventory","data_integrity")
        with np.load(io.BytesIO(raw),allow_pickle=False) as archive:
            used=set()
            for fid,desc in entries.items():
                if desc.get("status")!="constructed":
                    self.notes.append(fid+": construction did not complete")
                    continue
                if fid not in expected:
                    self.check(False,fid+": unexpected fixture")
                    continue
                k,d,seed,geometry,suite=expected[fid]
                self.same(desc,{key:value for key,value in zip(("K","d","seed","geometry","suite"),(k,d,seed,geometry,suite))},fid+": fixture identity","data_integrity")
                values={}
                for name in ("S","R","vocab","ids"):
                    item=desc["arrays"][name]
                    key=item["npz_key"]
                    a=np.asarray(archive[key])
                    used.add(key)
                    self.check(not a.dtype.hasobject and list(a.shape)==item["shape"] and a.dtype.str==item["dtype"],fid+"/"+name+": shape/dtype")
                    self.check(a.nbytes==item["data_bytes"] and sha(a.tobytes(order="C"))==item["data_sha256"],fid+"/"+name+": data hash")
                    values[name]=a
                s,r,v,ids=(values[n] for n in ("S","R","vocab","ids"))
                self.check(s.shape==(k,d) and r.shape==(k,d) and v.shape==(k+256,d) and ids.shape==(k,),fid+": fixed dimensions")
                self.check(all(np.isfinite(values[n]).all() for n in ("S","R","vocab")),fid+": finite retained arrays")
                intended=np.zeros(k,dtype=np.uint32) if geometry==5 else np.arange(k,dtype=np.uint32)
                self.check(np.array_equal(ids,intended) and np.array_equal(r,v[ids]),fid+": intended IDs/reference table")
                ss=np.geomspace(1,.2,k-1); rr=np.geomspace(1,.1,k-1)
                if geometry==1: ss[-1]=0
                elif geometry==2: rr[-1]=0
                elif geometry==3: ss[1:]=0; rr[1:]=0
                elif geometry==4: ss[:]=0
                elif geometry==5: rr[:]=0
                elif geometry==6: ss[-1]=1e-5
                elif geometry==7: rr[-1]=1e-5
                elif geometry==8:
                    ss=np.ones(k-1);rr=np.ones(k-1);ss[-1]=rr[0]=2e-4
                h=helmert(k)
                self.science(np.allclose(np.linalg.svd(h.T@s,compute_uv=False),sorted(ss,reverse=True),rtol=1e-8,atol=1e-10),fid+": source construction spectrum")
                self.science(np.allclose(np.linalg.svd(h.T@r,compute_uv=False),sorted(rr,reverse=True),rtol=1e-8,atol=1e-10),fid+": receiver construction spectrum")
                target=np.median(np.linalg.norm(r,axis=1))
                self.science(np.allclose(np.linalg.norm(v[k:],axis=1),target,rtol=1e-8,atol=1e-10),fid+": distractor norm contract")
                if geometry==8:
                    self.science(metrics((h.T@s)@(h.T@s).T,np.diag(ss*ss))["passed"],fid+": diagonal source Gram")
                    self.science(metrics((h.T@r)@(h.T@r).T,np.diag(rr*rr))["passed"],fid+": diagonal receiver Gram")
                self.fixtures[fid]={**values,"K":k,"d":d,"suite":suite,"geometry":geometry}
            self.check(set(archive.files)==used,"no unaccounted fixture NPZ keys")
        self.totals["retained_fixtures"]=len(self.fixtures)

    def load_references(self):
        root=Path(__file__).resolve().parent.parent
        core_path=root/"latent_bridge"/"alignment_core.py"
        frozen=None
        try:
            code=core_path.read_bytes()
            if sha(code)!=CORE_SHA256: raise EvidenceError("local frozen reference source hash differs")
            spec=importlib.util.spec_from_file_location("_coefficient_audit_frozen_core",core_path)
            frozen=importlib.util.module_from_spec(spec)
            sys.modules[spec.name]=frozen
            spec.loader.exec_module(frozen)
        except Exception as exc:
            self.check(False,"pinned frozen numerical reference unavailable: "+str(exc))
        seen=set()
        for row in self.rows("references"):
            fid=row.get("fixture_id")
            self.check(fid not in seen,"reference row unique: "+str(fid))
            seen.add(fid)
            fixture=self.fixtures.get(fid)
            if fixture is None:
                self.check(False,str(fid)+": reference lacks retained fixture")
                continue
            refmap={}
            fresh_dense=None
            if fixture["suite"]=="primary" and frozen is not None:
                fresh_dense=frozen.dense_reference(fixture["S"],fixture["R"],.001,1e-6,details=False).pre
                self.reference_calls+=1
            for name,item in row.get("references",{}).items():
                if item.get("status")!="computed": continue
                pre=self.array(item["pre"]);cal=self.array(item["calibrated"]);final=self.array(item["final"])
                ids=np.asarray(item["snap_ids"],dtype=np.int64)
                computed=postprocess(pre,fixture["vocab"])
                self.science(metrics(cal,computed["calibrated"])["passed"],fid+"/"+name+": retained reference calibration")
                if ids.shape!=(len(pre),) or np.any(ids<0) or np.any(ids>=len(fixture["vocab"])):
                    self.check(False,fid+"/"+name+": invalid reference snap IDs")
                    continue
                regret=computed["scores"].max(axis=1)-computed["scores"][np.arange(len(ids)),ids]
                self.science(np.all(regret<=1e-8),fid+"/"+name+": retained reference IDs maximize cosine")
                self.science(metrics(final,.7*cal+.3*fixture["vocab"][ids])["passed"],fid+"/"+name+": retained reference anchoring")
                self.same(item["margins"],computed["margins"].tolist(),fid+"/"+name+": reference margins")
                if fresh_dense is not None:
                    self.science(metrics(pre,fresh_dense)["passed"],fid+"/"+name+": fresh frozen dense primary reference")
                computed.update({"calibrated":cal,"final":final,"ids":ids})
                refmap[name]={"pre":pre,"post":computed}
            self.references[fid]=refmap
        self.totals["reference_fixture_rows"]=len(seen)

    def check_packet(self,raw,decoded,fid,arm,encoded_meta=None):
        fixture=self.fixtures[fid]
        packet=parse_packet(raw)
        k,d=fixture["K"],fixture["d"]
        self.check((packet["K"],packet["d"])==(k,d),fid+"/"+arm+": message dimensions")
        allowed={"coefficient":{3,6},"dual_direct":{3,4},"thin_direct":{3,4},
                 "gram_guard":{1,3},"gram_original":{1,3},"token_lookup":{5}}
        self.check(packet["mode"] in allowed[arm],fid+"/"+arm+": mode belongs to arm")
        self.check(packet["digest"]==hashlib.sha256(canonical(session_manifest(fixture["vocab"]))).digest(),fid+"/"+arm+": bound session fingerprint")
        final=self.array(decoded["final"])
        self.science(final.shape==(k,d) and np.isfinite(final).all(),fid+"/"+arm+": decoded shape/finite")
        if encoded_meta is not None:
            self.same(encoded_meta.get("fallback"),packet["mode"]==3,fid+"/"+arm+": actual fallback route")
            self.same(encoded_meta.get("packet_bytes"),len(raw),fid+"/"+arm+": encoded byte count")
        if "ids" in packet:
            self.check(np.array_equal(packet["ids"],fixture["ids"]),fid+"/"+arm+": actual transmitted IDs")
        raw_exact=None
        if packet["mode"] in {3,4}:
            raw_exact=np.asarray(final,dtype=packet["dtype"],order="C").tobytes()==packet["payload"]
            self.science(raw_exact,fid+"/"+arm+": exact direct/raw wire bits")
        elif packet["mode"]==5:
            self.science(np.array_equal(final,fixture["R"]),fid+": embedding lookup output")
        else:
            pre=self.array(decoded["pre"]) if decoded.get("pre") else None
            self.science(pre is not None,fid+"/"+arm+": native pre-prefix retained")
            if pre is not None:
                key=(sha(raw),fid)
                replay=self.packet_cache.get(key)
                if replay is None:
                    if packet["mode"]==6:
                        np.linalg.cholesky(packet["matrix"])
                        h=helmert(k);r=fixture["vocab"][packet["ids"]]
                        replay=(h@packet["matrix"])@(h.T@r)+r.mean(axis=0)
                        self.numeric_replays["coefficient_wire"]+=1
                    elif fixture["suite"]=="primary":
                        replay=gram_pre(packet["matrix"],fixture["vocab"][packet["ids"]])
                        self.numeric_replays["primary_gram_wire"]+=1
                    else:
                        self.numeric_replays["nonprimary_gram_decomposition_not_replayed"]+=1
                    if replay is not None: self.packet_cache[key]=replay
                if replay is not None:
                    self.science(metrics(pre,replay)["passed"],fid+"/"+arm+": independent actual-wire reconstruction")
                computed=postprocess(pre,fixture["vocab"])
                ids=decoded.get("metadata",{}).get("snap_ids")
                if ids is not None:
                    ids=np.asarray(ids,dtype=np.int64)
                    valid=ids.shape==(k,) and np.all(ids>=0) and np.all(ids<len(fixture["vocab"]))
                    self.check(valid,fid+"/"+arm+": receiver snap-ID shape/range")
                    if valid:
                        regret=computed["scores"].max(axis=1)-computed["scores"][np.arange(k),ids]
                        self.science(np.all(regret<=1e-8),fid+"/"+arm+": reported snap is near maximum")
                        self.science(metrics(final,.7*computed["calibrated"]+.3*fixture["vocab"][ids])["passed"],fid+"/"+arm+": actual decoder anchoring")
                else: self.science(False,fid+"/"+arm+": receiver snap IDs missing")
        return packet,final,raw_exact

    def numerical_diagnostics(self):
        rows=self.rows("diagnostics")
        expected={(fid,a,p) for fid,f in self.fixtures.items() if f["suite"]!="warmup"
                  for a in ARMS for p in ("float64","float32")}
        seen=set();self.diag_outputs={};self.diag_checks={};self.diag_required={};self.diag_rows=rows
        for row in rows:
            key=(row.get("fixture_id"),row.get("arm"),row.get("precision"))
            self.check(key not in seen and key in expected,"diagnostic identity/uniqueness: "+str(key))
            seen.add(key)
            fid,arm,precision=key
            if row.get("status")!="checked":
                self.same(row.get("required_passed"),False,str(key)+": unfinished diagnostic cannot pass")
                self.diag_checks[key]=False
                continue
            try:
                raw=self.reference_bytes(row["packet"]);decoded=row["decoded"]
                packet,final,exact=self.check_packet(raw,decoded,fid,arm,row.get("sender_metadata"))
                checks={"finite_final":bool(np.isfinite(final).all()),"exact_serialized_size":True}
                if exact is not None: checks["raw_roundtrip_exact_at_wire_dtype"]=exact
                fallback=packet["mode"]==3
                self.same(row.get("fallback"),fallback,str(key)+": fallback field")
                if arm=="token_lookup":
                    error=metrics(final,self.fixtures[fid]["R"])
                    checks["token_lookup_exact"]=np.array_equal(final,self.fixtures[fid]["R"])
                    actual_ids=None
                else:
                    preferred="dense" if fallback else ("thin" if arm=="thin_direct" else "dual")
                    available=preferred in self.references.get(fid,{})
                    name=preferred
                    if precision=="float32" and not available and "dense" in self.references.get(fid,{}):
                        name="dense"
                        self.same(row.get("evaluation_reference_reason"),
                                  "unrounded fast reference unsupported; descriptive dense reference",
                                  str(key)+": descriptive reference substitution")
                    self.same(row.get("preferred_reference_method"),preferred,str(key)+": preferred reference")
                    self.same(row.get("preferred_reference_supported"),available,str(key)+": preferred support")
                    self.same(row.get("reference_method"),name,str(key)+": actual evaluation reference")
                    ref=self.references[fid][name]
                    error=metrics(final,ref["post"]["final"])
                    mask=ref["post"]["stable"]
                    self.same(row.get("stable_rows"),mask.tolist(),str(key)+": stable mask")
                    self.same(row.get("stable_row_count"),int(mask.sum()),str(key)+": stable count")
                    actual_ids=decoded.get("metadata",{}).get("snap_ids")
                    if actual_ids is None: actual_ids=row.get("sender_metadata",{}).get("snap_ids")
                    self.same(row.get("actual_internal_snap_ids"),actual_ids,str(key)+": actual endpoint IDs")
                    if actual_ids is not None:
                        self.same(row.get("internal_snap_changes"),int(np.count_nonzero(np.asarray(actual_ids)!=ref["post"]["ids"])),str(key)+": snap changes")
                    if decoded.get("pre") is not None:
                        pre=self.array(decoded["pre"]);cal=self.array(row["actual_calibrated"])
                        pre_error=metrics(pre,ref["pre"]);cal_error=metrics(cal,ref["post"]["calibrated"])
                        self.same(row.get("pre_error"),pre_error,str(key)+": pre error")
                        self.same(row.get("calibrated_error"),cal_error,str(key)+": calibration error")
                        self.science(metrics(cal,postprocess(pre,self.fixtures[fid]["vocab"])["calibrated"])["passed"],str(key)+": diagnostic calibrated array")
                        if precision=="float64":
                            checks["pre_reference"]=pre_error["passed"]
                            checks["calibrated_reference"]=cal_error["passed"]
                    if precision=="float64":
                        if np.any(mask):
                            e=metrics(final[mask],ref["post"]["final"][mask])
                            self.same(row.get("stable_final_error"),e,str(key)+": stable final error")
                            checks["stable_final_reference"]=e["passed"]
                            if actual_ids is not None:
                                checks["stable_internal_snaps"]=np.array_equal(np.asarray(actual_ids)[mask],ref["post"]["ids"][mask])
                            elif not fallback: checks["native_internal_snaps_available"]=False
                        if fallback: checks["dense_fallback_target"]=error["passed"]
                    control=row["same_precision_direct"]
                    cpacket=parse_packet(self.reference_bytes(control["packet"]))
                    cfinal=self.array(control["decoded"])
                    cbits=np.asarray(cfinal,dtype=cpacket["dtype"],order="C").tobytes()
                    targetbits=np.asarray(ref["post"]["final"],dtype=cpacket["dtype"],order="C").tobytes()
                    ce=cbits==cpacket["payload"]==targetbits
                    self.science(ce,str(key)+": matched same-precision direct control")
                    self.same(control.get("error"),metrics(cfinal,ref["post"]["final"]),str(key)+": direct rounding error")
                    self.same(control.get("raw_roundtrip_exact"),ce,str(key)+": direct raw claim")
                    checks["direct_control_exact_at_wire_dtype"]=ce
                self.same(row.get("final_error"),error,str(key)+": full-vector final error")
                if self.fixtures[fid]["suite"]=="primary" and precision=="float64" and arm in {"coefficient","gram_guard"}:
                    checks["native_primary_route"]=not fallback
                self.same(row.get("required_checks"),{k:bool(v) for k,v in checks.items()},str(key)+": required checks")
                passed=all(checks.values())
                self.same(row.get("required_passed"),bool(passed),str(key)+": diagnostic outcome")
                for name,value in checks.items(): self.science(value,str(key)+"/"+name)
                self.diag_checks[key]=bool(passed)
                self.diag_required[key]={name:bool(value) for name,value in checks.items()}
                self.diag_outputs[key]=(final,actual_ids)
            except Exception as exc:
                self.check(False,str(key)+": diagnostic audit error: "+str(exc),"report_consistency")
                self.diag_checks[key]=False
        if not self.partial: self.check(seen==expected and len(rows)==300,"complete fixed 300-diagnostic coverage")
        self.totals["diagnostic_rows"]=len(rows)
        self.eligibility={};self.practical={}
        for fid,f in self.fixtures.items():
            if f["suite"]!="primary": continue
            c=self.diag_outputs.get((fid,"coefficient","float64"))
            d=self.diag_outputs.get((fid,"dual_direct","float64"))
            if c is not None and d is not None:
                error=metrics(c[0],d[0]);equal=c[1] is not None and d[1] is not None and c[1]==d[1]
                self.practical[fid]={"passed":bool(error["passed"] and equal),
                    "all_row_final_error":error,"all_internal_snaps_equal":bool(equal),
                    "coefficient_snap_ids":c[1],"dual_direct_snap_ids":d[1]}
            else: self.practical[fid]={"passed":False}
            self.eligibility[fid]={a:self.diag_checks.get((fid,a,"float64"),False) for a in ARMS}
            self.eligibility[fid]["coefficient"] &= self.practical[fid]["passed"]
        reported=self.json("eligibility.json") or {}
        if reported:
            for fid,mapping in reported.get("routes",{}).items():
                self.same(mapping,self.eligibility.get(fid,{}),fid+": eligibility")
            for fid,p in reported.get("practical",{}).items():
                if fid in self.practical:
                    self.same(p,self.practical[fid],fid+": practical all-row gate")
        self.totals["eligible_primary_routes"]=sum(sum(m.values()) for m in self.eligibility.values())
        self.totals["practical_gates_passed"]=sum(p["passed"] for p in self.practical.values())

    def check_setup(self,payload,fixture,arm):
        if len(payload)<4: raise EvidenceError("short setup")
        n=struct.unpack_from("<I",payload)[0]
        if not 1<=n<=4096 or 4+n>len(payload): raise EvidenceError("setup JSON length")
        raw=payload[4:4+n];meta=json.loads(raw);table=payload[4+n:]
        self.check(canonical(meta)==raw,"canonical setup JSON")
        expected_manifest=session_manifest(fixture["vocab"])
        digest=sha(canonical(expected_manifest))
        direct=arm in {"dual_direct","thin_direct"}
        expected={"version":2,"arm":arm,"d":fixture["d"],"digest":digest,
                  "kind":"direct_context" if direct else "table",
                  "table_bytes":0 if direct else fixture["vocab"].nbytes}
        if not direct: expected["manifest"]=expected_manifest
        self.same(meta,expected,"setup contract","data_integrity")
        self.check(set(meta)==set(expected),"exact setup schema keys")
        expected_table=b"" if direct else np.asarray(fixture["vocab"],dtype="<f8").tobytes()
        self.check(table==expected_table,"actual setup table/context bytes")
        self.totals["cold_setup_table_bytes"]+=len(table)

    def endpoint(self,io_record,sent,received,profile,is_receiver,label,complete):
        rate,hold=PROFILE[profile]
        for direction,raw in (("sent",sent),("received",received)):
            self.same(io_record.get(direction+"_bytes"),len(raw),label+"/"+direction+": actual bytes")
            self.same(io_record.get(direction+"_sha256"),sha(raw),label+"/"+direction+": stream hash")
            attempts=io_record.get(direction+"_frames",[])
            offset=0
            for i,attempt in enumerate(attempts):
                n=attempt.get("actual_wire_bytes")
                if type(n) is not int or n<0 or offset+n>len(raw):
                    self.check(False,label+": invalid frame actual byte count")
                    continue
                portion=raw[offset:offset+n];offset+=n
                declared=attempt.get("declared_wire_bytes");payload_n=attempt.get("payload_bytes")
                if payload_n is not None:
                    self.check(type(payload_n) is int and declared==payload_n+4,label+": frame declaration")
                if n>=4:
                    header=struct.unpack_from(">I",portion)[0]
                    self.check(header==payload_n,label+": actual length prefix")
                if attempt.get("complete"):
                    self.check(n==declared and n>=5 and 0<payload_n<=MAX_FRAME,label+": complete frame bytes")
                    self.check(attempt.get("error") is None,label+": completed frame has no error")
                elif complete: self.check(False,label+": incomplete frame in completed trial")
            self.check(offset==len(raw),label+"/"+direction+": all bytes assigned to attempts")
        expected=[]
        for attempt in io_record.get("sent_frames",[]):
            if is_receiver and hold>0: expected.append(("reply_holdoff",hold))
            if rate is not None:
                n=attempt.get("declared_wire_bytes",0)
                if type(n) is int:
                    expected.extend(("pacing",min(16384,n-i)/rate) for i in range(0,n,16384))
        waits=io_record.get("waits",[])
        self.check(len(waits)==len(expected) if complete else len(waits)<=len(expected),label+": wait schedule length")
        for i,wait in enumerate(waits):
            if i>=len(expected): break
            kind,seconds=expected[i]
            self.check(wait.get("kind")==kind and close_scalar(wait.get("requested_seconds"),seconds,1e-14,1e-12),
                       label+": actual ordered pacing/holdoff request")
            observed=wait.get("observed_ns")
            self.check(type(observed) is int and observed>=0,label+": observed wait nonnegative")
            if complete:
                self.check(wait.get("error") is None,label+": completed wait has no error")
                self.science(type(observed) is int and observed+1000>=seconds*1e9,label+": wait not materially shorter than request")
        for kind in ("pacing","reply_holdoff"):
            selected=[w for w in waits if w.get("kind")==kind]
            totals={"calls":len(selected),"requested_seconds":sum(w["requested_seconds"] for w in selected),
                    "observed_ns":sum(w["observed_ns"] for w in selected)}
            self.same(io_record.get("wait_totals",{}).get(kind),totals,label+"/"+kind+": wait sums")
            self.totals["observed_"+kind+"_ns"]+=totals["observed_ns"]
        return sum(w.get("observed_ns",0) for w in waits)

    def trial(self,row,warmup=False):
        fid,arm=row["fixture_id"],row["arm"]
        if not row.get("retained"):
            if row.get("status")=="completed": self.check(False,row.get("trial_id",fid)+": completed trial lacks raw evidence")
            return
        retained=row["retained"];fixture=self.fixtures[fid]
        payloads={k:self.reference_bytes(v) for k,v in retained.get("payloads",{}).items()}
        wire={k:self.reference_bytes(v) for k,v in retained.get("wire",{}).items()}
        complete=row.get("transport_status",row.get("status"))=="completed"
        cold=bool(row.get("cold",row.get("mode")=="cold"))
        label=row.get("trial_id","warmup/"+fid+"/"+arm)
        profile=row.get("profile","loopback")
        self.check(profile in PROFILE,label+": known profile")
        observed=0
        for endpoint in ("sender","receiver"):
            if endpoint+"_io" in row:
                observed+=self.endpoint(row[endpoint+"_io"],wire.get(endpoint+"_sent",b""),
                    wire.get(endpoint+"_received",b""),profile,endpoint=="receiver",label+"/"+endpoint,complete)
        if "sender_sent" in wire and "receiver_received" in wire:
            self.check(wire["sender_sent"].startswith(wire["receiver_received"]),label+": TCP forward prefix")
        if "receiver_sent" in wire and "sender_received" in wire:
            self.check(wire["receiver_sent"].startswith(wire["sender_received"]),label+": TCP reverse prefix")
        checks={}
        if complete:
            for name in ("sender_sent","sender_received","receiver_sent","receiver_received"):
                self.check(name in wire,label+": full trace "+name)
            checks["sender_to_receiver_exact"]=wire.get("sender_sent")==wire.get("receiver_received")
            checks["receiver_to_sender_exact"]=wire.get("receiver_sent")==wire.get("sender_received")
            forward,_=split_frames(wire["sender_sent"]);backward,_=split_frames(wire["sender_received"])
            count=2 if cold else 1
            self.check(len(forward)==len(backward)==count,label+": exact setup/message frame coverage")
            for endpoint in ("sender_io","receiver_io"):
                for direction in ("sent_frames","received_frames"):
                    attempts=row.get(endpoint,{}).get(direction,[])
                    checks[endpoint+"_"+direction+"_complete"]=(len(attempts)==count and all(
                        a.get("complete") and a.get("actual_wire_bytes")==a.get("declared_wire_bytes") for a in attempts))
            if cold:
                self.check_setup(forward[0],fixture,arm)
                status,digest=parse_ack(backward[0],b"SET2")
                self.check(status==0 and digest==hashlib.sha256(forward[0]).digest(),label+": exact SET2 receipt")
                for name in ("setup_sent","receiver_setup_payload"):
                    self.check(payloads.get(name)==forward[0],label+": setup payload retention")
                for name in ("setup_ack_received","receiver_setup_ack"):
                    self.check(payloads.get(name)==backward[0],label+": setup ACK retention")
            message=forward[-1];ack=backward[-1]
            for name in ("message_sent","receiver_message_payload"):
                self.check(payloads.get(name)==message,label+": actual message payload retention")
            for name in ("output_ack_received","receiver_output_ack"):
                self.check(payloads.get(name)==ack,label+": actual output ACK retention")
            packet,final,exact=self.check_packet(message,retained["decoded"],fid,arm,row.get("sender_metadata"))
            status,digest=parse_ack(ack,b"ACK2")
            checks["output_receipt_hash"]=status==0 and digest==hashlib.sha256(np.asarray(final,dtype="<f8").tobytes()).digest()
            self.check(checks["output_receipt_hash"],label+": exact ACK2 output hash")
            self.same(row.get("checks"),checks,label+": transport checks")
            logical=len(wire["sender_sent"])+len(wire["sender_received"])
            self.same(row.get("logical_wire_bytes"),logical,label+": logical framed bytes")
            self.totals["logical_wire_bytes"]+=logical
            self.totals["logical_frames"]+=len(forward)+len(backward)
            self.totals["decoded_handoffs"]+=1
            if not warmup:
                wall=row.get("wall_ns")
                self.check(type(wall) is int and wall>0,label+": positive measured wall time")
                self.same(row.get("end_perf_counter_ns")-row.get("start_perf_counter_ns"),wall,label+": actual clock difference")
                self.science(observed<=wall,label+": all observed imposed waits fit inside measured completion")
                phases=row.get("phases_ns",{})
                included=[v for k,v in phases.items() if not k.endswith("_outside_timer")]
                self.check(all(type(v) is int and v>=0 for v in included),label+": nonnegative timed phase durations")
                self.science(sum(included)<=wall,label+": sequential sender phases fit completion")
                self.trial_numerical(row,packet,final,retained["decoded"],fixture)
        else:
            # Partial traces remain evidence. Never require a successful ACK or
            # force missing/decode-error rows into the complete success cohort.
            for raw in wire.values():
                try: split_frames(raw,allow_partial=True)
                except EvidenceError as exc:
                    self.notes.append(label+": partial malformed frame retained: "+str(exc))
            self.totals["partial_transport_records"]+=1

    def trial_numerical(self,row,packet,final,decoded,fixture):
        fid,arm=row["fixture_id"],row["arm"];label=row["trial_id"]
        fallback=packet["mode"]==3
        checks={}
        if packet["mode"] in {3,4}:
            checks["raw_roundtrip_exact_at_wire_dtype"]=(
                np.asarray(final,dtype=packet["dtype"]).tobytes()==packet["payload"])
        if arm=="token_lookup":
            checks["token_lookup_exact"]=np.array_equal(final,fixture["R"])
        else:
            name="dense" if fallback else ("thin" if arm=="thin_direct" else "dual")
            ref=self.references[fid][name];mask=ref["post"]["stable"]
            allerror=metrics(final,ref["post"]["final"])
            self.same(row.get("numerical_details",{}).get("all_row_final_error"),allerror,label+": full final error")
            if decoded.get("pre") is not None:
                e=metrics(self.array(decoded["pre"]),ref["pre"])
                checks["pre_reference"]=e["passed"]
                self.same(row.get("numerical_details",{}).get("pre_error"),e,label+": timed pre error")
            actual_ids=decoded.get("metadata",{}).get("snap_ids")
            if actual_ids is None: actual_ids=(row.get("sender_metadata") or {}).get("snap_ids")
            if np.any(mask):
                e=metrics(final[mask],ref["post"]["final"][mask])
                checks["stable_final_reference"]=e["passed"]
                if actual_ids is not None:
                    checks["stable_internal_snaps"]=np.array_equal(np.asarray(actual_ids)[mask],ref["post"]["ids"][mask])
                self.same(row.get("numerical_details",{}).get("stable_final_error"),e,label+": timed stable error")
            if fallback: checks["dense_fallback_target"]=allerror["passed"]
            if arm in {"coefficient","gram_guard"}: checks["native_primary_route"]=not fallback
            if arm=="coefficient":
                direct=self.diag_outputs.get((fid,"dual_direct","float64"))
                if direct is None: checks["all_row_practical_comparison_available"]=False
                else:
                    e=metrics(final,direct[0])
                    checks["all_row_practical_final"]=e["passed"]
                    checks["all_row_practical_snaps"]=actual_ids is not None and direct[1] is not None and actual_ids==direct[1]
                    self.same(row.get("numerical_details",{}).get("all_row_error_to_actual_dual_direct"),e,label+": practical error")
        self.same(row.get("numerical_checks"),{k:bool(v) for k,v in checks.items()},label+": numerical checks")
        for name,value in checks.items(): self.science(value,label+"/"+name)
        if row.get("status")=="completed":
            self.check(bool(checks) and all(checks.values()),label+": completed row has valid recomputed numerical checks","report_consistency")

    def transport_rows(self):
        planned=self.json("plan.json") or {}
        if planned:
            self.same(planned.get("planned_trials"),self.plan,"exact fixed plan/order","data_integrity")
            self.check(planned.get("expected_count")==720 and planned.get("schema_version")==1,"plan header")
        self.trials=self.rows("trials");self.warmups=self.rows("warmups")
        byid={r["trial_id"]:r for r in self.plan}
        seen=[];started=[];end_of_previous=None
        for row in self.trials:
            ident=row.get("trial_id")
            self.check(ident in byid and ident not in seen,"trial identity/duplicate: "+str(ident))
            seen.append(ident)
            if ident not in byid: continue
            self.same(row,byid[ident],str(ident)+": immutable planned fields","data_integrity")
            self.check(row.get("status") in {"completed","error","numerically_ineligible","not_executed"},str(ident)+": legal outcome")
            if "start_perf_counter_ns" in row:
                started.append(byid[ident]["sequence_index"])
                start=row["start_perf_counter_ns"];end=row.get("end_perf_counter_ns")
                if end_of_previous is not None:
                    self.check(start>=end_of_previous,str(ident)+": no overlapping timed trials")
                if type(end) is int: end_of_previous=end
            if row.get("status")=="numerically_ineligible":
                self.check(not self.eligibility.get(row["fixture_id"],{}).get(row["arm"],False),str(ident)+": declared ineligibility")
            try: self.trial(row)
            except Exception as exc: self.check(False,str(ident)+": transport audit error: "+str(exc),"report_consistency")
        self.check(started==sorted(started),"actual measured trial order follows frozen schedule")
        if not self.partial: self.check(set(seen)==set(byid) and len(seen)==720,"all720 planned slots retained")
        self.totals["trial_rows"]=len(self.trials)
        warm_expected={(f"warmup-k{k}-d{d}-seed23041",a) for k,d in SHAPES for a in ARMS}
        warm_seen=set()
        for row in self.warmups:
            key=(row.get("fixture_id"),row.get("arm"))
            self.check(key in warm_expected and key not in warm_seen,"warmup identity/uniqueness")
            warm_seen.add(key)
            self.check(row.get("included_in_timing_analysis") is False and row.get("wall_ns") is None,
                       "warmup excluded from primary timing")
            try: self.trial(row,warmup=True)
            except Exception as exc: self.check(False,str(key)+": warmup audit error: "+str(exc),"report_consistency")
        if not self.partial: self.check(warm_seen==warm_expected,"all18 distinct warmups")
        self.totals["warmup_rows"]=len(self.warmups)

    def final_statistics(self):
        analysis=recompute_analysis(self.plan,self.trials)
        if self.results:
            self.same(self.results.get("analysis"),analysis,"independently rebuilt paired-time analysis")
        counts=dict(Counter(r.get("status") for r in self.trials))
        diagnostic_failed=sum(sum(not value for value in checks.values())
                              for checks in self.diag_required.values())
        for row in self.diag_rows:
            key=(row.get("fixture_id"),row.get("arm"),row.get("precision"))
            if key not in self.diag_required:
                # A failed/unfinished operation is not rerun. Its claimed check
                # details remain reported evidence; completeness cannot pass.
                diagnostic_failed+=sum(not bool(v) for v in row.get("required_checks",{}).values())
                self.notes.append(str(key)+": failed-operation check details not numerically replayed")
        coverage=(len(self.trials)==720 and counts.get("completed")==720 and
                  len({r.get("trial_id") for r in self.trials})==720 and
                  len(self.diag_rows)==300 and len(self.diag_required)==300 and diagnostic_failed==0 and
                  self.totals["eligible_primary_routes"]==54 and len(self.warmups)==18 and
                  all(r.get("status")=="completed" for r in self.warmups) and
                  not self.results.get("fatal_error"))
        shaped=analysis["aggregates"]["shaped_warm"]["comparisons"]
        speed=all(shaped[a]["complete"] and shaped[a]["geometric_mean_case_median"]<=.97
                  for a in COMPARATORS+("hindsight_reference",))
        loop=analysis["aggregates"]["loopback_warm"]["comparisons"]["hindsight_reference"]
        worst=analysis["worst_warm_case_paired_median_ratio"]
        overhead=bool(loop["complete"] and loop["geometric_mean_case_median"]<=1.10 and
                      analysis["all_warm_hindsight_complete"] and worst<=1.25)
        practical=len(self.practical)==9 and all(p["passed"] for p in self.practical.values())
        criteria={"complete_correct_720_slots_and_required_diagnostics":bool(coverage),
            "shaped_warm_against_all_three_and_hindsight_at_most_0.97":bool(speed),
            "loopback_at_most_1.10_and_worst_warm_case_at_most_1.25":bool(overhead),
            "all_nine_primary_all_row_fidelity_gates":bool(practical)}
        summary={"trial_status_counts":counts,"planned_trials":720,
                 "diagnostic_rows":len(self.diag_rows),"failed_diagnostic_checks":diagnostic_failed,
                 "eligible_primary_routes":self.totals["eligible_primary_routes"],
                 "practical_gates_passed":self.totals["practical_gates_passed"],
                 "warmup_count":len(self.warmups),"criteria":criteria,
                 "engineering_success":all(criteria.values())}
        if self.results: self.same(self.results,summary,"recomputed complete/negative outcomes")
        maximum={};routes=Counter()
        for row in self.diag_rows:
            if row.get("status")!="checked": continue
            key="/".join((row["suite"],row["arm"],row["precision"],"fallback" if row.get("fallback") else "native"))
            routes[key]+=1
            value=row.get("final_error",{}).get("relative_frobenius")
            if isinstance(value,(int,float)) and (key not in maximum or value>maximum[key]["relative_error"]):
                maximum[key]={"relative_error":value,"fixture_id":row["fixture_id"],
                              "internal_snap_changes":row.get("internal_snap_changes")}
        if self.results:
            self.same(self.results.get("diagnostic_route_counts"),dict(routes),"diagnostic native/fallback counts")
            self.same(self.results.get("descriptive_final_error_maxima"),maximum,"all descriptive precision maxima")
        complete_ids=len(self.trials)==720 and len({r.get("trial_id") for r in self.trials})==720 and set(
            r.get("trial_id") for r in self.trials)=={r["trial_id"] for r in self.plan}
        accounting=self.json("completion_accounting.json") or {}
        if accounting:
            self.same(accounting.get("status_counts"),dict(sorted(counts.items())),"supervisor outcome counts")
            self.same(accounting.get("all_slots_recorded"),bool(complete_ids),"supervisor complete slot inventory")
            self.same(accounting.get("all_trials_completed_successfully"),
                      bool(complete_ids and counts=={"completed":720}),"supervisor completed calls")
        self.same(self.execution.get("all_slots_recorded"),bool(complete_ids),"execution complete slot inventory")
        self.same(self.execution.get("all_trials_completed_successfully"),
                  bool(complete_ids and counts=={"completed":720}),"execution completed calls")
        self.analysis=analysis
        self.criteria=criteria
        self.scientific_complete=coverage
        return summary

    def run(self):
        self.run_inventory()
        self.load_fixtures()
        self.load_references()
        self.numerical_diagnostics()
        self.transport_rows()
        self.final_statistics()

    def receipt(self):
        for section in self.sections.values():
            section["passed"]=section["issue_count"]==0
        integrity=self.sections["data_integrity"]["passed"]
        consistency=self.sections["report_consistency"]["passed"]
        e=self.execution
        raw=self.raw_cache.get("execution.json",b"")
        criteria=getattr(self,"criteria",{})
        required_failed=(bool(self.science_failures) or
            not criteria.get("complete_correct_720_slots_and_required_diagnostics",False) or
            not criteria.get("all_nine_primary_all_row_fidelity_gates",False))
        return {
            "audit_status":"completed" if integrity and consistency else "completed_with_integrity_issues",
            "artifact_directory":str(self.root),
            "audit_runtime":{"script_sha256":sha(Path(__file__).read_bytes()),
                             "python":platform.python_version(),"numpy":np.__version__,
                             "platform":platform.platform()},
            "evidence":{"execution":{"sha256":sha(raw) if raw else None,"bytes":len(raw),
                "external_digest_supplied":True,"external_digest_matches":bool(raw and sha(raw)==self.expected_execution),
                "source_commit":e.get("github_commit"),"run_id":str(e.get("github_run_id",""))},
                "frozen_design_commit":DESIGN_COMMIT,"frozen_design_sha256":DESIGN_SHA256,
                "frozen_reference_core_sha256":CORE_SHA256},
            **self.sections,
            "scientific_validation":{
                "required_checks_failed":bool(required_failed),
                "complete_evidence":bool(getattr(self,"scientific_complete",False)),
                "checks":self.science_checks,
                "failed_checks":self.science_failures,
                "engineering_success":bool(criteria and all(criteria.values())),
                "recomputed_criteria":criteria,
                "recomputed_analysis":getattr(self,"analysis",None),
                "frozen_dense_primary_reference_evaluations":self.reference_calls,
                "independent_wire_reconstructions":dict(self.numeric_replays),
                "network_totals_including_warmups":dict(self.totals),
                "scope":[
                    "No imports of the new codec, study or transport modules.",
                    "Fixture arrays are read, hashed and spectrum-checked, never regenerated.",
                    "Only deterministic trial-order permutations are reconstructed.",
                    "Actual packet, setup, ACK and framed stream bytes are parsed independently.",
                    "Coefficient payload reconstruction is independent; native Gram decomposition is replayed for primary inputs.",
                    "The SHA-pinned original dense function is reused only for nine unique full-rank primary references; this is not an independently rewritten dense solver.",
                    "Fixed seed metadata and construction spectra are checked, but random fixture draws are not regenerated.",
                    "Nonprimary Gram decompositions are not replayed; retained pre/post arrays and descriptive errors are checked. Arbitrary dense null completions are not claimed cross-host bit-identical.",
                    "Stress/f32 vector errors and paired statistical gates are recomputed from retained actual arrays.",
                    "Near-tied replay snaps use actual retained decisions with a near-maximum score and anchored-vector check.",
                    "Observed wait/phase/clock arithmetic is checked; clocks are not independently remeasured.",
                    "Not every sender guard predicate is re-executed or independently certified, and descriptive f32 errors do not establish general fidelity.",
                    "A valid negative scientific result is not a data-integrity failure.",
                ]},
            "notes":list(dict.fromkeys(self.notes)),
            **self.forbidden,
        }


def self_test():
    """Tiny parser/statistic/algebra controls, never the empirical fixture grid."""
    raw=b"abc"
    framed=struct.pack(">I",len(raw))+raw
    assert split_frames(framed)==([raw],b"")
    assert split_frames(framed[:-1],True)==([],framed[:-1])
    try:
        split_frames(framed[:-1])
    except EvidenceError: pass
    else: raise AssertionError("truncated frame accepted")
    digest=hashlib.sha256(raw).digest()
    assert parse_ack(ACK.pack(b"SET2",0,digest),b"SET2")==(0,digest)
    try:
        parse_ack(ACK.pack(b"ACK2",1,digest),b"ACK2")
    except EvidenceError: pass
    else: raise AssertionError("invalid error digest accepted")
    r=np.array([[1.,0.,0.],[-1.,0.,0.]])
    payload=np.array([0,1],dtype="<u4").tobytes()+struct.pack("<d",1.5)
    parsed=parse_packet(WIRE.pack(b"SBG1",1,6,1,0,2,3,len(payload),bytes(32))+payload)
    h=helmert(2)
    pre=(h@parsed["matrix"])@(h.T@r)
    assert metrics(pre,1.5*r)["passed"]
    assert metrics(postprocess(pre,r)["final"],r)["passed"]
    bits=struct.pack("<6d",0.,-0.,1.,-2.,3.,-0.)
    raw_packet=WIRE.pack(b"SBG1",1,4,1,0,2,3,len(bits),bytes(32))+bits
    assert np.asarray(parse_packet(raw_packet)["raw"],dtype="<f8").tobytes()==bits
    plan=expected_plan()
    assert len(plan)==720 and len({r["trial_id"] for r in plan})==720
    assert Counter(r["mode"] for r in plan)=={"warm":648,"cold":72}
    selected=[r for r in plan if r["mode"]=="warm" and
              r["fixture_id"]=="primary-k12-d192-seed17011" and r["profile"]=="moderate"]
    walls={"coefficient":80,"dual_direct":100,"thin_direct":120,
           "gram_guard":50,"gram_original":90,"token_lookup":1}
    rows=[{**p,"status":"completed","wall_ns":walls[p["arm"]],"logical_wire_bytes":1} for p in selected]
    analysis=recompute_analysis(plan,rows)
    case=next(c for c in analysis["cases"] if c["mode"]=="warm" and
              c["fixture_id"]=="primary-k12-d192-seed17011" and c["profile"]=="moderate")
    assert case["hindsight_reference"]["chosen_arm"]=="gram_guard"
    assert case["paired"]["dual_direct"]["coefficient_over_comparator"]==[.8]*4
    assert case["hindsight_reference"]["median_ratio"]==1.6
    aggregate=analysis["aggregates"]["shaped_warm"]["comparisons"]["hindsight_reference"]
    assert not aggregate["complete"] and aggregate["geometric_mean_case_median"] is None
    assert close_scalar(aggregate["available_only_descriptive_geometric_mean"],1.6)
    assert statistics.median([.5,2.])==1.25
    print(json.dumps({"self_test":"passed","controls":["framing","ACK","raw_bits",
          "coefficient_reconstruction","paired_ratios","incomplete_denominator"],
          "model_imports_or_inference":0,"optimization_invocations":0,"timing_measurements":0}))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir")
    parser.add_argument("--execution-sha256")
    parser.add_argument("--output")
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if not all((args.artifact_dir,args.execution_sha256,args.output)):
        parser.error("--artifact-dir, --execution-sha256 and --output are required")
    digest=args.execution_sha256
    if len(digest)!=64 or any(c not in "0123456789abcdef" for c in digest):
        parser.error("execution SHA256 must be lowercase hexadecimal")
    root=Path(args.artifact_dir).resolve()
    output=Path(args.output).resolve()
    if not root.is_dir(): parser.error("artifact directory is absent")
    if output.is_relative_to(root): parser.error("receipt must be written outside the original artifact")
    audit=Audit(root,digest)
    try:
        audit.run()
    except Exception as exc:
        audit.check(False,"audit could not finish: "+type(exc).__name__+": "+str(exc))
        audit.notes.append(traceback.format_exc())
    receipt=audit.receipt()
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(clean(receipt),sort_keys=True,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"audit_status":receipt["audit_status"],
                      "required_checks_failed":receipt["scientific_validation"]["required_checks_failed"],
                      "engineering_success":receipt["scientific_validation"]["engineering_success"],
                      "output":str(output),**audit.forbidden},sort_keys=True))
    return 0 if receipt["audit_status"]=="completed" else 1


if __name__=="__main__":
    raise SystemExit(main())
