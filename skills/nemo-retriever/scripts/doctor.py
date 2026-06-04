#!/usr/bin/env python
"""Verify the installed `retriever` engine satisfies the skill's contract.

Usage: <RETRIEVER_VENV>/bin/python skills/nemo-retriever/scripts/doctor.py
Exits 0 if all checks pass, 1 otherwise. Always runs a LIVE ingest+query probe.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CONTRACT_DIR = os.path.join(os.path.dirname(HERE), "contract")
FIXTURE = os.path.join(os.path.dirname(HERE), "tests", "fixtures", "contract_probe.txt")
EMBED_MODEL = "nvidia/llama-nemotron-embed-1b-v2"

results = []  # (ok: bool, label: str, detail: str)


def check(ok, label, detail=""):
    results.append((bool(ok), label, detail))


def retriever_bin():
    path = shutil.which("retriever")
    if not path:
        return None
    return path


def help_text(bin_path, subcmd):
    # Force a wide terminal so the rich/click help box does not truncate long
    # flag names (e.g. "--embed-model-na…"), which would break substring checks.
    env = dict(os.environ, COLUMNS="200")
    try:
        out = subprocess.run([bin_path, subcmd, "--help"], capture_output=True, text=True,
                             timeout=60, env=env)
        return (out.stdout or "") + (out.stderr or "")
    except Exception as e:  # noqa: BLE001
        return f"__ERROR__ {e}"


def main():
    contract = json.load(open(os.path.join(CONTRACT_DIR, "cli-contract.json")))
    hit_schema = json.load(open(os.path.join(CONTRACT_DIR, "actual-hit.schema.json")))

    bin_path = retriever_bin()
    check(bin_path is not None, "retriever CLI on PATH",
          "" if bin_path else "run skills/nemo-retriever/references/install.md")
    if not bin_path:
        return report()

    # --- Static flag-surface checks (no GPU) ---
    qhelp = help_text(bin_path, "query")
    for flag in contract["query"]["required_flags"]:
        check(flag in qhelp, f"query has {flag}")
    ihelp = help_text(bin_path, "ingest")
    for flag in contract["ingest"]["required_flags"]:
        check(flag in ihelp, f"ingest has {flag}")
    for flag in contract["ingest"]["forbidden_flags"]:
        check(flag not in ihelp, f"ingest does NOT have {flag}",
              "engine changed: skill assumes single-pass auto-detect")

    # --- Live probe: ingest tiny fixture, query, validate hit schema (GPU) ---
    tmp = tempfile.mkdtemp(prefix="retriever_doctor_")
    try:
        corpus = os.path.join(tmp, "corpus")
        os.makedirs(corpus)
        shutil.copy(FIXTURE, corpus)
        uri = os.path.join(tmp, "lancedb")
        table = "contract_probe"
        ing = subprocess.run(
            [bin_path, "ingest", corpus + "/", "--table-name", table, "--lancedb-uri", uri,
             "--embed-model-name", EMBED_MODEL, "--quiet"],
            capture_output=True, text=True, timeout=900)
        check(ing.returncode == 0, "live ingest of fixture", ing.stderr.strip()[-300:])

        q = subprocess.run(
            [bin_path, "query", "What is the capital of the test corpus?", "--top-k", "3",
             "--table-name", table, "--lancedb-uri", uri, "--embed-model-name", EMBED_MODEL],
            capture_output=True, text=True, timeout=600)
        check(q.returncode == 0, "live query", q.stderr.strip()[-300:])
        hits = []
        if q.returncode == 0:
            try:
                hits = json.loads(q.stdout)
                check(isinstance(hits, list) and len(hits) > 0, "query returned hits")
            except Exception as e:  # noqa: BLE001
                check(False, "query stdout is JSON", str(e))
        # validate first hit against the actual-hit schema
        if hits:
            ok, why = validate(hits[0], hit_schema)
            check(ok, "hit matches actual-hit.schema.json", why)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return report()


def validate(obj, schema):
    """Tiny dependency-free validator for the subset of JSON Schema we use."""
    if not isinstance(obj, dict):
        return False, "hit is not an object"
    for req in schema.get("required", []):
        if req not in obj:
            return False, f"missing required field '{req}'"
    types = {"integer": int, "string": str, "number": (int, float), "object": dict, "array": list}
    for name, spec in schema.get("properties", {}).items():
        if name in obj and "type" in spec:
            py = types.get(spec["type"])
            if py and not isinstance(obj[name], py):
                return False, f"field '{name}' should be {spec['type']}, got {type(obj[name]).__name__}"
    return True, ""


def report():
    failed = [r for r in results if not r[0]]
    for ok, label, detail in results:
        mark = "PASS" if ok else "FAIL"
        line = f"[{mark}] {label}"
        if detail and not ok:
            line += f"  -- {detail}"
        print(line)
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
