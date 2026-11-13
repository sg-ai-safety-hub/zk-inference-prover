"""The attention output's no-wrap window on two real 512-token prompts.

The magnitude form refused a Llama 3.1 8B extraction of the first prompt in
`fixtures/no_wrap_prompts.json`, at `y-delay-vw12`, block 0 layer 1:

    the attention output's no-wrap window does not fit:
    1 + 16 + 6 + 9 = 32 > 31 (over by 1, magnitude form)

The prompt is a table of contents: short lines of titles and page numbers.
Layer 1's head 22 spreads its attention nearly evenly over them, so the
denominator `Z` reaches about 68,000 against under 10,000 on prose, and the
values it averages agree in sign, so `|acc| ~ Z |mean v|` takes 22 bits.  The
exponent window is the same 9 on both prompts.  What no slot holds is the
pair the window charged for: the largest `|acc|` sits at `e = 5`, and the
largest `mag num` any slot holds is 27 bits of the field's 30.  The product
form (`prover/src/compose.rs`, `Params::ao_acc_prod`) bounds that product slot
by slot, and this checks that the extraction now goes through in it, while
the second prompt, prose, keeps the form it always had.

It needs the model: `ZK_MODEL` naming a Llama 3.1 8B checkpoint, as a local
directory or as a hub id already in the local cache (layers 0 and 1 are all it
reads, so a checkpoint truncated to two or more layers gives the same dumps).
Without one it says so and exits 0, the way the CPU selftests treat a card they
do not have, and deciding that downloads nothing.

    ZK_MODEL=<llama 3.1 8b> uv run python -m harness.witness.no_wrap_regress
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

from .shapes import ao_acc_split, ao_no_wrap, ao_no_wrap_refusal

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "fixtures", "no_wrap_prompts.json")


def not_llama31_8b(model):
    """Why `model` is not a Llama 3.1 8B checkpoint, by its config alone, or
    `None` if it is one.

    `model` is `ZK_MODEL`, which `extract.py`'s `Model` hands to
    `from_pretrained`: a local directory or a hub id.  A directory's
    `config.json` is read as it is; a hub id is resolved the way `Model` would
    resolve it, but from the local cache only (`local_files_only`, as
    `eval/selftest.py`'s CPU path does), so a run that skips downloads nothing.
    """
    if not model:
        return "ZK_MODEL is not set"
    if os.path.isdir(model):
        try:
            with open(os.path.join(model, "config.json")) as f:
                c = json.load(f)
        except OSError as e:
            return f"no config.json in that directory ({type(e).__name__})"
        except ValueError as e:
            return f"its config.json does not parse ({e})"
    else:
        from transformers import AutoConfig
        try:
            c = AutoConfig.from_pretrained(model, local_files_only=True).to_dict()
        except Exception as e:             # any of: not cached, not a hub id, unreadable
            return (f"not a directory, and no config for it in the local cache "
                    f"({type(e).__name__})")
    if not isinstance(c, dict):
        return "its config.json is not an object"
    if not (c.get("model_type") == "llama" and c.get("hidden_size") == 4096
            and c.get("num_attention_heads") == 32 and c.get("vocab_size") == 128256):
        return (f"its config is a {c.get('model_type')!r} of hidden size "
                f"{c.get('hidden_size')}, {c.get('num_attention_heads')} heads and a "
                f"vocabulary of {c.get('vocab_size')}")
    return None


def window_of(ids, tmp):
    """Extract block 0 layer 1 over `ids` and return its manifest's window."""
    np.save(os.path.join(tmp, "ids.npy"), np.array(ids, dtype=np.int64))
    out = os.path.join(tmp, "w")
    cmd = [sys.executable, "-m", "harness.witness.extract", "--tokens", str(len(ids)),
           "--chunk", "512", "--layers", "1", "--spec", "y-delay-vw12", "--diagonal-block",
           "--slim", "--slim-num", "--ids", os.path.join(tmp, "ids.npy"),
           "--blocks", "all", "--out", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return None, (r.stderr.strip().splitlines() or ["(no output)"])[-1]
    man = json.load(open(os.path.join(f"{out}_B0_L1", "manifest.json")))
    ao = man["attn_out"]
    return (ao["ao_acc_bits"], ao["jv_max"], ao["jo_min"]), None


def main():
    model = os.environ.get("ZK_MODEL", "")
    why = not_llama31_8b(model)
    if why:
        print(f"SKIP  no Llama 3.1 8B checkpoint at ZK_MODEL={model!r}: {why}. The "
              f"window arithmetic on these prompts' numbers is in "
              f"`extract.py --selftest`")
        return 0
    prompts = {p["name"]: p["ids"] for p in json.load(open(FIXTURE))["prompts"]}
    bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        for name, ids in prompts.items():
            d = os.path.join(tmp, name)
            os.makedirs(d)
            got, err = window_of(ids, d)
            if got is None:
                bad += 1
                print(f"FAIL  {name:<18s} -> extraction refused: {err}")
                continue
            acc, jv_max, jo_min = got
            w = ao_no_wrap(acc, jv_max, jo_min, 8, 8)
            lo, hi = ao_acc_split(acc)
            mag_bits = 1 + lo + hi + max(jv_max - jo_min, 0)
            if name == "table_of_contents":
                # the refusal's own shape: the magnitude form past the field,
                # and the product form inside it
                ok = (mag_bits > 31 and w["form"] == "product"
                      and ao_no_wrap_refusal(w, name) is None)
            else:
                ok = w["form"] != "product" and ao_no_wrap_refusal(w, name) is None
            bad += not ok
            print(f"{'OK  ' if ok else 'FAIL'}  {name:<18s} -> |acc| {acc} bits, "
                  f"e_hi {jv_max - jo_min} (jv_max {jv_max}, jo_min {jo_min}); magnitude "
                  f"window {mag_bits}/31; taken: {w['bits']}/31, margin {w['margin']} "
                  f"({w['form']} form)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
