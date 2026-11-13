"""Dump the layer witness B6.5 composes into one proof.

Forked from `b6/extract.py`.  Re-runs B1's forward pass under the composed
layout's spec (`b65-design`: power-of-two linear scales on q/k/v/o, the bf16
chain on gate/up/down, the integer softmax at int12 x int11, and the attention
output rounded straight to int8 with a power-of-two `s_o`) and writes everything
the composed proof's components read:

  * **the attention block**, in `b5/extract.py`'s format (the S queries of the
    last prefill chunk against the C-key prefix) -- the packed key tiles load a key tile
    of heads from it;
  * **the matmul accumulators and scales**, as in B6 -- the dequant components;
  * **new against B6**: the quantized weights `qw_*.i8` (the preprocessed side of
    the r^T W Freivalds check), the quantized activations `qx_*.i8` (the Freivalds check's committed
    side), the bf16 input patterns `x_*.u16` (the requantize / max-argument
    witnesses), and the attention-output fold's scale exponents `jo.i32` (the
    per-token power-of-two `s_o`) and `jv.i32` (the per-(head, channel)
    power-of-two V scales).

    uv run python b65/extract.py --tokens 1536 --chunk 512 --layer 12

**Stream DD (E11): the block axis.**  The forward loop below already prefills the
whole concatenation in `--chunk` blocks, and `LastChunk.add` *assigns*, so every
recorded column is computed for every block and then overwritten by the next.
Dumping a prompt block is therefore a loop inversion and not a new harness: the
dump body moved inside the forward loop, the `P = C - S` assertion relaxed to a
whole number of blocks, and a block axis on the output directory.  `--blocks`
selects which blocks are written; its default, `last`, is exactly the pre-DD
behaviour, byte for byte, and is what the bit-for-bit control checks.

    uv run python b65/extract.py --tokens 4608 --chunk 512 --layers 0,1 \\
        --blocks all --diagonal-block --slim --share out/w/shared --out out/w/witness
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))

from .. import provenance  # noqa: E402
from .common import MODEL, SPECS, setup, tokens  # noqa: E402
from .model import BF16, Cache, Model, Recorder  # noqa: E402
from .quant import pow2_ceil, round_hafz  # noqa: E402
# issue #118: the bound below is the shape table's too, so it is defined once,
# in the module that has no torch and no numpy at import time
from .shapes import (ao_no_wrap, ao_no_wrap_refusal, ao_row_of,  # noqa: E402
                     ao_row_window, ao_states_window)

OUT = os.path.join(HERE, "out", "witness")
MATS = ["q", "k", "v", "o", "gate", "up", "down"]

# B10 step 1: the attention block's three *derived* (nh, S, C) columns.
# `AttnWitness::load` reads only `qq.i16`, `kk.i16`, `vv.i8` and `num.i32` from a
# dump -- the scores, the `d = max - sc` gaps and the bf16 exp-table inputs are
# what the AIR recomputes and commits itself, and nothing in the prover opens
# them.  They are kept for diagnosis, and at C=16K they are 2.50 GB of a 4.03 GB
# witness, which is the difference between the model's 32 layers fitting on a
# disk and not.  `--slim` hashes them into the manifest and does not write them.
# `mm` and `zz` stay: they are (nh, S), 64 KiB, and the model-wide shape table
# reads `zz` (`b10/gpu/shapes32.py` computes `ao_rem_bits` from it).
DERIVED = ("sc.i32", "dd.i32", "xx.u16", "scc.i32", "ddc.i32", "xxc.u16")
SLIM = False
# Stream DD: the second disk lever.  `num.i32` is 84.2% of the per-prefix-key
# term (nh*S*4 of nh*hd*3 + nh*S*4 = 65,536 of 77,824 bytes at 8B) and the
# prover reads it for one thing only -- `torch_num`, which becomes
# `AttnWitness::torch_mismatch` and feeds no constraint (`attn_witness.rs:111,
# 851-853`; `attn_run.rs:104` reports it).  `--slim-num` hashes it into the
# manifest and does not write it, exactly as `--slim` does for the derived
# columns.  It is OFF by default.  It needed a prover that reads the file as
# optional, and since `bda8ecc` (2026-09-01) it has one: `Half::read` takes a
# `slim_num` flag and `num32` is an `Option`, keyed off the manifest's
# `slim_num`.  A dump that does not declare it and is missing the file is still
# an error.  See `briefs/dd-extract.md`'s Record for the policy call.  (This
# comment used to say the loader opened it unconditionally; that outdated line
# is what put open item 11 in the queue after the fix had already landed.)
NUMFILES = ("num.i32", "numc.i32")
SLIM_NUM = False
# Stream DD: the first disk lever.  These files are a function of the *layer*
# and not of the token block -- the quantized weights, their scales, the two
# norm weights and the offline V scales -- so one copy serves every block.
# `--share DIR` writes them once under `DIR/L<k>/` and leaves a *relative*
# symlink in each block's dump (the pattern at `b10/out/x/ctl_witness_L1/`).
# The manifest's sha256 is taken from the array in memory either way, so a
# shared dump hashes identically to a copied one.
SHARE = None


def is_shared(name):
    """Is this file the same for every token block of one layer?"""
    # Stream BB: `embed.u16` is the model's embedding matrix -- the same bytes
    # for every block of every layer, and 1.05 GB at 8B, so it is shared like
    # the weights rather than copied per block.
    return (name.startswith(("qw_", "jw_", "sw_"))
            or name in ("jv.i32", "w1.u16", "w2.u16", "embed.u16"))


class LastChunk(Recorder):
    """Keep only the newest value of each column: the newest prefill chunk.

    Stream DD: this *assignment* (against `Recorder.add`'s append) is the whole
    reason the block axis is free.  The recorder holds exactly one block, so the
    dump body reads block `b` if it runs after block `b`'s forward call and
    block `B-1` if it runs after the loop -- and nothing else has to change.
    """

    def __init__(self, keep):
        super().__init__()
        self.keep = keep

    def add(self, key, t):
        if key in self.keep:
            # B9: B8 runs the forward pass on the GPU, so a recorded column is a
            # cuda tensor and every `.numpy()` below would fail; the dump is host
            # bytes either way, so it comes across here, once, per chunk.
            self.cols[key] = t.detach().to("cpu").clone()


def write(name, arr, digests):
    arr = np.ascontiguousarray(arr)
    held = (SLIM and name in DERIVED) or (SLIM_NUM and name in NUMFILES)
    link = None
    if not held and SHARE is not None and is_shared(name):
        # one copy per layer, symlinked into this block's dump.  Written only
        # when it is missing or the wrong length, so a re-run of one block does
        # not rewrite 218 MB of weights and a half-written file is not reused.
        src = os.path.join(SHARE, name)
        os.makedirs(SHARE, exist_ok=True)
        if not os.path.exists(src) or os.path.getsize(src) != arr.nbytes:
            arr.tofile(src)
        dst = os.path.join(OUT, name)
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
        link = os.path.relpath(src, OUT)
        os.symlink(link, dst)
    elif not held:
        arr.tofile(os.path.join(OUT, name))
    digests[name] = dict(values=int(arr.size), dtype=str(arr.dtype), shape=list(arr.shape),
                         sha256=hashlib.sha256(arr.tobytes()).hexdigest()[:16])
    if held:
        digests[name]["written"] = False
    if link:
        digests[name]["link"] = link
    note = ("  [derived, hashed only]" if held
            else f"  [-> {link}]" if link else "")
    print(f"  {name:<14} {arr.size:>12,} values ({arr.nbytes / 2 ** 20:6.1f} MiB)  {arr.shape}"
          f"{note}")


def parse_blocks(spec, n):
    """Which token blocks to dump.  `last` is the pre-DD behaviour."""
    if spec in ("", "last"):
        return {n - 1}
    if spec == "all":
        return set(range(n))
    if spec == "prompt":                       # every block but the decoded one
        return set(range(n - 1))
    want = set()
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            want |= set(range(int(lo), int(hi) + 1))
        else:
            want.add(int(part))
    bad = sorted(b for b in want if not 0 <= b < n)
    assert not bad, f"--blocks names {bad}, outside 0..{n - 1}"
    return want


def bits_for(v):
    return int(v).bit_length()


def bf16_parts(t):
    """(mantissa field, biased exponent) of a bf16-valued tensor."""
    # B9: on the B8 path the model's own tensors live on the GPU
    bits = t.detach().to("cpu").to(BF16).view(torch.int16).numpy().astype(np.uint16)
    return (bits & 0x7F).astype(np.int32), ((bits >> 7) & 0xFF).astype(np.int32)


def _pow2_mat(spec, name):
    """Does this spec quantize `name` with power-of-two scales? (`model.py`'s rule)"""
    return spec.pow2_scales in ("all", "linear") or name in spec.pow2_matmuls


def _rope_params_block(m):
    """Stream NN: `rope_theta`, the `inv_freq_of` branch and its constants.

    Returned as `{theta, rope_type, scale}` with `scale` in the order the
    prover's `rope::inv_freq_of` reads it: `llama3` takes
    `[factor, low_freq_factor, high_freq_factor, original_max_position_embeddings]`,
    `proportional` takes `[partial_rotary_factor, factor]`, `default` takes none.
    `None` for a model whose layers have different rope parameters -- the
    statement would have to name which, and this stream does not carry that.
    """
    if getattr(m, "inv_freq", None) is None:
        return None
    rp = dict(m.cfg.rope_parameters) if hasattr(m, "cfg") else None
    if rp is None:
        return None
    kind = rp.get("rope_type", "default")
    if kind == "llama3":
        scale = [float(rp["factor"]), float(rp["low_freq_factor"]),
                 float(rp["high_freq_factor"]),
                 float(rp["original_max_position_embeddings"])]
    elif kind == "proportional":
        scale = [float(rp.get("partial_rotary_factor", 1.0)), float(rp.get("factor", 1.0))]
    elif kind == "default":
        scale = []
    else:
        raise NotImplementedError(f"rope_type {kind}")
    return dict(theta=float(rp["rope_theta"]), rope_type=kind, scale=scale)


def pow2_exponent(t):
    """`j` such that the value is exactly `2^j`; NaN-safe, and -1000 if it is not."""
    v = t.detach().to("cpu").double().numpy()
    with np.errstate(divide="ignore"):
        j = np.log2(np.abs(v))
    exact = np.isfinite(j) & (np.abs(j - np.round(j)) < 1e-9)
    return np.where(exact, np.round(j), -1000).astype(np.int32)


def dump_dir(base, block, layer, *, tag_blocks, tag_layer):
    """The directory one (block, layer) dump is written to.

    Both axes are in the *name* and nowhere else: the prover knows no directory
    grammar (`grep -rn "witness_L" prover/` returns nothing; every path is
    `format!("{dir}/<literal>")`), so a block is a directory and a layer is a
    directory.  `_B<b>` goes before `_L<k>`, which is the order `shapes.py`
    walks (`shape_of(f"{base}{k}")` over `<out>_B<b>_L`) and the order
    `plan.py` builds (`wpre = f"{a.dir}/witness_B{b}_L"`).

    The layer is tagged whenever `--layers` chose it, one layer or eight.  A
    sequence extraction has to run layer 0 in a pass of its own -- `--embed`
    writes the embedding table into every dump of the pass it is given, and the
    prover turns the embedding gadget on for any dump that carries one -- so a
    one-layer pass is the normal case, not the odd one, and it is named like
    every other.  `--layer N`, the singular form the head and the reference
    dumps use, is untagged as it was.
    """
    d = f"{base}_B{block}" if tag_blocks else base
    return f"{d}_L{layer}" if tag_layer else d


def dump_digest(files):
    """The dump's identity: the per-array digests its manifest lists, in one number.

    A manifest already carries one entry per array under `files` -- the name,
    the element count, the dtype, the shape and the array's sha256 -- and
    nothing rolled them into a number a record can quote.  So a recorded proof
    digest named the commit that made it and not the dump it was made from, and
    when the 8B reference digest moved with no prover commit behind it, nothing
    in either record could say that the dump had moved (issue #130).

    The roll-up is over the arrays alone: the entries sorted by name, one line
    each, `name values dtype shape sha256`.  Not over the manifest's own bytes,
    which carry the ids path this host read and the torch version it ran, so two
    hosts that wrote the same arrays would get two numbers out of them.  Not
    over `written` or `link` either: `--slim` and `--share` decide what is on
    disk and not what the extraction computed, and a shared dump already hashes
    like a copied one.

    It is a claim about the arrays and not a re-reading of them -- each entry's
    sha256 is taken from the array in memory as it is written -- so a reader
    that wants the bytes checked has to hash the files itself.

    `prover/src/util.rs` rolls the same lines up the same way, over the `files`
    map of any dump it reads, and writes the result into the run JSON's
    provenance block.  The two are pinned against one number: the case below
    and `prover/tests/dump_identity.rs` name the same map and the same digest.
    """
    h = hashlib.sha256()
    h.update(b"zk-dump-v1\n")
    for name in sorted(files):
        f = files[name]
        shape = ",".join(str(int(x)) for x in f["shape"])
        h.update(f"{name} {int(f['values'])} {f['dtype']} {shape} {f['sha256']}\n".encode())
    return h.hexdigest()


def attn_ranges_source(path, given):
    """`attn_ranges`: the attention calibration this run quantized with.

    `out/attn_ranges.json` holds max|k| per (layer, head), max|v| per (layer,
    head, channel) and max|q| per (layer, head), from a forward pass over the
    calibration slice.  The K half of it becomes `jk.i32`, the per-head
    power-of-two K scale exponents, and those move the integer scores, the shape
    table's `z_max` and the model tree's root.  It is computed on the host when
    it is absent and reused when it is present, so it is a per-host input that
    no commit pins and, until this, no dump recorded (issue #130).

    `given` is whether `--attn-ranges FILE` named it, which is the difference
    between a dump quantized under a calibration a reader can get back and one
    quantized under whatever the host had.
    """
    return {**provenance.digest(path), "path": os.path.realpath(path), "given": bool(given)}


def ids_source(path, ids):
    """`ids_from`: the ids file this run actually read, by path and by digest.

    A path is not enough to name the ids a dump was extracted from.  The
    one-spec re-decode wrote a second ids file over the first path at some
    points, so a manifest that names a path names whichever file was there when
    it was read, and a dump directory reused for the second decode kept the
    first decode's `ids_from`.  The digest is of the file this run read, and
    `ids_stale` compares it, so the manifest either names the ids the dump was
    extracted from or the extraction does not happen.
    """
    if path is None:
        return None
    return {**provenance.digest(path), "path": os.path.realpath(path),
            "tokens": int(len(ids))}


def ids_stale(manifest_path, source):
    """Why this dump directory's manifest names other ids than this run reads, or None.

    The digest decides: the same bytes under another name are the same ids, and
    re-extracting over them is the ordinary resume every driver does.  A
    manifest written before this field carried a digest names a bare path, and
    then the file name is all there is to compare -- which the reason says,
    rather than the check assuming either way.
    """
    try:
        with open(manifest_path) as f:
            old = json.load(f)["ids_from"]
    except (OSError, ValueError, KeyError):
        return None                        # no dump here, or none this can read
    if old is None and source is None:
        return None
    if old is None:
        return f"was extracted from the corpus; this run reads {source['path']}"
    if source is None:
        return (f"was extracted from {old['file'] if isinstance(old, dict) else old}; "
                f"this run reads the corpus")
    if isinstance(old, str):               # an extractor before the digest
        if os.path.basename(old) == source["file"]:
            return None
        return (f"names {old}, from before the manifest carried a digest, and this "
                f"run reads {source['path']}: only the file name could be compared")
    if old.get("sha256") != source["sha256"]:
        return (f"was extracted from {old.get('path', old['file'])} "
                f"(sha256 {str(old.get('sha256'))[:16]}), and this run reads "
                f"{source['path']} (sha256 {source['sha256'][:16]})")
    return None


def clear_manifest(out):
    """Drop the manifest before the dump beside it is rewritten.

    A manifest is written when a dump is finished, so a manifest beside a
    half-written dump is the one thing an interrupted run must not leave.  The
    drivers read exactly that -- a dump directory with a manifest is done and is
    skipped -- so removing it here is what makes their resume true.
    """
    path = os.path.join(out, "manifest.json")
    if os.path.exists(path):
        os.remove(path)


def selftest_dirs():
    """The directory grammar, against the two readers.  No model, no card."""
    base = "out/witness"
    cases = [
        # (the call, block-tagged, layer-tagged, block, layer, the name)
        ("--layers 0 --blocks all --embed", True, True, 0, 0, "out/witness_B0_L0"),
        ("the same pass, last block", True, True, 8, 0, "out/witness_B8_L0"),
        ("--layers 1,2,3,4,5,6,7 --blocks all", True, True, 0, 7, "out/witness_B0_L7"),
        ("--layers 12, last block only", False, True, 0, 12, "out/witness_L12"),
        ("--layer 12 --blocks all", True, False, 3, 12, "out/witness_B3"),
        ("--layer 12, the reference dump", False, False, 0, 12, "out/witness"),
    ]
    bad = 0
    for label, tag_blocks, tag_layer, b, k, want in cases:
        got = dump_dir(base, b, k, tag_blocks=tag_blocks, tag_layer=tag_layer)
        bad += got != want
        print(f"{'OK  ' if got == want else 'FAIL'}  {label:<40s} -> {got}")
    # The claim the defect broke: a 32-layer shape table over a block reads
    # exactly what 32 one-layer passes wrote, and layer 0 is one of them.
    n = 32
    read = [f"{base}_B0_L{k}" for k in range(n)]
    wrote = [dump_dir(base, 0, k, tag_blocks=True, tag_layer=True) for k in range(n)]
    ok = read == wrote
    bad += not ok
    print(f"{'OK  ' if ok else 'FAIL'}  {n} one-layer passes == what shapes.py and "
          f"plan.py read over <base>_B0_L")
    return bad


def selftest_ids():
    """`ids_from` and the reuse it goes stale under.  No model, no card."""
    tmp = tempfile.mkdtemp(prefix="extract-selftest-")
    try:
        first, second = os.path.join(tmp, "ids_a.npy"), os.path.join(tmp, "ids_b.npy")
        np.save(first, np.arange(8, dtype=np.int64))
        np.save(second, np.arange(8, dtype=np.int64) + 1)
        copy = os.path.join(tmp, "ids_a_again.npy")
        shutil.copyfile(first, copy)
        link = os.path.join(tmp, "ids_link.npy")
        os.symlink(first, link)
        ids = np.load(first)
        src = ids_source(first, ids)
        corpus = ids_source(None, ids)

        out = os.path.join(tmp, "witness")
        os.makedirs(out)
        man = os.path.join(out, "manifest.json")

        def manifest(ids_from):
            with open(man, "w") as f:
                json.dump({"ids_from": ids_from, "layer": 0}, f)

        cases = []
        cases.append(("a directory with no dump in it is not a reuse",
                      ids_stale(man, src) is None))
        manifest(src)
        cases.append(("the same ids file again is the ordinary resume",
                      ids_stale(man, src) is None))
        cases.append(("REFUSED: the same directory over the re-decode's ids",
                      bool(ids_stale(man, ids_source(second, np.load(second))))))
        cases.append(("the same bytes under another name are the same ids",
                      ids_stale(man, ids_source(copy, np.load(copy))) is None))
        cases.append(("REFUSED: a dump from an ids file, re-extracted from the corpus",
                      bool(ids_stale(man, corpus))))
        manifest(None)
        cases.append(("REFUSED: a dump from the corpus, re-extracted from an ids file",
                      bool(ids_stale(man, src))))
        manifest("ids_a.npy")
        cases.append(("an older manifest naming the same file: the name is all there is",
                      ids_stale(man, src) is None))
        cases.append(("REFUSED: an older manifest naming another file",
                      bool(ids_stale(man, ids_source(second, np.load(second))))))
        with open(first, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        through = ids_source(link, ids)
        cases.append(("`ids_from` is the digest of the file the run read, "
                      "resolved through the link it was given",
                      through["sha256"] == sha and through["path"] == os.path.realpath(first)
                      and through["tokens"] == len(ids)))
        manifest(src)
        cases.append(("the link and the file behind it are one ids file",
                      ids_stale(man, through) is None))
        clear_manifest(out)
        cases.append(("the manifest goes before the dump beside it is rewritten, "
                      "so an interrupted run leaves none",
                      not os.path.exists(man) and ids_stale(man, corpus) is None))
        bad = 0
        for label, ok in cases:
            bad += not ok
            print(f"{'OK  ' if ok else 'FAIL'}  {label}")
        return bad
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# Issue #130: the one number the prover and the extractor have to agree on.
# `prover/tests/dump_identity.rs` names this map and this digest too; the prover
# rolls the map up in Rust and writes the result into its run JSON, and if the
# two implementations ever part company, one of the two tests says so.
GOLDEN_FILES = {
    "tok_head.i32": dict(values=8, dtype="int32", shape=[8],
                         sha256="e5e9f2e7cf0c271d"),
    "key_head.u16": dict(values=4096, dtype="uint16", shape=[8, 512],
                         sha256="132ce9b6a2a54e71"),
}
GOLDEN_DUMP = "badad7e68e76e9c83289dc3f18abcab667fb3a4039677edaf9ff371607a71f26"


def selftest_dump():
    """The dump digest and the calibration it is pinned beside.  No model, no card."""
    from .ppl import attention_ranges     # noqa: E402  (no model: it reads a file)

    tmp = tempfile.mkdtemp(prefix="extract-selftest-")
    try:
        base = dump_digest(GOLDEN_FILES)
        moved = dict(GOLDEN_FILES)
        moved["key_head.u16"] = dict(GOLDEN_FILES["key_head.u16"],
                                     sha256="0000000000000000")
        added = dict(GOLDEN_FILES, **{"jk.i32": dict(values=32, dtype="int32", shape=[32],
                                                     sha256="1111111111111111")})
        # the same two arrays in the other order, one of them held by `--slim`
        # and one of them a `--share` symlink
        other = {k: dict(GOLDEN_FILES[k]) for k in reversed(list(GOLDEN_FILES))}
        other["tok_head.i32"]["written"] = False
        other["key_head.u16"]["link"] = "../shared/L0/key_head.u16"

        acal = os.path.join(tmp, "attn_ranges.json")
        with open(acal, "w") as f:
            json.dump({"L0.k": [1.0, 2.0], "L0.q": [3.0], "L0.v": [[4.0]]}, f)
        with open(acal, "rb") as f:
            asha = hashlib.sha256(f.read()).hexdigest()
        link = os.path.join(tmp, "given.json")
        os.symlink(acal, link)

        cases = [
            ("the dump digest is the roll-up `prover/tests/dump_identity.rs` names",
             base == GOLDEN_DUMP),
            ("the same arrays in the other order, one of them held by `--slim` and "
             "one of them a `--share` link, are the same dump",
             dump_digest(other) == base),
            ("one array's digest moves it", dump_digest(moved) != base),
            ("an array the extraction did not write before moves it",
             dump_digest(added) != base),
            ("a dump that lists no array at all is neither of them",
             dump_digest({}) not in (base, dump_digest(moved))),
            ("the calibration is named by the digest of the file that was read, "
             "resolved through the link it was given",
             attn_ranges_source(link, True)["sha256"] == asha
             and attn_ranges_source(link, True)["path"] == os.path.realpath(acal)),
            ("a calibration this host computed is recorded too, and says it was not given",
             attn_ranges_source(acal, False)["given"] is False),
            ("`--attn-ranges FILE` is what the quantization then reads",
             attention_ranges(acal)["L0.k"] == [1.0, 2.0]),
        ]
        bad = 0
        for label, ok in cases:
            bad += not ok
            print(f"{'OK  ' if ok else 'FAIL'}  {label}")
        return bad
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def selftest():
    return 1 if selftest_dirs() + selftest_ids() + selftest_dump() + no_wrap_selftest() else 0


def no_wrap_selftest():
    """Issue #111: the no-wrap bound, against the shape the prover pins it on.

    The first case is `params_no_prefix`, the synthetic shape
    `prover/src/compose.rs` pins its own window on
    (`the_no_wrap_window_reports_its_margin`, PR #110): two limbs of 16 and 4
    with `e_hi = 4` is 25 of the field's 31, so six bits of margin, and the
    sibling bound is 15 of 30.  This has no `Params`, so it reads the same shape
    off the numbers a dump carries -- an accumulator of 19 bits, `jv_max = -6`,
    `jo_min = -10`, a fold at int8 and a divisor range of 8 bits -- and has to
    land on the same window.  The rest walk the margin down to nothing and one
    bit past it, on each of the two bounds.
    """
    cases = [
        # (the label, the dump's numbers, the window, its margin, the form)
        ("the synthetic shape the prover pins its window on",
         (19, -6, -10, 8, 8), 25, 6, "offset", 15, 15, "plain"),
        # the margin spent down to zero still composes: this is where layer 0
        # sits under the paper's quantization
        ("the margin spent to zero, in the offset form",
         (25, -6, -10, 8, 8), 31, 0, "offset", 15, 15, "plain"),
        # one more bit of accumulator and the offset form no longer fits, so the
        # magnitude form is taken -- the one bit it is worth, and no more
        ("one bit more, and the magnitude form is what fits",
         (26, -6, -10, 8, 8), 31, 0, "magnitude", 15, 15, "plain"),
        # and one more past that: the product form bounds `mag num` slot by
        # slot, so the limbs stop adding and the window is `1 + max(lo, hi) + e_hi`
        ("one bit past the magnitude form, in the product form",
         (27, -6, -10, 8, 8), 21, 10, "product", 15, 15, "plain"),
        # the Llama 3.1 8B dump this form was written for: block 0 layer 1 of a
        # 512-token table of contents (`table_of_contents` in
        # `harness/witness/fixtures/no_wrap_prompts.json`), whose
        # 22-bit accumulator the magnitude form refused at `1 + 16 + 6 + 9 = 32`
        # (the window's three numbers are that dump's; the quotient's two are the
        # synthetic shape's, since that half is the pin's and not this form's)
        ("the 8B table of contents, block 0 layer 1",
         (22, -5, -14, 8, 8), 26, 5, "product", 15, 15, "plain"),
        # the sibling bound, at the E11 set's fold width: 11 + 20 is past the
        # plain form's 30, so the denp2 form is taken and lands on its own cap
        ("the quotient's margin spent to zero, in the denp2 form",
         (19, -6, -10, 12, 20), 25, 6, "offset", 31, 0, "denp2"),
    ]
    bad = 0
    for label, args, bits, margin, form, q_bits, q_margin, q_form in cases:
        w = ao_no_wrap(*args)
        got = (w["bits"], w["margin"], w["form"],
               w["quotient"]["bits"], w["quotient"]["margin"], w["quotient"]["form"])
        want = (bits, margin, form, q_bits, q_margin, q_form)
        ok = got == want and ao_no_wrap_refusal(w, "L0") is None
        bad += not ok
        print(f"{'OK  ' if ok else 'FAIL'}  {label:<52s} -> {w['bits']}/{w['limit']} "
              f"margin {w['margin']} ({w['form']}), quotient "
              f"{w['quotient']['bits']}/{w['quotient']['limit']} margin "
              f"{w['quotient']['margin']} ({w['quotient']['form']})")
    # and one bit past each of the two, which is the refusal this check exists
    # for: an extraction that would not prove, named with its numbers
    for label, args, want in [
        ("an exponent window one bit past the product form",
         (27, -6, -21, 8, 8), "1 + max(16, 11) + 15 = 32 > 31 (over by 1, product form)"),
        ("a divisor range one bit past the quotient term",
         (19, -6, -10, 12, 21), "11 + 21 = 32 > 31 (over by 1, denp2 form)"),
    ]:
        got = ao_no_wrap_refusal(ao_no_wrap(*args), "block 0 layer 0") or ""
        ok = got.startswith("block 0 layer 0: ") and want in got
        bad += not ok
        print(f"{'OK  ' if ok else 'FAIL'}  {label:<52s} -> "
              f"{'refused: ' + want if ok else 'NOT refused: ' + got}")
    # Issue #121: and the same window off the manifest this extraction writes.
    # The check above runs on numbers that are in memory; the shape table and
    # the plan writer run on the manifest, so what the manifest carries has to
    # come back as the same window -- otherwise the two evaluate different
    # bounds and only the one nobody reads is right.
    acc, jv_max, jo_min, out_bits, rem = 19, -6, -10, 8, 8
    man = dict(attn_out=dict(jo_min=jo_min, jo_max=-10, jv_min=-8, jv_max=jv_max,
                             fold_q_max=127, ao_acc_bits=acc))
    row = dict(ao_row_of(man), attn_out_bits=out_bits, ao_rem_bits=rem)
    w, direct = ao_row_window(row, rem), ao_no_wrap(acc, jv_max, jo_min, out_bits, rem)
    ok = ao_states_window(row) and w == direct
    bad += not ok
    print(f"{'OK  ' if ok else 'FAIL'}  {'the manifest block is the same window':<52s} -> "
          f"{w['bits']}/{w['limit']} margin {w['margin']} ({w['form']})")
    # and a manifest written before this field existed states no window at all,
    # which is the case the shape table reports rather than guesses
    old_man = dict(attn_out={k: v for k, v in man["attn_out"].items() if k != "ao_acc_bits"})
    old_row = dict(ao_row_of(old_man), attn_out_bits=out_bits, ao_rem_bits=rem)
    ok = not ao_states_window(old_row) and ao_row_window(old_row, rem) is None
    bad += not ok
    print(f"{'OK  ' if ok else 'FAIL'}  {'a manifest without it states no window':<52s} -> "
          f"ao_acc_bits {old_row['ao_acc_bits']}")
    return bad


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=1536)
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--layer", type=int, default=12)
    # B7b: several layers from one forward pass, each to `<out>_L<k>`, and one
    # layer is named the same way -- see `dump_dir`
    p.add_argument("--layers", default=None,
                   help="comma-separated layer indices (overrides --layer); every "
                        "layer of the pass is written to <out>[_B<b>]_L<k>")
    p.add_argument("--out", default=OUT, help="output directory (base, for --layers)")
    p.add_argument("--spec", default="b65-design")
    p.add_argument("--mats", default=",".join(MATS))
    # B10 step 1: the sequence's witness is a real *decode*, so the ids are the
    # prompt plus `b10/gpu/gen_ids.py`'s greedy continuation rather than the
    # eval corpus's next 512 tokens (B8 step 1: decode is bit-identical to
    # prefill, so the extractor still prefills the concatenation).
    p.add_argument("--ids", default=None,
                   help="token ids (.npy) to run instead of the corpus's first --tokens")
    # Issue #130: the attention calibration is an input, so it can be given like
    # one.  Without this the file is whatever the host computed the first time
    # anything asked for it, and two hosts quantize the same weights differently.
    p.add_argument("--attn-ranges", default=None,
                   help="the attention calibration (out/attn_ranges.json) to quantize "
                        "K, V and Q with, instead of the one this host has or would "
                        "compute; the dump's manifest records its digest either way")
    p.add_argument("--slim", action="store_true",
                   help="hash the derived attention columns into the manifest, do not write them")
    # ---- stream DD (E11): the block axis and the two disk levers -------------
    p.add_argument("--blocks", default="last",
                   help="which token blocks to dump: 'last' (the pre-DD behaviour, and "
                        "the only value that leaves the output paths unchanged), 'all', "
                        "'prompt' (every block but the decoded one), or a list/range "
                        "like '1-8' or '0,3,4'")
    p.add_argument("--share", default=None,
                   help="write the block-invariant files (qw_*, jw_*, sw_*, jv, w1, w2) "
                        "once under DIR/L<k>/ and symlink them into every block's dump; "
                        "177.5 -> 121.6 GB on a 4096+512 sequence at 32 layers")
    p.add_argument("--slim-num", action="store_true",
                   help="hash num.i32/numc.i32 into the manifest and do not write them "
                        "(121.6 -> ~74 GB, and it needs a prover that reads them as "
                        "optional -- see the module docstring)")
    p.add_argument("--residual-stream", action="store_true",
                   help="also dump B10 step 2's wiring columns (residual stream, "
                        "norm weights and rsqrt/silu lookups)")
    # Stream BB (E11): the embedding gadget's two inputs.  Only meaningful on a
    # layer-0 dump -- `xin` is the embedding's output there and nothing else's --
    # and the prover's `EWitness::from_cols` refuses a dump where it is not.
    p.add_argument("--embed", action="store_true",
                   help="also dump the embedding matrix and this block's input token "
                        "ids (stream BB; layer 0 only)")
    p.add_argument("--head", action="store_true",
                   help="also dump B10 step 4's `lm_head` chunk (the accumulators, the "
                        "quantized weights, the logits' order keys and the emitted tokens)")
    p.add_argument("--diagonal-block", action="store_true",
                   help="also dump B10 step 3's diagonal block (the S decoded tokens "
                        "against each other): one joint row maximum, the total "
                        "denominator, and the diagonal block's own keys and values")
    p.add_argument("--selftest", action="store_true",
                   help="check the output-directory grammar against what the shape "
                        "table and the plan read; no model and no card")
    a = p.parse_args()
    if a.selftest:
        sys.exit(selftest())
    SLIM = a.slim
    SLIM_NUM = a.slim_num
    setup()
    layers = [int(x) for x in a.layers.split(",")] if a.layers else [a.layer]
    TAG_LAYER = a.layers is not None
    OUT_BASE = a.out
    spec = SPECS[a.spec]
    assert spec.int_softmax, f"{a.spec} is not an integer-softmax spec"
    assert spec.attn_out_int8 and spec.attn_out_pow2, \
        f"{a.spec} does not use the provable int8 attention-output fold"
    mats = a.mats.split(",")

    from .ppl import ACALIB, attention_scales  # noqa: E402

    # Issue #130: which calibration this extraction quantizes with, resolved
    # before the model like the ids are.  `--attn-ranges` is read and never
    # written, so a file that is not there is a refusal and not a fresh
    # calibration under that name; without the flag it is the host's own, which
    # `attention_ranges` computes if the host has none yet.
    ATTN_RANGES = a.attn_ranges
    if ATTN_RANGES is not None and not os.path.exists(ATTN_RANGES):
        sys.exit(f"--attn-ranges {ATTN_RANGES}: no such file.  Name a calibration that "
                 f"exists, or leave the flag off to use this host's own.")

    # Stream DD: the ids, the block axis and the reuse check all come before the
    # model, so a refusal costs the operator nothing.
    if a.ids:
        ids = torch.from_numpy(np.load(a.ids).astype(np.int64))
        assert len(ids) == a.tokens, f"--ids has {len(ids)} tokens, --tokens says {a.tokens}"
        print(f"  ids from {a.ids}: {len(ids)} tokens, the last {a.chunk} decoded")
    else:
        ev, _ = tokens()
        ids = ev[: a.tokens]

    nblocks = (len(ids) + a.chunk - 1) // a.chunk
    # Stream DD: the block axis is only well defined when every block is the
    # same width -- a ragged last block would give it a different `S` from the
    # one its own prefix is a whole number of, and every driver in the tree
    # already passes a multiple.
    assert len(ids) % a.chunk == 0, \
        f"--tokens {len(ids)} is not a whole number of --chunk {a.chunk} blocks"
    want = parse_blocks(a.blocks, nblocks)
    # `last` leaves the output paths exactly where they were, which is what the
    # byte-for-byte regression check (a re-extraction reproduces the existing
    # dump) compares against.  Any other value tags the block into the path.
    tag_blocks = a.blocks not in ("", "last")
    print(f"  {nblocks} blocks of {a.chunk}; dumping {sorted(want)}"
          f"{' (block-tagged paths)' if tag_blocks else ''}")

    # A manifest and the files beside it are one dump, and a reader takes the
    # ids the manifest names (`boundary-grid --verify --prompt`).  A directory
    # reused for an extraction over *other* ids would hold one decode's dump
    # under the other's manifest, which is what the one-spec re-decode left
    # behind.  So the reuse is refused, and refused here: before the model is
    # loaded, not after an hour of card time.
    ids_from = ids_source(a.ids, ids)
    stale = []
    for bi in sorted(want):
        for L in layers:
            d = dump_dir(OUT_BASE, bi, L, tag_blocks=tag_blocks, tag_layer=TAG_LAYER)
            why = ids_stale(os.path.join(d, "manifest.json"), ids_from)
            if why:
                stale.append(f"  {d}: the manifest there {why}")
    if stale:
        sys.exit("refusing to reuse a dump directory whose manifest names other ids "
                 "than this run reads:\n" + "\n".join(stale)
                 + "\nRemove those directories, or dump to another --out, and extract again.")

    m = Model(MODEL, spec, attn_scales=attention_scales(spec, ATTN_RANGES), track_acc=True)
    # after the model, because a host with no calibration yet writes it on the
    # way in and the digest has to be of the file this extraction quantized with
    attn_ranges = attn_ranges_source(ATTN_RANGES or ACALIB, ATTN_RANGES is not None)
    print(f"  attn_ranges {attn_ranges['path']}: sha256 {str(attn_ranges['sha256'])[:16]}"
          f" ({'given' if attn_ranges['given'] else 'calibrated on this host'})")
    m.clamped = {}
    keep = set()
    for L in layers:
        # B10 step 1: `expnum/in` and `expnum/out` are (nh, S, C) and nothing
        # below reads them -- `num` is recomputed from `xx` here.  At C=16K they
        # are 1.5 GB of host memory per layer, and holding 32 layers from one
        # forward pass is what makes the model's 32 dumps one pass and not eight.
        keep |= {f"L{L}.qq", f"L{L}.kk", f"L{L}.vv", f"L{L}.qkacc", f"L{L}.sqk",
                 f"L{L}.attnq", f"L{L}.s_o", f"L{L}.zz"}
        keep |= {f"L{L}.{n}/acc" for n in mats} | {f"L{L}.{n}/s_x" for n in mats}
        keep |= {f"L{L}.{n}/qx" for n in mats} | {f"L{L}.{n}/x" for n in mats}
        # B10 step 2: the residual stream's own columns -- the residual stream at the three
        # points the two adds join, each add's second operand, the two norms'
        # rsqrt lookups and the SwiGLU's.  `--residual-stream` gates them because they are
        # 5 (S, d) + 2 (S, dff) bf16 columns, which at C=16K is another 1.2 GB a
        # layer and nothing in steps 0-1 reads them.
        if a.residual_stream:
            keep |= {f"L{L}.xin", f"L{L}.xmid", f"L{L}.x", f"L{L}.o_out", f"L{L}.d_out",
                     f"L{L}.norm1.rsqrt/in", f"L{L}.norm1.rsqrt/out",
                     f"L{L}.norm2.rsqrt/in", f"L{L}.norm2.rsqrt/out",
                     f"L{L}.silu/in", f"L{L}.silu/out"}
    if a.head:
        # B10 step 4: the head is a `Linear` like any other, but it has no layer
        # prefix -- `m.head = Linear(..., "lm_head", ...)` -- so its recorder keys
        # are bare.
        keep |= {"lm_head/acc", "lm_head/qx", "lm_head/s_x"}
        # Stream II (E11): the head chunk's own norm scale.  `lm_head` consumes
        # the int8 activation of `rmsnorm(x, final_norm)`, and no dump carried
        # either operand, so the final RMSNorm was proved by nobody (stream KK).
        # The norm scale needs the last layer's committed residual (its input), the
        # model's final-norm weight (preprocessed, tree 0) and B1's own rsqrt
        # lookup and head input as the two references the component reproduces.
        keep |= {f"L{m.L - 1}.x", "final_norm.rsqrt/in", "final_norm.rsqrt/out",
                 "lm_head/x"}
    m.rec = LastChunk(keep)
    cache = Cache(m.L)
    t0 = time.time()
    logits = None

    def dump_blocks(bi, base):
        """Dump every requested layer's witness for token block `bi`.

        Stream DD's loop inversion.  `bi` is the block index and `base` the
        position of its first token, so the recorder holds the S queries of
        positions `base .. base + S - 1` against the whole `base + S` key
        prefix.  Called from inside the forward loop; called after it for the
        last block only, it is the pre-DD dump.
        """
        global OUT, SHARE

        for L in layers:
            OUT = dump_dir(OUT_BASE, bi, L, tag_blocks=tag_blocks, tag_layer=TAG_LAYER)
            SHARE = os.path.join(a.share, f"L{L}") if a.share else None
            os.makedirs(OUT, exist_ok=True)
            clear_manifest(OUT)
            print(f"block {bi} layer {L} -> {OUT}")
            digests = {}

            # ---- the attention block, in b5/extract.py's format ---------------------
            sc_all = m.rec.cols[f"L{L}.qkacc"].to(torch.int64)      # (nh, S, C)
            nh, S, C = sc_all.shape
            hd = m.hd
            # The block's rectangle: key c is visible to query t iff c <= (C - S) + t,
            # so the C - S prefix is fully visible and the S x S diagonal block is the only
            # masked part.  Stream DD: `P` is the *block index* times `S` -- block 3
            # of a 512-token sweep has a 1,536-key prefix -- so the pre-DD
            # power-of-two assertion was a restriction to the last block of a
            # power-of-two prompt and nothing deeper.  What the rest of the body
            # actually needs is that the prefix is a whole number of blocks.
            P = C - S
            assert P >= 0 and P % S == 0, \
                f"prefix {P} must be a whole number of {S}-token blocks"
            assert P or a.diagonal_block, \
                "block 0 has no prefix at all: it is diagonal-block-only and needs --diagonal-block"
            sc = sc_all[:, :, :P].contiguous()
            # B10 step 3: the diagonal block.  Key k of it *is* decoded token k, so token
            # t attends to it exactly when `k <= t`; `vis[t, k]` is that mask, and it is
            # a function of the position alone, which is why the AIR can preprocess it.
            scc = sc_all[:, :, P:].contiguous() if a.diagonal_block else None
            vis = (torch.arange(S).unsqueeze(0) <= torch.arange(S).unsqueeze(1)) if a.diagonal_block \
                else None

            # ONE row maximum for the pair.  `m_t` is a property of the token, not of a
            # half: the softmax is shift-invariant only when the same maximum is taken
            # against every key of the row, and the two halves' numerators would
            # otherwise be on two different scales (B10 rule / README 6.2).
            # `amax` has no identity on an empty axis, so block 0's maximum comes
            # from the diagonal block alone.  Everything downstream of `mm` is written
            # against a zero-width prefix without further help: `dd`, `num` and
            # `xx` are empty, `zz` is the diagonal block's sum, and the A.V einsum over a
            # zero-length key axis is the zero it should be.
            mm = (sc.amax(-1) if P else
                  torch.full((nh, S), torch.iinfo(torch.int64).min, dtype=torch.int64))
            if a.diagonal_block:
                mm = torch.maximum(mm, scc.masked_fill(~vis, torch.iinfo(torch.int64).min)
                                   .amax(-1))
            sqk = m.rec.cols[f"L{L}.sqk"].double()
            nmax = (1 << spec.num_bits) - 1

            def numer(d):
                """The spec's numerator for a shift array, and the bf16 exp input."""
                x = (-d.double() * sqk).float().to(BF16)
                n = round_hafz(nmax * torch.exp(x.double())).clamp(0, nmax).to(torch.int64)
                return x, n

            dd = mm.unsqueeze(-1) - sc
            xx, num = numer(dd)
            zz = num.sum(-1)
            if a.diagonal_block:
                # the true shift, which the AIR commits for *every* slot -- Freivalds is
                # over the whole rectangle -- and which is negative wherever a masked
                # key outscores the maximum the token was shifted by
                ddc = mm.unsqueeze(-1) - scc
                zero = torch.zeros_like(ddc)
                xxc, numc = numer(torch.where(vis, ddc, zero))
                numc = torch.where(vis, numc, torch.zeros_like(numc))
                zz = zz + numc.sum(-1)

            write("qq.i16", m.rec.cols[f"L{L}.qq"].numpy().astype(np.int16), digests)
            write("kk.i16", m.rec.cols[f"L{L}.kk"][:, :P].numpy().astype(np.int16), digests)
            write("vv.i8", m.rec.cols[f"L{L}.vv"][:, :P].numpy().astype(np.int8), digests)
            write("sc.i32", sc.numpy().astype(np.int32), digests)
            write("dd.i32", dd.numpy().astype(np.int32), digests)
            write("num.i32", num.numpy().astype(np.int32), digests)
            write("xx.u16", xx.contiguous().view(torch.int16).numpy().view(np.uint16), digests)
            write("mm.i32", mm.numpy().astype(np.int32), digests)
            write("zz.i32", zz.numpy().astype(np.int32), digests)
            if a.diagonal_block:
                write("kkc.i16", m.rec.cols[f"L{L}.kk"][:, P:].numpy().astype(np.int16), digests)
                write("vvc.i8", m.rec.cols[f"L{L}.vv"][:, P:].numpy().astype(np.int8), digests)
                write("scc.i32", scc.numpy().astype(np.int32), digests)
                write("ddc.i32", ddc.numpy().astype(np.int32), digests)
                write("numc.i32", numc.numpy().astype(np.int32), digests)
                write("xxc.u16", xxc.contiguous().view(torch.int16).numpy().view(np.uint16),
                      digests)

            # ---- the attention-output fold's scales ---------------------------------
            # The composed attnout component proves the *rectangle* sub-computation (the
            # same statement the key tiles prove; the diagonal block stays outside every AIR
            # in the project, B9's job), so its accumulators and denominators are the
            # recomputed `out`/`zz` above.  The V scales are the forward pass's,
            # power-of-two per (head, channel).
            s_v = attention_scales(spec, ATTN_RANGES)[f"L{L}.v"]    # (nh, 1, hd)
            jv = pow2_exponent(s_v.reshape(nh, hd))
            assert (jv != -1000).all(), "V scales are not powers of two -- wrong spec?"
            write("jv.i32", jv.reshape(-1), digests)

            # ---- stream NN: what the rotation and the q/k/v quantize need ------------
            # `jv.i32` above is `s_v`'s exponent.  `s_k` and (under
            # `q_scale_mode == "head"`) `s_q` are the other two scales
            # `attention_int` applies, and `inv_freq` is the model constant
            # `rope(pos)` reads.  All three are per *query* head, because B1
            # applies them after `repeat_interleave`; the prover asserts that a
            # GQA group's heads agree before it states a gadget over the kv grid.
            #
            # Absent files are not an error: without `jq.i32` the q scale is the
            # dynamic per-token one and its component is `requant.rs`'s `with_max`
            # shape, which stream NN designed and did not build; without
            # `inv_freq.f64` the model has per-layer-type frequencies and the
            # statement would have to name which. The prover proves no rotation
            # in either case and says so, rather than guessing one.
            ascl = attention_scales(spec, ATTN_RANGES)
            s_k = ascl[f"L{L}.k"]
            jk = pow2_exponent(s_k.reshape(nh, -1).expand(nh, hd).contiguous())
            assert (jk != -1000).all(), "K scales are not powers of two -- wrong spec?"
            write("jk.i32", jk.reshape(-1), digests)
            s_q = ascl.get(f"L{L}.q") if spec.q_scale_mode == "head" else None
            if s_q is not None:
                jq = pow2_exponent(s_q.reshape(nh, -1).expand(nh, hd).contiguous())
                assert (jq != -1000).all(), "Q scales are not powers of two"
                write("jq.i32", jq.reshape(-1), digests)
            else:
                print(f"  NN: q_scale_mode={spec.q_scale_mode!r} -- no jq.i32, so no "
                      "q quantize component on this dump")
            # The angle table itself is NOT written: the statement carries
            # `rope_theta`, the branch of `inv_freq_of` this model takes and that
            # branch's own constants, and the prover rebuilds `inv_freq` from
            # them.  Stream LL's `pre-split` then classifies tree 0's cos/sin
            # columns `needs: Params, varies: Block` instead of `needs: Model`,
            # which is where the weights-tree design's margin lives.  What IS
            # written is B1's own fp64 vector, as a *reference*: the prover
            # compares its derivation against it and reports the mismatch count,
            # so a branch this repo implements differently shows up as a number
            # and not as a rejected proof.
            inv = getattr(m, "inv_freq", None)
            if inv is not None:
                write("inv_freq.f64", inv.detach().to("cpu").double().numpy(), digests)
            else:
                print("  NN: this model has per-layer-type rope frequencies; "
                      "inv_freq.f64 not written")

            # B9 step 1b: so must the *output* scale be the rectangle's.  Taking `s_o`
            # from the forward pass -- which attended to the prefix *and* the causal
            # diagonal block -- calibrates it on a different quantity: the diagonal block's numerators
            # enlarge `Z` more than they enlarge `A.V` on some rows, so dropping them
            # makes `A.V s_v / Z` *larger* than the scale was cut for, the quotient
            # leaves the fold's 2^`ao_q_bits` (q, nz) table, its LogUp sum does not
            # cancel and the honest proof is rejected.  That is 1 to 662 outputs of
            # ~33 M on 14 of Qwen-0.5B's 24 layers, with quotients up to 370 against a
            # 127 window.  The gadget's completeness precondition is
            # `max |A.V s_v / Z| <= 127 s_o` over the rows it proves, so the scale is
            # recalibrated on those rows here.  `s_o_fwd` is kept in the manifest so
            # the two are comparable; nothing else in the dump moves.
            out_rect = torch.einsum("hsc,hcd->hsd", num.double(),
                                    m.rec.cols[f"L{L}.vv"][:, :P].double())
            if a.diagonal_block:
                # with the diagonal block proved, the AIR's attention output is the forward
                # pass's: both halves, causally masked
                out_rect = out_rect + torch.einsum("hsc,hcd->hsd", numc.double(),
                                                   m.rec.cols[f"L{L}.vv"][:, P:].double())
            real_rect = out_rect * s_v.double().reshape(nh, 1, hd) / zz.double().unsqueeze(-1)
            flat_rect = real_rect.transpose(0, 1).reshape(S, nh * hd)
            # stream Z: the fold's target width is the spec's, not int8's -- the
            # scale calibrates against `qmax_o` and the whole dump declares it
            qmax_o = (1 << (spec.attn_out_bits - 1)) - 1
            s_o_rect = pow2_ceil(flat_rect.abs().amax(-1, keepdim=True).clamp_min(1e-30)
                                 / float(qmax_o))
            s_o_fwd = m.rec.cols[f"L{L}.s_o"].float()                # (S, 1)
            jo = pow2_exponent(s_o_rect.reshape(-1).float())
            jo_fwd = pow2_exponent(s_o_fwd.reshape(-1))
            assert (jo != -1000).all(), "the rectangle's s_o is not a power of two"
            assert (jo_fwd != -1000).all(), "s_o is not a power of two -- wrong spec?"
            write("jo.i32", jo, digests)
            # the quotient the fold will build, so the precondition is checked here and
            # not four hours later inside a prover
            q_rect = round_hafz(flat_rect.abs() / s_o_rect.double())
            assert int(q_rect.max()) <= qmax_o, \
                f"fold quotient {int(q_rect.max())} > {qmax_o}"
            jo_bumped = int((jo != jo_fwd).sum())
            # the fold's exponent e = jv[h, ch] - jo[t]: its range decides the int8
            # gadget's num/den split (num = 2^max(e,0), den = Z * 2^max(-e,0))
            e_min = int(jv.min()) - int(jo.max())
            e_max = int(jv.max()) - int(jo.min())
            # Issue #111: and the no-wrap bound the same identity has to hold
            # inside, checked here for the same reason the fold quotient is --
            # every input to it is on this desk, and the prover's own refusal
            # comes hours later, after the whole dump has been written and a
            # chunk loop has loaded its first block.  The four numbers are
            # `AttnOut`'s: `acc_bits` is the bit length of `1 + max |A.V|`,
            # which is the accumulator the two limbs range-check; `rem_bits` is
            # `2 max_(h,t,j) (Z 2^max(-e, 0))`; `jv` and `jo` are the exponents
            # themselves.  A driver that pins wider model-wide values can only
            # spend more of the window, so a dump refused here is refused under
            # every plan.
            ao_acc_bits = bits_for(int(out_rect.abs().max()) + 1)
            zzn = zz.numpy().astype(np.int64)                              # (nh, S)
            shift = np.maximum(jo[None, :] - jv.min(-1)[:, None], 0).astype(np.int64)
            ao_rem_bits = bits_for(max(2, 2 * int((zzn << shift).max())))
            no_wrap = ao_no_wrap(ao_acc_bits, int(jv.max()), int(jo.min()),
                                 spec.attn_out_bits, ao_rem_bits)
            refusal = ao_no_wrap_refusal(no_wrap, f"block {bi} layer {L}")
            assert refusal is None, refusal
            print(f"  no-wrap window: {no_wrap['bits']}/{no_wrap['limit']} bits, margin "
                  f"{no_wrap['margin']} ({no_wrap['form']} form); quotient "
                  f"{no_wrap['quotient']['bits']}/{no_wrap['quotient']['limit']}, margin "
                  f"{no_wrap['quotient']['margin']} ({no_wrap['quotient']['form']} form)")

            # ---- the accumulators the dequant rounds, and the Freivalds check's operands -------
            accs = {}
            for name in mats:
                key = f"L{L}.{name}/acc"
                if key not in m.rec.cols:
                    print(f"  (no accumulator recorded for {name})")
                    continue
                acc = m.rec.cols[key]
                lin = m.layers[L][name]
                # zk-prover #172: the dequant gadget states `acc s_x s_w` and
                # nothing else.  A projection bias sits between that product and
                # whatever reads it -- the rotation for q and k, the quantize for
                # v -- so every integer the prover derives from this matmul is off
                # by the bias and the segment is rejected after it is proved.
                # Qwen2.5 gives q, k and v one; Llama gives none, which is why
                # every witness extracted before this carried none.  Refuse here,
                # where the model is still in hand, rather than dumping tens of
                # gigabytes nothing can prove.
                assert lin.bias is None, (
                    f"{name} carries a bias -- the statement has no site for it, so no "
                    f"proof of this layer can be accepted (zk-prover #172)")
                s_x = m.rec.cols[f"L{L}.{name}/s_x"]
                write(f"acc_{name}.i32", acc.numpy().astype(np.int32), digests)
                # stream Z: the widened attention output makes o's quantized input
                # int16; the file name carries the storage width so a loader cannot
                # misread it
                if name == "o" and spec.attn_out_bits > 8:
                    write(f"qx_{name}.i16",
                          m.rec.cols[f"L{L}.{name}/qx"].numpy().astype(np.int16), digests)
                else:
                    write(f"qx_{name}.i8",
                          m.rec.cols[f"L{L}.{name}/qx"].numpy().astype(np.int8), digests)
                write(f"qw_{name}.i8", lin.qw.detach().to("cpu").numpy().astype(np.int8),
                      digests)
                if f"L{L}.{name}/x" in m.rec.cols:
                    write(f"x_{name}.u16",
                          m.rec.cols[f"L{L}.{name}/x"].contiguous().view(torch.int16)
                          .numpy().view(np.uint16), digests)
                jx, jw = pow2_exponent(s_x.reshape(-1)), pow2_exponent(lin.s_w.reshape(-1))
                mx, ex = bf16_parts(s_x.reshape(-1))
                mw, ew = bf16_parts(lin.s_w.reshape(-1))
                write(f"jx_{name}.i32", jx, digests)
                write(f"jw_{name}.i32", jw, digests)
                write(f"sx_{name}.i32", np.stack([mx, ex], -1), digests)
                write(f"sw_{name}.i32", np.stack([mw, ew], -1), digests)
                av = acc.abs().amax().item()
                accs[name] = dict(
                    n_in=int(lin.n_in), n_out=int(lin.n_out), tokens=int(acc.shape[0]),
                    abs_max=int(av), bits=bits_for(av),
                    kmax_needed=max(0, bits_for(av) - 9),
                    worst_case=int(127 * 127 * lin.n_in),
                    kmax_worst_case=max(0, bits_for(127 * 127 * lin.n_in) - 9),
                    scales_pow2=bool((jx != -1000).all() and (jw != -1000).all()),
                    zero_frac=float((acc == 0).double().mean()),
                )
                print(f"  {name:<5} |acc| <= {av:>12,.0f} ({bits_for(av)} bits) "
                      f"kmax {accs[name]['kmax_needed']} "
                      f"(worst case {accs[name]['kmax_worst_case']}), "
                      f"pow2 scales {accs[name]['scales_pow2']}")

            # ---- stream Z: the fold's own identity, at the witness -------------------
            # `y-delay-*` carries v's output as the exact product and the A.V quantize
            # is the one rounding.  The AIR will prove exactly
            # `round_hafz(acc s_x s_w / s_v)`, so that quantity is recomputed here from
            # the dumped arrays alone and byte-compared against the recorder's own
            # `vvc` (the S decoded tokens' values are the diagonal block's) -- the identity
            # gate's witness half, checked at extraction and not hours later.  The
            # no-clamp precondition rides along: a clamped site has no honest witness.
            fold_stats = {}
            if "v" in spec.delay_matmuls:
                assert a.diagonal_block, "the v fold's identity check reads the diagonal block's vvc"
                accv = m.rec.cols[f"L{L}.v/acc"].double()               # (S, nkv*hd)
                linv = m.layers[L]["v"]
                # the fold's statement is `round_hafz(acc s_x s_w / s_v)` and nothing
                # else -- a v_proj bias (Qwen) sits between the product and the
                # quantize, and no bias-free gadget can state that site
                assert linv.bias is None, \
                    "v_proj carries a bias -- the fold cannot state this model's v site"
                sxv = m.rec.cols[f"L{L}.v/s_x"].double()                # (S, 1)
                swv = linv.s_w.detach().to("cpu").double().reshape(1, -1)
                rep = nh * hd // linv.n_out
                svq = s_v.detach().to("cpu").double().reshape(nh // rep, rep, hd)
                assert bool((svq == svq[:, :1]).all()), "s_v disagrees inside a GQA group"
                sv_flat = svq[:, 0].reshape(1, -1)                      # (1, nkv*hd)
                qv = round_hafz(accv * sxv * swv / sv_flat)
                # the AIR proves the true rounding and the clamp beside it, so the
                # completeness precondition is the quotient *domain*, not 127
                q_dom = (1 << (spec.attn_out_bits - 1)) - 1
                assert int(qv.abs().max()) <= q_dom, \
                    f"v fold quotient {int(qv.abs().max())} past the {q_dom} domain"
                clamped = int((qv.abs() > 127).sum())
                q_true_max = int(qv.abs().max())
                qv = qv.clamp(-127, 127)
                vvc = m.rec.cols[f"L{L}.vv"][:, P:].double()            # (nh, S, hd)
                vvc_kv = vvc.reshape(nh // rep, rep, S, hd)[:, 0]       # (nkv, S, hd)
                vvc_flat = vvc_kv.permute(1, 0, 2).reshape(S, -1)       # (S, nkv*hd)
                assert bool((qv == vvc_flat).all()), \
                    "v fold identity: clamp(round_hafz(acc s_x s_w / s_v)) != vvc"
                gt = 134 - bf16_parts(m.rec.cols[f"L{L}.v/s_x"].reshape(-1))[1]
                gc = 134 - bf16_parts(linv.s_w.reshape(-1))[1] \
                    + pow2_exponent(sv_flat.reshape(-1).float())
                assert int(gt.min() + gc.min()) >= 14, "fold divisor below 2^14"
                fold_stats = dict(
                    q_max=q_true_max, clamped=clamped,
                    gt_min=int(gt.min()), gt_max=int(gt.max()),
                    gc_min=int(gc.min()), gc_max=int(gc.max()),
                    fold_rem_bits=int(gt.max() + gc.max() - 13),
                    fold_alpha=int(max(14 - gc.min(), 0)),
                )
                print(f"  v fold identity holds on {qv.numel():,} sites "
                      f"({clamped} clamped); rem {fold_stats['fold_rem_bits']} bits, "
                      f"alpha {fold_stats['fold_alpha']}")

            # ---- B10 step 2: the residual stream's columns ----------------------
            # Every one is a bf16 *pattern* column, which is what the wiring family
            # commits: the residual stream in, mid and out, the two adds' second
            # operands, the two norm weights (preprocessed -- they are weights), the
            # two rsqrt lookups and the SwiGLU's.  `eps` and `1/d` are bf16 scalars and
            # go in the manifest.
            residual_stream = {}
            if a.residual_stream:
                def pat(t):
                    return (t.detach().to("cpu").contiguous().to(BF16)
                            .view(torch.int16).numpy().view(np.uint16))

                for key, name in (("xin", "xin"), ("xmid", "xmid"), ("x", "xout"),
                                  ("o_out", "o_out"), ("d_out", "d_out"),
                                  ("norm1.rsqrt/in", "n1_in"), ("norm1.rsqrt/out", "n1_out"),
                                  ("norm2.rsqrt/in", "n2_in"), ("norm2.rsqrt/out", "n2_out"),
                                  ("silu/in", "silu_in"), ("silu/out", "silu_out")):
                    write(f"{name}.u16", pat(m.rec.cols[f"L{L}.{key}"]), digests)
                write("w1.u16", pat(m.layers[L]["norm1"]), digests)
                write("w2.u16", pat(m.layers[L]["norm2"]), digests)
                residual_stream = dict(d=int(m.d), dff=int(m.layers[L]["gate"].n_out),
                            eps=int(pat(m.eps.reshape(1))[0]),
                            inv_d=int(pat(m.inv_d.reshape(1))[0]),
                            reduce=spec.reduce, rms_norm_eps=float(m.eps.item()))

            # ---- B10 step 4: the `lm_head` chunk -------------------------------------
            # The head is a `Linear` like every other matmul, recorded under a bare
            # `lm_head/...` key because it has no layer.  What is new here is the
            # *token* side: the order key the argmax component compares, and which token
            # each row emits.
            head = {}
            if a.head:
                hacc = m.rec.cols["lm_head/acc"].to(torch.int64)          # (S, vocab)
                hsx = m.rec.cols["lm_head/s_x"]
                lin = m.head
                # the files carry the *mat* name, so `Layer::load` reads the head with
                # the same code every other matmul goes through
                write("acc_lm_head.i32", hacc.numpy().astype(np.int32), digests)
                write("qx_lm_head.i8", m.rec.cols["lm_head/qx"].numpy().astype(np.int8), digests)
                write("qw_lm_head.i8", lin.qw.detach().to("cpu").numpy().astype(np.int8), digests)
                jxh = pow2_exponent(hsx.reshape(-1))
                jwh = pow2_exponent(lin.s_w.reshape(-1))
                write("jx_lm_head.i32", jxh, digests)
                write("jw_lm_head.i32", jwh, digests)
                # and the bf16 scale pairs, which the chain rung reads and the
                # power-of-two one ignores -- the dump does not know which it will be
                # composed under, only which the spec says (`deq_variants`)
                mxh, exh = bf16_parts(hsx.reshape(-1))
                mwh, ewh = bf16_parts(lin.s_w.reshape(-1))
                write("sx_lm_head.i32", np.stack([mxh, exh], -1), digests)
                write("sw_lm_head.i32", np.stack([mwh, ewh], -1), digests)
                # B1's own logits for these rows, and the **order key** the argmax
                # compares: bf16 magnitude order is pattern order, but logits are
                # signed, so negatives are flipped into a single ascending 16-bit key.
                lg = logits.detach().to("cpu").to(BF16)
                pat = lg.view(torch.int16).numpy().view(np.uint16).astype(np.int64)
                sign, mag = pat >> 15, pat & 0x7FFF
                key = np.where(sign == 1, 0x7FFF - mag, 0x8000 + mag)
                write("key_head.u16", key.astype(np.uint16), digests)
                tokh = lg.float().argmax(-1).numpy().astype(np.int64)
                write("tok_head.i32", tokh.astype(np.int32), digests)

                # ---- stream II (E11): the head chunk's norm scale ------------
                # Five bf16 pattern columns and two bf16 scalars, which is what
                # a chunk needs to prove `h = (x rsqrt(ms inv_d + eps)) w` for
                # the *final* norm exactly as a layer proves it for norm1 and
                # norm2: the residual in, the weight (tree 0, under `pre_root`),
                # and B1's `r`, its `ms + eps` and its `h` as references.  The
                # residual-in claim is also where the head chunk's tree-3
                # boundary lands -- stream KK's, hooked here by writing the
                # column the boundary would be taken over.
                def hpat(t):
                    return (t.detach().to("cpu").contiguous().to(BF16)
                            .view(torch.int16).numpy().view(np.uint16))

                write("xin_head.u16", hpat(m.rec.cols[f"L{m.L - 1}.x"]), digests)
                write("w_head.u16", hpat(m.final_norm), digests)
                write("rin_head.u16", hpat(m.rec.cols["final_norm.rsqrt/in"]), digests)
                write("r_head.u16", hpat(m.rec.cols["final_norm.rsqrt/out"]), digests)
                write("h_head.u16", hpat(m.rec.cols["lm_head/x"]), digests)

                # Which tokens do these rows emit?  This block covers positions
                # `base .. base + S - 1`, and row `t` predicts the token at position
                # `base + t + 1` -- so the rows emit the window *shifted by one* from
                # the tokens the layer rectangle is extracted at.  Both alignments
                # are counted rather than assumed, because getting this wrong is
                # silent: the head would prove a real argmax of the wrong row.
                # Stream DD: block-relative.  On the last block `base = N - S` and
                # these are the pre-DD slices to the element.  The comparison stays
                # at `S - 1` rows on every block so the number means one thing:
                # row `S - 1` of the last block predicts a position that is not in
                # `ids` at all.
                idn = ids.numpy().astype(np.int64)
                nxt = int((tokh[:-1] == idn[base + 1:base + S]).sum())
                same = int((tokh == idn[base:base + S]).sum())
                accs["lm_head"] = dict(
                    n_in=int(lin.n_in), n_out=int(lin.n_out), tokens=int(hacc.shape[0]),
                    abs_max=int(hacc.abs().amax()), bits=bits_for(int(hacc.abs().amax())),
                    kmax_needed=max(0, bits_for(int(hacc.abs().amax())) - 9),
                    worst_case=int(127 * 127 * lin.n_in),
                    kmax_worst_case=max(0, bits_for(127 * 127 * lin.n_in) - 9),
                    scales_pow2=bool((jxh != -1000).all() and (jwh != -1000).all()),
                    zero_frac=float((hacc == 0).double().mean()),
                )
                head = dict(
                    vocab=int(lin.n_out), d_in=int(lin.n_in), tokens=int(hacc.shape[0]),
                    # stream II: what the head chunk's norm scale is built from.
                    # `eps` and `inv_d` are bf16 patterns of the model, because
                    # a table is only as pinned as the parameters the verifier
                    # builds it from, and `reduce` says whether the AIR's
                    # balanced bf16 tree is B1's own accumulation.
                    final_norm=True,
                    eps=int(hpat(m.eps.reshape(1))[0]),
                    inv_d=int(hpat(m.inv_d.reshape(1))[0]),
                    reduce=spec.reduce, rms_norm_eps=float(m.eps.item()),
                    tied_embedding=bool(getattr(m, "tied", False)),
                    abs_max=int(hacc.abs().amax()), bits=bits_for(int(hacc.abs().amax())),
                    kmax_needed=max(0, bits_for(int(hacc.abs().amax())) - 9),
                    scales_pow2=bool((jxh != -1000).all() and (jwh != -1000).all()),
                    zero_frac=float((hacc == 0).double().mean()),
                    # the alignment, both ways round
                    emits_next_token=nxt, emits_next_of=int(S - 1),
                    emits_same_position=same, emits_same_of=int(S),
                    # what the binding has to carry (README 9.2 measured this separately)
                    co_max_rows=int((lg.float() == lg.float().amax(-1, keepdim=True)).sum(-1).gt(1).sum()),
                )
                print(f"  head  vocab {lin.n_out:,} |acc| <= {int(hacc.abs().amax()):,} "
                      f"({head['bits']} bits) pow2 {head['scales_pow2']}")
                print(f"        rows emit the *next* token on {nxt}/{S - 1}; "
                      f"the same position on {same}/{S}")

            # ---- Stream BB (E11): the embedding gadget's two inputs -----------------
            # `embed.u16` is the table the residual stream entering layer 0 is a row
            # of, as bf16 *patterns* -- the same encoding `xin.u16` carries, because
            # the gadget's lookup compares the two cell for cell.  On a model that
            # scales the embedding before layer 0 (Gemma's sqrt(d)) the scaled table
            # is what `xin` is a row of, so that is what is written and the manifest
            # says so.  `tok_in.i32` is this block's own input ids: row `t` of `xin`
            # is position `base + t`.
            emb = {}
            if a.embed:
                scale = getattr(m, "embed_scale", None)
                etab = m.emb.detach().to("cpu")
                if scale:
                    etab = (etab.float() * float(scale)).to(BF16)
                else:
                    etab = etab.to(BF16)
                epat = etab.contiguous().view(torch.int16).numpy().view(np.uint16)
                write("embed.u16", epat, digests)
                idn_e = ids.numpy().astype(np.int64)
                write("tok_in.i32", idn_e[base:base + S].astype(np.int32), digests)
                emb = dict(vocab=int(etab.shape[0]), d=int(etab.shape[1]),
                           layer=int(L), block_base=int(base), tokens=int(S),
                           embed_scale=(float(scale) if scale else None))
                # the premise the prover then re-checks value by value: on a
                # layer-0 dump `xin` *is* a row of this table
                if a.residual_stream and L == 0:
                    xp = (m.rec.cols[f"L{L}.xin"].detach().to("cpu").contiguous().to(BF16)
                          .view(torch.int16).numpy().view(np.uint16))
                    want = epat.reshape(int(etab.shape[0]), -1)[idn_e[base:base + S]]
                    canon = lambda v: np.where(v == 0x8000, 0, v)
                    bad = int((canon(xp.reshape(S, -1)) != canon(want)).sum())
                    emb["xin_mismatch"] = bad
                    print(f"  embed vocab {etab.shape[0]:,} x {etab.shape[1]} "
                          f"scale {scale} | xin mismatches {bad}")
                else:
                    print(f"  embed vocab {etab.shape[0]:,} x {etab.shape[1]} "
                          f"scale {scale} | layer {L}: xin is not the embedding, "
                          f"not checked")

            frac = (-torch.log2(sqk)).reshape(nh, -1)
            fpi = frac[:, 0].tolist()
            # B10 step 3: the master table is sized by the largest shift the prover
            # *looks up*, so the diagonal block's visible half counts and its masked half does
            # not -- a masked shift is taken against a maximum over keys it is not one
            # of, can be negative, and is never a table row.
            d_max = int(dd.max()) if dd.numel() else 0
            diagonal_block_stats = {}
            if a.diagonal_block:
                dvis = ddc[vis.expand_as(ddc)]
                d_max = max(d_max, int(dvis.max()))
                diagonal_block_stats = dict(
                    keys=int(S),
                    masked_frac=float((~vis).double().mean()),
                    d_max_visible=int(dvis.max()), d_min_visible=int(dvis.min()),
                    d_min_masked=int(ddc[~vis.expand_as(ddc)].min()),
                    numerator_zero_frac=float((numc[vis.expand_as(numc)] == 0).double().mean()),
                    z_diagonal_block_max=int(numc.sum(-1).max()),
                    z_diagonal_block_min=int(numc.sum(-1).min()),
                    # which half the row maximum came from, per token: the reason
                    # `Z >= nmax` is a statement about the total and not about a key tile
                    # 2026-09-01: block 0 has no prefix, so `sc` is
                    # (nh, S, 0) and `amax(-1)` raises "Expected reduction dim
                    # -1 to have non-zero size".  With no prefix keys every row
                    # maximum comes from the diagonal block, which is what `nh * S` says.
                    max_from_diagonal_block=(
                        int((scc.masked_fill(~vis, torch.iinfo(torch.int64).min)
                             .amax(-1) > sc.amax(-1)).sum()) if P else int(nh * S)),
                    rows=int(nh * S))
            stats = dict(
                heads=nh, s=S, c=P, hd=hd, layer=L, spec=a.spec, qk_bits=list(spec.qk_bits),
                # stream NN: the key/value head count, so the prover sizes the
                # rotation's components from the model rather than by dividing a
                # matmul's width by `hd` and guessing which head count it is
                nkv=int(m.nkv),
                # stream NN: everything the verifier needs to build the rotation's
                # angle table itself.  `scale` is the branch's own constants in
                # `inv_freq_of`'s order, so the prover reads a list and not four
                # differently-named keys.
                rope=_rope_params_block(m),
                # Stream DD: which token block this is.  `c` already says it (a
                # block's prefix is `block * s`), but a driver that walks a
                # sequence should not have to divide to find out, and `blocks`
                # is what says the dump is one of a set rather than a decode.
                block=int(bi), blocks=int(nblocks), base_pos=int(base),
                seq_tokens=int(len(ids)), chunk=int(a.chunk),
                slim_num=bool(SLIM_NUM), shared=bool(a.share),
                num_bits=spec.num_bits, model=MODEL,
                diagonal_block=bool(a.diagonal_block), diagonal_block_stats=diagonal_block_stats,
                d_max=d_max, d_bits=bits_for(d_max),
                d_fits_16bit=bool(d_max < (1 << 16)),
                z_max=int(zz.max()), z_worst_case=int(nmax * (P + S if a.diagonal_block else P)),
                numerator_zero_frac=(float((num == 0).double().mean()) if num.numel()
                                     else 0.0),
                frac_per_head=fpi,
                frac_shared_per_head=bool(frac.shape[1] == 1 or bool((frac == frac[:, :1]).all())),
                frac_is_integer=bool(((frac - frac.round()).abs() < 1e-9).all()),
                frac_spread=float(max(fpi) - min(fpi)),
                frac_integer_separated=bool(
                    all(abs((max(fpi) - f) - round(max(fpi) - f)) < 1e-9 for f in fpi)),
                master_table_bits=int(bits_for(d_max) + round(max(fpi) - min(fpi))),
                attn_out=dict(jo_min=int(jo.min()), jo_max=int(jo.max()),
                              jv_min=int(jv.min()), jv_max=int(jv.max()),
                              e_min=e_min, e_max=e_max,
                              # B9 step 1b: the fold's scale, calibrated on the
                              # rectangle the AIR proves rather than on the forward
                              # pass's masked output
                              jo_fwd_min=int(jo_fwd.min()), jo_fwd_max=int(jo_fwd.max()),
                              jo_tokens_bumped=jo_bumped, jo_tokens=int(S),
                              fold_q_max=int(q_rect.max()),
                              # Issue #121: the accumulator's bit length, beside
                              # the two exponent extremes above.  The check
                              # below evaluates the window half here, where
                              # every input is on the desk; the shape table and
                              # the plan writer evaluate it again under the pins
                              # THEY write, and until this was recorded they had
                              # no accumulator to evaluate it with and could
                              # only do the quotient half.
                              ao_acc_bits=ao_acc_bits),
                accumulators=accs,
                # B10 step 0: which dequant gadget each matmul needs is a property of
                # the *spec*, not of the prover's build.  `b65-design` puts the
                # power-of-two dequant on q/k/v/o and the bf16 chain on gate/up/down;
                # `b10-design` (= `b6-design-p2o` + away + 9 bits) puts the chain on
                # all seven, which is the layout B8 licensed and the denominator
                # charges.  The witness says so, so a prover cannot compose it under
                # the other layout by accident.
                # B10 step 4: `lm_head` is named beside the seven, because the head
                # chunk's dequant is 79% of it and the two rungs differ by 52% -- a
                # prover must not have to guess (rule 1).  Note that `b10-design` puts
                # it on the *chain*, while `b2/cost_model.py` prices the design B8
                # licensed (power-of-two on the head, +0.02% at 8B).  That divergence
                # is named in `B10.md` §3; the manifest states what the dump is.
                deq_variants={**{n: ("fold" if n in spec.delay_matmuls
                                     else "pow2" if _pow2_mat(spec, n) else "chain")
                                 for n in mats},
                              **({"lm_head": ("pow2" if _pow2_mat(spec, "lm_head") else "chain")}
                                 if a.head else {})},
                # stream Z: the fold's target width and its measured divisor range --
                # the witness is the knob, so the prover reads both from here
                attn_out_bits=spec.attn_out_bits,
                fold=fold_stats,
                dequant_round=spec.dequant_round,
                clamped={k: dict(over=v[0], total=v[1], frac=v[0] / v[1])
                         for k, v in (m.clamped or {}).items()},
                ids_from=ids_from,
                # Issue #130: the attention calibration this dump was quantized
                # under, by digest.  `jk.i32` below is its K half and the model
                # tree's root moves with it, so a dump that does not name it is
                # a dump whose statement nothing can reproduce.
                attn_ranges=attn_ranges,
                slim=bool(SLIM),
                residual_stream=residual_stream, head=head, embed=emb,
                torch=torch.__version__, files=digests,
                # Issue #130: the whole `files` map above, in one number a
                # record can quote.  Last, because it is a function of every
                # array this dump wrote and of nothing else in the manifest.
                dump_sha256=dump_digest(digests),
            )
            json.dump(stats, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
            print(json.dumps({k: v for k, v in stats.items() if k != "files"}, indent=1))
            # B10 step 1: a layer's recorded columns are ~1.4 GB at C=16K and its
            # working set another ~10 GB, so both go before the next layer starts --
            # 32 layers out of one forward pass against a 128 GiB cgroup.
            del sc, dd, num, xx, mm, zz, out_rect, real_rect, flat_rect, residual_stream
            for key in [k for k in m.rec.cols if k.startswith(f"L{L}.")]:
                del m.rec.cols[key]

    for bi, i in enumerate(range(0, len(ids), a.chunk)):
        logits, cache = m.forward(ids[i:i + a.chunk], cache)
        print(f"  prefilled {min(i + a.chunk, len(ids))}/{len(ids)} "
              f"({time.time() - t0:.0f}s)", flush=True)
        # The inversion: the dump body used to run once, here, after the loop.
        # `LastChunk` holds exactly this block, so running it here dumps this
        # block and running it after the loop dumps the last -- same code.
        if bi in want:
            dump_blocks(bi, i)

    print(f"wrote {OUT} in {time.time() - t0:.0f}s")
