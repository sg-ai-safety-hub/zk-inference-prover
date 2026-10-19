"""The witness-calibrated fields of a segment's `params` block, read off the dump.

    uv run python -m harness.eval.predict.witness_fields \
        --dir DIR --layers 48 --numbits 9 --tight-table --out fields.json
    uv run python -m harness.eval.predict.witness_fields --selftest

WHY THIS EXISTS.  `harness/costmodel/exact_cells.py` counts a composed segment's
committed cells from its `params` block, and part of that block is not the
model's shape: it is the calibration the run took off its own witness
(`log_table`, `kmax`, the five `ao_*` exponent and accumulator widths, the
fold's two pins).  A registration made BEFORE any proof therefore has to state
them, and the answer so far was to assume the modal value of the recorded 8B
runs.  At a new model that is wrong by a whole table: Qwen2.5-14B's dumps ask
for a master exponent table of 2^24 and 2^25 rows where the 8B's modal value is
2^23, and five cells a table row makes that 41.9 M and 125.8 M cells a segment.

They do not have to be assumed.  Every one of them is a property of the dump,
and the prover derives all of them inside `Layer::load` and `params_of` --
before `compose::generate` first touches the card and before any proof exists.
So this module reads them the same way, from the same files, and writes them
down.  It proves nothing, opens no card and writes nothing into the dump.

WHAT IT READS, AND THE RULE FOR EACH FIELD.  Every rule below is the prover's
own, named where it lives.

    field              rule                                        prover source
    ---------------    -----------------------------------------   -------------
    log_table          without `--tight-table`, the manifest's      cli.rs
                       `master_table_bits`.  With it (slack 0),     `tighten_of`,
                       `min(master_table_bits, bits(key_max))`      layer.rs
                       -- `set_master_table` clamps to the size     `set_master_table`,
                       the dump asks for and only ever shrinks.     `key_bits`
    key_max            `max` over every key tile and every          attn_witness.rs
                       VISIBLE grid entry of `dd * delta`:          `key_max`,
                       `dd[h,t,c] = mm[h,t] - sc[h,t,c]` with       `col_idx`,
                       `sc = qq . kk^T` recomputed here as the      `visible`
                       loader recomputes it, `mm` the dump's own
                       joint row maximum (`mm.i32`, shared by the
                       prefix and the diagonal block), and
                       `delta[h] = 2^round(F - frac_h)` for the
                       layer-wide master exponent `F = max frac`.
                       The diagonal block's masked half is not in
                       it: token `t` sees key `c` only for `c <= t`.
    kmax               `min(max_m kmax_worst_case(m), KMAX_SOUND)`  cli.rs
                       over the dump's own accumulators; 21 is      `params_of`,
                       `bf16::KMAX_SOUND`.                          bf16.rs
    ao_rem_bits        `bits(2 * max_{h,t} (zz[h,t] <<              layer.rs
                       max(jo[t] - min_j jv[h,j], 0)))`, which is   `rem_bits`
                       what `harness/witness/shapes.py` already
                       reads for the chunk loop's pin.  With
                       `--ao-rem-bits`, that pin.                   cli.rs
    ao_q_bits          `attn_out_bits - 1`, the fold's target       cli.rs
                       width minus one.                             `params_of`
    ao_jo_min/max      the extremes of `jo.i32`, the attention      manifest
    ao_jv_min/max      output's per-token and per-channel scale     `attn_out`
                       exponents.
    ao_acc_lo_bits     `params_of`'s two limbs of the committed     cli.rs
    ao_acc_hi_bits     accumulator: the low one takes what it can   `params_of`
    ao_acc_mag         up to 16, the high one the rest, neither     `split`,
                       below `MIN_LOG_ROWS`.  The offset form       shapes.py
                       needs `acc_bits + 1` and the magnitude       `ao_no_wrap`
                       form `acc_bits`, and the prover takes the
                       magnitude form exactly when
                       `1 + lo + hi + e_hi > 31` refuses the other.
    fold_rem_bits      the dump's own fold pins, zero on a dump     cli.rs
    fold_alpha         whose `deq_variants` names no `fold`         `params_of`,
                       matmul -- which every `b10-final` dump is.   layer.rs
                                                                    `fold_pins`
    k_bits, q_bits     the manifest's `qk_bits`, which is where     attn_witness.rs
                       `Attn::load_from` reads them.                `load_from`
    nmax, z_min        `2^numbits - 1`: the run's `--numbits`, a    cli.rs
                       command-line knob and not a witness field,   `src_of`
                       so it is an argument here and recorded as
                       a knob.
    deq_variants       which dequantize gadget each matmul is on.   layer.rs
                       Not a witness-calibrated field either, and   `mat_variants`,
                       read for the same reason: at `b10-final`     extract.py
                       every matmul is on the chain, and the 8B
                       segments the cost model was read off have
                       `v` on the fold, so a count that assumes
                       the 8B's layout prices the wrong gadget.

WHAT IT DOES NOT READ.  The run knobs that are the prover's command line and not
the dump: the layout targets, the head grouping, the attention variant and the
rounding.  Those are `exact_cells.statement_knobs()` and the driver's own flags,
and a caller states them rather than reading them off a witness.

THE ONE EXPENSIVE STEP.  `key_max` is the only field that is not in the
manifest, and the two passes `--tight-table` makes over every shift a layer
holds are what make it expensive: the loader recomputes `Q.K^T` for it.  This
does the same, once per head, in float64 -- `hd * max|q| * max|k|` is under
2^53 at every shape this project quantizes, which is asserted before the first
product, so the sum is exact and not approximate.  A dump written with `--slim`
has no `dd.i32` to read instead; `--check-derived` compares against it where it
is on disk.
"""
import argparse
import hashlib
import json
import os
import sys

from harness.witness import shapes

HERE = os.path.dirname(os.path.abspath(__file__))

SCHEMA = "witness-fields/1"

# `prover/src/bf16.rs`: the widest `kmax` the normalisation argument stays sound
# at, which is the cap `params_of` takes the dump's own worst case against.
KMAX_SOUND = 21

# The fields `exact_cells` reads out of a `params` block that a model and a
# context do not fix.  `exact_cells.WITNESS_FIELDS` is the same list; it is not
# imported, because this module writes a file that outlives the run and the two
# lists agreeing is checked (`--selftest`) rather than assumed.
WITNESS_FIELDS = ("log_table", "kmax", "ao_acc_hi_bits", "ao_acc_lo_bits",
                  "ao_rem_bits", "ao_q_bits", "ao_jo_min", "ao_jo_max",
                  "ao_jv_min", "ao_jv_max", "fold_rem_bits")

# What is in a row beside those: the fold's second pin, the accumulator form,
# the two quantize widths, the denominator's floor and the dequantize layout.
# All of them are the dump's or the run's and none of them is a fit.
ROW_FIELDS = WITNESS_FIELDS + ("fold_alpha", "ao_acc_mag", "ao_denp2",
                               "k_bits", "q_bits", "nmax", "z_min",
                               "deq_variants")


def _man(d):
    with open(os.path.join(d, "manifest.json")) as f:
        return json.load(f)


def kmax_of(man):
    """`params_of`'s `kmax`: the dump's own blind worst case, capped.

    `kmax_blind` is per matmul and the layer is proved at the largest of them;
    `bf16::KMAX_SOUND` is where the normalisation argument stops being sound, so
    a witness that wants more is proved at the cap and not at what it wants.
    """
    accs = man["accumulators"]
    return min(max(v["kmax_worst_case"] for v in accs.values()), KMAX_SOUND)


def fold_pins(man):
    """The fold's two pins, and zero for a dump with no fold matmul.

    `params_of` writes `fold_rem_bits: 0` and `fold_alpha: 0` where
    `Layer::fold_pins` is `None`, which is every dump whose spec delays no
    matmul.  `b10-final` delays none; the 8B headline segments were extracted at
    `y-delay-vw12`, which delays `v`, and that is why they carry 15.
    """
    fold = man.get("fold") or {}
    if not fold.get("fold_rem_bits"):
        return 0, 0
    rem = int(fold["fold_rem_bits"])
    # `worst_of`'s joint condition, at one layer: alpha covers this layer's
    # min G_c and has to sit under its min G_t
    alpha = max(14 - int(fold.get("gc_min", 0)), 0)
    if alpha > int(fold.get("gt_min", 0)):
        raise SystemExit(f"the fold's alpha does not fit this dump: need "
                         f"<= {fold.get('gt_min')}, want {alpha}")
    return rem, alpha


def accumulator_limbs(man, ao_rem_bits):
    """`params_of`'s `ao_acc_lo_bits`, `ao_acc_hi_bits`, `ao_acc_mag`, `ao_denp2`.

    The decision between the two forms is the no-wrap window's, which
    `harness/witness/shapes.py` already evaluates for the plan writer, so it is
    read from there rather than written twice.  A dump older than the extractor
    that recorded `ao_acc_bits` states no accumulator width, and this refuses
    rather than guessing one: a guessed limb is a guessed table.
    """
    ao = man.get("attn_out") or {}
    acc = ao.get("ao_acc_bits")
    if acc is None:
        raise SystemExit(
            "this dump records no `attn_out.ao_acc_bits`, so the attention "
            "output's two accumulator limbs cannot be derived: it was extracted "
            "before the extractor recorded the accumulator's bit length. "
            "Re-extract it, or state the two limbs on the command line of the "
            "run that proves it.")
    w = shapes.ao_no_wrap(acc, ao["jv_max"], ao["jo_min"],
                          man.get("attn_out_bits", 8), ao_rem_bits)
    return dict(ao_acc_bits=acc, ao_acc_lo_bits=w["acc_lo_bits"],
                ao_acc_hi_bits=w["acc_hi_bits"],
                ao_acc_mag=(w["form"] == "magnitude"),
                ao_denp2=(w["quotient"]["form"] == "denp2"),
                no_wrap_margin=w["margin"], quotient_margin=w["quotient"]["margin"])


def key_max(d, man, check_derived=False):
    """The largest exp-table key this layer's key tiles look up.

    `--tight-table` sizes the master exponent table by this rather than by the
    manifest's `d_bits + spread`, and the two differ whenever the largest raw
    shift does not come from the lowest-exponent head -- which is every layer of
    this model.  The rule is `attn_witness.rs`'s `key_max` over `col_idx`, for
    the `intexp` variant: `dd * delta`, over the visible half alone.

    Returned with the per-head maxima beside it, so a reader can see which head
    and which scale the table's size came from.
    """
    import numpy as np

    nh, s, c, hd = man["heads"], man["s"], man["c"], man["hd"]
    fpi = list(man["frac_per_head"])[:nh]
    if not man.get("frac_shared_per_head", False):
        raise SystemExit(
            "this dump's Q scale is not shared per head, so there is no "
            "layer-wide master exponent and every key tile would build its own "
            "table: `Layer::load` refuses it too")
    # `Attn::build`: one master exponent for the layer, every instance's shift
    # scaled up to it by a public power of two
    master = max(fpi)
    delta = [1 << min(max(int(round(master - f)), 0), 30) for f in fpi]
    qq = np.fromfile(os.path.join(d, "qq.i16"), dtype=np.int16).reshape(nh, s, hd)
    mm = np.fromfile(os.path.join(d, "mm.i32"), dtype=np.int32).reshape(nh, s)
    kk = (np.fromfile(os.path.join(d, "kk.i16"), dtype=np.int16).reshape(nh, c, hd)
          if c else None)
    kkc = (np.fromfile(os.path.join(d, "kkc.i16"), dtype=np.int16).reshape(nh, s, hd)
           if man.get("diagonal_block") else None)
    lim = 1 << 53
    top = int(np.abs(qq).max()) * int(np.abs(kk if kk is not None else kkc).max()) * hd
    assert top < lim, (f"the scores do not fit a float64 mantissa at this shape: "
                       f"{top} >= 2^53")
    # token `t` sees key `c` of the diagonal block only for `c <= t`
    vis = np.arange(s)[None, :] <= np.arange(s)[:, None]
    per_head, best = [], 0
    for h in range(nh):
        q = qq[h].astype(np.float64)
        row = mm[h].astype(np.int64)[:, None]
        dmax = 0
        if kk is not None:
            dd = row - (q @ kk[h].astype(np.float64).T).astype(np.int64)
            dmax = max(dmax, int(dd.max()))
            if check_derived:
                _check_derived(d, h, nh, s, c, dd, "dd.i32")
        if kkc is not None:
            ddc = row - (q @ kkc[h].astype(np.float64).T).astype(np.int64)
            dmax = max(dmax, int(ddc[vis].max()))
            if check_derived:
                _check_derived(d, h, nh, s, s, ddc, "ddc.i32")
        per_head.append(dict(head=h, shift_max=dmax, delta=delta[h],
                             key=dmax * delta[h]))
        best = max(best, dmax * delta[h])
    best = max(best, 1)
    return dict(key_max=best, key_bits=best.bit_length(),
                master_exponent=master, shift_max=max(p["shift_max"] for p in per_head),
                from_head=max(per_head, key=lambda p: p["key"])["head"],
                deltas=sorted({d_ for d_ in delta}), per_head=per_head)


def _check_derived(d, h, nh, s, c, got, name):
    """Compare one head's recomputed shifts against the dump's own, where the
    dump carries them.  A `--slim` dump does not, and then there is nothing to
    compare: the derived columns are what the prover recomputes anyway."""
    import numpy as np

    p = os.path.join(d, name)
    if not os.path.exists(p):
        return
    a = np.memmap(p, dtype=np.int32, mode="r", shape=(nh, s, c))
    bad = int((a[h].astype(np.int64) != got).sum())
    assert not bad, f"{name}: head {h} disagrees with the recomputed shifts on {bad} entries"


def fields_of(d, numbits=9, tight=True, table_bits=0, check_derived=False,
              ao_rem_bits=0):
    """One dump's row: the witness-calibrated fields, and what they were read off.

    `numbits`, `tight`, `table_bits` and `ao_rem_bits` are the run's own command
    line, because `nmax`, the master table's size and the remainder's width are
    decided there as well as by the witness.  `tight` is `--tight-table` at
    slack 0, which is what every campaign at this context passes.  `ao_rem_bits`
    is a chunk loop's shared pin; one below the dump's own is refused.
    """
    man = _man(d)
    r = shapes.shape_of(d)
    ao = man["attn_out"]
    rem = r["ao_rem_bits"]
    if ao_rem_bits:
        if int(ao_rem_bits) < rem:
            raise SystemExit(f"--ao-rem-bits {ao_rem_bits} is below this witness's {rem}")
        rem = int(ao_rem_bits)
    limbs = accumulator_limbs(man, rem)
    before = man["master_table_bits"]
    row = dict(
        layer=man["layer"], block=man.get("block"), c=man["c"], s=man["s"],
        nh=man["heads"], hd=man["hd"], nkv=man.get("nkv"),
        d=(man.get("residual_stream") or {}).get("d"),
        dff=(man.get("residual_stream") or {}).get("dff"),
        vocab=(man.get("embed") or {}).get("vocab"),
        spec=man.get("spec"), model=man.get("model"),
        dump_sha256=man.get("dump_sha256"),
        kmax=kmax_of(man), ao_rem_bits=rem, ao_q_bits=man.get("attn_out_bits", 8) - 1,
        ao_jo_min=ao["jo_min"], ao_jo_max=ao["jo_max"],
        ao_jv_min=ao["jv_min"], ao_jv_max=ao["jv_max"],
        k_bits=man["qk_bits"][1], q_bits=man["qk_bits"][0],
        nmax=(1 << numbits) - 1, z_min=(1 << numbits) - 1,
        deq_variants=dict(man.get("deq_variants") or {}),
        master_table_bits=before, d_bits=man["d_bits"], d_max=man["d_max"],
        frac_spread=man["frac_spread"],
    )
    row.update(limbs)
    row["fold_rem_bits"], row["fold_alpha"] = fold_pins(man)
    row["ao_rem_bits_from"] = (f"--ao-rem-bits {ao_rem_bits}" if ao_rem_bits
                               else "the dump's own rem_bits")
    if table_bits:
        row["log_table"] = int(table_bits)
        row["log_table_from"] = f"--table-bits {table_bits}"
    elif tight:
        k = key_max(d, man, check_derived=check_derived)
        row["log_table"] = min(k["key_bits"], before)
        row["log_table_from"] = "--tight-table"
        row["key_max"] = k["key_max"]
        row["key_bits"] = k["key_bits"]
        row["key_max_from_head"] = k["from_head"]
        row["shift_max"] = k["shift_max"]
        row["table_scales"] = k["deltas"]
    else:
        row["log_table"] = before
        row["log_table_from"] = "the manifest's master_table_bits"
    return row


def witness_block(row):
    """The `params` fields `exact_cells` reads, out of a row of this file."""
    return {f: row[f] for f in WITNESS_FIELDS}


def mats_of(row, mats):
    """`mats`, with each matmul on the dequantize gadget this dump names.

    `exact_cells.layer_mats` writes the layout of the segments the cost model
    was read off, where `v` is folded.  A `b10-final` dump has every matmul on
    the chain, and the fold's gadget is not the chain's: counting one for the
    other is a different component and a different table.
    """
    v = row.get("deq_variants") or {}
    return [dict(m, variant=v.get(m["name"], m["variant"])) for m in mats]


def load(path):
    """Read a fields file and refuse one this predictor cannot use.

    The refusals are the ones a registration needs: the schema, a field that is
    missing on a layer (which would otherwise be filled in from the assumed
    block without saying so) and a layer index that is not in the file.
    """
    with open(path) as f:
        doc = json.load(f)
    if doc.get("schema") != SCHEMA:
        raise SystemExit(f"{path}: the document's schema is {doc.get('schema')!r} "
                         f"and not {SCHEMA}")
    rows = doc.get("layers") or {}
    if not rows:
        raise SystemExit(f"{path}: the document names no layer")
    for k, row in sorted(rows.items()):
        miss = [f for f in ROW_FIELDS if f not in row]
        if miss:
            raise SystemExit(f"{path}: layer {k} states no {', '.join(miss)}. "
                             f"Every field a segment is counted at is read off the "
                             f"dump, and a field that is missing is not assumed here.")
    return doc


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def summary(doc):
    """The ranges a reader of the record wants: one line per field that moves."""
    rows = list(doc["layers"].values())
    out = {}
    for f in WITNESS_FIELDS:
        vals = sorted({r[f] for r in rows})
        out[f] = dict(values=vals, min=vals[0], max=vals[-1],
                      histogram={str(v): sum(1 for r in rows if r[f] == v) for v in vals})
    return out


def build(base, layers, numbits=9, tight=True, table_bits=0, check_derived=False,
          dump=None, head_groups=None, absent=(), ao_rem_bits=0):
    rows = {}
    absent = sorted(set(int(k) for k in absent))
    if dump:
        r = fields_of(dump, numbits, tight, table_bits, check_derived, ao_rem_bits)
        rows[str(r["layer"])] = r
    else:
        for k in range(layers):
            if k in absent:
                print(f"  L{k:<3} absent: the extractor refused this layer, and the "
                      f"file says so rather than counting it", flush=True)
                continue
            d = f"{base}_L{k}"
            if not os.path.exists(os.path.join(d, "manifest.json")):
                raise SystemExit(f"{d}: no dump there")
            rows[str(k)] = fields_of(d, numbits, tight, table_bits, check_derived,
                                     ao_rem_bits)
            print(f"  L{k:<3} log_table {rows[str(k)]['log_table']:>3} "
                  f"(of {rows[str(k)]['master_table_bits']})  kmax "
                  f"{rows[str(k)]['kmax']:>3}  ao_rem_bits "
                  f"{rows[str(k)]['ao_rem_bits']:>3}  ao_q_bits "
                  f"{rows[str(k)]['ao_q_bits']:>3}", flush=True)
    doc = dict(schema=SCHEMA, layers=rows,
               # The layers this file does NOT carry, named rather than left to
               # be noticed as a gap.  A dump that is not there is still a
               # refusal; this is the other case, a layer the extractor itself
               # refused to write, which a reader of the registration has to be
               # told about because the per-token figure is then over fewer
               # segments than the model has.
               absent=absent,
               read_by="harness/eval/predict/witness_fields.py",
               # the run's own command line, beside the dump's fields: `nmax`,
               # the master table's size, the loop's remainder pin and how the
               # heads are grouped into key tiles are the prover's flags and a
               # reader of the file should not have to find the driver to learn
               # which ones these rows are for
               knobs=dict(numbits=numbits, tight_table=bool(tight),
                          table_bits=int(table_bits), ao_rem_bits=int(ao_rem_bits),
                          head_groups=list(head_groups) if head_groups else None),
               witness_fields=list(WITNESS_FIELDS), row_fields=list(ROW_FIELDS))
    doc["summary"] = summary(doc)
    return doc


# ------------------------------------------- a segment the prover BUILDS --
# The `--synthetic` half of this module.  Everything above reads a dump; a
# synthetic layer segment has none, and a registration made before it is proved
# still has to state the fields it will be counted at.
#
# They are not guesses in the same way for every field, and the difference is
# the whole value of writing them down:
#
#   * `log_table` and `kmax` are the run's own flags.  `params_of` takes
#     `--table-bits` and `--kmax` verbatim, so a driver that passes them has
#     stated them and nothing is assumed;
#   * `ao_q_bits` and the fold's two pins are the generator's.  A synthetic
#     layer sets `attn_out_bits` to 8 and `--layout b10` delays no matmul, so
#     the first is 7 and the other two are 0;
#   * the attention output's measured windows -- `ao_rem_bits`, the two
#     accumulator limbs, the `jo` and `jv` extremes -- are the witness's own,
#     and the witness does not exist yet.  Those are the assumption, and the
#     caller says where each one came from in `stated_from`, which is written
#     into the file and travels into the registration with it.
#
# Every row of the file is the same block, because every segment of a synthetic
# stack at one width IS the same segment: there is no model behind it for one
# layer to differ from another in.
SYNTHETIC_READ_BY = (
    "not read off a dump: the prover's own fixed-seed synthetic layer generator "
    "builds this segment (`Layer::synthetic` and `synth_residual_stream` in "
    "prover/src/layer.rs), so no witness of this shape exists and none was read. "
    "`stated_from` says where each field came from")


def synthetic_fields(stated, layers, head_groups=None):
    """A fields file for a stack of synthetic segments, from the stated block.

    `stated` is one segment's row -- the shape it is built at, the run knobs and
    the witness-calibrated fields -- plus an optional `stated_from` naming where
    each field came from.  Nothing here derives a field: a synthetic segment's
    are the driver's to state, and this refuses a block that leaves one out
    rather than filling it in from a dump that does not exist.
    """
    stated_from = stated.get("stated_from")
    row = {k: v for k, v in stated.items() if k != "stated_from"}
    miss = [f for f in ROW_FIELDS if f not in row]
    if miss:
        raise SystemExit(f"the stated block names no {', '.join(miss)}. A synthetic "
                         f"segment has no dump to read them off, so every field it "
                         f"is counted at is stated or the file is not written.")
    miss = [f for f in ("c", "s", "model", "spec") if f not in row]
    if miss:
        raise SystemExit(f"the stated block names no {', '.join(miss)}: the context "
                         f"and the block length are part of what the fields are for, "
                         f"and the model and the spec are what the registration says "
                         f"it counted")
    if row.get("dump_sha256"):
        raise SystemExit("the stated block names a dump digest, and a synthetic "
                         "segment has no dump: what is registered is the shape")
    if layers < 1:
        raise SystemExit("--layers N: a stack has at least one segment")
    rows = {str(k): dict(row, layer=k, synthetic=True, dump_sha256=None)
            for k in range(layers)}
    doc = dict(schema=SCHEMA, layers=rows,
               # no segment is missing: nothing refused to write a dump, because
               # there is no dump
               absent=[],
               synthetic=True, read_by=SYNTHETIC_READ_BY, stated_from=stated_from,
               knobs=dict(numbits=int(row["nmax"] + 1).bit_length() - 1,
                          tight_table=False, table_bits=int(row["log_table"]),
                          head_groups=list(head_groups) if head_groups else None),
               witness_fields=list(WITNESS_FIELDS), row_fields=list(ROW_FIELDS))
    doc["summary"] = summary(doc)
    return doc


# ------------------------------------------------------------- the self-test --
def _synthetic(d, nh=4, s=8, c=16, hd=4, fracs=(12.0, 13.0, 15.0, 15.0), seed=7):
    """A dump with the files this module reads and no others.

    It is not a proof fixture: what it checks is that the rules above are the
    rules, on a shape small enough to compute the answer twice -- once through
    `key_max`'s per-head maxima and once by brute force over every entry.
    """
    import numpy as np

    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(seed)
    qq = rng.integers(-2048, 2048, size=(nh, s, hd), dtype=np.int16)
    kk = rng.integers(-1024, 1024, size=(nh, c, hd), dtype=np.int16)
    kkc = rng.integers(-1024, 1024, size=(nh, s, hd), dtype=np.int16)
    sc = np.einsum("hti,hci->htc", qq.astype(np.int64), kk.astype(np.int64))
    scc = np.einsum("hti,hci->htc", qq.astype(np.int64), kkc.astype(np.int64))
    vis = np.arange(s)[None, :] <= np.arange(s)[:, None]
    mm = np.maximum(sc.max(-1), np.where(vis, scc, np.iinfo(np.int64).min).max(-1))
    qq.tofile(os.path.join(d, "qq.i16"))
    kk.tofile(os.path.join(d, "kk.i16"))
    kkc.tofile(os.path.join(d, "kkc.i16"))
    mm.astype(np.int32).tofile(os.path.join(d, "mm.i32"))
    (mm[:, :, None] - sc).astype(np.int32).tofile(os.path.join(d, "dd.i32"))
    (mm[:, :, None] - scc).astype(np.int32).tofile(os.path.join(d, "ddc.i32"))
    zz = rng.integers(1, 4000, size=(nh, s), dtype=np.int32)
    zz.tofile(os.path.join(d, "zz.i32"))
    jo = rng.integers(-8, -4, size=(s,), dtype=np.int32)
    jv = rng.integers(-11, -5, size=(nh, hd), dtype=np.int32)
    jo.tofile(os.path.join(d, "jo.i32"))
    jv.tofile(os.path.join(d, "jv.i32"))
    d_max = int((mm[:, :, None] - sc).max())
    man = dict(
        layer=0, block=8, c=c, s=s, hd=hd, heads=nh, nkv=2, spec="b10-final",
        model="selftest", dump_sha256="0" * 64, qk_bits=[12, 11], num_bits=9,
        diagonal_block=True, slim=False, frac_per_head=list(fracs),
        frac_shared_per_head=True, frac_spread=float(max(fracs) - min(fracs)),
        d_max=d_max, d_bits=int(max(d_max, 1)).bit_length(),
        master_table_bits=int(int(max(d_max, 1)).bit_length()
                              + round(max(fracs) - min(fracs))),
        z_max=int(zz.max()), attn_out_bits=8, fold={},
        attn_out=dict(jo_min=int(jo.min()), jo_max=int(jo.max()),
                      jv_min=int(jv.min()), jv_max=int(jv.max()),
                      fold_q_max=100, ao_acc_bits=19),
        accumulators={n: dict(kmax_needed=9, kmax_worst_case=w)
                      for n, w in (("q", 18), ("k", 18), ("v", 18), ("o", 18),
                                   ("gate", 18), ("up", 18), ("down", 19))},
        deq_variants={n: "chain" for n in ("q", "k", "v", "o", "gate", "up", "down")},
        residual_stream=dict(d=32, dff=64), embed=dict(vocab=256),
    )
    with open(os.path.join(d, "manifest.json"), "w") as f:
        json.dump(man, f)
    return man, dict(qq=qq, kk=kk, kkc=kkc, mm=mm, fracs=fracs, vis=vis)


def selftest():
    import tempfile

    import numpy as np

    from harness.costmodel import exact_cells as ec

    bad = []
    if list(WITNESS_FIELDS) != list(ec.WITNESS_FIELDS):
        bad.append(f"the field list is not `exact_cells`': {WITNESS_FIELDS} against "
                   f"{ec.WITNESS_FIELDS}")
    with tempfile.TemporaryDirectory() as t:
        d = os.path.join(t, "witness_L0")
        man, w = _synthetic(d)
        # (1) the key max, against a brute force over every visible entry
        nh, s, c = man["heads"], man["s"], man["c"]
        master = max(w["fracs"])
        delta = np.array([1 << min(max(int(round(master - f)), 0), 30)
                          for f in w["fracs"]])[:, None, None]
        sc = np.einsum("hti,hci->htc", w["qq"].astype(np.int64), w["kk"].astype(np.int64))
        scc = np.einsum("hti,hci->htc", w["qq"].astype(np.int64), w["kkc"].astype(np.int64))
        mm = w["mm"][:, :, None]
        want = max(int(((mm - sc) * delta).max()),
                   int((np.where(w["vis"], (mm - scc) * delta, 0)).max()))
        got = key_max(d, man, check_derived=True)
        if got["key_max"] != want:
            bad.append(f"key_max is {got['key_max']} and the brute force says {want}")
        if got["key_bits"] != max(want, 1).bit_length():
            bad.append(f"key_bits is not the bit length of {want}")
        # (2) the whole row, and the two ways the table size is decided
        row = fields_of(d, numbits=9, tight=True, check_derived=True)
        if row["log_table"] != min(got["key_bits"], man["master_table_bits"]):
            bad.append("the tight table is not the smaller of the key bits and the "
                       "size the dump asks for")
        loose = fields_of(d, numbits=9, tight=False)
        if loose["log_table"] != man["master_table_bits"]:
            bad.append("without --tight-table the table is not the manifest's")
        pinned = fields_of(d, numbits=9, table_bits=21)
        if pinned["log_table"] != 21:
            bad.append("--table-bits does not pin the table")
        # --ao-rem-bits pins the remainder, and the row says so
        own = row["ao_rem_bits"]
        if row["ao_rem_bits_from"] != "the dump's own rem_bits":
            bad.append("an unpinned row does not say its width is the dump's own")
        at_pin = fields_of(d, numbits=9, ao_rem_bits=own + 3)
        if at_pin["ao_rem_bits"] != own + 3:
            bad.append("--ao-rem-bits does not pin the remainder")
        if at_pin["ao_rem_bits_from"] != f"--ao-rem-bits {own + 3}":
            bad.append("a pinned row does not say where its width came from")
        try:
            fields_of(d, numbits=9, ao_rem_bits=own - 1)
            bad.append("a pin below the dump's own was accepted")
        except SystemExit:
            pass
        # (3) the rules that are one line of the prover each
        if row["kmax"] != 19:
            bad.append(f"kmax is {row['kmax']} and the dump's worst case is 19")
        if row["ao_q_bits"] != man["attn_out_bits"] - 1:
            bad.append("ao_q_bits is not the fold's width minus one")
        if (row["ao_jo_min"], row["ao_jo_max"]) != (man["attn_out"]["jo_min"],
                                                    man["attn_out"]["jo_max"]):
            bad.append("the jo window is not the dump's")
        if row["fold_rem_bits"] or row["fold_alpha"]:
            bad.append("a dump with no fold matmul carries a fold pin")
        if (row["ao_acc_lo_bits"], row["ao_acc_hi_bits"]) != (16, 4):
            bad.append(f"the accumulator limbs are {row['ao_acc_lo_bits']} + "
                       f"{row['ao_acc_hi_bits']} and 19 + 1 bits split 16 + 4")
        if row["ao_acc_mag"]:
            bad.append("the offset form fits here and the magnitude form was taken")
        # (4) a witness block out of it is one `exact_cells` counts a segment at
        p = ec.segment_params("layer", d=5120, nh=40, nkv=8, hd=128, dff=13824,
                              vocab=152064, s=512, c=4096,
                              witness=witness_block(row), head_groups=[16, 16, 8])
        p["mats"] = mats_of(row, p["mats"])
        got_cells = ec.cells_of(p)
        if not got_cells["fresh"] > 0:
            bad.append("the segment built from this row counts no cells")
        if any(m["variant"] == "fold" for m in p["mats"]):
            bad.append("a `b10-final` row left a matmul on the fold")
        # at the pin, the segment builds the pin's range table, five cells a row
        q = ec.segment_params("layer", d=5120, nh=40, nkv=8, hd=128, dff=13824,
                              vocab=152064, s=512, c=4096,
                              witness=witness_block(at_pin), head_groups=[16, 16, 8])
        q["mats"] = mats_of(at_pin, q["mats"])
        tabs, tabs_pin = ec.table_rows(p), ec.table_rows(q)
        if (f"range{own}" not in tabs or f"range{own}" in tabs_pin
                or f"range{own + 3}" not in tabs_pin):
            bad.append("the pinned segment does not build the pin's range table")
        if (ec.cells_of(q)["fresh"] - got_cells["fresh"]
                != 5 * ((1 << (own + 3)) - (1 << own))):
            bad.append("the pin does not move the count by five cells a row")
        # (5) the file, and what reading it back refuses
        doc = build(None, 0, dump=d)
        pinned_doc = build(None, 0, dump=d, ao_rem_bits=own + 3)
        if doc["knobs"]["ao_rem_bits"] != 0 or pinned_doc["knobs"]["ao_rem_bits"] != own + 3:
            bad.append("the file's knobs do not carry the remainder pin")
        f = os.path.join(t, "fields.json")
        with open(f, "w") as fh:
            json.dump(doc, fh, indent=1, sort_keys=True)
        if load(f)["layers"]["0"]["log_table"] != row["log_table"]:
            bad.append("the file does not read back")
        broken = dict(doc, layers={"0": {k: v for k, v in doc["layers"]["0"].items()
                                         if k != "kmax"}})
        with open(f, "w") as fh:
            json.dump(broken, fh)
        try:
            load(f)
            bad.append("a row with no `kmax` was accepted")
        except SystemExit:
            pass
        # (6) the stated block of a segment that has no dump: the same schema,
        # every row the same, and a block short of a field refused
        stated = dict(row, c=4096, s=128, model="a synthetic layer", spec="b10-final",
                      stated_from={"log_table": "--table-bits"})
        stated.pop("dump_sha256", None)
        syn = synthetic_fields(stated, 3, head_groups=[16, 16, 8])
        with open(f, "w") as fh:
            json.dump(syn, fh, indent=1, sort_keys=True)
        back = load(f)
        if sorted(back["layers"]) != ["0", "1", "2"]:
            bad.append("the synthetic file does not carry one row per layer")
        if not all(r["synthetic"] and r["dump_sha256"] is None
                   for r in back["layers"].values()):
            bad.append("a synthetic row does not say it is one, or names a dump")
        if witness_block(back["layers"]["2"]) != witness_block(row):
            bad.append("the synthetic rows are not the stated block")
        try:
            synthetic_fields({k: v for k, v in stated.items() if k != "kmax"}, 1)
            bad.append("a stated block with no `kmax` was accepted")
        except SystemExit:
            pass
        try:
            synthetic_fields(dict(stated, dump_sha256="0" * 64), 1)
            bad.append("a stated block naming a dump digest was accepted")
        except SystemExit:
            pass
    for b in bad:
        print("FAIL:", b)
    print(f"witness_fields selftest: {'FAILED' if bad else 'ok'} -- the key max against a "
          f"brute force over every visible entry, the three ways the table size is "
          f"decided, the remainder pin, the per-field rules, the file's refusals, and "
          f"the stated block a synthetic segment is counted at")
    return 1 if bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", metavar="BASE", help="the witness base: <BASE>_L<k>")
    ap.add_argument("--dump", metavar="DIR", help="one dump directory instead")
    ap.add_argument("--layers", type=int, default=0, help="how many layers to read")
    ap.add_argument("--numbits", type=int, default=9,
                    help="the run's `--numbits`: nmax = 2^numbits - 1 (default 9)")
    ap.add_argument("--tight-table", dest="tight", action="store_true", default=True,
                    help="the master table is sized by the largest key the key tiles "
                         "look up, which is what the campaigns pass (default)")
    ap.add_argument("--no-tight-table", dest="tight", action="store_false",
                    help="the master table is the size the dump asks for")
    ap.add_argument("--table-bits", type=int, default=0,
                    help="the run's `--table-bits`, which pins the table instead")
    ap.add_argument("--ao-rem-bits", type=int, default=0,
                    help="the chunk loop's `--ao-rem-bits`, which pins the "
                         "remainder instead of the dump's own width")
    ap.add_argument("--check-derived", action="store_true",
                    help="compare the recomputed shifts against the dump's own "
                         "`dd.i32`, where it was not written with --slim")
    ap.add_argument("--head-groups", metavar="A,B,C",
                    help="the run's `--head-groups`: how the attention heads are "
                         "grouped into key tiles. A run knob and not a witness "
                         "field, recorded here because the count needs it and the "
                         "rule a projection assumes is not what a driver passes "
                         "(16,16,8 at 40 heads, where the rule would say 20,10,10 "
                         "and the prover refuses a group that is not a power of two)")
    ap.add_argument("--absent", metavar="A,B",
                    help="layers the extractor refused to dump, named here so the "
                         "file states which segments it does not carry. A layer "
                         "that is merely missing is still a refusal; this is for "
                         "one the harness itself would not write, and the "
                         "registration counted from the file says so")
    ap.add_argument("--synthetic", metavar="FILE",
                    help="the stated block of a segment the prover BUILDS rather "
                         "than reads: its shape, the run knobs and the witness "
                         "fields, with a `stated_from` saying where each one came "
                         "from. Every one of the --layers rows is that block, "
                         "because a synthetic stack at one width is one segment")
    ap.add_argument("--out", metavar="FILE", help="write the fields file here")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.synthetic:
        if a.dir or a.dump:
            raise SystemExit("--synthetic states a segment that has no dump, so "
                             "there is nothing for --dir or --dump to read")
        if not a.layers:
            raise SystemExit("--synthetic needs --layers N: the stack's own length")
        with open(a.synthetic) as fh:
            stated = json.load(fh)
        hg = [int(x) for x in a.head_groups.split(",")] if a.head_groups else None
        doc = synthetic_fields(stated, a.layers, head_groups=hg)
        print(f"  {a.layers} synthetic segment(s) at C={doc['layers']['0']['c']:,}, "
              f"S={doc['layers']['0']['s']}, none of them read off a dump")
    else:
        if not a.dir and not a.dump:
            raise SystemExit("--dir BASE (with --layers N), --dump DIR, or "
                             "--synthetic FILE")
        if a.dir and not a.layers:
            raise SystemExit("--dir needs --layers N")
        hg = [int(x) for x in a.head_groups.split(",")] if a.head_groups else None
        ab = [int(x) for x in a.absent.split(",") if x != ""] if a.absent else ()
        doc = build(a.dir, a.layers, numbits=a.numbits, tight=a.tight,
                    table_bits=a.table_bits, check_derived=a.check_derived,
                    dump=a.dump, head_groups=hg, absent=ab, ao_rem_bits=a.ao_rem_bits)
    print()
    for f, v in doc["summary"].items():
        if v["min"] != v["max"]:
            print(f"  {f:<16} {v['min']} .. {v['max']}   {v['histogram']}")
        else:
            print(f"  {f:<16} {v['min']} on every layer")
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(doc, fh, indent=1, sort_keys=True)
            fh.write("\n")
        print(f"\nwrote {a.out}  sha256 {sha256(a.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
