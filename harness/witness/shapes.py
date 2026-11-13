"""The model-wide shape table, read off the layer witnesses' manifests.

A chunk loop over a real model may mix table sizes but not two statements
(`prover/src/chunks.rs`), so `kmax` and `ao_rem_bits` are pinned at the
model-wide worst case while `master_table_bits` varies per layer.  This reads
what each layer needs -- before a loop is launched, not from the assert it
trips -- and prints the table B10's record carries.

`ao_rem_bits` is `Layer::rem_bits()` in `prover/src/layer.rs`:

    bits( 2 * max_{h,t,j} ( Z[h,t] << max(-(jv[h,j] - jo[t]), 0) ) )

which for a fixed (h, t) is maximised at the smallest `jv[h, .]`, so the whole
maximum is over (h, t) alone.

    uv run python -m harness.witness.shapes <witness base> 32 [<out.json>]

**Stream DD (E11): the block axis.**  A sequence has one of these per token
block, so the output path is no longer derivable from the base alone -- every
block's base lives in the same directory -- and the *model-wide* pins are no
longer a maximum over 32 layers but over `B x 32`.  Two additions, both
backwards compatible:

    ... shapes <base> <n> [<out.json>]        one block, named output
    ... shapes --merge a.json b.json ... --out shapes_all.json

The merge recomputes `worst` over every row of every input -- `ao_rem_bits`
grows with the prefix, so the last block's is the model-wide one and a plan
pinned from block 0 would be rejected by `params_of`'s
`--ao-rem-bits {want} is below this witness's {}` assertion at the last chunk,
after hours of proving.

**Issue #118: the bound the pin has to leave room for.**  `ao_rem_bits` is one
number for the whole loop and every layer is proved at it, so the model-wide
worst case can push a layer past the quotient cap the attention output's
rounding identity has to fit inside -- `q_bits + rem_bits`, against 30 in the
plain form and 31 in the `denp2` form (`Plan::new`) -- even though that layer's
own divisor range fits.  `worst_of` therefore evaluates that bound, in both
forms, on every row it took the pin over, and refuses the pin with the layer and
the numbers rather than leaving the refusal to the first chunk of a loop that is
hours old.  Every table also prints the tightest window and what is left of it:
the recorded C=4,096 set sits at `11 + 20 = 31` of 31, which composes and admits
nothing wider.  The layer's own value is legal, so what the refusal names is
`plan.py --ao-rem-per-line`, which writes it on that layer's own line.

**Issue #121: the other half, where the row states it.**  The bound has two
halves and only the quotient one moves with the pin; the window half,
`1 + lo + hi + e_hi`, is the dump's own accumulator and its own exponents.  A
dump records all three (`ao_acc_bits`, `jv_max` and `jo_min`, in the manifest's
`attn_out` block), `shape_of` carries them onto the row, and every row that
states them is evaluated on both halves here and in the plan writer -- so a
shape that does not compose is refused at plan time and not by `Plan::new`.  A
row extracted before the extractor recorded the accumulator width states no
`ao_acc_bits`, and its window half is then **not checked** rather than checked
against a guess: the tables print how many rows that was.  The recorded
4,096-token set's 288 rows are all of that kind, so its plan is unchanged.

**Issue #125: the head line, which this table has no row for.**  A head chunk
proves the attention output's rounding like any layer chunk -- its dump is the
layer dump of the layer it was extracted at with a `head` block added -- so both
halves apply to it.  This table cannot hold it: `shape_of` measures a divisor
range off a layer dump's own columns, the sweep walks one block's layers, and
the head dump is written by its own pass into a directory of its own.  Folding
it in would also widen the loop's shared pin to a value no layer needs.  So the
head line's bound is evaluated where its pin is decided, in the plan writer, off
the manifest that plan already opens; the arithmetic and the way a head row is
named live here, beside the layers' (`ao_head_row`, `ao_head_report`).

    ... shapes --selftest                     the bound and its refusal, no dump
"""
import argparse
import json
import os
import sys


# ---- the attention output's no-wrap bound (issues #111, #118) ---------------
# The rounding identity the attention-output fold proves multiplies `2 mag num`,
# and it is an identity over the integers only while that product stays under
# the field's `2^31 - 1`.  `prover/src/cli.rs`'s `params_of` sizes the two
# accumulator limbs, picks each form off the numbers it is given, and
# `Plan::new` (`prover/src/compose.rs`) refuses a statement whose window does
# not fit -- at proving time, hours after the extraction and the plan that
# decided it.  The functions below are that arithmetic, so `extract.py` refuses
# a dump that would not compose and this table refuses a pin that would not.
# They live here because this module imports neither torch nor numpy at import
# time, and the plan writer does not want either.
#
# `MIN_LOG_ROWS` is `prover/src/common.rs`: a range table is its own `1 << bits`
# rows and the LogUp builder has no shorter column, so no limb is narrower.
MIN_LOG_ROWS = 4
# `p = 2^31 - 1`, so the product has 31 bits to fit in.  The sibling bound's cap
# is 30 in the plain form and the same 31 in the `denp2` form.
NO_WRAP_BITS = 31


def ao_acc_split(need):
    """`params_of`'s two accumulator limbs for a witness needing `need` bits.

    The low limb takes what it can up to 16, the widest range table the layer
    already builds; the high limb takes the rest; neither goes below the
    shortest column the builder has.
    """
    lo = min(max(need, MIN_LOG_ROWS), 16)
    return lo, max(need - lo, MIN_LOG_ROWS)


def ao_quotient(attn_out_bits, rem_bits):
    """The no-wrap bound's sibling term `q_bits + rem_bits`, in its two forms.

    `q_bits` is the fold's target width minus one, which is the dump's own, and
    `rem_bits` is the divisor range the chunk is proved at -- its own, or the
    wider one a plan pins it at, and the only one of the two a plan can move.
    The plain form's cap is 30 and the `denp2` form's is 31; `params_of` takes
    the `denp2` form exactly when the plain form no longer fits.
    """
    q_bits = attn_out_bits - 1
    used = q_bits + rem_bits
    denp2 = used > 30
    limit = NO_WRAP_BITS if denp2 else 30
    return dict(bits=used, limit=limit, margin=limit - used,
                q_bits=q_bits, rem_bits=rem_bits,
                form="denp2" if denp2 else "plain")


def ao_no_wrap(acc_bits, jv_max, jo_min, attn_out_bits, rem_bits):
    """The attention output's no-wrap bound, in the two forms the prover uses.

    The accumulator's two limbs bound one factor of `2 mag num` and the exponent
    table's own domain bounds the other, so what has to hold is
    `1 + lo + hi + e_hi <= 31` with `e_hi = jv_max - jo_min`.  The offset form
    spends the accumulator's sign bit on the offset and the magnitude form
    spends it on the window instead, which is worth one bit and no more;
    `params_of` takes the magnitude form exactly when the offset form does not
    fit, so that every dump that composed before it existed keeps its columns.

    The product form (`Params::ao_acc_prod`) is taken exactly when the
    magnitude form does not fit either.  It range-checks `ahi num` slot by
    slot, so the two limbs stop adding: the largest `|acc|` and the largest
    exponent are a chunk-wide pair that no slot holds, and what has to fit is
    `1 + max(lo, hi) + e_hi <= 31`.  Its soundness argument is
    `prover/src/compose.rs`'s, beside the field.

    The sibling bound of the same identity is `ao_quotient` above, and it is the
    half a plan's pins move.

    Returns what each window comes to, its limit and its **margin**: how many
    bits a wider context, another model or a different quantization may add
    before this layer stops composing.  A negative margin is a dump that would
    not prove.
    """
    e_hi = max(jv_max - jo_min, 0)
    lo_off, hi_off = ao_acc_split(acc_bits + 1)
    lo_mag, hi_mag = ao_acc_split(acc_bits)
    mag = 1 + lo_off + hi_off + e_hi > NO_WRAP_BITS
    prod = 1 + lo_mag + hi_mag + e_hi > NO_WRAP_BITS
    lo, hi = (lo_mag, hi_mag) if mag else (lo_off, hi_off)
    bits = 1 + (max(lo, hi) if prod else lo + hi) + e_hi
    return dict(
        bits=bits, limit=NO_WRAP_BITS, margin=NO_WRAP_BITS - bits,
        acc_bits=acc_bits, acc_lo_bits=lo, acc_hi_bits=hi, e_hi=e_hi,
        form="product" if prod else "magnitude" if mag else "offset",
        quotient=ao_quotient(attn_out_bits, rem_bits),
    )


def ao_quotient_refusal(q, where):
    """What `where`'s quotient term is refused with, or `None` if it fits."""
    if q["margin"] >= 0:
        return None
    return (f"{where}: the attention output's quotient term does not fit: "
            f"{q['q_bits']} + {q['rem_bits']} = {q['bits']} > {q['limit']} (over by "
            f"{-q['margin']}, {q['form']} form). `q_bits` is the fold's target width "
            f"minus one and `rem_bits` is the divisor range this chunk is proved at, "
            f"and the one bit the `denp2` form is worth is already taken.")


def ao_no_wrap_refusal(w, where):
    """What `where`'s window is refused with, or `None` if it fits.

    The message carries the numbers.  Only the product form can be refused
    here, since it is taken whenever the other two do not fit, and what it asks
    is that each limb times `2^e_hi` fits on its own: what is left is a
    narrower exponent window or a narrower high limb.
    """
    if w["margin"] < 0:
        return (f"{where}: the attention output's no-wrap window does not fit: "
                f"1 + max({w['acc_lo_bits']}, {w['acc_hi_bits']}) + {w['e_hi']} = "
                f"{w['bits']} > {w['limit']} (over by {-w['margin']}, {w['form']} form). "
                f"Each limb of |acc| ({w['acc_bits']} bits) times 2^e_hi, with "
                f"e_hi = jv_max - jo_min, has to stay under the field. What is left is a "
                f"narrower exponent window or a narrower high limb.")
    return ao_quotient_refusal(w["quotient"], where)


# ---- issue #121: the window half, where the row states it -------------------
# `ao_no_wrap` above needs three numbers no pin moves: the accumulator's bit
# length and the two exponent extremes.  All three are the dump's own and all
# three are in its manifest, so a shape row can carry them and the table and the
# plan writer can evaluate the window half at the same time as the quotient one.
# The two exponents have been in the manifest all along; `ao_acc_bits` was
# computed during extraction and thrown away until #121, so a row read off an
# older dump states no accumulator width.  The window half is then NOT CHECKED,
# and said to be: an accumulator guessed from a default would refuse a set that
# composes or admit one that does not, and either is worse than the silence the
# reader can see.


def ao_row_of(man):
    """The three window-half numbers a shape row takes from a dump manifest.

    `ao_acc_bits` is `None` on a dump extracted before the extractor recorded
    it.  The row keys are the prover's own field names for these
    (`prover/src/compose.rs`'s `Params`), so the shape table, the plan and the
    statement say them the same way.
    """
    ao = man.get("attn_out") or {}
    return dict(ao_acc_bits=ao.get("ao_acc_bits"),
                ao_jv_max=ao.get("jv_max"), ao_jo_min=ao.get("jo_min"))


def ao_states_window(r):
    """Does this row state the window half at all?"""
    return all(r.get(k) is not None for k in ("ao_acc_bits", "ao_jv_max", "ao_jo_min"))


def ao_row_window(r, rem_bits):
    """One row's whole bound at `rem_bits`, or `None` if it states no window.

    The returned window is the row's own and does not depend on `rem_bits`;
    only the `quotient` inside it does.
    """
    if not ao_states_window(r):
        return None
    return ao_no_wrap(r["ao_acc_bits"], r["ao_jv_max"], r["ao_jo_min"],
                      r.get("attn_out_bits", 8), rem_bits)


# ---- issue #118: the bound under the pins this table writes -----------------
# `ao_rem_bits` below is a model-wide maximum, and it goes on the chunk loop's
# command line, so every layer of every block is proved at it.  A layer whose
# own `attn_out_bits` is wide can be pushed past the quotient cap by that pin
# even though its own divisor range fits, and until this check existed the
# refusal came from `Plan::new` at the first chunk of a loop that was hours old.
# The layer's own value is legal, so the fix is a per-line pin, which is what
# the message says: `plan.py --ao-rem-per-line`.


def ao_where(r):
    """A row, named as a refusal names it.

    Issue #125: a head row names the line it is, because the head chunk is not
    a layer of the model and no layer coordinate of the dump it was extracted at
    identifies it.
    """
    if r.get("head"):
        return ("the head line" if r.get("block") is None
                else f"block {r['block']}'s head line")
    return (f"layer {r['layer']}" if r.get("block") is None
            else f"block {r['block']} layer {r['layer']}")


def ao_pin(rows):
    """The `--ao-rem-bits` a chunk loop over `rows` is pinned at.

    `params_of` refuses a pin below the witness's own (`--ao-rem-bits {want} is
    below this witness's {}`), so the model-wide pin is the maximum over every
    row, which is what `worst_of` writes and what the drivers read.
    """
    return max(r["ao_rem_bits"] for r in rows)


def ao_pin_row(r, rem_bits):
    """One row's two windows at `rem_bits`, and what it is refused with.

    Issue #121: both halves, where the row states the window one.  The window
    half comes first because no pin moves it -- a refusal there is a shape that
    does not compose, and naming `--ao-rem-per-line` for it would name a flag
    that cannot help.  The quotient half is the one the pin moves, and its
    refusal says which of the two it is: a pin this layer cannot take although
    its own divisor range fits, or a shape that does not compose at any pin.

    The returned window is the quotient one, as it was before #121, with the
    row's whole bound under `window` (`None` when the row states none).
    """
    where = ao_where(r)
    w = ao_row_window(r, rem_bits)
    q = dict(w["quotient"] if w else ao_quotient(r.get("attn_out_bits", 8), rem_bits),
             window=w)
    if w is not None and w["margin"] < 0:
        return q, ao_no_wrap_refusal(w, where)
    msg = ao_quotient_refusal(q, where)
    if msg is None:
        return q, None
    # Issue #125: a head row's `ao_rem_bits` IS the pin the line resolves -- the
    # head manifest states no divisor range of its own -- so there is no "its
    # own value fits" case to name here, and `--ao-rem-per-line` is a layer
    # line's flag and could not move a head line at all.
    if r.get("head"):
        if r.get("head_flag_pin") is None:
            msg += (" The head line carries no `--ao-rem-bits` of its own, so it is proved "
                    "at the loop's shared pin, which is the layers' worst case. "
                    "`plan.py --head-flags='--ao-rem-bits N'` writes the head dump's own "
                    "divisor range on the head line instead, and a line's arguments are "
                    "read after the shared ones; `params_of` refuses any N below the head "
                    "dump's own.")
        else:
            msg += (f" {r['ao_rem_bits']} is the pin `--head-flags` names, and `params_of` "
                    f"refuses a pin below the head dump's own divisor range, so no pin "
                    f"recovers this: what is left is a head spec whose fold target width "
                    f"is narrower, or a quantization whose divisor range is.")
        return q, msg
    own = ao_quotient(r.get("attn_out_bits", 8), r["ao_rem_bits"])
    if own["margin"] >= 0:
        msg += (f" This layer's own divisor range is {own['rem_bits']} bits and does fit "
                f"({own['bits']}/{own['limit']}, margin {own['margin']}, {own['form']} "
                f"form), so what does not fit is the model-wide pin. A line's arguments "
                f"are read after the shared ones and the last one wins, so "
                f"`plan.py --ao-rem-per-line` writes this layer's own value on this "
                f"layer's own line and the loop keeps the shared pin for the rest.")
    else:
        msg += (f" This layer's own divisor range is {own['rem_bits']} bits and does not "
                f"fit either ({own['bits']}/{own['limit']}), so no pin recovers it: what "
                f"is left is a narrower context or a quantization whose divisor range is "
                f"narrower.")
    return q, msg


def ao_pin_check(rows, rem_bits):
    """Every row at `rem_bits`: the tightest row, its window, and the refusals."""
    seen = [(r,) + ao_pin_row(r, rem_bits) for r in rows]
    tight = min(seen, key=lambda t: t[1]["margin"])
    return tight[0], tight[1], [m for _, _, m in seen if m]


def ao_pin_refusals(bad):
    """The refusals as one message: three of them, and then how many more.

    Every layer of a set is refused by one pin for the same reason, and the
    three that are printed carry the numbers of all of them.
    """
    return "\n\n".join(bad[:3] + ([f"... and {len(bad) - 3} more layers, at the same pin"]
                                   if len(bad) > 3 else []))


def ao_pin_report(rows, rem_bits):
    """The line the table prints about the pin it writes: the tightest window.

    A margin of zero is a pin that composes and admits nothing wider, which is
    where the recorded C=4,096 set sits: `11 + 20 = 31` in the `denp2` form.
    """
    r, q, _ = ao_pin_check(rows, rem_bits)
    tail = ("; a wider context, another model or a different quantization adds a bit "
            "and this layer stops composing" if q["margin"] == 0 else "")
    return (f"attention output: at --ao-rem-bits {rem_bits} the tightest quotient window "
            f"is {q['bits']}/{q['limit']} bits, margin {q['margin']} ({q['form']} form), "
            f"at {ao_where(r)}{tail}")


def ao_window_report(rows):
    """Issue #121: the line a table prints about the window half.

    No pin moves it, so it is one line per table rather than one per pin: the
    tightest window over the rows that state it, and how many stated none.
    """
    have = [(r, ao_row_window(r, r["ao_rem_bits"])) for r in rows if ao_states_window(r)]
    if not have:
        return (f"attention output: none of the {len(rows)} rows states ao_acc_bits, so "
                f"the no-wrap window half was NOT CHECKED -- these rows were read off "
                f"dumps extracted before the extractor recorded the accumulator width "
                f"(zk-prover #121). A re-extracted set carries it and gets both halves.")
    r, w = min(have, key=lambda t: t[1]["margin"])
    tail = ("; a wider context, another model or a different quantization adds a bit "
            "and this layer stops composing" if w["margin"] == 0 else "")
    if len(have) < len(rows):
        tail += (f". {len(rows) - len(have)} of the {len(rows)} rows state no "
                 f"ao_acc_bits and their window half was NOT CHECKED")
    return (f"attention output: the tightest no-wrap window is {w['bits']}/{w['limit']} "
            f"bits, margin {w['margin']} ({w['form']} form), at {ao_where(r)}{tail}")


# ---- issue #125: the head line, a chunk of the same kind --------------------
# A head dump is the layer dump of the layer it was extracted at with a `head`
# block added: `extract.py --head` writes into the same directory and the same
# `manifest.json` as that layer's own, so the dump carries that layer's
# attention columns and the manifest carries `attn_out_bits` and the whole
# `attn_out` block beside the vocabulary and the norm parameters.  The prover
# reads them like any other dump's -- `Layer::load` builds the segment's
# `AttnOut` out of those columns, and `prover/src/compose.rs` builds the
# attention-output component and its interaction unconditionally (lines 4535
# and 6288).  So a head chunk proves the same rounding identity a layer chunk
# does, both halves of the no-wrap bound apply to it, and its line is evaluated
# exactly as a layer line's is.
#
# It has no row in this table, and it should not have one.  `shape_of` measures
# a layer dump's divisor range off its `zz.i32`, `jo.i32` and `jv.i32`, and the
# sweep that calls it walks one block's layers; the head dump lives in a
# directory of its own, written by its own extraction pass, and is never in that
# sweep.  Folding it in would also be wrong twice over: what this table writes
# is the pin the whole loop is launched with, and the head line resolves its own
# through `--head-flags`, so a head dump's wider divisor range would push every
# layer of the set past the quotient cap without any layer needing it.  The head
# line's pin is decided where the head line is written, so `plan.py` is where it
# is checked, off the manifest it already opens for `master_table_bits`.


def ao_head_refusal(man, where):
    """What an incomplete head manifest is refused with, or `None`.

    `attn_out_bits` and the `attn_out` block's two exponent extremes have been
    written for every dump `extract.py` has ever written, head dumps included,
    so a manifest without them is not a dump manifest.  It is refused rather
    than skipped: skipping it would leave the one line of a plan whose bound
    nothing else evaluates unevaluated and silent.  `ao_acc_bits` is the one
    field that may be absent -- a dump extracted before it was recorded states
    none -- and the window half is then not checked, as a layer row's is not.
    """
    missing = []
    if man.get("attn_out_bits") is None:
        missing.append("attn_out_bits")
    ao = man.get("attn_out")
    if not isinstance(ao, dict):
        missing.append("attn_out")
    else:
        missing += [f"attn_out.{k}" for k in ("jv_max", "jo_min") if ao.get(k) is None]
    if not missing:
        return None
    return (f"{where}: the head dump's manifest states no {', '.join(missing)}, so the "
            f"attention output's no-wrap bound cannot be evaluated for it -- and the head "
            f"chunk proves that rounding like any layer chunk does. A head dump is the "
            f"layer dump of the layer it was extracted at with a `head` block added, and "
            f"`extract.py` writes these fields into every manifest it writes, so a "
            f"manifest without them is not one it wrote: extract the head dump again.")


def ao_head_row(man, rem_bits, block=None, flag_pin=None):
    """Issue #125: the shape row of a head line, off the head dump's manifest.

    The fold's target width and the three window-half numbers are the head
    dump's own, exactly as a layer row's are.  `ao_rem_bits` is not: the
    manifest records no divisor range -- `shape_of` measures that off the dump's
    columns and no plan reads those -- and the head line is proved at the pin it
    resolves, which `params_of` refuses to place below the dump's own.  So the
    resolved pin is the row's value, and `flag_pin` is the one `--head-flags`
    named, or `None` when the line carries none and takes the shared pin.
    """
    return dict(ao_row_of(man), head=True, layer=None, block=block,
                attn_out_bits=man.get("attn_out_bits", 8),
                ao_rem_bits=rem_bits, head_flag_pin=flag_pin)


def ao_head_report(r, q):
    """Issue #125: the head line's two margins, printed beside the layer lines'.

    One line per head chunk, because there is one head line per head block and a
    plan's reader wants its numbers whether or not it is the tightest of the
    plan.  The recorded head configuration is a `b10-final` dump, whose fold
    target width is 8, pinned at the 23 bits its own divisor range needs:
    `7 + 23 = 30` is the plain form's cap exactly.
    """
    w = q["window"]
    limbs = (f"max({w['acc_lo_bits']}, {w['acc_hi_bits']})" if w and w["form"] == "product"
             else f"{w['acc_lo_bits']} + {w['acc_hi_bits']}" if w else "")
    half = (f"the no-wrap window is 1 + {limbs} + "
            f"{w['e_hi']} = {w['bits']}/{w['limit']} bits, margin {w['margin']} "
            f"({w['form']} form)" if w else
            "the no-wrap window half is NOT CHECKED -- this head dump states no "
            "ao_acc_bits (zk-prover #121)")
    return (f"attention output, {ao_where(r)}: at --ao-rem-bits {q['rem_bits']} the "
            f"quotient window is {q['q_bits']} + {q['rem_bits']} = {q['bits']}/{q['limit']} "
            f"bits, margin {q['margin']} ({q['form']} form); {half}")


def shape_of(d):
    # Stream DD: imported here rather than at the top so the merge mode, which
    # reads JSON and nothing else, runs where numpy is not installed (the desk
    # self-test at `b12/e11/dd_selftest.py`).  It is cached after the first of
    # the 32 calls.
    import numpy as np

    man = json.load(open(os.path.join(d, "manifest.json")))
    nh, s = man["heads"], man["s"]
    zz = np.fromfile(os.path.join(d, "zz.i32"), dtype=np.int32).reshape(nh, s).astype(np.int64)
    jo = np.fromfile(os.path.join(d, "jo.i32"), dtype=np.int32).astype(np.int64)      # (S,)
    jv = np.fromfile(os.path.join(d, "jv.i32"), dtype=np.int32).reshape(nh, -1)        # (nh, hd)
    shift = np.maximum(jo[None, :] - jv.min(-1)[:, None], 0)                           # (nh, S)
    rem = int(2 * (zz << shift).max())
    accs = man["accumulators"]
    # stream Z: a dump extracted with a fold matmul carries the fold's own
    # divisor range; the model-wide pins come off these exactly as `ao_rem_bits`
    fold = man.get("fold") or {}
    return dict(
        layer=man["layer"], c=man["c"], s=man["s"],
        # Stream DD: absent on a pre-DD dump, and `None` then rather than 0 --
        # a decode dump is not "block 0".
        block=man.get("block"),
        table_bits=man["master_table_bits"], spread=man["frac_spread"],
        d_bits=man["d_bits"], z_max=man["z_max"],
        ao_rem_bits=int(rem).bit_length(),
        attn_out_bits=man.get("attn_out_bits", 8),
        # issue #121: the window half's own three numbers, so this table and the
        # plan writer evaluate both halves and not just the one the pin moves
        **ao_row_of(man),
        kmax_needed=max(v["kmax_needed"] for v in accs.values()),
        kmax_blind=max(v["kmax_worst_case"] for v in accs.values()),
        fold_q_max=man["attn_out"]["fold_q_max"],
        fold_rem_bits=fold.get("fold_rem_bits", 0),
        fold_gt_min=fold.get("gt_min", 0), fold_gc_min=fold.get("gc_min", 0),
        slim=man.get("slim", False),
    )


def worst_of(rows):
    """The model-wide pins a chunk loop has to be launched with.

    Stream DD: taken over whatever rows it is given, which is one block's 32
    layers in the pre-DD call and `B x 32` under the merge mode.
    """
    tb = sorted({r["table_bits"] for r in rows})
    hist = {b: sum(1 for r in rows if r["table_bits"] == b) for b in tb}
    worst = dict(ao_rem_bits=ao_pin(rows),
                 kmax=max(r["kmax_blind"] for r in rows),
                 fold_q_max=max(r["fold_q_max"] for r in rows),
                 table_bits=hist)
    # stream Z: the fold's model-wide pins.  `fold_rem_bits` is a plain maximum;
    # `fold_alpha` must sit under every layer's min G_t while covering every
    # layer's min G_c, which is a joint condition and is asserted, not assumed.
    if any(r["fold_rem_bits"] for r in rows):
        alpha = max(max(14 - r["fold_gc_min"], 0) for r in rows if r["fold_rem_bits"])
        gt_lo = min(r["fold_gt_min"] for r in rows if r["fold_rem_bits"])
        assert alpha <= gt_lo, f"fold: no alpha fits (need <= {gt_lo}, want {alpha})"
        worst["fold_rem_bits"] = max(r["fold_rem_bits"] for r in rows)
        worst["fold_alpha"] = alpha
    # stream Z: the attnout fold's quotient clamps at the dump's own width
    assert all(r["fold_q_max"] <= (1 << (r["attn_out_bits"] - 1)) - 1 for r in rows), \
        "a layer's fold quotient is past its clamp"
    # issue #118: and the bound the pin above has to leave room for, on every
    # row it was taken over.  `ao_rem_bits` is one number for the whole loop, so
    # a layer is proved at the model-wide worst case and not at its own; a layer
    # that does not fit it is refused here rather than by `Plan::new`, hours in.
    if all(r.get("ao_rem_bits") for r in rows):
        _, _, bad = ao_pin_check(rows, worst["ao_rem_bits"])
        assert not bad, ao_pin_refusals(bad)
    return worst, tb


def _row(layer, rem, bits=12, block=None, table_bits=23, acc=None, jv=-6, jo=-10):
    """A shape table row with the fields the pins are taken over.

    Issue #121: `acc` is the accumulator's bit length, and `None` -- the
    default -- is a row read off a dump that recorded none, whose window half
    is not checked.
    """
    return dict(layer=layer, block=block, c=512, s=512, table_bits=table_bits,
                spread=0.0, d_bits=16, z_max=1, ao_rem_bits=rem, attn_out_bits=bits,
                ao_acc_bits=acc, ao_jv_max=(jv if acc else None),
                ao_jo_min=(jo if acc else None),
                kmax_needed=19, kmax_blind=19, fold_q_max=(1 << (bits - 1)) - 1,
                fold_rem_bits=15, fold_gt_min=8, fold_gc_min=8, slim=True)


def selftest():
    """Issue #118: the bound under the pin this table writes.

    The numbers are the recorded C=4,096 set's: `attn_out_bits` 12, so `q_bits`
    11, and divisor ranges reaching the 20 whose `denp2` window is the field's
    31 exactly.  `plan.py --selftest` checks the same bound where a plan line
    resolves its pin; this checks it where the pin is decided.
    """
    bad = 0

    def case(label, ok, note):
        nonlocal bad
        bad += not ok
        print(f"{'OK  ' if ok else 'FAIL'}  {label:<52s} -> {note}")

    # the two forms, at the two fold widths the sets use
    got = [(ao_quotient(8, 23)["form"], ao_quotient(8, 23)["margin"]),
           (ao_quotient(8, 24)["form"], ao_quotient(8, 24)["margin"]),
           (ao_quotient(12, 20)["form"], ao_quotient(12, 20)["margin"]),
           (ao_quotient(12, 21)["form"], ao_quotient(12, 21)["margin"])]
    want = [("plain", 0), ("denp2", 0), ("denp2", 0), ("denp2", -1)]
    case("the quotient term's two forms and their margins", got == want, got)

    # the recorded set: the pin is the maximum over every row and it composes
    rows = [_row(k, r, block=b) for b in (0, 1)
            for k, r in enumerate((14, 15, 19) if b == 0 else (15, 16, 20))]
    worst, _ = worst_of(rows)
    _, q, refusals = ao_pin_check(rows, worst["ao_rem_bits"])
    ok = (worst["ao_rem_bits"] == 20 and not refusals and q["bits"] == 31
          and q["margin"] == 0 and q["form"] == "denp2")
    case("the recorded pin, at the denp2 form's cap exactly", ok,
         ao_pin_report(rows, worst["ao_rem_bits"]))

    # and the layer this check exists for: one row extracted at the narrower
    # fold width carries a wider divisor range, so the pin it sets is one no
    # 12-bit row can take, although every one of them takes its own
    rows[-1] = _row(2, 23, bits=8, block=1)
    try:
        worst_of(rows)
        got = "NOT refused"
    except AssertionError as e:
        got = str(e).splitlines()[0]
    ok = (got.startswith("block 0 layer 0: ")
          and "11 + 23 = 34 > 31 (over by 3, denp2 form)" in got
          and "--ao-rem-per-line" in got)
    case("a pin no layer of the set can take is refused", ok, got.split(".")[0])

    # a table whose rows are one block's, as the pre-DD call writes it: the
    # refusal names the layer and no block, because the row carries none
    one = [_row(k, r) for k, r in enumerate((15, 16, 21))]
    try:
        worst_of(one)
        got = ["NOT refused"]
    except AssertionError as e:
        got = str(e).split("\n\n")
    ok = (len(got) == 3 and got[0].startswith("layer 0: ")
          and "--ao-rem-per-line" in got[0]
          and got[-1].startswith("layer 2: ") and "no pin recovers it" in got[-1])
    case("one block's table names the layer, and each case", ok, got[0].split(".")[0])

    # ---- issue #121: the other half, where the row states it ----------------
    # A row that carries `ao_acc_bits` and the two exponents is evaluated on the
    # window half as well, and the accumulator's width is the only input to it a
    # quantization moves.

    # the recorded set's rows say nothing about the accumulator, so the window
    # half is not checked and the table says which rows those were
    rows = [_row(k, r, block=b) for b in (0, 1)
            for k, r in enumerate((14, 15, 19) if b == 0 else (15, 16, 20))]
    got = ao_window_report(rows)
    ok = ("NOT CHECKED" in got and "none of the 6 rows states ao_acc_bits" in got
          and worst_of(rows)[0]["ao_rem_bits"] == 20)
    case("a row that states no accumulator is not checked", ok, got.split(" -- ")[0])

    # the same set re-extracted: 19 bits of accumulator against the synthetic
    # shape's window is 25 of 31, and both halves are evaluated at the pin
    rows = [_row(k, r, block=b, acc=19) for b in (0, 1)
            for k, r in enumerate((14, 15, 19) if b == 0 else (15, 16, 20))]
    worst, _ = worst_of(rows)
    _, q, refusals = ao_pin_check(rows, worst["ao_rem_bits"])
    got = ao_window_report(rows)
    ok = (not refusals and q["window"]["bits"] == 25 and q["window"]["margin"] == 6
          and "NOT CHECKED" not in got and "25/31 bits, margin 6 (offset form)" in got)
    case("a re-extracted set is checked on both halves", ok, got)

    # an accumulator one bit past the magnitude form's window composes in the
    # product form, which bounds `mag num` slot by slot
    rows[-1] = _row(2, 20, block=1, acc=27)
    worst_of(rows)
    w = ao_row_window(rows[-1], 20)
    ok = w["form"] == "product" and w["bits"] == 21 and w["margin"] == 10
    case("an accumulator past the magnitude form, in the product", ok,
         f"{w['bits']}/{w['limit']} margin {w['margin']} ({w['form']} form)")

    # and the layer the window half exists for: an exponent window past even
    # the product form.  No pin moves it, so the refusal names neither the pin
    # nor `--ao-rem-per-line`, which could not recover a bit of it.
    rows[-1] = _row(2, 20, block=1, acc=27, jv=-5, jo=-20)
    try:
        worst_of(rows)
        got = "NOT refused"
    except AssertionError as e:
        got = str(e).splitlines()[0]
    ok = (got.startswith("block 1 layer 2: ") and "no-wrap window does not fit" in got
          and "1 + max(16, 11) + 15 = 32 > 31 (over by 1, product form)" in got
          and "--ao-rem-per-line" not in got)
    case("an exponent window past the product form is refused", ok, got.split(".")[0])

    # a set half re-extracted: the rows that state it are checked, the rest are
    # counted, and the pin is taken over all of them either way
    rows = ([_row(k, r, block=0, acc=19) for k, r in enumerate((14, 15, 19))]
            + [_row(k, r, block=1) for k, r in enumerate((15, 16, 20))])
    got = ao_window_report(rows)
    ok = ("margin 6 (offset form), at block 0 layer 0" in got
          and "3 of the 6 rows state no ao_acc_bits" in got
          and not ao_pin_check(rows, worst_of(rows)[0]["ao_rem_bits"])[2])
    case("a set half re-extracted says how many were not", ok, got.split(". ")[-1])

    return 1 if bad else 0


def merge_tables(paths, out):
    """Stream DD: one `worst` over several blocks' shape tables."""
    rows = []
    for q in paths:
        d = json.load(open(q))
        for r in d["rows"]:
            r = dict(r)
            if r.get("block") is None:
                r["block"] = d.get("block")
            rows.append(r)
    worst, _ = worst_of(rows)
    blocks = sorted({r.get("block") for r in rows}, key=lambda b: (b is None, b))
    json.dump(dict(rows=rows, worst=worst, merged_from=list(paths), blocks=blocks),
              open(out, "w"), indent=1)
    print(f"merged {len(paths)} tables, {len(rows)} rows, blocks {blocks}")
    print(f"worst case: {json.dumps(worst)}")
    print(ao_pin_report(rows, worst["ao_rem_bits"]))
    print(ao_window_report(rows))
    print(f"-> {out}")


if __name__ == "__main__":
    # issue #118: the bound's own test, read before the parser because the
    # positional arguments below are a witness set and this needs none
    if "--selftest" in sys.argv[1:]:
        sys.exit(selftest())
    # Stream DD: the merge mode folds several blocks' tables into the one
    # `worst` a sequence's chunk loop has to be pinned from.
    if "--merge" in sys.argv:
        ap = argparse.ArgumentParser()
        ap.add_argument("--merge", nargs="+", required=True)
        ap.add_argument("--out", required=True)
        aa = ap.parse_args()
        merge_tables(aa.merge, aa.out)
        sys.exit(0)
    base, n = sys.argv[1], int(sys.argv[2])
    rows = [shape_of(f"{base}{k}") for k in range(n)]
    print(f"{'L':>3} {'C':>6} {'S':>4} {'tbl':>4} {'sprd':>5} {'dbits':>6} {'zmax':>8} "
          f"{'aorem':>6} {'kmax':>5} {'blind':>6} {'foldq':>6}")
    for r in rows:
        print(f"{r['layer']:>3} {r['c']:>6} {r['s']:>4} {r['table_bits']:>4} "
              f"{r['spread']:>5.1f} {r['d_bits']:>6} {r['z_max']:>8} {r['ao_rem_bits']:>6} "
              f"{r['kmax_needed']:>5} {r['kmax_blind']:>6} {r['fold_q_max']:>6}")
    worst, tb = worst_of(rows)
    # 5.010 fresh cells per master-table row (B9 1.5), so what the tables cost
    # per-layer against forcing the model's largest on all of them
    per = {b: 5.010 * (1 << b) for b in tb}
    own = sum(per[r["table_bits"]] for r in rows)
    one = per[max(tb)] * len(rows)
    print(f"\nworst case: {json.dumps(worst)}")
    print(ao_pin_report(rows, worst["ao_rem_bits"]))
    print(ao_window_report(rows))
    print(f"master table, per-layer {own / 1e9:.3f} G cells vs one shape at 2^{max(tb)} "
          f"{one / 1e9:.3f} G = x{one / own:.2f}")
    # Stream DD: the output path.  Pre-DD it was `<dir>/shapes.json` and with two
    # arguments it still is; a sequence passes a third, because every block's
    # base lives in the same directory and the default would have nine blocks
    # overwriting one file.
    out = (sys.argv[3] if len(sys.argv) > 3
           else os.path.join(os.path.dirname(base) or ".", "shapes.json"))
    json.dump(dict(rows=rows, worst=worst, block=rows[0].get("block")),
              open(out, "w"), indent=1)
    print(f"-> {out}")
