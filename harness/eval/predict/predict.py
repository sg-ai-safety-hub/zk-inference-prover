"""E4 -- the unfitted-scale prediction, registered BEFORE it is measured.

    uv run python -m harness.eval.predict.predict --constants PATH \
        --artifacts DIR --inputs DIR          # print the registration
    uv run python -m harness.eval.predict.predict ... --json   # and rewrite
                                              # out/registration.json
    uv run python -m harness.eval.predict.predict ... \
        --shape Llama-3.1-70B --json          # another registration, into
                                              # out/registration-<shape>.json

What this is.  `experiments/b2/cost_model.py` was built from gadget marginals
measured at exactly two model scales: Qwen2.5-0.5B (B1-B6.5) and Llama-3.1-8B
(B8-B10).  Nothing in it has ever been checked against a third.  This script
freezes what the model says a *third* scale costs, with a pass/fail bracket
fixed in advance, so that the later measurement is a test and not a fit.

The rule the plan attaches to E4 (eval-plan.md section 4, row 9) is in bold:
**"Must be registered before it is measured."**  At the moment this file was
committed no witness, no proof and no cell count existed at any of the three
target scales, on either box -- see the `provenance` block in the emitted JSON.

Nothing here re-derives a frozen constant, and nothing here falls back to a
number.  The anchor that carries the weight is the 8B layer block zk-results
records per proved position, one row per context, with a measured 8B layer
chunk read from `--artifacts` beside it; a constants key that this constants
file does not name is recorded as absent and nothing stands in for it.  The
cost model's own inputs come through `cost_model.PROVENANCE`, constants first
and the micro-benchmark outputs second, and the whole record is frozen with the
prediction.
"""
import argparse
import hashlib
import json
import os
import re
import sys

from harness import constants
from harness import inputs
from harness.constants import get
from harness.costmodel import cost_model as cm
from harness.costmodel import exact_cells as ec
from harness.costmodel import key_tile
from harness.eval.predict import witness_fields as wf

HERE = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------
# HOW THE CELLS ARE PRICED.  Two paths, and the choice is the run's.
#
#   fitted  the per-unit constants a measured 8B segment was read off -- cells
#           per accumulator, per weight, per lookup -- times the shape's own
#           counts.  This is what the three registrations already recorded were
#           made on, and it stays the DEFAULT so that none of them moves.
#   exact   the cells are COUNTED from the prover's trace layout
#           (`harness/costmodel/exact_cells.py`), segment by segment, with each
#           segment's witness-calibrated fields READ OFF ITS OWN DUMP
#           (`--witness-fields FILE`, written by `witness_fields.py`).  Nothing
#           is fitted and nothing is assumed: the fields exist before any proof
#           of them does, which is what keeps the registration a registration.
#
# The owner's ruling of 2026-09-11 is the exact formula wherever we can
# (`docs/decisions/2026-09-11-cells-are-counted-from-the-layout-wherever-a-
# shape-can-be-stated.md`), and the 14B is where it matters most: that model's
# dumps ask for a master exponent table of 2^21 to 2^25 rows where the fitted
# model charges the 8B's 2^23 everywhere, and five cells a table row makes the
# difference tens of millions of cells a segment.
# --------------------------------------------------------------------------
CELLS = ("fitted", "exact")

S = 512                     # decoded tokens per chunk: the frozen spec's chunking
B = cm.BATCH                # the LogUp lookup-batch size the model charges (2)

# --------------------------------------------------------------------------
# The target shapes.  Chosen for one reason: `experiments/b1/model.py` runs
# them with no new architecture (an explicit Qwen2 forward pass, the same code
# B8 pointed at Llama-3.1-8B through `ZK_MODEL`), and `hd` is a power of two,
# which the model asserts.  Gemma needs E5a's embedding normalizer, QK-norm and
# dual-theta local/global attention first, so it is not an E4 candidate.
#
# 7-tuple order is `cost_model`'s: (L, d, nh, nkv, hd, dff, vocab).
# Every field is transcribed from the model's own published config.json.
# --------------------------------------------------------------------------
SHAPES = {
    "Qwen2.5-14B-Instruct": dict(
        cfg=(48, 5120, 40, 8, 128, 13824, 152064), tied=False,
        role="primary", hf="Qwen/Qwen2.5-14B-Instruct",
        why="above both fitted scales, so the claim is extrapolation and not "
            "interpolation; 29.5 GB of bf16 weights and a layer chunk ~1.3x the "
            "8B's, both comfortable on the H200"),
    "Qwen2.5-3B-Instruct": dict(
        cfg=(36, 2048, 16, 2, 128, 11008, 151936), tied=True,
        role="fallback-small", hf="Qwen/Qwen2.5-3B-Instruct",
        why="if the H200 window is under an hour: interpolates between the two "
            "fitted scales rather than extrapolating, so it is a weaker test -- "
            "registered so that falling back is not a post-hoc choice"),
    "Qwen2.5-32B-Instruct": dict(
        cfg=(64, 5120, 40, 8, 128, 27648, 152064), tied=False,
        role="fallback-large", hf="Qwen/Qwen2.5-32B-Instruct",
        why="the strongest test if the window is generous, and the shape closest "
            "to the paper's 27B frontier row; C=4096 only -- a 16K layer chunk "
            "extrapolates past the card"),
    # The shape E4's third scale actually measures.  Qwen2.5 puts a bias on q, k
    # and v, the statement has no site for a projection bias and an honest
    # segment is rejected on the one equality that compares the derived integer
    # with the key tile's (#172; #174 is the component that would be needed),
    # so no Qwen layer can be proved and the extractor now refuses one by name.
    # This is the nearest bias-free model of the same class: the same forty
    # heads of 128, the same hidden size, the same inner width, eight layers
    # fewer, and one key/value head per attention head where the 14B shares one
    # per five.  Whether it becomes E4's third-scale row rather than the 14B is
    # the owner's call (PR 173, decision 3); the row is here because the shape
    # is what a registration prices.
    "CodeLlama-13b-hf": dict(
        cfg=(40, 5120, 40, 40, 128, 13824, 32016), tied=False,
        role="stand-in", hf="codellama/CodeLlama-13b-hf",
        why="above both fitted scales and provable, which the 14B is not: no "
            "bias on any matmul, a head dimension that is a power of two, and a "
            "16,384-token trained window, so the 4,608-token sequence the "
            "measurement proves is inside it.  Llama-2-13B is the same seven "
            "numbers and was not taken: its window is 4,096 and its published "
            "weights are fp16, which the harness would round on the way in"),
    # zk-results #93.  Not an E4 candidate and not registered by an E4 run: it is
    # the shape of the 70B campaign, whose measurement is a SYNTHETIC layer
    # segment at this width and not the model, so `hf` names where the seven
    # numbers were read and not a checkpoint anything here loads.
    "Llama-3.1-70B": dict(
        cfg=(80, 8192, 64, 8, 128, 28672, 128256), tied=False,
        role="campaign", hf="meta-llama/Llama-3.1-70B",
        why="the shape zk-results #93 measures and projects: 64 attention heads "
            "of 128 with 8 key/value heads, twice the 8B's hidden size and twice "
            "its FFN. 64 heads is a power of two, so it clears the rotation "
            "gadget's requirement that the 14B's 40 do not, and the width is the "
            "one `SHAPES` in `harness/costmodel/frontier.py` now carries a row "
            "for"),
}

# --------------------------------------------------------------------------
# A COUNTED REGISTRATION, shape by shape.  `--cells exact` registers ONE shape,
# because the witness fields it counts with are one model's dumps, and what the
# document has to say about that shape's standing is here rather than in
# `REGISTERED`: the priced registration above was made on 2026-09-01 with
# nothing at any of those three shapes in existence, and a counted one is made
# after that shape's witnesses were extracted and before any proof of them was
# accepted.  Those are two different standings and they are not one row.
#
# Everything but the date and the disclosure is the same sentence for every
# shape, so it is written once, in `counted_standing`, and built from the fields
# file itself -- the model it names, the spec it was extracted at, the context
# and the per-dump digests.  ADDING A MODEL IS THEREFORE ONE ROW IN `SHAPES`:
# the seven-tuple from the model's published config, `hf`, `role`, `why`, and a
# `counted` block with the date it is registered on, who registers it, the
# driver that will measure it and any disclosure that shape needs.  Nothing else
# in this file is shape-specific.
COUNTED = {
    "Qwen2.5-14B-Instruct": dict(
        utc_date="2026-09-11", by="stream E4 (desk lane), the counted registration",
        driver="tools/drivers/register-14b.sh (zk-results), which passes `--cells "
               "exact` and the fields file, and tools/drivers/predict-14b.sh, which "
               "proves every layer of the target shape with these flags",
        disclosure=(
            "Two of the 48 layers are NOT blind, and this document says so rather "
            "than letting a reader assume otherwise: the first attempt to prove "
            "layers 0 and 1 was REJECTED by the prover's own verification "
            "(zk-prover issue #172, an honest proof rejected at 40 attention "
            "heads), and those two rejected documents reported `cells.fresh` "
            "1,793,630,880 and 1,824,875,168, which that issue states and which "
            "was visible before this registration was made. Layers 2 to 47 are "
            "blind. The counted total for those two segments landing on those two "
            "integers is the expected result of counting a layout, not evidence of "
            "a fit: nothing here is fitted to them, and the same code reproduces "
            "all 84 recorded 8B segments to the cell without ever having seen this "
            "shape.")),
    "CodeLlama-13b-hf": dict(
        utc_date="2026-09-11", by="stream E4 (desk lane), the counted registration",
        driver="tools/drivers/register-13b.sh (zk-results), which passes `--cells "
               "exact` and the fields file, and tools/drivers/predict-13b.sh, which "
               "proves every layer of the target shape with these flags",
        disclosure=(
            "Layer 0 of this model cannot be proved and is not in this "
            "registration: the harness refuses to write its dump, because the "
            "attention output's quotient term is 7 + 26 = 33 bits against the "
            "field's 31 in the widest form there is, and the divisor range is the "
            "layer's own, which a driver may only widen. So what is registered and "
            "what will be measured is layers 1 to 39, with the embedding segment "
            "in neither. "
            "One of those 39 layers is NOT blind, and this document says so rather "
            "than letting a reader assume otherwise: before this registration was "
            "made, layer 16 of this model was extracted and proved once as a probe "
            "of whether a bias-free model of this class proves at all, after the "
            "14B's rejection left E4's third scale with no measurement. That proof "
            "accepted and reported `cells.fresh` 2,290,115,312, at C=4,096, S=512, "
            "spec b10-final, head groups 16,16,8 and the same statement. It made no "
            "run record and the dump it proved is not one of the dumps this "
            "registration is counted from. Layers 1 to 15 and 17 to 39 are blind. "
            "The counted total for layer 16 landing on that integer is the expected "
            "result of counting a layout, not evidence of a fit: nothing here is "
            "fitted to it, and the same code reproduces all 84 recorded 8B segments "
            "to the cell without ever having seen this shape.")),
    # zk-results #93.  The one counted registration here whose segments are not
    # dumps: what is measured against it is a SYNTHETIC layer segment at this
    # width, built by the prover's own generator, so there is no witness of the
    # model to read fields off and the fields are stated instead.  Which of them
    # is a flag the prover takes verbatim, which is the generator's own and
    # which is the assumption is in the fields file's `stated_from` and travels
    # into this document with it.
    "Llama-3.1-70B": dict(
        utc_date="2026-09-13", by="zk-results #93, the 70B campaign (desk lane)",
        driver="tools/drivers/register-70b.sh (zk-results), which passes `--cells "
               "exact`, the stated fields file and the block length, and "
               "tools/drivers/predict-70b.sh, which proves ONE synthetic layer "
               "segment at this width per context with these flags -- not the "
               "model's layers, and not a weight of it",
        disclosure=(
            "Nothing at this width has been proved, counted or measured, so no "
            "segment of this registration is anything but blind. What the "
            "registration assumes, and what the measurement therefore tests, is "
            "two things and they are not the same thing. The first is the count "
            "itself: the trace layout at this shape, which is the same code that "
            "reproduces all 84 recorded 8B segments to the cell. The second is the "
            "attention output's measured windows -- the divisor's range and the "
            "accumulator's two limbs -- which no flag pins and which a synthetic "
            "segment only settles when it is built; they are stated here at the "
            "values the same generator produced at the 8B attention width, and a "
            "run that produces others is a run at other fields and not a miss of "
            "the count. The master exponent table and the normalisation window are "
            "not in that second group: the driver passes both and the prover takes "
            "both verbatim.")),
}


def counted_record(name):
    """The shape's own counted-registration record, or the refusal that says what
    a new shape needs.  Looked up before anything is read, so a missing row costs
    nothing."""
    rec = COUNTED.get(name)
    if rec is None:
        raise SystemExit(
            f"--cells exact --shape {name}: this document carries no counted "
            f"registration record for that shape. A counted registration states "
            f"when it was made and what was and was not in existence at the time, "
            f"and neither is derivable from a witness. Add a row to `COUNTED` "
            f"naming the date, who registers it, the driver that measures it and "
            f"any disclosure the shape needs -- and a row to `SHAPES` if the shape "
            f"itself is new, which is the seven-tuple from the model's published "
            f"config.\n"
            f"  the shapes it holds: {', '.join(sorted(COUNTED))}")
    return rec


def counted_standing(name, fields):
    """What a counted registration's provenance says about its own standing.

    Generic over the shape table on purpose: a counted registration of any shape
    makes the same claim, and the numbers that back it come out of the fields
    file rather than out of a sentence somebody has to remember to edit.
    """
    rec = counted_record(name)
    rows = list(fields["layers"].values())
    model = sorted({r.get("model") for r in rows if r.get("model")}) or ["unnamed"]
    spec = sorted({r.get("spec") for r in rows if r.get("spec")}) or ["unnamed"]
    ctx = fields_context(fields)
    absent = sorted(fields.get("absent") or ())
    gap = ("" if not absent else
           f"The stack has {len(rows) + len(absent)} layers and this counts "
           f"{len(rows)}: layer{'s' if len(absent) > 1 else ''} "
           f"{', '.join(str(k) for k in absent)} "
           f"{'are' if len(absent) > 1 else 'is'} NOT in it, because the harness "
           f"refused to write that dump, so no witness of it exists and no proof "
           f"of it will. Every per-token figure here is over the segments this "
           f"file names and not over the stack. ")
    # A synthetic segment has no dump, so the sentence above would be false of
    # it in every clause: there is no witness set, nothing was extracted, and no
    # digest names anything.  What the registration is made from is a block of
    # fields somebody stated, and the file says which of them the prover takes
    # verbatim and which are the assumption.
    if fields.get("synthetic"):
        what = (f"What does exist, and what this registration is made from, is a "
                f"STATED block of fields for a segment the prover BUILDS: "
                f"{len(rows)} synthetic segments of one token block at "
                f"C={ctx:,}, spec {spec[0]}, every one of them the same segment, "
                f"because a synthetic stack at one width has one. No weight of "
                f"this model is read anywhere, no dump of it exists and no digest "
                f"of one is in this document. The block is in "
                f"`frozen_inputs.witness_fields` and `stated_from` beside it says "
                f"of each field whether the prover takes it from the driver's own "
                f"flag, whether it is a property of the generator, or whether it "
                f"is the value a recorded synthetic segment of the same generator "
                f"carried -- which is the assumption this registration makes. ")
    else:
        what = (f"What does exist, and what this registration is made from, is the "
                f"extracted witness set: {len(rows)} layer dumps of one token "
                f"block at C={ctx:,}, spec {spec[0]}, whose per-layer digests are "
                f"in `frozen_inputs.witness_fields`. A dump is not a measurement: "
                f"every field read off it is a property of the forward pass and is "
                f"fixed before any proof is built. ")
    return dict(
        utc_date=rec["utc_date"], by=rec["by"], driver=rec["driver"],
        layers_counted=sorted(int(k) for k in fields["layers"]),
        layers_absent=absent,
        no_measurement_exists=(
            f"NO accepted proof and NO measured cell count of {model[0]} exists in "
            f"either repository or on any host. " + what + gap
            + rec.get("disclosure", "No total at this shape was visible when this "
                                    "registration was made.")))


# The shapes a run registers when it is given no `--shape`: E4's three, in the
# order E4 registered them.  A run of this document is a registration, so which
# shapes it prices is named rather than defaulted to whatever `SHAPES` holds --
# adding a row above must not move the document E4 registered.
E4_SHAPES = ("Qwen2.5-14B-Instruct", "Qwen2.5-3B-Instruct", "Qwen2.5-32B-Instruct")

# What the provenance block says about each selection this document knows how to
# register: the date the prediction was written, who wrote it, and the claim a
# registration exists to make -- that nothing is measured at the shapes it
# prices.  A selection that is in no row here is refused rather than registered
# under E4's dated sentence, and E4's own row is left exactly as it was written.
REGISTERED = {
    E4_SHAPES: dict(
        utc_date="2026-09-01", by="stream E4 (desk lane)",
        driver="tools/drivers/predict-14b.sh (zk-results), which proves every "
               "layer of the target shape with these flags",
        no_measurement_exists=(
            "At registration there is NO witness, NO proof and NO cell count for "
            "Qwen2.5-3B / 14B / 32B anywhere in this repo or on either box. The "
            "only scales ever extracted are Qwen2.5-0.5B (experiments/b65/out/) "
            "and Llama-3.1-8B (experiments/b8, b9, b10). The H200 queue at "
            "registration time is bring-up -> E8 seq6 -> E2 measurement -> E4 "
            "extraction, and E8 is the item in progress; the H100 is on E11's "
            "third fit point. Neither box has been asked for any number in this "
            "file.")),
    ("Llama-3.1-70B",): dict(
        utc_date="2026-09-10", by="zk-results #93, the 70B campaign (desk lane)",
        driver="tools/drivers/predict-70b.sh (zk-results), which proves a "
               "SYNTHETIC layer segment at this shape's width, one per context, "
               "with these flags -- not the model's layers, and not a weight of "
               "it",
        no_measurement_exists=(
            "At registration there is NO witness, NO proof and NO cell count at "
            "the Llama-3.1-70B width anywhere in either repository or on any "
            "box, and no box is booked. Every measured number either repository "
            "holds was taken at the Llama-3.1-8B width or below. What the "
            "campaign will measure against this is a SYNTHETIC layer segment at "
            "this width -- the shape and not the model -- so the run it is "
            "registered against proves no 70B weight and the bracket's "
            "invalidation clause, which asks for a real extracted witness, is "
            "not met by it: what the comparison tests here is the closed form "
            "and the cell counts at a width nothing was fitted at, and the "
            "document says so rather than being read as the E4 bracket.")),
}

CONTEXTS = (4096, 16384)


_SEQ5 = {}


def _seq5(ctx, layer=16):
    """One measured 8B layer chunk, for scaling only (memory, ratios).

    Not a frozen constant -- `constants.json` freezes the 32-layer sums and the
    per-token figures, not L16 alone -- so it is read from the artifact and
    cited as such.  Nothing measured lives in this repository, so the artifact
    comes from `--artifacts DIR` (or `ZK_ARTIFACTS`), the directory the rest of
    the cost model already reads run outputs from, laid out as
    `<DIR>/seq5/seq_c<C>_L<k>.json`.  It is read when a caller asks for it and
    not when this module is imported.
    """
    if (ctx, layer) not in _SEQ5:
        f = inputs.artifacts_dir() / "seq5" / f"seq_c{ctx}_L{layer}.json"
        if not f.exists():
            raise SystemExit(f"missing {f}: the measured 8B layer chunk at C={ctx}, "
                             "layer 16, which the shape ratios are stated against")
        _SEQ5[(ctx, layer)] = json.loads(f.read_text())
    return _SEQ5[(ctx, layer)]["result"]["point"]


def _seq5_source(ctx, layer=16):
    """Which run, block and segment the reference chunk was derived from.

    The chunk carries its own `derived_from` block -- the run id, the slice
    output, the segment tag and the proof digest -- so the registration records
    where the number came from instead of naming a file path."""
    _seq5(ctx, layer)
    d = dict(_SEQ5[(ctx, layer)].get("derived_from") or {})
    d.pop("file", None)
    return d


def ref8b():
    """The 8B L16 chunk's fresh cells at both contexts: the scaling reference."""
    return {c: _seq5(c)["cells"]["fresh"] for c in CONTEXTS}


REF8B_SOURCE = "<--artifacts>/seq5/seq_c{4096,16384}_L16.json (result.point.cells.fresh)"

# --------------------------------------------------------------------------
# The predictor.  Two halves, deliberately kept apart because they have very
# different standing:
#
#   key tiles   the attention key tile family.  A CLOSED FORM read off the layout
#           (`prover/src/attn_trace.rs` x `layout.rs` x `compose.rs`'s `Plan`),
#           not a fit -- see `attention_grid.{base,fresh}_cell_coeffs`.  It
#           reproduces the measured `family_cells.key_tiles` of every recorded
#           layer chunk EXACTLY, at both contexts, base and interaction, with
#           and without the five reductions
#           (`harness.costmodel.key_tile --selftest`).  So the E4 claim on this
#           half is exact, not bracketed.
#
#           The family is `C/S` prefix key tiles of S keys each plus the S x S
#           diagonal block, with one per-token component across the lot, and it
#           costs `nh*hd*(C - S)` base cells per chunk more than one untiled
#           prefix key tile would (issue #141).  `selfcheck` prices the statement
#           the measured chunk names; `predict` prices `STATEMENT` below, the
#           statement the run this is registered against proves.
#
#   rest    everything else in a layer chunk: the dequant, the Freivalds check, the
#           requantize, the q/k quantize, the int8 range checks, the
#           attention-output fold, RoPE, the lookup tables, the boundary trees
#           and the residual stream.  Two of those are closed forms of the
#           layout like the key tile family (the boundary trees and the lookup
#           tables); the rest is arithmetic in the 7-tuple times a gadget rung.
#           This is the half that can be wrong at a new scale.
#
#           Both halves are priced at the SAME statement, and `cells_per_token`
#           is told which one: `--gkr-dequant`, `--gkr-residual-stream`,
#           `--gkr-freivalds` and `--gkr-attnout` each prove a family's LogUp
#           running sums instead of committing them, and the driver passes all
#           four beside `--gkr-key-tiles`.  What the model does NOT carry is
#           `gkrlink`, the reductions' own link -- 12,529,664 cells on both
#           recorded 8B layer chunks, with no form in the shape -- so this half
#           is short by about that much per chunk and `cost_model.PROVENANCE`
#           records it as unpriced.
#
# `rest` is the cost model's layers-only per-token total minus its `attn` term,
# plus the residual stream's NET cost (`residual_stream_cells(...)['net']` --
# total minus the stub the composed statement replaces).
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# THE STATEMENT THE REGISTRATION PRICES.  `tools/drivers/predict-14b.sh` in
# zk-results proves every layer of the target shape with `--diagonal-block
# --gkr-key-tiles --gkr-dequant --gkr-residual-stream --gkr-freivalds
# --gkr-attnout`, the same configuration as both headline runs, so that is the
# statement the key tile half is priced at.  The closed form's own defaults are
# a different statement -- the five reductions off -- and would register a
# larger interaction at the same shape, so it is asked for this one explicitly
# and never left to default.
#
# `kv_boundary` and `bind` are not flags: the prover derives `kv_boundary =
# diagonal_block && !head` (`cli.rs`) and publishes the root-equality claims
# whenever the segment has `Params::rope`, which every layer segment does.
# --------------------------------------------------------------------------
STATEMENT = dict(diagonal_block=True, kv_boundary=True, bind=True, gkr=True)

# The same statement for the other half: every one of the prover's five batched
# reductions, which is what the driver passes.  `STATEMENT` is the key tile
# family's own configuration and `gkr=True` in it is `--gkr-key-tiles` alone;
# this is the set `cells_per_token` and `residual_stream_cells` are told about.
STATEMENT_REDUCTIONS = cm.REDUCTIONS

STATEMENT_FLAGS = dict(
    diagonal_block="--diagonal-block",
    kv_boundary="not a flag: `kv_boundary = diagonal_block && !head` (`cli.rs`), "
                "so every layer segment of the run has it",
    bind="not a flag: `Params::rope`, which every layer segment has and no "
         "output head does",
    gkr="--gkr-key-tiles",
)

# What the driver passes beside `--gkr-key-tiles`, and what the model does with
# it.  Named here so a reader of the registration is not left to infer it from
# the integers.
STATEMENT_NOT_PRICED = dict(
    flags="--gkr-dequant --gkr-residual-stream --gkr-freivalds --gkr-attnout",
    effect="each proves the LogUp running sums of the family it names instead "
           "of committing them, and moves what it does commit into `gkrlink`. "
           "`cells_per_token(gkr=...)` prices all four; `gkrlink` itself is the "
           "one family the model does not price, and "
           "`cost_model.PROVENANCE['gkrlink_cells_per_chunk']` says so",
)


def key_tiles_per_chunk(nh, hd, s, c, base_slope=2.0, **kw):
    """`cm.key_tile_family`, with the ONE coefficient caveat 1 disputes exposed.

    The family is the prefix tiled into key blocks of `s` keys plus the diagonal
    block, not one prefix key tile over `c` keys; `**kw` is the statement's own
    configuration (`diagonal_block`, `kv_boundary`, `bind`, `gkr`), which
    `selfcheck` reads off the measured chunk and `predict` takes from
    `STATEMENT`.  There is no default here worth having: the caller says which
    statement it is pricing or the integers mean nothing.

    `base_slope` is the key tile family's committed BASE cells per (head, key, token)
    -- the `2 * nh * s * c` score-and-numerator term of `cm.key_tile_cells`, i.e. the
    64 in `attention_grid.key_tile_model_base_coeffs` divided by NH=32.  The refuted
    `key tiles = p * S^2` reading of B9's two prefill points puts it at
    72.083 / 32 = 2.2526 instead.  Everything else is untouched: this is a
    sensitivity, not a second model.  The family commits that term over `c + s`
    keys however the prefix is tiled, so the shift is unchanged.
    """
    b, i, sat = cm.key_tile_family(nh, hd, s, c, **kw)
    extra = (base_slope - 2.0) * nh * (s * c + s * s)
    return b + extra, i, sat


def exact_segments(cfg, ctx, fields, seq=S, reductions=None, head_groups=None):
    """Every layer segment of the stack, counted from its own witness fields.

    The stack is one segment that carries the embedding lookup and `L - 1` layer
    segments, each over the `seq` positions it covers -- the same split
    `cost_model.exact_cells_per_token` makes, except that the witness block is
    per segment here rather than one assumed block for all of them.  That is the
    whole point: `log_table` is 21 to 25 over this model's 48 layers and a single
    value for the stack is wrong on 46 of them.

    Three things a `params` block needs that the fields file carries and the
    modal assumption does not, all of them the dump's own:

      * `ao_acc_mag` and `ao_acc_prod`, the attention output's accumulator
        form.  `params_of` takes the magnitude form exactly when the offset
        window does not fit, and the product form on top of it exactly when the
        magnitude window does not fit either, and that is a question about the
        segment's own accumulator and exponents;
      * the dequantize gadget each matmul is on (`deq_variants`).  A
        `b10-final` dump has every matmul on the chain; the 8B segments the cost
        model was read off were extracted at a spec that folds `v`, and the fold
        is not the chain;
      * the fold's pins, which are zero wherever nothing is folded.

    `bound` beside each count is the same segment with every witness-calibrated
    field at the dearest value the recorded 8B runs carry
    (`exact_cells.witness_bounds`): what a projection at this shape would have to
    state if no witness existed.  It is here to be compared against, not added
    up.
    """
    L, d, nh, nkv, hd, dff, vocab = cfg
    red = STATEMENT_REDUCTIONS if reductions is None else reductions
    hg = list(head_groups if head_groups is not None else ec.head_groups_for(nh))
    if sum(hg) != nh or any(g & (g - 1) for g in hg):
        raise SystemExit(
            f"--head-groups {hg} does not group {nh} attention heads into key "
            f"tiles the prover would build: the groups have to add up to the head "
            f"count and each one has to be a power of two (`Attn::build`'s "
            f"`insts must be a power of two`). The run's own grouping belongs in "
            f"the fields file's `knobs.head_groups`.")
    rows = []
    absent = set(fields.get("absent") or ())
    for k in range(L):
        row = (fields.get("layers") or {}).get(str(k))
        if row is None and k in absent:
            # A layer the extractor refused to write.  It is not counted at
            # another layer's fields and it is not counted at a guess: the
            # registration carries fewer segments than the model has layers, and
            # `counted_standing` says which ones and that the per-token figure
            # is over those.
            continue
        if row is None:
            raise SystemExit(
                f"the fields file names no layer {k}, and this shape has {L}. Every "
                f"segment is counted at its own witness fields, and a layer that is "
                f"not in the file is not counted at another layer's. A layer the "
                f"extractor refused belongs in the file's `absent` list, which "
                f"`witness_fields --absent` writes.")
        w = dict(wf.witness_block(row), ao_acc_mag=row["ao_acc_mag"],
                 ao_acc_prod=row.get("ao_acc_prod", False))
        p = ec.segment_params("first" if k == 0 else "layer", d=d, nh=nh, nkv=nkv,
                              hd=hd, dff=dff, vocab=vocab, s=seq, c=ctx,
                              witness=w, gkr=red, head_groups=hg)
        p["mats"] = wf.mats_of(row, p["mats"])
        got = ec.cells_of(p)
        rows.append(dict(
            layer=k, kind=ec.segment_kind(p), fresh=got["fresh"], base=got["base"],
            interaction=got["interaction"], boundary=got["boundary"],
            bound_fresh=ec.cells_of(p, bound=True)["fresh"],
            table_rows=got["table_rows"],
            components={n: b + i for n, (b, i) in sorted(got["components"].items())},
            fields=w, deq_variants=dict(row["deq_variants"]),
            dump_sha256=row.get("dump_sha256")))
    return rows


def fields_context(fields):
    """The context the dumps in a fields file were extracted at.

    A witness field is a property of one forward pass over one prefix, and the
    prefix is part of it: the attention output's divisor range grows with the
    context, and so does the master exponent table.  So a fields file is for one
    context and this is how a caller finds out which.
    """
    cs = sorted({r["c"] for r in fields["layers"].values() if r.get("c") is not None})
    if len(cs) != 1:
        raise SystemExit(f"the fields file names {len(cs)} contexts ({cs}) and a "
                         f"registration is counted at one: every dump of one block "
                         f"reads the same prefix")
    return cs[0]


def predict_exact(cfg, ctx, fields, seq=S, statement=None, reductions=None,
                  head_groups="knobs"):
    """The same quantity as `predict`, counted instead of priced.

    `total` is the layers-only fresh cells per decoded token: the shape's own
    segments' counts over the `seq` positions one token block proves.  The key tile
    half is the layout's own `key_tiles` component rather than the closed form,
    which is the same integer wherever the three grids tile exactly and is not
    where they pad.
    """
    L = cfg[0]
    st = STATEMENT if statement is None else statement
    red = STATEMENT_REDUCTIONS if reductions is None else reductions
    if head_groups == "knobs":
        head_groups = (fields.get("knobs") or {}).get("head_groups")
    at = fields_context(fields)
    if at != ctx:
        raise SystemExit(
            f"the fields file was read off dumps at C={at:,} and this asks for "
            f"C={ctx:,}. The attention output's divisor range and the master "
            f"exponent table both grow with the prefix, so a segment at one "
            f"context is not counted at another's fields: extract that context and "
            f"read its own.")
    rows = exact_segments(cfg, ctx, fields, seq=seq, reductions=red,
                          head_groups=head_groups)
    fresh = sum(r["fresh"] for r in rows)
    kt = {r["components"]["key_tiles"] for r in rows}
    assert len(kt) == 1, f"the key tile count is not one integer over the layers: {kt}"
    comps = {}
    for r in rows:
        for n, v in r["components"].items():
            comps[n] = comps.get(n, 0) + v
    key_tiles = sum(r["components"]["key_tiles"] for r in rows) / seq
    ktb, kti = ec.key_tile_cells(_kt_shape(cfg, ctx, seq, red, head_groups))
    return dict(key_tiles=key_tiles, rest=fresh / seq - key_tiles, total=fresh / seq,
                key_tiles_base_per_chunk=ktb, key_tiles_interaction_per_chunk=kti,
                per_layer_chunk=fresh / L, saturated=False,
                statement=dict(st), reductions=tuple(red), cells="exact",
                layers=rows, layers_fresh_sum=fresh,
                bound_total=sum(r["bound_fresh"] for r in rows) / seq,
                components={n: v / seq for n, v in sorted(comps.items())},
                reductions_not_priced=())


def _kt_shape(cfg, ctx, seq, red, head_groups):
    """The one layer segment the key tile integers are read off, as a block."""
    L, d, nh, nkv, hd, dff, vocab = cfg
    p = ec.segment_params("layer", d=d, nh=nh, nkv=nkv, hd=hd, dff=dff, vocab=vocab,
                          s=seq, c=ctx, witness=ec.modal_assumption("layer"), gkr=red,
                          head_groups=list(head_groups if head_groups is not None
                                           else ec.head_groups_for(nh)))
    return p


def recorded_8b_fields(ctx):
    """A fields block for the 8B, out of the recorded segments' own `params`.

    The anchor under `--cells exact` is the same count against the same 32
    segments the recorded row divides, with each segment's OWN witness fields --
    which is the strongest statement this document can make about the path it
    registers with: given the fields, the count is the recorded number.
    """
    rows = {}
    for p in ec.headline_chunks(("layer", "first segment")):
        q = p["params"]
        if q["c"] != ctx:
            continue
        rows[str(int(p["segment"].split("L")[1]))] = dict(
            {f: q[f] for f in wf.WITNESS_FIELDS}, c=q["c"],
            ao_acc_mag=bool(q.get("ao_acc_mag", False)),
            ao_acc_prod=bool(q.get("ao_acc_prod", False)),
            deq_variants={m["name"]: m["variant"] for m in q["mats"]},
            dump_sha256=None)
    hg = sorted({tuple(p["params"]["head_groups"])
                 for p in ec.headline_chunks(("layer", "first segment"))
                 if p["params"]["c"] == ctx})
    assert len(hg) == 1, f"the recorded 8B segments disagree on the head grouping: {hg}"
    return dict(schema=wf.SCHEMA, layers=rows, knobs=dict(head_groups=list(hg[0])),
                read_by="the recorded segments' own `params`, out of "
                        "harness/costmodel/exact_chunks.json")


def predict(cfg, ctx, seq=S, base_slope=2.0, statement=None, reductions=None,
            cells="fitted", fields=None):
    """Layers-only fresh cells per decoded token, split key tiles / rest.

    `statement` is the key tile configuration being priced, `STATEMENT` by
    default.  The cost model's own `attn` term is subtracted whole and replaced
    by `key_tiles_per_chunk`, so it is that argument and nothing else that says
    which statement the key tile half of the prediction is for.

    `reductions` is the same statement for the other half -- which of the
    prover's five batched reductions the run proves -- and it defaults to
    `STATEMENT_REDUCTIONS`, all five, which is what the driver passes.  The two
    arguments have to agree: pricing the key tile family at a reduced statement
    and everything else at an un-reduced one is not a statement any run proves.

    `cells="exact"` counts the segments from the layout instead, at the witness
    fields `fields` carries.  The default is `fitted` because this document is a
    registration and the three already recorded were made on that path.
    """
    if cells not in CELLS:
        raise SystemExit(f"--cells {cells!r}: the cell side is counted (`exact`) or "
                         f"priced from per-unit constants (`fitted`)")
    if cells == "exact":
        if not fields:
            raise SystemExit(
                "--cells exact needs --witness-fields FILE: every segment is counted "
                "at the fields its own dump carries, and this document assumes none "
                "of them")
        return predict_exact(cfg, ctx, fields, seq=seq, statement=statement,
                             reductions=reductions)
    st = STATEMENT if statement is None else statement
    red = STATEMENT_REDUCTIONS if reductions is None else reductions
    L, d, nh, nkv, hd, dff, vocab = cfg
    r = cm.cells_per_token(cfg, ctx, B, seq=seq, head=False, gkr=red)
    gl = cm.residual_stream_cells(cfg, "tree", gkr=red)
    bb, bi, sat = key_tiles_per_chunk(nh, hd, seq, ctx, base_slope=base_slope, **st)
    key_tiles = L * (bb + bi) / seq
    rest = r["total"] - r["attn"] + gl["net"]
    return dict(key_tiles=key_tiles, rest=rest, total=key_tiles + rest,
                key_tiles_base_per_chunk=bb, key_tiles_interaction_per_chunk=bi,
                per_layer_chunk=(key_tiles + rest) * seq / L,
                saturated=sat, residual_stream_net=gl["net"], bare=r["total"],
                statement=dict(st), reductions=tuple(red),
                reductions_not_priced=r["gkr_not_priced"],
                rest_rows={k: r[k] for k in ("acc", "act", "nl", "logup", "freivalds",
                                             "range", "limbs", "attn_out", "requant",
                                             "rope", "quant", "tables", "boundary")})


def head_chunk(cfg, seq=S):
    return cm.lm_head_chunk(cfg, seq)


def exact_head_segment(cfg, ctx, seq=S, reductions=None, head_groups=None):
    """The output head segment, counted at this context.

    Its witness block IS an assumption, and that is said here in as many words:
    the 14B campaign extracts the 48 layer dumps and no head dump, so there is no
    head witness of this shape to read, and the modal block of the recorded 8B
    head segments stands in.  It is outside the registered quantity either way --
    `cells_per_token_layers` is layers only, and so is the comparison -- and it is
    counted because a reader of the document asks what the whole model costs.
    """
    L, d, nh, nkv, hd, dff, vocab = cfg
    red = STATEMENT_REDUCTIONS if reductions is None else reductions
    p = ec.segment_params("head", d=d, nh=nh, nkv=nkv, hd=hd, dff=dff, vocab=vocab,
                          s=seq, c=ctx, witness=ec.modal_assumption("head"),
                          gkr=red, head_groups=list(head_groups or ec.head_groups_for(nh)))
    got = ec.cells_of(p)
    return dict(fresh=got["fresh"], per_token=got["fresh"] / seq,
                fields=ec.modal_assumption("head"),
                fields_source="ASSUMED: the modal block of the recorded 8B output "
                              "head segments. This campaign extracts no head dump "
                              "at the registered shape, so the head's fields are "
                              "the only ones in this document that are not read "
                              "off a witness",
                preprocessed_note="the weight trace the Freivalds check reads is "
                                  "committed offline and is no committed cell of "
                                  "this segment, so the count says nothing about it")


# --------------------------------------------------------------------------
# Anchors: what the same predictor says about the two scales it WAS built from.
# This is the only honest basis for a bracket, so it is computed here rather
# than asserted.
#
# The anchor that carries the weight is the RECORDED LAYER BLOCK, one
# constants row per context: the fresh cells of the model's 32 layer segments
# of one token block over the 512 positions that block proves.  The block is
# the one whose prefix is the whole context, so both sides of the comparison
# are 32 layers at C over a block of S, and neither side carries an output head
# segment: it is the quantity `predict` states, position for position.  The
# other `cells_per_position.headline.*` rows of the same run are NOT that
# quantity -- they divide every segment of the run, output heads and shorter
# prefixes included, by every position the run proves -- and are not read here.
#
# One difference between the two sides is named rather than adjusted away: the
# measured block's layer segment 0 carries the embedding lookup and one more
# cell per attention-output accumulator, which the model prices as no layer's
# cost.  `layer_chunks.json` holds the same 32 segments, so both the size of
# that difference and the recorded figure itself are recomputed here, to the
# cell, instead of transcribed.
#
# The second anchor per context is the DERIVED REFERENCE CHUNK: one 8B layer
# chunk out of that same block, read from `--artifacts`, scaled to 32 layers.
# It is what a constants file older than the row above leaves, and it splits
# the residual into the key tile half and the rest, which the row cannot.
# --------------------------------------------------------------------------
FROZEN_8B_KEYS = {4096: "cells_per_position.headline.c4096.fresh_layers_block",
                  16384: "cells_per_position.headline.c16384.fresh_layers_block"}
FROZEN_8B_ABSENT = (
    "this constants file does not name it. A constants file older than the "
    "record of this row has only the `cells_per_position.headline.*` rows over "
    "the whole decode sequence, which are a different quantity -- every segment "
    "of the run, output heads and shorter prefixes included, over every "
    "position it proves -- and none of them stands in for it. The registration "
    "is anchored on the derived reference chunk beside this instead.")


def entry_provenance(e):
    """Where a constants entry came from, in the fields zk-results names.

    That file's entries carry `run` and `artifact`; none of them has a `source`
    field and its own check refuses one that does. An entry that names neither
    -- an assumption, or a figure read from a public source -- says so here
    rather than being given a path it does not have."""
    run, artifact = e.get("run"), e.get("artifact")
    if run and artifact:
        return f"runs/{run}/{artifact}"
    return run or artifact or "the constants file, which names no run for this entry"


def recorded_layer_block(ctx):
    """The measured 8B layer block, out of this repository's copy of its segments.

    `layer_chunks.json` holds every segment of the token block each recorded
    row divides, so the row can be recomputed here to the cell instead of
    trusted, and the embedding lookup that separates the two sides of the
    comparison can be sized without transcribing a number."""
    ch = [q for q in key_tile.chunks() if not q["head"] and q["c"] == ctx]
    if not ch:
        return None
    ident = key_tile.chunk_identity(ch)
    exact = len(ident["layer"]) == 1 and len(ident["embedding"]) == 1
    fresh, s = sum(q["fresh"] for q in ch), ch[0]["s"]
    return dict(layers=len(ch), positions=s, fresh=fresh, per_position=fresh / s,
                embedding_lookup_per_position=(ident["embedding"][0] - ident["layer"][0]) / s
                if exact else None,
                run=ch[0]["run"], artifact=ch[0]["output"],
                block=ch[0]["segment"].split("L")[0],
                note="the same 32 segments the recorded row divides, copied into "
                     "this repository; `embedding_lookup_per_position` is what "
                     "layer segment 0 builds above the other 31, which is the "
                     "embedding lookup and one more cell per attention-output "
                     "accumulator")


def anchors(cells="fitted"):
    """What the predictor says about the two scales it WAS built from.

    Under `--cells exact` the 8B side is counted on exactly the path the
    registration is made on, with each recorded segment's own witness fields
    (`recorded_8b_fields`).  That is the anchor's whole value there: the same
    function, the same fields, against the number the runs recorded.
    """
    out = {}
    for ctx in CONTEXTS:
        key = FROZEN_8B_KEYS[ctx]
        p = predict(cm.LLAMA8B, ctx, cells=cells,
                    fields=recorded_8b_fields(ctx) if cells == "exact" else None)
        e = constants.find(key)
        if e is None:
            out[f"llama8b_c{ctx}"] = dict(
                scale="Llama-3.1-8B", ctx=ctx, predicted=p["total"], measured=None,
                ratio=None, key=key, source="absent", grade="absent",
                statement=dict(STATEMENT), note=FROZEN_8B_ABSENT)
        else:
            m = e["value"]
            rec = recorded_layer_block(ctx)
            emb = rec and rec["embedding_lookup_per_position"]
            out[f"llama8b_c{ctx}"] = dict(
                scale="Llama-3.1-8B", ctx=ctx, predicted=p["total"], measured=m,
                ratio=p["total"] / m, key=key, source=entry_provenance(e),
                run=e.get("run"), artifact=e.get("artifact"),
                grade=e["grade"], card=e["card"], statement=dict(STATEMENT),
                recorded_chunks=rec,
                measured_matches_recorded_chunks=(rec is not None
                                                  and rec["fresh"] == m * rec["positions"]),
                embedding_lookup_per_position=emb,
                ratio_without_the_embedding_layer=None if not emb else p["total"] / (m - emb),
                note="both sides are the fresh cells of the model's 32 layer segments "
                     "over the 512 positions one token block proves, at this context "
                     "and at the statement this registration prices, with no output "
                     "head segment on either. The measured block is the one whose "
                     "prefix is the whole context: the earlier blocks of that sequence "
                     "read a shorter prefix and cost less, which is the comparison the "
                     "prediction does not make. Its layer segment 0 carries the "
                     "embedding lookup and one more cell per attention-output "
                     "accumulator, which the model prices as no layer's cost -- that "
                     "is `embedding_lookup_per_position`, and the ratio beside it is "
                     "the same comparison without it")
        # The second anchor: one layer chunk out of the same block, scaled to
        # the model's 32.  It splits the residual into the key tile half, which
        # is exact to the cell, and the rest, which the recorded row cannot.
        pt = _seq5(ctx)
        mref = pt["cells"]["fresh"] * cm.LLAMA8B[0] / S
        mkt = (pt["family_cells"]["key_tiles"]
               + pt["family_cells"]["interaction_by_family"]["key_tiles"]["cells"])
        pkt = cm.LLAMA8B[0] * (p["key_tiles_base_per_chunk"]
                               + p["key_tiles_interaction_per_chunk"]) / S
        mrest = mref - mkt * cm.LLAMA8B[0] / S
        out[f"llama8b_c{ctx}_registered_statement"] = dict(
            scale="Llama-3.1-8B", ctx=ctx, predicted=p["total"], measured=mref,
            ratio=p["total"] / mref, source=REF8B_SOURCE, grade="measured",
            card="n/a (a cell count is the same on either card)",
            statement=dict(STATEMENT),
            key_tiles_exact=int(round(pkt)) == int(round(mkt * cm.LLAMA8B[0] / S)),
            rest_ratio=p["rest"] / mrest,
            rest_residual_per_chunk=(p["rest"] - mrest) * S / cm.LLAMA8B[0],
            chunk=_seq5_source(ctx),
            note="one layer chunk per context, scaled to 32 layers, at exactly the "
                 "statement this registration prices. The key tile half of this "
                 "ratio is exact to the cell; every cell of the difference is in "
                 "`rest`, and the largest single term the model does not price is "
                 "`gkrlink`, the reductions' own link")
    # Qwen2.5-0.5B: B6.5's composed layer, a DIFFERENT statement -- no
    # residual-stream family, no diagonal block key tile, none of the five reductions,
    # and q/k/v/o on the pow2 dequant
    # rung while gate/up/down are on the chain.  Evaluated the same way, as a weak anchor.
    #
    # Four families that prover generation did not build are taken back off the
    # predicted side, because charging a chunk for a family it has no entry for
    # is not a residual of the model: its `cells.fresh` is exactly its base plus
    # its interaction, so it commits no boundary tree at all, and its
    # `family_cells` names no lookup-table, quantize or residual-stream family.
    #
    # It stays on the PRICED path under `--cells exact`, and the entry says so:
    # that chunk's witness is in neither repository, so there are no fields to
    # read, and a segment counted at another shape's fields is not this one.
    weak = dict(cells="fitted",
                cells_note="the priced path, whatever this document registers "
                           "with: no witness of this chunk is in either "
                           "repository, so its fields cannot be read") \
        if cells == "exact" else {}
    L, d, nh, nkv, hd, dff, vocab = cm.QWEN
    r = cm.cells_per_token(cm.QWEN, 1024, B, seq=S, head=False, diagonal_block=False)
    mixed = L * (((nh + 2 * nkv) * hd + d) * cm.HEAD_DEQUANT_PER_ACC
                 + (2 * dff + d) * cm.LAYER_DEQUANT_PER_ACC)
    not_built = r["boundary"] + r["tables"] + r["quant"] + r["rope"]
    tot = (r["total"] - r["limbs"] + mixed - not_built) * S / L
    meas = (cm.B65C or {}).get("compose", {}).get("fresh")
    if meas is None:
        out["qwen0p5b_c1024"] = dict(
            scale="Qwen2.5-0.5B", ctx=1024, predicted=tot, measured=None, ratio=None,
            source="b65/compose_host.json under --inputs DIR", grade="absent",
            note="the composed 0.5B layer is in no constants file and no key names "
                 "it; without the inputs directory this anchor is not evaluated",
            **weak)
        return out
    out["qwen0p5b_c1024"] = dict(
        scale="Qwen2.5-0.5B", ctx=1024, predicted=tot, measured=meas,
        ratio=tot / meas, source="experiments/b65/out/compose_host.json "
        "(result.point.cells.fresh)", grade="measured", card="none (host, macOS)",
        statement=dict(diagonal_block=False, kv_boundary=False, bind=True, gkr=False),
        statement_note="what `cells_per_token` prices here, not what B6.5 proved: "
                       "that prover generation published no root-equality claim, "
                       "so `bind` is charged against a chunk that has none",
        families_not_built=dict(
            boundary=r["boundary"] * S / L, tables=r["tables"] * S / L,
            quant=r["quant"] * S / L, rope=r["rope"] * S / L,
            why="this chunk's `cells.fresh` is exactly its base plus its "
                "interaction, so it commits no boundary tree, and its "
                "`family_cells` names no lookup-table, quantize or "
                "residual-stream family. Charged, they are most of the chunk."),
        note="weak anchor: no residual-stream family, no diagonal block key tile, mixed dequant rungs, "
             "and a prover generation older than the seq5 statement",
        **weak)
    return out


# --------------------------------------------------------------------------
# What one layer chunk is, and how far one layer is from the mean of them.
# Both are computed off `harness/costmodel/layer_chunks.json` -- every chunk of
# one token block of each headline run, copied into the repository -- so they
# move with the runs and never with a sentence somebody forgot to edit.
# --------------------------------------------------------------------------
def chunk_identity():
    """The whole-chunk identity, solved on the recorded chunks.

    A composed layer chunk's fresh cells are its key tile family, its boundary
    trees, its lookup tables and a rest the shape fixes:

        cells.fresh == key_tile_family(nh, hd, S, C) + boundary_cells(...)
                       + 5 * (sum of 2^log_rows over the chunk's tables) + rest

    and `rest` is one integer on every recorded non-embedding layer chunk at
    both contexts.  A single-layer run can check it on its own chunk, which is
    what protocol C leans on.
    """
    ident = key_tile.chunk_identity()
    return dict(
        form="cells.fresh == key_tile_family(nh, hd, S, C) + boundary_cells(nh, hd, d, S, C)"
             " + 5 * sum_t 2^log_rows(t) + rest",
        rest=ident["layer"][0] if len(ident["layer"]) == 1 else ident["layer"],
        rest_embedding_layer=(ident["embedding"][0] if len(ident["embedding"]) == 1
                              else ident["embedding"]),
        exact_to_the_cell=len(ident["layer"]) == 1,
        chunks=sum(1 for q in key_tile.chunks() if not q["head"] and not q["embedding"]),
        note="8B, S=512, spec b10-final, both contexts. The embedding layer's rest is "
             "higher by its embedding lookup and one more cell per attention-output "
             "accumulator. `boundary_cells` is the term no `family_cells` entry names: "
             "what `cells.fresh` has above `base + interaction`.",
        checked_by="harness.costmodel.key_tile --selftest")


def layer_spread():
    """How far one layer chunk is from the mean of a context's 32, measured.

    This is the basis the protocol widening was set from and it is recomputed
    here rather than quoted, at both contexts, for one layer and for the worst
    evenly-spaced subsample of eight over every phase offset.  The registered
    widening percentages are NOT derived from it: they are what they were.
    """
    out = {}
    for ctx in CONTEXTS:
        ch = sorted((q for q in key_tile.chunks()
                     if not q["head"] and q["c"] == ctx),
                    key=lambda q: int(q["segment"].split("L")[1]))
        if not ch:
            continue
        fresh = [q["fresh"] for q in ch]
        n = len(fresh)
        mean = sum(fresh) / n
        worst8 = max(abs(sum(fresh[o::4]) / len(fresh[o::4]) / mean - 1) for o in range(4))
        no0 = fresh[1:]
        out[str(ctx)] = dict(
            layers=n, mean_fresh_per_chunk=mean,
            single_layer_pct=[100 * (min(fresh) / mean - 1), 100 * (max(fresh) / mean - 1)],
            single_layer_pct_without_the_embedding_layer=[
                100 * (min(no0) / mean - 1), 100 * (max(no0) / mean - 1)],
            worst_even_8_subsample_pct=100 * worst8,
            widest_layer=ch[fresh.index(max(fresh))]["segment"],
            narrowest_layer=ch[fresh.index(min(fresh))]["segment"],
            run=ch[0]["run"],
            note="the widest layer is the embedding one: it carries the embedding "
                 "lookup, the largest master exponent table of the 32 and one more "
                 "cell per attention-output accumulator. Without it the spread is "
                 "the row above.")
    return out


# --------------------------------------------------------------------------
# The pre-committed bracket.  R's prize-floor / floor / outer pattern.
# --------------------------------------------------------------------------
BRACKET = dict(
    quantity="rho = measured / predicted, on layers-only fresh cells per decoded "
             "token at the registered (shape, S, C)",
    prize=[0.97, 1.03],
    pass_=[0.90, 1.10],
    marginal=[0.80, 1.25],
    rule=(
        "PRIZE  rho in [0.97, 1.03] -- the model predicts a never-fitted scale to "
        "3%. The extrapolation licence is claimed without qualification and the "
        "section 6.3 frontier table keeps its [projected] grade.\n"
        "PASS   rho in [0.90, 1.10] -- the range as registered, unchanged. What "
        "the same predictor shows at the two scales it WAS built from is not "
        "quoted here: it moves whenever the model does, and a number in this "
        "sentence went stale once already. It is computed into the `anchors` "
        "block of this document, per scale and per context, with the statement "
        "each measured side was proved at beside it. The range is NOT rebased "
        "on any of them. The licence is claimed with the measured residual "
        "quoted beside it.\n"
        "MARGINAL rho in [0.80, 0.90) or (1.10, 1.25] -- the licence is claimed "
        "only with an explicit +/-25% key tile printed on every frontier row.\n"
        "FAIL   rho < 0.80 or rho > 1.25 -- the cost model is REFUTED as an "
        "extrapolator. Section 6.3 is regraded from [projected] to "
        "[illustrative], the 'you didn't build it' mitigation falls back to "
        "publishing the model runnable, and E4 is written up as a negative "
        "result. No re-fit may be presented as if it were this prediction."),
    protocol_widening=dict(
        A_all_layers=dict(
            what="extract and prove every layer at the registered C, the seq5 "
                 "recipe at a new shape; compare layers-only cells/token",
            widen_pct=0.0),
        B_even_subset=dict(
            what=">= 8 EVENLY SPACED layers (L0, L*1/8, ... by a rule fixed "
                 "before the run), compare the mean scaled to L layers",
            widen_pct=1.0,
            basis="the percentages here are the ones registered and are NOT "
                  "re-derived; `layer_spread` in this document is what the two "
                  "headline runs' own 32 chunks per context say today, computed "
                  "rather than quoted"),
        C_single_layer=dict(
            what="one layer only -- the minimum that still tests anything",
            widen_pct_upper=9.0, widen_pct_lower=1.5,
            basis="the percentages here are the ones registered and are NOT "
                  "re-derived; `layer_spread` in this document is what the two "
                  "headline runs say today, and the run MUST report "
                  "`params.log_table` and `interaction_by_family.table.cells` "
                  "for the layer it proved. The whole chunk is the exact "
                  "identity `chunk_identity` states, which a single-layer run "
                  "can check on its own chunk."),
    ),
    also=(
        "SEPARATE and EXACT: the key tiles family. `family_cells.key_tiles` (base) and "
        "`family_cells.interaction_by_family.key_tiles.cells` must equal the registered "
        "integers TO THE CELL, as they do on every recorded layer chunk at both contexts "
        "(`harness.costmodel.key_tile --selftest`). The registered interaction is "
        "the statement WITH the five reductions -- `--diagonal-block --gkr-key-tiles "
        "--gkr-dequant --gkr-residual-stream --gkr-freivalds --gkr-attnout`, the "
        "configuration `tools/drivers/predict-14b.sh` proves every layer under and "
        "the one both headline runs used. A run WITHOUT `--gkr-key-tiles` commits the "
        "LogUp running sums instead, reports a LARGER interaction, and is compared "
        "against `key_tile_family(..., gkr=False)` at the same shape. The base is the "
        "same integer either way. Any nonzero deviation is a finding in its own right "
        "and must be reported even if rho passes -- it means the layout identities "
        "in `key_tile_cells` do not hold at a shape whose `nh` and `d` are not "
        "powers of two, which is the one thing the closed form's own docstring "
        "flags.\n"
        "SEPARATE and EXACT, the second one: the BOUNDARY TREES. "
        "`cells.fresh - cells.base - cells.interaction` must equal "
        "`2*d*S + 2*nh*hd*(2*C + S)` on every layer chunk and `d*S` on every "
        "output head chunk, as it does on all 68 recorded 8B chunks of the two "
        "headline runs (`harness.costmodel.key_tile --selftest`). No "
        "`family_cells` entry names this family, so a run that does not report "
        "`cells.base` and `cells.interaction` beside `cells.fresh` cannot be "
        "checked against it. Like the key tiles, a deviation is a finding in "
        "its own right and is reported whatever rho does; unlike a "
        "configuration miss, it does not void the run."),
    invalidation=(
        "The comparison is void, not failed, if the run does not reproduce the "
        "registered configuration: spec `b10-final` (round even, all seven "
        "matmuls on the `chain` dequant rung), S=512, diagonal block key tile ON, "
        "grid_target 64 / chan_target 128 / rw 16, k_bits 11 / q_bits 12, "
        "bounds on, and a REAL extracted witness (not `ZK_SYNTHETIC_DIMS`). "
        "A void run is re-run, never re-bracketed."),
)


def registered_statement(names=E4_SHAPES, cells="fitted", block=S):
    """What the registered key tile integers price, said in the document itself.

    Four settings and the driver flags they correspond to, so that a reader of
    `registration.json` can see which statement the integers assume without
    reading this file.  It is part of the hashed content: a registration that
    priced another statement is another registration.

    The flags are the same set for every selection -- the five reductions and the
    diagonal block -- and the driver that proves them is not, so the driver is
    read from the selection's own record rather than named once here.

    `block` is the run's own `--block`, and it is in the statement exactly when
    it is not the 512 every recorded run proves: the registered quantity is per
    decoded token at a (shape, block, context), a segment over a shorter block
    is a different segment, and a document that did not say which one it counted
    would be comparable with a run it does not describe.  Said only where it
    moves, so that every registration made before this flag existed is the
    document it was, key for key.
    """
    return dict(
        STATEMENT,
        **({} if block == S else dict(block=block)),
        driver=(counted_record(names[0])["driver"] if cells == "exact"
                else REGISTERED[tuple(names)]["driver"]),
        driver_flags="--diagonal-block --gkr-key-tiles --gkr-dequant "
                     "--gkr-residual-stream --gkr-freivalds --gkr-attnout",
        flags=dict(STATEMENT_FLAGS),
        key_tiles_half="priced at exactly this statement: "
                       "`key_tile_family(nh, hd, S, C, **STATEMENT)`",
        rest_half=dict(STATEMENT_NOT_PRICED),
    )


def statement(pt):
    """The key tile configuration the measured chunk was proved under, off its own
    `params`: whether the prefix is tiled into key blocks, whether the chunk binds
    its K/Q/V to the boundary roots, and whether the five reductions were on.

    Read rather than assumed, because the closed form prices a different family
    under each and the chunk is the only thing that says which it is.
    """
    return dict(grid_target=pt["params"]["grid_target"],
                chan_target=pt["params"]["chan_target"], rw=pt["params"]["rw"],
                diagonal_block=bool(pt["params"]["diagonal_block"]),
                kv_boundary=bool(pt["params"].get("kv_boundary")),
                bind=pt["params"].get("rope") is not None,
                gkr=bool(pt["params"].get("gkr_key_tiles")))


def selfcheck():
    """The predictor must reproduce the 8B chunk's key tiles family TO THE CELL.

    This is what licenses registering the key tiles half with no bracket at all: the
    same code path, at the shape it was derived from, against the artifact.
    """
    out = []
    for ctx in CONTEXTS:
        pt = _seq5(ctx)
        nh, hd = pt["params"]["nh"], pt["params"]["hd"]
        assert (nh, hd) == (cm.LLAMA8B[2], cm.LLAMA8B[4]), (nh, hd)
        st = statement(pt)
        bb, bi, sat = key_tiles_per_chunk(nh, hd, S, ctx, **st)
        mb = pt["family_cells"]["key_tiles"]
        mi = pt["family_cells"]["interaction_by_family"]["key_tiles"]["cells"]
        assert int(bb) == mb, (ctx, bb, mb)
        assert int(round(bi)) == mi, (ctx, bi, mi)
        # and the local reimplementation must equal cost_model's own key_tile_fresh
        bf, _ = cm.key_tile_fresh(nh, hd, S, ctx, **st)
        assert abs(bf - (bb + bi)) < 1e-6, (ctx, bf, bb + bi)
        out.append(dict(ctx=ctx, key_tiles_base=mb, key_tiles_interaction=mi, exact=True,
                        saturated=sat, statement=st))
    return out


def _closed_form_key_tiles(cfg, ctx, seq=S):
    """The key tile family's closed form at this shape, beside the counted pair.

    They are the same integers wherever the three grids tile exactly and they are
    not where the grids pad, which is what a head count that is not a power of
    two does.  Both are in the document so the difference is visible rather than
    inferred: the closed form is what `compare.py` used to check a chunk
    against, and the count is what the layout builds.
    """
    _, _, nh, _, hd, _, _ = cfg
    b, i, sat = key_tiles_per_chunk(nh, hd, seq, ctx, **STATEMENT)
    return dict(base=int(b), interaction=int(round(i)), saturated=sat)


def canonical(block):
    return json.dumps(block, sort_keys=True, separators=(",", ":")).encode()


def build(names=E4_SHAPES, cells="fitted", fields=None, block=S):
    """The registration's `shapes` block, one entry per shape in `names`.

    The selection is an argument and not `SHAPES` itself: this document is a
    registration, and a row added to the registry for another campaign must not
    change the one E4 registered.

    Under `--cells exact` the entry carries one row per layer segment -- the
    fields it was counted at, its count and its bound -- and the fields file is
    the shape's own, so only a shape whose dumps it names can be counted.  A
    shape it does not name is priced by nothing rather than assumed.

    `block` is how many decoded tokens one segment covers, 512 everywhere it is
    not said otherwise.  It is an argument because a shape whose segment does
    not fit the card at 512 is measured at a shorter block, and then the
    registration has to be counted at the block the run will prove: the cells of
    a segment are not linear in it -- the key tile family's diagonal block goes
    with the square of it and the prefix's tiles with the ratio -- so a
    registration at another block is a registration of another segment."""
    reg = {}
    ref8b_cells = ref8b()
    for name in names:
        meta = SHAPES[name]
        cfg = meta["cfg"]
        L, d, nh, nkv, hd, dff, vocab = cfg
        n = cm.n_params(cfg) + (0 if meta["tied"] else vocab * d)
        entry = dict(hf=meta["hf"], role=meta["role"], why=meta["why"],
                     cfg=list(cfg), tied_embeddings=meta["tied"],
                     params_total=n, params_b=round(n / 1e9, 3),
                     weights_gib_bf16=round(n * 2 / 2 ** 30, 2), contexts={})
        for ctx in CONTEXTS:
            if meta["role"] == "fallback-large" and ctx == 16384:
                continue
            if cells == "exact" and fields_context(fields) != ctx:
                # not registered rather than counted at the other context's
                # fields: the divisor range and the table both move with the
                # prefix, and a number under a heading that assumed they did not
                # is worse than no number
                entry.setdefault("contexts_not_registered", {})[str(ctx)] = (
                    f"the fields these segments are counted at are stated for "
                    f"C={fields_context(fields):,}, and this document registers only "
                    f"the context they are stated for: the attention output's "
                    f"divisor range and the master exponent table both grow with "
                    f"the prefix, and no synthetic segment at another context has "
                    f"been built for anyone to state them from"
                    if fields.get("synthetic") else
                    f"the fields these segments are counted at were read off dumps "
                    f"at C={fields_context(fields):,}, and this document registers "
                    f"only the context its witnesses were extracted at")
                continue
            if ctx % block:
                raise SystemExit(
                    f"--block {block}: C={ctx:,} is not a multiple of it, so the "
                    f"prefix is not tiled into blocks of it and the segment this "
                    f"would count is one no run proves")
            p = predict(cfg, ctx, seq=block, cells=cells, fields=fields)
            # caveat 1: what the refuted `p * S^2` reading's base slope would do.
            # A counted key tile family has no slope to move: the grids are built
            # and their padding is in the number, so the row below is the witness
            # bound instead, which is the assumption the counted path still makes.
            hi = (predict(cfg, ctx, seq=block, base_slope=72.083 / 32.0)
                  if cells != "exact" else None)
            ref8 = ref8b_cells[ctx]
            entry["contexts"][str(ctx)] = dict(
                **({} if block == S else dict(block=block)),
                cells_per_token_layers=p["total"],
                cells_per_token_layers_key_tiles=p["key_tiles"],
                cells_per_token_layers_rest=p["rest"],
                per_layer_chunk_fresh=p["per_layer_chunk"],
                key_tiles_family_base_per_chunk=int(p["key_tiles_base_per_chunk"]),
                key_tiles_family_interaction_per_chunk=int(round(p["key_tiles_interaction_per_chunk"])),
                key_tiles_family_fresh_per_chunk=int(round(p["key_tiles_base_per_chunk"]
                                                       + p["key_tiles_interaction_per_chunk"])),
                key_tile_layout_saturated=p["saturated"],
                chunk_over_8b_l16_chunk=p["per_layer_chunk"] / ref8,
                **({} if cells == "exact" else dict(
                    sensitivity_base_slope_2p2526=dict(
                        cells_per_token_layers=hi["total"],
                        shift_pct=100.0 * (hi["total"] / p["total"] - 1.0)))),
                **({} if cells != "exact" else dict(
                    cells="exact",
                    cells_per_token_layers_bound=p["bound_total"],
                    bound_shift_pct=100.0 * (p["bound_total"] / p["total"] - 1.0),
                    layers_fresh_sum=p["layers_fresh_sum"],
                    components_per_token=p["components"],
                    key_tiles_closed_form_per_chunk=_closed_form_key_tiles(cfg, ctx,
                                                                          seq=block),
                    head_segment=exact_head_segment(
                        cfg, ctx, seq=block,
                        head_groups=(fields.get("knobs") or {}).get("head_groups")),
                    layers=p["layers"])),
            )
        h = head_chunk(cfg)
        entry["head_chunk"] = dict(fresh=h["fresh"], per_token=h["per_token"],
                                   preprocessed=h["preprocessed"])
        reg[name] = entry
    return reg


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def provenance(names=E4_SHAPES, cells="fitted", fields=None, fields_path=None,
               block=S):
    rec = (counted_standing(names[0], fields) if cells == "exact"
           else REGISTERED[tuple(names)])
    return dict(
        # the same digests the frozen inputs carry, so the two cannot drift
        inputs_sha256=dict(frozen_inputs(names, cells, fields, fields_path,
                                         block)["code_sha256"],
                           **{"constants.json": _sha(constants.path())}),
        registers=list(names),
        registered_utc_date=rec["utc_date"],
        registered_by=rec["by"],
        no_measurement_exists=rec["no_measurement_exists"],
        predictor=dict(
            engine="experiments/b2/cost_model.py",
            entry_points=["cells_per_token(cfg, ctx, batch, seq, head=False, gkr=...)",
                          "residual_stream_cells(cfg, 'tree', gkr=...)",
                          "key_tile_family", "key_tile_cells", "key_tile_fresh",
                          "boundary_cells", "lookup_table_cells", "lm_head_chunk"],
            reductions_priced=list(STATEMENT_REDUCTIONS),
            not_the_engine="experiments/budget.py is an early one-shape proxy "
                           "(eval-plan.md section 3) and is not used here",
            attention_term="the KEY_TILE MODEL (alpha=None). The legacy single-slope "
                           "alpha=4.00 path is NOT used -- section 7.5(e)'s ruling "
                           "is that the paper uses the composed-chunk value.",
            lookup_batch=B, seq_per_chunk=block,
            # said only where it is true, so that a priced registration is the
            # document it was before this flag existed, key for key
            **({} if cells != "exact" else dict(
                cells=cells,
                counted_by="harness/costmodel/exact_cells.py, which rebuilds the "
                           "prover's trace layout component by component and "
                           "reproduces every recorded chunk in zk-results to the "
                           "cell",
                counted_entry_points=["exact_cells.segment_params", "exact_cells",
                                      "exact_cells.bound_cells",
                                      "witness_fields.fields_of"],
                fields_read_from="the witness dumps themselves, before any proof "
                                 "of them exists")),
        ),
    )


def frozen_inputs(names=E4_SHAPES, cells_mode="fitted", fields=None,
                  fields_path=None, block=S):
    """What this prediction actually read, frozen with it.

    It used to be a list of constants keys, and most of them were in no
    constants file: three of them named the key tile model's fitted
    coefficients, which this predictor does not read at all -- the key tile
    half is a closed form, so its coefficients are an output of the layout and
    not an input to the prediction.  What the prediction does read is three
    things, and all three are here:

      * every cell cost the model charged, with the source each came from --
        `cost_model.PROVENANCE`, which #153 made the model's own record of
        constants-first reading with an archive fallback.  A cost the model
        could not price is in it too, marked absent;
      * the statement priced, which is what makes the integers mean anything;
      * the code that computed it, by digest, and the two measured chunks it
        was anchored on, by run and segment.

    A key that IS in the constants file and was read still appears, under
    `constants_entries`, with its grade -- that is the graded provenance the
    freeze was for -- and one that is not is not silently dropped: the
    `cell_costs` entry for it says `absent` and names the key.
    """
    cells = {n: dict(e) for n, e in sorted(cm.PROVENANCE.items())}
    # `boundary` is what an older constants file called the timing scope; a
    # cell count has neither, so both are optional here and neither defaults.
    def entry(k):
        e = get(k)
        return {f: e[f] for f in ("value", "grade", "card", "context", "unit",
                                  "timing_scope", "boundary", "run", "artifact")
                if f in e}

    from_constants = {e["key"]: entry(e["key"])
                      for e in cells.values() if e["source"] == "constants" and e["key"]}
    unpriced = sorted(n for n, e in cells.items() if e["source"] == "absent")
    code = {
        "harness/costmodel/cost_model.py": _sha(cm.__file__),
        "harness/costmodel/key_tile.py": _sha(key_tile.__file__),
        "harness/costmodel/layer_chunks.json": _sha(str(key_tile.CHUNKS)),
        "harness/costmodel/composed_rungs.json": _sha(str(cm.COMPOSED_RUNGS)),
        "harness/eval/predict/predict.py": _sha(os.path.join(HERE, "predict.py")),
    }
    if cells_mode == "exact":
        # the two files that do the counting, and the one that reads the fields
        # off the dumps, by digest -- the counted path reads no constant on the
        # committed side, so these are what the numbers are a function of
        code["harness/costmodel/exact_cells.py"] = _sha(ec.__file__)
        code["harness/costmodel/exact_chunks.json"] = _sha(str(ec.CHUNKS))
        code["harness/eval/predict/witness_fields.py"] = _sha(wf.__file__)
    out = dict(
        cell_costs=cells,
        constants_entries=from_constants,
        constants_file_sha256=_sha(constants.path()),
        n_from_constants=len(from_constants), n_inputs=len(cells),
        not_priced=unpriced,
        statement=registered_statement(names, cells_mode, block),
        reference_chunks={str(c): _seq5_source(c) for c in CONTEXTS},
        code_sha256=code)
    if cells_mode == "exact":
        out["cells"] = "exact"
        out["witness_fields"] = witness_fields_record(fields, fields_path)
        out["cell_costs_charged_on_the_counted_side"] = []
        out["cell_costs_note"] = (
            "the counted path charges NO per-unit cell cost on the committed "
            "side: every cell of every layer segment is counted from the trace "
            "layout. `cell_costs` above is the model's own record of what it "
            "COULD price and what each of those would have come from, kept so a "
            "reader can see the two paths side by side and so the driver's gate "
            "still reads one record; the only figure in this document that still "
            "comes from a per-unit constant is the output head's preprocessed "
            "weight trace, which is no committed cell of any segment")
        out["constants_keys_read"] = sorted(from_constants)
    return out


def witness_fields_record(fields, path):
    """What the counted registration read, frozen with it.

    The fields file by digest, the knobs it was read under, and every layer's
    block in full: the document has to carry the numbers themselves, or the
    digest names a file and the reader still cannot check the count.
    """
    if not fields:
        raise SystemExit("--cells exact needs --witness-fields FILE")
    rows = fields["layers"]
    return dict(
        file=os.path.basename(path) if path else None,
        file_sha256=wf.sha256(path) if path else None,
        schema=fields.get("schema"), knobs=fields.get("knobs"),
        # The layers the file does not carry, carried here too: a reader of the
        # registration must not have to open the fields file to learn that the
        # count is over fewer segments than the model has layers.
        absent=sorted(fields.get("absent") or ()),
        read_by=fields.get("read_by"),
        # A stated block says where each of its fields came from; a block read
        # off dumps does not, and carries no such key rather than a null one.
        **({} if not fields.get("stated_from") else
           dict(synthetic=True, stated_from=fields["stated_from"])),
        summary=fields.get("summary"),
        layers={k: dict(wf.witness_block(r), ao_acc_mag=r["ao_acc_mag"],
                        ao_acc_prod=r.get("ao_acc_prod", False),
                        deq_variants=dict(r["deq_variants"]),
                        dump_sha256=r.get("dump_sha256"))
                for k, r in sorted(rows.items())},
        note="every field here is a property of the dump and exists before any "
             "proof of it does. The rule for each one, and the line of the prover "
             "it comes from, is in `harness/eval/predict/witness_fields.py`")


# The card rate the secondary tok/s column would need, and the key that would
# hold it.  Neither is part of the prediction: a cell count is the same on
# either card, and the rate is a card measurement the registration is not
# making.  zk-results holds `rate.headline.*` for the H100 and no H200 rate at
# all, so on that constants file this column is not printed rather than
# computed from something else.
RATE_KEYS = {"4096": "rate.h200.c4096.commit_layers32",
             "16384": "rate.h200.c16384.commit_layers32"}


def implied_tok_s(reg):
    """Secondary, and conditional on a rate that is NOT part of the prediction."""
    out = {}
    for name, e in reg.items():
        row = {}
        for ctx, key in RATE_KEYS.items():
            if ctx not in e["contexts"]:
                continue
            r = constants.find(key)
            if r is None:
                row[ctx] = dict(tok_s_commit_boundary=None, rate_key=key, rate=None,
                                note="this constants file does not name the H200 "
                                     "commit rate, so no tok/s is implied here")
                continue
            rate = r["value"] * 1e9
            row[ctx] = dict(tok_s_commit_boundary=rate / e["contexts"][ctx]["cells_per_token_layers"],
                            rate_key=key, rate=r["value"], card=r["card"],
                            boundary=r["boundary"])
        out[name] = row
    return out


def integrator_notes(reg):
    """Two things E4 found on the way past, for the coordinator -- not predictions.

    (1) section 7.5(e): the frontier rows really are still on the 4.00.
    (2) the card footprint the run-book has to plan around.
    """
    g = cm.GEMMA
    rows = {}
    for ctx, key in ((4096, "cells_per_token.27b.c4096.model"),
                     (16384, "cells_per_token.27b.c16384.model")):
        alpha4 = cm.cells_per_token(g, ctx, B, seq=S, alpha=4.0)["total"]
        kt = cm.cells_per_token(g, ctx, B, seq=S, gkr=STATEMENT_REDUCTIONS)["total"]
        e = constants.find(key)
        if e is None:
            rows[str(ctx)] = dict(frozen=None, key=key, rerun_alpha_4p00=alpha4,
                                  rerun_key_tile_model=kt,
                                  note="this constants file does not name the frozen "
                                       "27B row, so there is nothing to compare the "
                                       "re-run against")
            continue
        rows[str(ctx)] = dict(frozen=e["value"], rerun_alpha_4p00=alpha4,
                              rerun_key_tile_model=kt,
                              key_tile_over_frozen=kt / e["value"],
                              understated_pct=100.0 * (kt / e["value"] - 1.0))
    mem = {}
    for name, e in reg.items():
        mem[name] = {ctx: dict(chunk_over_8b_l16=c["chunk_over_8b_l16_chunk"])
                     for ctx, c in e["contexts"].items()}
    return dict(
        frontier_rows_still_on_4p00=dict(
            finding="Every `cells_per_token.27b.*` row in paper/constants.json "
                    "reproduces EXACTLY when the cost model is re-run with the "
                    "legacy alpha=4.00 path, and not otherwise. Re-running the "
                    "same shape on the key tile model -- the composed-chunk value "
                    "W0's freeze ruled the paper must use (eval-plan.md 7.5(e)) "
                    "-- raises them as below. E4 reports this and does not patch "
                    "it: `paper/constants.json` is owned elsewhere this session.",
            grades_still_wrong=[
                dict(key=k, grade=(constants.find(k) or {}).get("grade", "absent"))
                for k in ("attention_grid.b5_standalone_alpha",
                          "model_constants.attention_cells_per_head_key_token")
            ],
            rows=rows),
        card_footprint=dict(
            basis="seq5's single 8B layer chunk peaked 76.8 GiB on the card at "
                  "C=16384 (eval-plan.md section 8). E4's chunks scale roughly "
                  "with fresh cells; the H200 has 141 GB and the C=16384 arm must "
                  "run SERIAL, never pipelined two-ahead.",
            ratios=mem),
    )


def out_name(names):
    """The file `--json` writes the document to.

    E4's three shapes keep `registration.json`, which is the name
    `harness/eval/predict/compare.py` reads and the one
    `tools/drivers/register-14b.sh` in zk-results copies a recorded registration
    back to.  Any other selection writes BESIDE it and never over it: a
    registration is the document that must not move once it is made, and two of
    them in one checkout must not be able to overwrite each other."""
    if tuple(names) == E4_SHAPES:
        return "registration.json"
    slug = "-".join(re.sub(r"[^a-z0-9]+", "-", n.lower()).strip("-") for n in names)
    return f"registration-{slug}.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="rewrite the registration file")
    ap.add_argument("--shape", action="append", metavar="NAME",
                    help="the shape to register, repeatable; default: the three "
                         f"E4 registered ({', '.join(E4_SHAPES)}). Known: "
                         f"{', '.join(SHAPES)}")
    ap.add_argument("--cells", choices=CELLS, default="fitted",
                    help="`fitted`: the committed cells are priced from the "
                         "per-unit constants a measured 8B segment was read off, "
                         "which is what the three recorded registrations were made "
                         "on and is the default. `exact`: they are COUNTED from the "
                         "prover's trace layout, segment by segment, at the witness "
                         "fields --witness-fields names")
    ap.add_argument("--witness-fields", metavar="FILE",
                    help="the per-layer witness-calibrated fields, read off the "
                         "dumps by `harness.eval.predict.witness_fields`. Required "
                         "by --cells exact and refused without it")
    ap.add_argument("--block", type=int, default=S, choices=(16, 32, 64, 128, 256, 512),
                    help=f"decoded tokens per segment ({S} by default, which is what "
                         f"every recorded run proves and what the three recorded "
                         f"registrations were made at). A shape whose segment does "
                         f"not fit the card at {S} is measured at a shorter block, "
                         f"and then this is the block the run will prove: the "
                         f"registration is counted at it and says so")
    constants.add_argument(ap)
    ap.add_argument("--inputs", metavar="DIR", help="cost-model inputs (read at import; or set ZK_COSTMODEL_INPUTS)")
    inputs.add_arguments(ap)
    a = ap.parse_args()
    constants.load(a.constants)

    fields = fields_path = None
    if a.cells == "exact":
        if not a.witness_fields:
            raise SystemExit(
                "--cells exact needs --witness-fields FILE: every layer segment is "
                "counted at the fields its own dump carries, and this document "
                "assumes none of them. Write the file first:\n"
                "  uv run python -m harness.eval.predict.witness_fields "
                "--dir <witness base> --layers <L> --out fields.json")
        fields_path = a.witness_fields
        fields = wf.load(fields_path)
    elif a.witness_fields:
        raise SystemExit("--witness-fields is read only by --cells exact: the priced "
                         "path charges per-unit constants and reads no witness")

    names = tuple(a.shape) if a.shape else E4_SHAPES
    unknown = [n for n in names if n not in SHAPES]
    if unknown:
        raise SystemExit(f"--shape {', '.join(unknown)}: this document holds no such "
                         f"shape. Known: {', '.join(SHAPES)}")
    if a.cells == "exact" and len(names) != 1:
        raise SystemExit(
            "--cells exact registers ONE shape: the witness fields it counts with "
            "are one model's dumps, and a document that counted two shapes from "
            "one file would be counting one of them at the other's witness.\n"
            "  pass --shape <name> once. Known: " + ", ".join(SHAPES))
    if a.cells == "exact":
        # the record has to exist before anything is counted, so the refusal
        # comes with nothing read rather than after the work
        counted_record(names[0])
    elif names not in REGISTERED:
        raise SystemExit(
            f"--shape {', '.join(names)}: this document carries no registration record "
            "for that selection, and a prediction registered under another selection's "
            "date and standing is not this one's.\n"
            "  the selections it holds: "
            + "; ".join(", ".join(k) for k in REGISTERED)
            + "\n  add a row to REGISTERED naming the date, who registers it and what is "
              "measured at those shapes, and register that.")

    chk = selfcheck()
    reg = build(names, cells=a.cells, fields=fields, block=a.block)
    anc = anchors(cells=a.cells)
    # The falsifiable content, hashed ALONE.  This must not move when an
    # unrelated constant is edited, or the hash cannot answer the only question
    # a later reader asks of it: is this the prediction that was registered?
    # The acceptance range is the one E4 registered and is not rebased here.  What
    # a counted registration adds to it is one more way for a run to be VOID: the
    # fields were read off named dumps, so a run that proves other dumps is not
    # the run this was registered against.
    br = BRACKET if a.cells != "exact" else dict(
        BRACKET,
        counted_invalidation=(
            # a stated block names no dump, so what would void the run is the
            # other half of the same rule: a segment built at other fields
            "Also void, and for the same reason the clause above is: a run whose "
            "segments were not built at the fields this registration was counted "
            "at. `frozen_inputs.witness_fields.layers[<k>]` states each one and "
            "the run's own `params` block reports what it proved; the two have to "
            "agree, and where they do not the comparison is against a segment "
            "this document does not describe. A segment built at other fields is "
            "reported, never adjusted for."
            if fields.get("synthetic") else
            "Also void, and for the same reason the clause above is: a run whose "
            "layer segments were not proved from the dumps this registration was "
            "counted from. `frozen_inputs.witness_fields.layers[<k>].dump_sha256` "
            "names each one, the run's own manifest names what it proved, and the "
            "two have to be the same 48 dumps. A re-extraction at the same spec is "
            "a different witness and a different registration."))
    content = dict(schema="e4-registration/1",
                   statement=registered_statement(names, a.cells, a.block),
                   shapes=reg, bracket=br,
                   anchors=anc, key_tiles_selfcheck_8b=chk,
                   chunk_identity=chunk_identity(), layer_spread=layer_spread(),
                   frozen_inputs=frozen_inputs(names, a.cells, fields, fields_path,
                                               a.block))
    content_digest = hashlib.sha256(canonical(content)).hexdigest()
    whole = dict(content, provenance=provenance(names, a.cells, fields, fields_path,
                                                a.block))
    digest = hashlib.sha256(canonical(whole)).hexdigest()
    doc = dict(whole, prediction_sha256=content_digest,
               prediction_block_sha256=digest,
               implied_tok_s_secondary=implied_tok_s(reg),
               integrator_notes=integrator_notes(reg),
               reference_8b_chunk=dict(cells=ref8b(), source=REF8B_SOURCE))

    print("=" * 78)
    print("E4 -- unfitted-scale prediction, registered before measurement")
    print("=" * 78)
    print()
    st = registered_statement(names, a.cells, a.block)
    print(f"the cells: {a.cells}" + ("" if a.cells != "exact" else
          f" -- counted from the trace layout, at the witness fields of "
          f"{a.witness_fields} (sha256 {wf.sha256(fields_path)[:12]}...)"))
    if a.block != S:
        print(f"the block: {a.block} decoded tokens a segment, not the {S} every "
              f"recorded run proves -- every integer below is at that block")
    print("the statement priced:", " ".join(f"{k}={STATEMENT[k]}" for k in sorted(STATEMENT)))
    print(f"  the run it is registered against: {st['driver_flags']}")
    print(f"  not priced: {STATEMENT_NOT_PRICED['flags']} -- {STATEMENT_NOT_PRICED['effect']}")
    print()
    print("key tiles self-check against the 8B artifacts (must be exact to the cell):")
    for c in chk:
        print(f"  C={c['ctx']:<6} base {c['key_tiles_base']:>13,}  interaction "
              f"{c['key_tiles_interaction']:>13,}  exact={c['exact']}  saturated={c['saturated']}")
    print()
    print("the predictor's residual at the two scales it WAS built from:")
    for k, v in anc.items():
        if v["measured"] is None:
            print(f"  {k:<34} predicted {v['predicted']:>15,.0f}  "
                  f"measured    not in this constants file ({v.get('key') or v['source']})")
            continue
        print(f"  {k:<34} predicted {v['predicted']:>15,.0f}  "
              f"measured {v['measured']:>15,.0f}  ratio {v['ratio']:.4f}  "
              f"({100*(v['ratio']-1):+.2f}%)")
        print(f"  {'':<34} measured side, {v['grade']}: {v['source']}")
        if v.get("embedding_lookup_per_position"):
            print(f"  {'':<34} the embedding lookup in it, which the model prices as "
                  f"no layer's cost: {v['embedding_lookup_per_position']:,.0f} a position, "
                  f"ratio without it {v['ratio_without_the_embedding_layer']:.4f}"
                  f"   recomputed from this repository's copy of the 32 segments: "
                  f"{v['measured_matches_recorded_chunks']}")
    print()
    ci = chunk_identity()
    print(f"one layer chunk, to the cell on {ci['chunks']} recorded chunks at both contexts:")
    print(f"  {ci['form']}")
    print(f"  rest = {ci['rest']:,}   (the embedding layer: {ci['rest_embedding_layer']:,})")
    ls = layer_spread()
    print("how far one layer chunk is from the mean of its 32, measured:")
    for ctx, v in ls.items():
        lo, hi = v["single_layer_pct"]
        nlo, nhi = v["single_layer_pct_without_the_embedding_layer"]
        print(f"  C={ctx:<6} one layer {lo:+.2f}% .. {hi:+.2f}% "
              f"(without the embedding layer {nlo:+.2f}% .. {nhi:+.2f}%), "
              f"worst even 8 of 32 {v['worst_even_8_subsample_pct']:.2f}%")
    w = BRACKET["protocol_widening"]
    print(f"  registered widening, unchanged: protocol B {w['B_even_subset']['widen_pct']}%, "
          f"protocol C -{w['C_single_layer']['widen_pct_lower']}% / "
          f"+{w['C_single_layer']['widen_pct_upper']}%")
    print()
    for name, e in reg.items():
        print(f"{name}  [{e['role']}]  L={e['cfg'][0]} d={e['cfg'][1]} nh={e['cfg'][2]} "
              f"nkv={e['cfg'][3]} hd={e['cfg'][4]} dff={e['cfg'][5]} vocab={e['cfg'][6]}")
        for ctx, c in e["contexts"].items():
            print(f"    C={ctx:<6} cells/token (layers) {c['cells_per_token_layers']:>15,.0f}"
                  f"   = key tiles {c['cells_per_token_layers_key_tiles']:>14,.0f}"
                  f" + rest {c['cells_per_token_layers_rest']:>13,.0f}")
            print(f"    {'':<8} one layer chunk, fresh {c['per_layer_chunk_fresh']:>13,.0f}"
                  f"   key tiles family/chunk: base {c['key_tiles_family_base_per_chunk']:,}"
                  f"  interaction {c['key_tiles_family_interaction_per_chunk']:,}")
            if a.cells == "exact":
                lt = sorted({r["fields"]["log_table"] for r in c["layers"]})
                print(f"    {'':<8} counted from the layout at each layer's own "
                      f"witness fields: log_table {lt[0]}..{lt[-1]}, "
                      f"the bound {c['bound_shift_pct']:+.2f}%"
                      f"   chunk vs 8B L16 x{c['chunk_over_8b_l16_chunk']:.3f}")
                cf = c["key_tiles_closed_form_per_chunk"]
                same = (cf["base"] == c["key_tiles_family_base_per_chunk"]
                        and cf["interaction"] == c["key_tiles_family_interaction_per_chunk"])
                print(f"    {'':<8} key tile family, counted against the closed form: "
                      f"{'the same integers' if same else 'they differ'} "
                      f"({cf['base']:,} / {cf['interaction']:,} closed form)")
                lo = min(c["layers"], key=lambda r: r["fresh"])
                hi_ = max(c["layers"], key=lambda r: r["fresh"])
                print(f"    {'':<8} the {len(c['layers'])} segments: {lo['fresh']:,} (L{lo['layer']}) "
                      f"to {hi_['fresh']:,} (L{hi_['layer']}), "
                      f"{100 * (hi_['fresh'] / lo['fresh'] - 1):.1f}% apart")
                continue
            print(f"    {'':<8} caveat-1 sensitivity (base slope 2.0 -> 2.2526): "
                  f"{c['sensitivity_base_slope_2p2526']['shift_pct']:+.2f}%"
                  f"   saturated={c['key_tile_layout_saturated']}"
                  f"   chunk vs 8B L16 x{c['chunk_over_8b_l16_chunk']:.3f}")
        print()
    _n = integrator_notes(reg)["frontier_rows_still_on_4p00"]["rows"]
    print("section 7.5(e) check -- the frozen 27B frontier rows:")
    for ctx, r in _n.items():
        if r["frozen"] is None:
            print(f"  C={ctx:<6} not in this constants file ({r['key']}); the re-runs are "
                  f"alpha 4.00 {r['rerun_alpha_4p00']:,.0f} and key tile model "
                  f"{r['rerun_key_tile_model']:,.0f}")
            continue
        print(f"  C={ctx:<6} frozen {r['frozen']:>13,}  = alpha 4.00 rerun "
              f"{r['rerun_alpha_4p00']:>13,.0f}   key tile model {r['rerun_key_tile_model']:>13,.0f}"
              f"  ({r['understated_pct']:+.2f}% understated)")
    print()
    print(f"prediction sha256 (content only): {content_digest}")
    print(f"prediction block sha256 (incl. provenance): {digest}")
    if a.json:
        os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
        p = os.path.join(HERE, "out", out_name(names))
        with open(p, "w") as f:
            json.dump(doc, f, indent=1, sort_keys=True)
            f.write("\n")
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
