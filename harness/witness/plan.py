#!/usr/bin/env python3
"""Write the segment loop's plan file for `zk-prover compose-chunks --plan`.

One line per chunk, in the order the sequence proves them: the 32 layers and
then the head. A line carries only what that chunk decides for itself -- its
witness, its master table, and for the head the argmax and its row count.
Everything the whole sequence shares (the round, the key tile sizes, the pins, the
tight table) stays on the command line, because that is where the loop's
shared arguments live and repeating them per line is how a driver drifts.

The lines are the same arguments `n_seq.sh`'s `seq` step passes to 33 separate
`compose` processes, which is what makes the loop's digests comparable to
theirs chunk by chunk.

**Stream DD (E11): the block axis.**  A full prompt+output proof is `B` token
blocks of 32 layer chunks, plus one `lm_head` chunk, and the writer learns that with
one extra loop and no new argument *kind*:

    python -m harness.witness.plan --dir D --blocks 9 --head D/witness_B8_head --out plan.txt

reads `D/shapes_B<b>.json` and writes `--witness D/witness_B<b>_L<k>
--ser-tag B<b>L<k>`.  What a prompt block's line carries that a decode line
does not is **nothing new** -- the prefix length is `Params.c`, which
`params_of` reads off the witness's own manifest (`cli.rs`'s `params_of`,
`c: b0.shape.c`), so a block is a directory and a table size and that is all.
No `Params` field is added, which matters because `boundary.rs:253-257`'s
`shape_ok` compares `params.s`, `params.residual_stream`, the blowup and the two boundary
trees' sizes and would reject every cross-block link if a block index entered
the statement.

Two block-axis facts the writer enforces rather than leaves to the runner:

* **Block 0 has no prefix.**  `c = 0` makes `Grid::new(groups = c * insts, ..)`
  divide by zero (`layout.rs:117`), so a diagonal-block-only chunk does not compose
  today -- that configuration is stream CC's.  Block 0 is therefore skipped
  unless `--with-block-0` is passed, and the plan says so in a comment.
* **The head goes on the last block only.**  A prompt block emits no token.
  `--head-every-block` is there for the variant that proves one per block, and
  it is off, because nothing downstream asks for it yet.

`--emit-out` writes the *emit* set: one representative line per distinct
constraint-kernel shape rather than one per chunk.  `emitset` runs one
`cudaemit` per plan line and the store dedups by fingerprint afterwards, so at
257 lines it launches 257 processes to install ~40 shapes.  Within one model
and one set of shared pins the only shape-bearing things that vary down a plan
are the prefix length and the master table size, so `(c, table_bits)` -- plus
the head's two extra arguments -- is the whole distinct set.  A head line is
keyed by its prefix the way a layer line is, and its prefix is the one its own
dump declares (issue #131): with `--head-per-block` the head chunks of two
blocks read two dumps of two prefixes and are two kernel shapes, and a key that
gave a head line no prefix at all listed the first of them and left the second
block's chunk to prove against kernels nobody generated.

**Stream EE (E11): the flags a line carries.**  `--layer-flags` goes on every
layer line, `--l0-flags` on the layer-0 lines after it, `--head-flags` on the
head line, and `--l0-dir` points the layer-0 lines at another dump set (a
layer-0 re-extract with `--embed`).  The E11 plan at C=4096 is

    python -m harness.witness.plan --dir D --blocks 9 --with-block-0 --head H \
        --layer-flags='--round even --diagonal-block' --head-flags='--ao-rem-bits 23'

(the `=` form, because the value starts with a dash), and a line's flags are
part of its emit key, so the emit set grows by the lines whose flags make a
new kernel shape and by nothing else.

**Issue #118: the pin every line is proved at.**  `--ao-rem-bits` is one number
for the whole loop, taken over every row of every block, and a layer whose fold
is wide can be pushed past the quotient cap by it -- `q_bits + rem_bits`, 30 in
the plain form and 31 in the `denp2` form (`Plan::new`) -- although its own
divisor range fits.  So each layer line's pin is resolved here as the loop will
resolve it (the shared one, unless the line carries one of its own, which is
what `--ao-rem-per-line` writes), the bound is evaluated at it, and a plan with a
layer over the cap is refused with the layer and the numbers instead of being
written and refused four hours later at the loop's first chunk.  What every plan
prints is the tightest window and what is left of it; the recorded C=4,096 plan
sits at `11 + 20 = 31` of 31 and is unchanged, line for line.

    python -m harness.witness.plan --selftest     the check itself, no witness set

**Issue #121: both halves of the bound, where the row states them.**  The pin a
line resolves moves only the quotient half.  The window half is the dump's own
accumulator and its own exponents, and a shape table whose rows carry the three
numbers is evaluated on it here too, so a shape that does not compose is refused
before the plan file is written instead of by `Plan::new` at the loop's first
chunk.  A row read off a dump extracted before the extractor recorded the
accumulator width states none, and the window half is then not checked: the plan
says so rather than checking it against a guess, and the recorded 4,096-token
plan is unchanged line for line.

**Issue #125: the head line, at the pin it resolves.**  A head chunk proves the
attention output's rounding exactly as a layer chunk does.  Its dump is the
layer dump of the layer it was extracted at with a `head` block added, so
`Layer::load` builds the segment's `AttnOut` out of that layer's attention
columns and `prover/src/compose.rs` builds the attention-output component and
its interaction unconditionally (lines 4535 and 6288).  Both halves of the bound
therefore apply to the head line, and the manifest this writer already opens for
`master_table_bits` states every number they need: `attn_out_bits`, and the
`attn_out` block's `ao_acc_bits`, `jv_max` and `jo_min`.  So a head row is built
here and evaluated at the pin the head line resolves -- `--head-flags`', or the
shared one when the line carries none -- and a head line over either cap is
refused with the numbers, beside the layers'.  The shape table has no row for it
and should not: it walks one block's layer dumps, and the pin it writes is the
loop's shared one, which the head line overrides.  A head manifest that states
neither the fold width nor the `attn_out` block is not a dump manifest and is
refused rather than skipped; one that predates `ao_acc_bits` leaves the window
half not checked, as a layer row of the same age does.  The recorded head
configuration is a `b10-final` dump at `--ao-rem-bits 23`, whose fold target
width is 8: `7 + 23 = 30` is the plain form's cap exactly.

**Stream MM (E11): what the model manifest needs of a plan.**  `zk-prover
model-roots` requires `master_table_bits` to be **one value per layer across
blocks** -- `table/exp` is `1 << log_table` rows, so a layer whose blocks
declare two widths has two tree-0 roots and no per-layer model root exists at
all.  Each block's shape table gives that block's own tightest width, and on
the recorded C=4096 extraction seven of the 33 layers carry two, so the
recorded plan is rejected (stream LL's open item 3).  `--table-bits-per-layer`
writes, for every block of a layer, the widest width any block of that layer
asked for; the extra rows are the price of a manifest, and the plan must then
run **without `--tight-table`**, which re-sizes the table off the witness
after the flag is read and which no manifest and no flag names.

`--head-per-block` says the head dump for block `b` is `<--head>_B<b>` rather
than `--head` itself: a head dump carries its own `base_pos`, so the two seam
blocks a `--verify --prompt` grid needs (stream KK) are two dumps, written by
one `extract.py --blocks 7,8 --head --diagonal-block` pass (stream OO).
"""

import argparse
import contextlib
import io
import json
import os
import sys
import tempfile

from .shapes import (ao_head_refusal, ao_head_report, ao_head_row, ao_pin, ao_pin_refusals,
                     ao_pin_row, ao_where, ao_window_report)


def read_rows(path):
    return {r["layer"]: r for r in json.load(open(path))["rows"]}


def want_layers(spec, rows):
    if not spec:
        return sorted(rows)
    want = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            want.extend(range(int(lo), int(hi) + 1))
        else:
            want.append(int(part))
    return want


def layer_line(tb, wdir, tag, extra=""):
    return f"--table-bits {tb}{extra} --witness {wdir} --ser-tag {tag}"


def ao_rem_of(extra, pin):
    """Issue #118: the `--ao-rem-bits` this line is proved at.

    A line's arguments are placed after the shared ones (`chunks.rs::plan_of`)
    and `arg_scalar` takes the last match, so a pin on the line wins over the
    shared one and the shared one is what a line that names none is proved at.
    `--ao-rem-per-line` puts one there, and so may `--layer-flags` or
    `--l0-flags`; whichever is last on the line is the one the chunk gets.
    """
    parts = extra.split()
    for i in range(len(parts) - 1, 0, -1):
        if parts[i - 1] == "--ao-rem-bits":
            return int(parts[i])
    return pin


def head_line(hdir, htb, tag="head", extra="", layer=None, block=None):
    """The `lm_head` line.

    **Stream KK (E11): the coordinate.**  A head dump's manifest says which
    layer it was *extracted* at (16, for the C=4K set) and carries no block at
    all, but the grid puts the head chunk after the last layer of the block it
    reads.  `--layer` / `--block` declare that, `Fixed` carries it, and
    `boundary-grid` places the chunk by the statement rather than by the tag --
    which is what lets the head take a cell at all.

    **Stream UU (E11): the residual boundary.**  The head chunk commits the residual its
    final norm consumed as tree 3, exactly as a layer chunk commits its
    `stream_in`, so the last layer's out root and the head's in root are
    compared by the pair check like every other layer link.  It commits no tree
    4 -- its norm's output is nobody's residual -- so the head is the last chunk
    of the layer axis and never a producer.  A head dump too old to carry the
    final norm has no wiring family and commits nothing here; the grid reports
    that as an unchecked end rather than as a link that failed.
    """
    coord = ""
    if layer is not None:
        coord += f" --layer {layer}"
    if block is not None:
        coord += f" --block {block}"
    return (f"--table-bits {htb} --deq-per-row 501{extra}{coord} "
            f"--witness {hdir} --ser-tag {tag}")


def flags(s):
    """A flag string as written on a line: one leading space, or nothing."""
    s = " ".join(s.split())
    return f" {s}" if s else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True,
                    help="the layer witness directory, holding shapes.json and witness_L*")
    ap.add_argument("--head", default="", help="the head witness directory, if it is to be proved")
    ap.add_argument("--layers", default="", help="which layers, e.g. 0-31 or 0,1,2 (default all)")
    ap.add_argument("--out", required=True)
    # ---- stream DD (E11): the block axis ------------------------------------
    ap.add_argument("--blocks", default="",
                    help="how many token blocks the sequence has: an integer, or 'auto' to "
                         "count DIR/shapes_B*.json. Empty (the default) is the pre-DD "
                         "single-block plan and its lines are unchanged.")
    ap.add_argument("--with-block-0", action="store_true",
                    help="include block 0's layer chunks. It has c = 0 and does not compose "
                         "today (layout.rs:117 divides by c * insts); stream CC owns the "
                         "diagonal-block-only configuration.")
    ap.add_argument("--head-every-block", action="store_true",
                    help="an lm_head chunk after every block, not only the last one")
    # ---- stream KK (E11): which blocks need a head chunk, and why ------------
    ap.add_argument("--head-blocks", default="",
                    help="the blocks that get an lm_head chunk, e.g. '7,8' or '0-8' "
                         "(default: the last block; --head-every-block is '0-B'). "
                         "A GENERATED position's id has to equal the argmax the head "
                         "proved for the position before it, so the block that owns the "
                         "position before the first generated one needs a head chunk too "
                         "-- for the 9-block C=4K sequence (8 prompt blocks + 1 decode "
                         "block) that is block 7, and '--head-blocks 7,8' is what makes "
                         "`boundary-grid --verify` report no unchecked generated position")
    ap.add_argument("--emit-out", default="",
                    help="also write the distinct emit shapes, one representative plan line "
                         "each, for `emitset` to walk instead of the plan")
    # ---- stream EE (E11): what every line carries besides its own shape ------
    # The E11 grid wants `--round even --diagonal-block` on every layer line (the
    # boundaries and the embedding gadget are on by default wherever the dump
    # lets them be) and the head pinned where seq6 ran it.  They go on the LINE,
    # not the shared command line, because `boundary-grid` reads a chunk's flags
    # back off its statement and the head must not inherit `--diagonal-block`.  A line's
    # flags are part of its emit key, and so is what the dump lets the defaults
    # turn on: a layer-0 line whose dump carries the embedding rows is a
    # different kernel shape from a layer-1 line at the same (c, table bits).
    ap.add_argument("--layer-flags", default="",
                    help="appended to every layer line, e.g. "
                         "'--round even --diagonal-block' (E11)")
    ap.add_argument("--l0-flags", default="",
                    help="appended to every layer-0 line after --layer-flags, e.g. "
                         "(E11 needs none: the embedding gadget is on by default on a "
                         "layer-0 dump that carries embed.u16 and tok_in.i32)")
    ap.add_argument("--l0-dir", default="",
                    help="take the layer-0 witnesses from this directory instead of --dir "
                         "(a layer-0 re-extract with --embed next to the full dump set)")
    ap.add_argument("--head-flags", default="",
                    help="appended to the head line, e.g. '--ao-rem-bits 23' to keep the "
                         "head at its own pin when the shared pins are the layers'")
    # ---- stream SS (E11): the attention-output remainder, per line ----------
    # A shared `--ao-rem-bits` is the model-wide worst case, and a chunk whose
    # own `ao_q_bits` is wide refuses it: `compose.rs` asserts
    # `ao_q_bits + ao_rem_bits <= 30`, so the E11 set's `attn_out_bits 12`
    # (`ao_q_bits` 11) refuses the shared 20 that its own worst block needs,
    # while its own 19 is legal.  A line's arguments are placed AFTER the
    # shared ones (`chunks.rs::plan_of`) and `arg_scalar` takes the last match,
    # so a per-line pin simply wins and the shared one stays as the default for
    # the lines that do not carry one.  That is the order a one-shot
    # `compose $SHARED $line` writes too, so a line rerun on its own resolves
    # its flags exactly as the loop did (zk-prover #30).  Off by default: with
    # it absent every plan this repo has recorded is byte-identical.
    ap.add_argument("--ao-rem-per-line", action="store_true",
                    help="write each layer line's own --ao-rem-bits, off the shape table's "
                         "row, instead of leaving it to the shared pin")
    # ---- stream MM (E11): one master table width per layer ------------------
    ap.add_argument("--table-bits-per-layer", action="store_true",
                    help="give every block of a layer the widest --table-bits any block of "
                         "that layer asked for, so `zk-prover model-roots` can derive one model "
                         "root per layer (it rejects a layer whose blocks declare two "
                         "widths). Run the plan WITHOUT --tight-table, which re-sizes the "
                         "table off the witness and which no manifest names.")
    ap.add_argument("--head-per-block", action="store_true",
                    help="the head dump for block b is <--head>_B<b>, not --head itself "
                         "(one `extract.py --blocks 7,8 --head --diagonal-block` pass writes both)")
    ap.add_argument("--table-bits-map", default="",
                    help="a JSON {\"<layer>\": bits, ...} (and optionally \"head\") giving "
                         "each layer's stated master table width, instead of the shape "
                         "table's. The shape table carries the EXTRACTION's "
                         "`master_table_bits`, which stream J measured 1-3 bits loose, and "
                         "declaring that without --tight-table put the C=4096 grid 137.9 GB "
                         "into a 150.1 GB card (stream MM). A map built from a previous "
                         "run's own tightened `params.log_table`, maxed over the blocks of "
                         "each layer, is both tight and one width per layer, which is what "
                         "`zk-prover model-roots` needs.")
    a = ap.parse_args()
    lf, l0f, hf = flags(a.layer_flags), flags(a.l0_flags), flags(a.head_flags)

    if not a.blocks:
        blocks = [None]
    elif a.blocks == "auto":
        import glob
        blocks = sorted(int(os.path.basename(q)[len("shapes_B"):-len(".json")])
                        for q in glob.glob(f"{a.dir}/shapes_B*.json"))
        assert blocks, f"--blocks auto: no {a.dir}/shapes_B*.json"
    else:
        blocks = list(range(int(a.blocks)))

    # the blocks whose layer lines this plan writes: block 0's are skipped
    # unless it is asked for, and what it asked for is then not this plan's
    written = [b for b in blocks if not (b == 0 and not a.with_block_0)]
    tables = {}

    def table_path(b):
        return f"{a.dir}/shapes.json" if b is None else f"{a.dir}/shapes_B{b}.json"

    def has_table(b):
        return os.path.exists(table_path(b))

    def table(b):
        """One block's shape table, read once."""
        if b not in tables:
            tables[b] = read_rows(table_path(b))
        return tables[b]

    # stream KK: which blocks carry a head chunk.  `None` keeps the pre-KK rule
    # (the last block, tagged `head`); a list tags them `B<b>head` unless there
    # is only one.
    if a.head_blocks:
        head_blocks = set(want_layers(a.head_blocks, {b: None for b in blocks if b is not None}))
    elif a.head_every_block:
        head_blocks = {b for b in blocks if b is not None}
    else:
        head_blocks = None
    # the layer coordinate the head chunk sits at: one past the last layer the
    # plan writes, read off the dump's own shapes rather than assumed to be 32
    n_layers = 1 + max(table(None if blocks == [None] else blocks[-1]))

    # stream MM: the widest `table_bits` any block of a layer asked for, so one
    # model root per layer exists.  Built over the blocks the plan actually
    # writes -- a width from a block this plan skips is not this plan's.
    widest = {}
    if a.table_bits_map:
        m = json.load(open(a.table_bits_map))
        widest = {int(k): int(v) for k, v in m.items() if k != "head"}
    elif a.table_bits_per_layer:
        for b in written:
            for k, r in table(b).items():
                widest[k] = max(widest.get(k, 0), r["table_bits"])

    # ---- issue #118: the pin a line that names none is proved at ------------
    # `--ao-rem-bits` is one number for the whole loop and the drivers read it
    # off the shape table's worst case, which is `worst_of`'s maximum over every
    # row of every block -- a block whose lines this plan skips included, since
    # the pin is on the loop's command line and not on a line.  So it is taken
    # here over every block that has a table, and a table too old to carry
    # `ao_rem_bits` is not checked at all rather than checked against a default.
    pin_rows = [r for b in blocks if has_table(b) for r in table(b).values()]
    pin = (ao_pin(pin_rows) if pin_rows and all(r.get("ao_rem_bits") for r in pin_rows)
           else None)
    checked, refused = [], []
    # issue #125: one line per head chunk, printed beside the layers' report --
    # a head line's numbers are wanted whether or not it is the tightest of the
    # plan, because it is the one line the shape table has no row for
    head_notes = []

    lines, emit = [], {}
    n_rows, n_layer_rows = 0, 0
    for b in blocks:
        if b is None:
            rows = table(None)
            pre, wpre = "", f"{a.dir}/witness_L"
        else:
            if b == 0 and not a.with_block_0:
                lines.append("# block 0 skipped: c = 0 is the diagonal-block-only chunk (stream CC); "
                             "pass --with-block-0 once it composes")
                continue
            rows = table(b)
            pre, wpre = f"B{b}", f"{a.dir}/witness_B{b}_L"
            lines.append(f"# block {b}: prefix c = {rows[sorted(rows)[0]]['c']}")
        want = want_layers(a.layers, rows)
        for k in want:
            tb = widest.get(k, rows[k]["table_bits"])
            extra = lf + (l0f if k == 0 else "")
            if a.ao_rem_per_line:
                rem = rows[k].get("ao_rem_bits")
                assert rem, f"--ao-rem-per-line: {a.dir}'s shape table has no ao_rem_bits"
                extra += f" --ao-rem-bits {rem}"
            # issue #118: what this line will be proved at, before it is
            # written -- the shared pin unless the line carries one of its own
            if pin is not None:
                row = rows[k] if b is None else dict(rows[k], block=b)
                q, msg = ao_pin_row(row, ao_rem_of(extra, pin))
                checked.append((row, q))
                if msg:
                    refused.append(msg)
            wdir = f"{wpre}{k}"
            if k == 0 and a.l0_dir:
                wdir = f"{a.l0_dir}/{os.path.basename(wpre)}{k}"
            line = layer_line(tb, wdir, f"{pre}L{k}", extra)
            lines.append(line)
            n_rows += 1
            n_layer_rows += 1
            emit.setdefault((rows[k]["c"], tb, False, extra), line)
        if a.head and (b in head_blocks if head_blocks is not None else b == blocks[-1]):
            hdir = f"{a.head}_B{b}" if a.head_per_block else a.head
            hman = json.load(open(f"{hdir}/manifest.json"))
            htb = hman["master_table_bits"]
            if a.table_bits_map:
                hm = json.load(open(a.table_bits_map)).get("head")
                if hm is not None:
                    htb = int(hm)
            elif a.table_bits_per_layer and a.head_per_block:
                # the head chunks sit at one layer coordinate too, so they need
                # one width between them for the same reason the layers do
                htb = max(json.load(open(f"{a.head}_B{hb}/manifest.json"))
                          ["master_table_bits"] for hb in sorted(head_blocks))
            lines.append("# the head, last, and it is the same compose family with two extra "
                         "arguments")
            solo = head_blocks is None or len(head_blocks) == 1
            hl = head_line(hdir, htb, "head" if solo else f"{pre}head", hf,
                           layer=n_layers, block=b)
            lines.append(hl)
            n_rows += 1
            # issue #131: a head chunk's prefix is the one its own dump
            # declares (`params_of` takes `c` off the witness manifest), and
            # with `--head-per-block` the dumps of two blocks are two prefixes
            # -- 3,584 and 4,096 tokens for the C=4,096 pair.  Keyed with no
            # prefix at all they were one entry, only the first block's line
            # reached the emit list, and the second head chunk proved against a
            # feed kernel that was never generated.
            emit.setdefault((hman["c"], htb, True, hf), hl)
            # issue #125: the head line's own bound, both halves, at the pin
            # this line resolves.  The head chunk proves the attention output's
            # rounding like any layer chunk -- its dump is that layer's dump
            # with a `head` block added, and `compose.rs` builds the component
            # and its interaction unconditionally -- so the manifest opened
            # above for the table width states the fold's target width and the
            # three window-half numbers, and the head line is checked here
            # exactly as a layer line is.  It is the only line whose bound
            # nothing else evaluates: the shape table walks layer dumps.
            hflag = ao_rem_of(hf, None)
            hpin = ao_rem_of(hf, pin)
            hrow = ao_head_row(hman, hpin, block=b, flag_pin=hflag)
            bad = ao_head_refusal(hman, ao_where(hrow))
            if bad:
                refused.append(bad)
            elif hpin is None:
                # neither the shape tables nor `--head-flags` name a pin, so the
                # loop leaves the chunk at the head dump's own divisor range and
                # no plan can say what that is -- the manifest records none
                head_notes.append(
                    f"attention output, {ao_where(hrow)}: it resolves no --ao-rem-bits "
                    f"(the shape tables carry none and --head-flags names none), so it is "
                    f"proved at the head dump's own divisor range, which the manifest does "
                    f"not state, and its bound is not checked here")
            else:
                hq, hmsg = ao_pin_row(hrow, hpin)
                checked.append((hrow, hq))
                head_notes.append(ao_head_report(hrow, hq))
                if hmsg:
                    refused.append(hmsg)
    # issue #118: refused here, with the layer and the numbers, and before the
    # plan file is written -- a plan that would be refused by `Plan::new` at the
    # first chunk of the loop is not written at all
    assert not refused, ao_pin_refusals(refused)

    lines.insert(0, f"# stream O: {n_layer_rows} layer chunks from {a.dir}"
                    if blocks == [None] else
                    f"# stream DD: {n_layer_rows} layer chunks over {len(blocks)} token "
                    f"blocks from {a.dir}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{a.out}: {n_rows} chunks")
    if checked:
        r, q = min(checked, key=lambda t: t[1]["margin"])
        own = " (the line's own)" if a.ao_rem_per_line else ""
        print(f"attention output: at --ao-rem-bits {q['rem_bits']}{own} the tightest "
              f"quotient window is {q['bits']}/{q['limit']} bits, margin {q['margin']} "
              f"({q['form']} form), at {ao_where(r)}")
        # issue #121: and the half no pin moves, over the same rows
        print(ao_window_report([row for row, _ in checked]))
    elif pin is None:
        print("attention output: the shape tables carry no ao_rem_bits, so the pin the "
              "loop is launched with is not checked here")
    # issue #125: the head lines' own margins, beside the layers' above
    for note in head_notes:
        print(note)

    if a.emit_out:
        el = [f"# stream DD: {len(emit)} distinct emit shapes for the {n_rows} chunks of "
              f"{a.out}; the key is (prefix c, master table bits, head, the line's flags)"]
        for key in sorted(emit, key=lambda k: (k[2], k[0], k[1], k[3])):
            el.append(emit[key])
        os.makedirs(os.path.dirname(a.emit_out) or ".", exist_ok=True)
        with open(a.emit_out, "w") as f:
            f.write("\n".join(el) + "\n")
        print(f"{a.emit_out}: {len(emit)} emit shapes (of {n_rows} chunks, "
              f"x{n_rows / max(len(emit), 1):.1f} fewer cudaemit launches)")


def _shape_table(d, b, rems, bits=12, acc=None, jv=-6, jo=-10):
    """One block's shape table, with only the fields a plan reads.

    `bits` is the fold's target width, one value or one per row: a set whose
    layers were not all extracted at the same width is what `--l0-dir` points a
    plan at, and it is where the model-wide pin and a layer's own cap part.

    Issue #121: `acc` is the accumulator's bit length, one value or one per row,
    and `None` -- the default -- is the recorded set, whose rows state none.
    """
    if isinstance(bits, int):
        bits = [bits] * len(rems)
    if not isinstance(acc, list):
        acc = [acc] * len(rems)
    # the two exponent extremes, one value or one per row, as `acc` is
    jv = jv if isinstance(jv, list) else [jv] * len(rems)
    jo = jo if isinstance(jo, list) else [jo] * len(rems)
    rows = [dict(layer=k, block=b, c=512, table_bits=23, ao_rem_bits=rem,
                 attn_out_bits=w, ao_acc_bits=a,
                 ao_jv_max=(v if a else None), ao_jo_min=(o if a else None))
            for k, (rem, w, a, v, o) in enumerate(zip(rems, bits, acc, jv, jo))]
    with open(os.path.join(d, f"shapes_B{b}.json"), "w") as f:
        json.dump(dict(rows=rows, block=b), f)


def _head_dump(d, b, c, tb=25, bits=8, acc=19, jv=-6, jo=-10, attn_out=True):
    """One head dump's manifest, with only the fields a plan reads.

    A head dump carries its own prefix, and the dumps of two blocks carry two.

    Issue #125: and it carries the fold's target width and the `attn_out` block,
    because a head dump is the layer dump of the layer it was extracted at with
    a `head` block added and `extract.py` writes one manifest per dump.  `bits`
    8 and a 23-bit pin are the recorded head configuration's numbers.  `acc`
    `None` is a dump extracted before the accumulator's width was recorded, and
    `attn_out` false is a manifest that is not a dump's at all.
    """
    hd = os.path.join(d, f"head_B{b}")
    os.makedirs(hd, exist_ok=True)
    man = dict(layer=16, block=b, c=c, master_table_bits=tb)
    if attn_out:
        man["attn_out_bits"] = bits
        man["attn_out"] = dict(jv_max=jv, jo_min=jo,
                               **({"ao_acc_bits": acc} if acc is not None else {}))
    with open(os.path.join(hd, "manifest.json"), "w") as f:
        json.dump(man, f)


def _plan(argv):
    """`main()` on `argv`: (what it printed, or the refusal it raised)."""
    buf = io.StringIO()
    argv0 = sys.argv
    try:
        sys.argv = ["plan"] + argv
        with contextlib.redirect_stdout(buf):
            main()
    except AssertionError as e:
        return None, str(e)
    finally:
        sys.argv = argv0
    return buf.getvalue(), None


def selftest():
    """Issue #118: the pin every line is proved at, and the layer it refuses.

    No dump, no model and no card: the shape tables are written here with the
    numbers the recorded C=4,096 set carries -- `attn_out_bits` 12, so `q_bits`
    11, and divisor ranges up to the 20 whose `denp2` window is the field's 31
    exactly.  The layer this check exists for is the one whose own range fits
    and whose model-wide pin does not, which is what `--ao-rem-per-line` is for
    and what the refusal names.
    """
    bad = 0

    def case(label, ok, note):
        nonlocal bad
        bad += not ok
        print(f"{'OK  ' if ok else 'FAIL'}  {label:<52s} -> {note}")

    def _head_note(printed):
        """Issue #125: the head line's own report line, out of what was printed."""
        return next((l for l in (printed or "").splitlines()
                     if l.startswith("attention output, ")), "")

    def last(printed, refusal):
        return (printed.strip().splitlines()[-1] if printed
                else (refusal or "NOT refused").split(".")[0])

    # 1. which pin a line resolves to: the shared one, unless the line names one
    got = (ao_rem_of(" --round even", 20), ao_rem_of(" --round even --ao-rem-bits 19", 20),
           ao_rem_of(" --ao-rem-bits 19 --ao-rem-bits 17", 20))
    case("a line's pin wins over the shared one, last first", got == (20, 19, 17), got)

    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "plan.txt")
        args = ["--dir", d, "--blocks", "2", "--with-block-0", "--out", out]

        # 2. the recorded set's own numbers: the pin is 20, `11 + 20 = 31` is the
        #    denp2 form's cap exactly, and the lines carry no pin of their own
        _shape_table(d, 0, [14, 15, 19])
        _shape_table(d, 1, [15, 16, 20])
        printed, refusal = _plan(args)
        body = [l for l in open(out).read().splitlines() if not l.startswith("#")]
        ok = (refusal is None and len(body) == 6 and "--ao-rem-bits" not in "".join(body)
              and body[0] == f"--table-bits 23 --witness {d}/witness_B0_L0 --ser-tag B0L0"
              and "margin 0 (denp2 form)" in printed)
        case("the recorded pin composes, and adds nothing to a line", ok,
             last(printed, refusal))

        # 3. the layer this check exists for: one layer of the set was extracted
        #    at the narrower fold width and its own divisor range is 23, so the
        #    model-wide pin is 23 and every 12-bit layer is over the cap at it --
        #    although every one of them fits at its own
        _shape_table(d, 1, [15, 16, 23], bits=[12, 12, 8])
        os.remove(out)          # a refused plan is not written at all
        printed, refusal = _plan(args)
        ok = (refusal is not None and refusal.startswith("block 0 layer 0: ")
              and "11 + 23 = 34 > 31 (over by 3, denp2 form)" in refusal
              and "does fit" in refusal and "--ao-rem-per-line" in refusal
              and not os.path.exists(out))
        case("a merged pin past a layer's cap is refused", ok, last(printed, refusal))

        # 4. and the per-line pin is what admits it, which is what the refusal
        #    named: every line is proved at its own range
        printed, refusal = _plan(args + ["--ao-rem-per-line"])
        body = [l for l in open(out).read().splitlines() if not l.startswith("#")]
        ok = (refusal is None and len(body) == 6
              and body[0] == (f"--table-bits 23 --ao-rem-bits 14 --witness "
                              f"{d}/witness_B0_L0 --ser-tag B0L0")
              and body[-1] == (f"--table-bits 23 --ao-rem-bits 23 --witness "
                               f"{d}/witness_B1_L2 --ser-tag B1L2"))
        case("--ao-rem-per-line writes each layer's own", ok, last(printed, refusal))

        # 5. a layer whose own range does not fit either: no pin recovers it,
        #    and the message says so rather than naming a flag that cannot help
        _shape_table(d, 1, [15, 16, 22])
        printed, refusal = _plan(args + ["--ao-rem-per-line"])
        ok = (refusal is not None and refusal.startswith("block 1 layer 2: ")
              and "11 + 22 = 33 > 31 (over by 2, denp2 form)" in refusal
              and "no pin recovers it" in refusal)
        case("a shape that fits at no pin says so", ok, last(printed, refusal))

        # 6. the two forms, at the fold's int8 width: `7 + 23 = 30` is the plain
        #    form's cap, and the bit past it is the denp2 form's, taken the way
        #    `params_of` takes it
        _shape_table(d, 0, [20, 20, 20], bits=8)
        _shape_table(d, 1, [20, 20, 23], bits=8)
        printed, refusal = _plan(args)
        ok = refusal is None and "30/30 bits, margin 0 (plain form)" in printed
        case("the plain form's cap, spent to zero", ok, last(printed, refusal))

        _shape_table(d, 1, [20, 20, 24], bits=8)
        printed, refusal = _plan(args)
        ok = refusal is None and "31/31 bits, margin 0 (denp2 form)" in printed
        case("and the denp2 form's one bit past it", ok, last(printed, refusal))

        # 7. issue #131: every head block's shape is in the emit list.  Two head
        #    dumps of one master table width are one entry only when their
        #    prefixes agree, and the two `--head-blocks` a `--verify --prompt`
        #    grid needs do not agree -- the recorded C=4,096 pair is 3,584
        #    tokens of prefix for block 7 and 4,096 for block 8.  Keyed with no
        #    prefix they were one entry, only the first block's line was
        #    emitted, and the second head chunk was proved against a feed kernel
        #    nobody had been given the shape of.
        em = os.path.join(d, "emit.txt")
        heads = ["--head", os.path.join(d, "head"), "--head-per-block",
                 "--head-blocks", "0,1", "--emit-out", em]

        def head_shapes():
            """(the plan's head lines, the emit list's lines)."""
            return ([l for l in open(out).read().splitlines() if l.endswith("head")],
                    [l for l in open(em).read().splitlines() if not l.startswith("#")])

        _shape_table(d, 0, [14, 15, 19])
        _shape_table(d, 1, [14, 15, 19])
        _head_dump(d, 0, 512)
        _head_dump(d, 1, 1024)
        printed, refusal = _plan(args + heads)
        hl, el = head_shapes()
        ok = (refusal is None and len(hl) == 2 and len(el) == 3
              and all(l in el for l in hl))
        case("every head block's shape is in the emit list", ok,
             f"{len(hl)} head chunks, {len(el)} emit shapes")

        # 8. and the other way round: two head chunks that read one dump's
        #    prefix at one width are one shape, and the list holds one of them
        _head_dump(d, 1, 512)
        printed, refusal = _plan(args + heads)
        hl, el = head_shapes()
        ok = (refusal is None and len(hl) == 2 and len(el) == 2
              and sum(l in el for l in hl) == 1)
        case("two head dumps of one prefix are one shape", ok,
             f"{len(hl)} head chunks, {len(el)} emit shapes")

        # ---- issue #121: the half no pin moves ---------------------------
        # 9. the recorded set: its rows carry no accumulator width, so the plan
        #    is written and says which half of the bound it could not check
        _shape_table(d, 0, [14, 15, 19])
        _shape_table(d, 1, [15, 16, 20])
        printed, refusal = _plan(args)
        ok = (refusal is None and "none of the 6 rows states ao_acc_bits" in printed
              and "NOT CHECKED" in printed and os.path.exists(out))
        case("a table with no accumulator width says so", ok, last(printed, refusal))

        # 10. the same set re-extracted: the rows state it and both halves are
        #    evaluated, at the pin each line resolves
        _shape_table(d, 0, [14, 15, 19], acc=19)
        _shape_table(d, 1, [15, 16, 20], acc=19)
        printed, refusal = _plan(args)
        ok = (refusal is None and "NOT CHECKED" not in printed
              and "the tightest no-wrap window is 25/31 bits, margin 6" in printed
              and "margin 0 (denp2 form)" in printed)
        case("a re-extracted table is checked on both halves", ok, last(printed, refusal))

        # 11. and a shape whose window half does not fit: refused before the file
        #    is written, and not by naming a pin, which cannot recover a bit of it.
        #    A 27-bit accumulator composes in the product form, so what is refused
        #    here is an exponent window past even that one.
        _shape_table(d, 1, [15, 16, 20], acc=[19, 19, 27], jv=[-6, -6, -5],
                     jo=[-10, -10, -20])
        os.remove(out)
        printed, refusal = _plan(args)
        ok = (refusal is not None and refusal.startswith("block 1 layer 2: ")
              and "no-wrap window does not fit" in refusal
              and "1 + max(16, 11) + 15 = 32 > 31 (over by 1, product form)" in refusal
              and "--ao-rem-per-line" not in refusal and not os.path.exists(out))
        case("a window half past the field is refused", ok, last(printed, refusal))

        # ---- issue #125: the head line, evaluated like a layer line -------
        # The head chunk proves the attention output's rounding as any layer
        # chunk does, and it is the one line no shape table has a row for, so
        # this is where both halves of its bound are evaluated.  The numbers
        # are the recorded head configuration's: a `b10-final` dump, whose fold
        # target width is 8, at the 23 bits its own divisor range needs.
        _shape_table(d, 0, [14, 15, 19])
        _shape_table(d, 1, [15, 16, 20])
        head = ["--head", os.path.join(d, "head"), "--head-per-block",
                "--head-blocks", "1"]
        pin23 = ["--head-flags=--ao-rem-bits 23"]

        # 12. the recorded head configuration: `7 + 23 = 30` is the plain form's
        #     cap exactly, the window half fits with six bits to spare, and the
        #     plan is written
        _head_dump(d, 1, 512)
        printed, refusal = _plan(args + head + pin23)
        ok = (refusal is None and os.path.exists(out)
              and "attention output, block 1's head line: at --ao-rem-bits 23 the "
                  "quotient window is 7 + 23 = 30/30 bits, margin 0 (plain form)" in printed
              and "the no-wrap window is 1 + 16 + 4 + 4 = 25/31 bits, margin 6 "
                  "(offset form)" in printed)
        case("the head line's two margins, beside the layers'", ok,
             _head_note(printed) or last(printed, refusal))

        # 13. one bit past the quotient cap: refused, and the refusal names the
        #     head line rather than a layer, and names the flag that pinned it
        os.remove(out)
        printed, refusal = _plan(args + head + ["--head-flags=--ao-rem-bits 25"])
        ok = (refusal is not None and refusal.startswith("block 1's head line: ")
              and "7 + 25 = 32 > 31 (over by 1, denp2 form)" in refusal
              and "`--head-flags` names" in refusal
              and "--ao-rem-per-line" not in refusal and not os.path.exists(out))
        case("a head line past the quotient cap is refused", ok, last(printed, refusal))

        # 14. and the half no pin moves: an exponent window one bit past the
        #     product form is refused too, naming neither pin, since neither
        #     could recover it
        _head_dump(d, 1, 512, acc=27, jv=-5, jo=-20)
        printed, refusal = _plan(args + head + pin23)
        ok = (refusal is not None and refusal.startswith("block 1's head line: ")
              and "no-wrap window does not fit" in refusal
              and "1 + max(16, 11) + 15 = 32 > 31 (over by 1, product form)" in refusal
              and "--head-flags" not in refusal and not os.path.exists(out))
        case("a head line past the window half is refused", ok, last(printed, refusal))

        # 15. a manifest that states neither the fold width nor the `attn_out`
        #     block is not a dump manifest: refused, not silently skipped
        _head_dump(d, 1, 512, attn_out=False)
        printed, refusal = _plan(args + head + pin23)
        ok = (refusal is not None and refusal.startswith("block 1's head line: ")
              and "states no attn_out_bits, attn_out" in refusal
              and "extract the head dump again" in refusal and not os.path.exists(out))
        case("an incomplete head manifest is refused", ok, last(printed, refusal))

        # 16. and a head dump that predates the accumulator's width: the
        #     quotient half is checked, the window half is not, and the plan
        #     says which it was rather than guessing an accumulator
        _head_dump(d, 1, 512, acc=None)
        printed, refusal = _plan(args + head + pin23)
        ok = (refusal is None and os.path.exists(out)
              and "7 + 23 = 30/30 bits, margin 0 (plain form)" in printed
              and "the no-wrap window half is NOT CHECKED" in printed)
        case("a head dump with no accumulator width says so", ok,
             _head_note(printed) or last(printed, refusal))

    return 1 if bad else 0


if __name__ == "__main__":
    # the self-test takes no other argument, and is read before the parser so
    # that --dir and --out stay required of a plan
    sys.exit(selftest() if "--selftest" in sys.argv[1:] else main())
