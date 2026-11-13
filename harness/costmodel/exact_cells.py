"""A composed segment's committed cells, counted from its `params` block alone.

    uv run python -m harness.costmodel.exact_cells --selftest

`cost_model.py` prices a segment from per-unit constants read off measured runs
-- cells per accumulator, per weight, per lookup -- and the frontier calibrates
the total against the recorded sequence.  This module does not: it rebuilds the
trace layout the prover builds, component by component, and multiplies columns
by rows.  Nothing here is fitted and nothing here is measured; every line is an
identity of `prover/src`, named in the comment above it.

`key_tile.py` solved three pieces of this already: the attention key tile
family, the boundary trees and the lookup tables.  The boundary form is
imported from it unchanged.  The other two are rebuilt here -- the tables
because the set a segment registers and the height of each is derivable rather
than a fixed list, and the key tiles because `key_tile.py`'s closed form holds
only where the three grids tile exactly, which the output head's C = 3,584 does
not.  What is new is everything else: the dequantize components, the attention
output, the Freivalds checks, the residual stream, the rotation's quantize
components, the requantize seams, their own tables, the GKR link, the argmax
and the embedding gadget.  That is the whole chunk, which is why `--selftest`
compares against `cells.base`, `cells.interaction` and `cells.fresh` rather
than against one family.

WHAT IT IS EXACT IN.  A `params` block, not an architecture.  Some of its
fields are not the model's shape at all: they are the calibration a run took
off its own witness (`kmax`, `log_table`, `ao_jo_min/max`, `ao_jv_min/max`,
`ao_acc_hi_bits`, `ao_acc_lo_bits`, `ao_rem_bits`, `ao_q_bits`, `fold_alpha`,
`fold_rem_bits`), and three groups of them move the count: see `witness_terms`.
A caller who has a model and a context but no witness has to assume them, and
the largest of the three -- the master exponent table at `2^log_table` -- is
between 1.5 and 11 per cent of a recorded chunk.  `head_groups` is a run knob
rather than a shape, and it decides the grids' heights: a `params` block
without it is refused rather than guessed at, and so is the `bf16x` attention
variant, which no recorded segment proves.

    exact_cells(params) -> {"base": ..., "interaction": ..., "boundary": ...,
                            "fresh": ..., "components": {name: (base, inter)}}
"""
import argparse
import json
import pathlib
import sys

from harness.costmodel.attention import latent as latent_attention
from harness.costmodel.key_tile import boundary_cells

HERE = pathlib.Path(__file__).parent
CHUNKS = HERE / "exact_chunks.json"

# stwo's SimdBackend needs N_LANES = 16 rows (`common.rs`, `MIN_LOG_ROWS`).
MIN_LOG_ROWS = 4
# The five batched GKR reductions, by the name a `params` block's flag carries
# (`gkr_<name>`).  The two headline runs were proved with all five on.
REDUCTIONS_ON = ("key_tiles", "dequant", "residual_stream", "freivalds", "attnout")

# A QM31 column is four M31 cells: every interaction column below is counted in
# QM31 and multiplied by this at the end (`plogup.rs`, `devfrac.rs`).
M31_PER_QM31 = 4


def log_rows_for(n, per_row=1):
    """`common.rs`, `log_rows_for`: rows for `n` items at `per_row` each, at or
    above the SIMD floor."""
    rows = max(1, -(-n // max(1, per_row)))
    return max(MIN_LOG_ROWS, (rows - 1).bit_length())


# ------------------------------------------------------------- the layouts --
# Every family below tiles its values into rows the same way: a width that
# divides the value count, a row count rounded up to a power of two, and the
# slack carried as inert lead rows.  The four choosers differ only in what they
# score and what they are allowed to pad.


def mat_layout(n_out, tokens, target):
    """`layer.rs`, `MatLayout::choose_or_padded`: the exact tiling when one
    exists, the padded one otherwise.  Returns (per_row, chunks, log_rows)."""
    best = None
    for pr in range(1, min(n_out, 4 * max(target, 1)) + 1):
        if n_out % pr:
            continue
        chunks = n_out // pr
        rows = tokens * chunks
        if rows & (rows - 1) or rows < 16:
            continue
        score = abs(pr - target)
        if best is None or score < best[0]:
            best = (score, (pr, chunks, rows.bit_length() - 1))
    if best is not None:
        return best[1]
    return _mat_layout_padded(n_out, tokens, target)


def _mat_layout_padded(n_out, tokens, target):
    """`layer.rs`, `MatLayout::choose_padded`: a vocabulary does not tile a
    power of two at any width, so the row count is rounded up and a candidate
    whose lead exceeds a quarter of its rows is refused."""
    max_lead_frac = 0.25
    best = None
    for pr in range(1, n_out + 1):
        if n_out % pr:
            continue
        chunks = n_out // pr
        active = tokens * chunks
        log_rows = max(MIN_LOG_ROWS, (active - 1).bit_length())
        lead = (1 << log_rows) - active
        if lead / (1 << log_rows) > max_lead_frac:
            continue
        score = abs(pr - target)
        if best is None or score < best[0]:
            best = (score, (pr, chunks, log_rows), lead)
        if pr > target and best is not None and best[2] == 0:
            break
    assert best is not None, f"no layout for n_out={n_out}"
    return best[1]


def ao_layout(nh, s, hd, target):
    """`layer.rs`, `AoLayout::choose`: the largest divisor of `hd` at most the
    target, one row per (head, token, chunk), inert lead first."""
    pr = max(1, min(target, hd))
    while hd % pr:
        pr -= 1
    chunks = hd // pr
    active = nh * s * chunks
    return pr, chunks, max(MIN_LOG_ROWS, (active - 1).bit_length())


def q_layout(n, hd, tokens, target):
    """`rope.rs`, `q_layout`: a quantize width divides both the head width and
    the matmul's output width; an exact tiling beats a padded one."""
    best = None
    for pr in range(1, hd + 1):
        if hd % pr or n % pr:
            continue
        chunks = n // pr
        active = tokens * chunks
        log_rows = log_rows_for(active)
        lead = (1 << log_rows) - active
        key = (1 if lead > 0 else 0, abs(pr - target))
        if best is None or key < best[0]:
            best = (key, (pr, chunks, log_rows))
    assert best is not None, f"no quantize layout for n={n}, hd={hd}"
    return best[1]


def freivalds_layout(n_in, n_out_max, tokens, target):
    """`layer.rs`, `FreivaldsLayout::choose`: one row group per input channel,
    `pk` weights and `pq` activations per row, the chunk count scored on `pk`
    and the least padded winning among equals.  Returns (pk, pq, log_rows)."""
    best = None
    for chunks in range(1, n_out_max + 1):
        pk = -(-n_out_max // chunks)
        pq = -(-tokens // chunks)
        rows = n_in * chunks
        log_rows = max(MIN_LOG_ROWS, (rows - 1).bit_length())
        # `Grid::new` caps the chunk count at `rows / groups`; a candidate the
        # cap would cut back is not the layout it asked for, and is dropped
        if (1 << log_rows) // n_in < chunks:
            continue
        pad = (1 << log_rows) - rows
        key = (abs(pk - target), pad)
        if best is None or key < best[0]:
            best = (key, (pk, pq, log_rows))
    assert best is not None, "no Freivalds layout"
    return best[1]


# `layer.rs`, `FREIVALDS_GROUPS`: the matmuls sharing one committed input.
FREIVALDS_GROUPS = [("qkv", ("q", "k", "v")), ("o", ("o",)), ("gateup", ("gate", "up")),
                    ("down", ("down",)), ("head", ("lm_head",))]


# ------------------------------------------------------- the dequantize family --
# `dequant.rs`, `Map::new`: a row carries the variant's shared columns and then
# one slot per accumulator.  The slot is the whole statement of one
# dequantization, so its width is a property of the variant and of the rounding
# alone -- not of the data, not of `kmax`, and not of the matmul's shape.


def deq_columns(variant, sticky, per_row):
    """`dequant.rs`, `Map::new`: committed columns of one dequantize component."""
    if variant == "fold":
        shared, per_slot = 2, 9          # flo fhi sg fa fb qm qc fe fdp
    elif variant == "chain":
        # acc sg m2 ph [st] r m7 ef nz, then two (mantissa, carry, exponent)
        shared, per_slot = 2, 8 + int(sticky) + 6
    elif variant == "pow2":
        shared, per_slot = 1, 8 + int(sticky) + 1
    elif variant == "int8":
        shared, per_slot = 1, 5
    elif variant == "limbs":
        shared, per_slot = 0, 2
    else:
        raise ValueError(f"unknown dequant variant {variant!r}")
    return shared + per_row * per_slot


def deq_looks(variant, per_row):
    """`dequant.rs`, `look_units`: how many lookups one row's slots draw.  Only
    the count matters for cells; whether a lookup is at unit multiplicity does
    not."""
    per_slot = {"fold": 8, "pow2": 3, "chain": 6}[variant]
    return per_slot * per_row


# ----------------------------------------------------------- the residual stream --
# `compose.rs`, `ResidualStreamPlan::new` and `residual_stream.rs`, `Prog`.  A
# component's committed width is `leaves_width + slots * slot_width`, and both
# are counted off the program the plan builds -- an add slot is 13 columns (14
# with the sticky bit round-to-even needs), a multiply slot is 4, a leaf is two
# columns or four when a unary lookup reads it.
ADD_SLOT, MUL_SLOT, LEAF, UN_LEAF = 13, 4, 2, 4


def _tree_slots(n):
    """`Prog::tree`: `n - 1` real adds however far the padding reaches, because
    `x + 0` is pruned rather than proved."""
    return n - 1


def residual_stream_components(p):
    """The wiring family's components: (tag, columns, log_rows, bindings,
    lookups per row).  `head_norm` is the output head's three; every layer
    segment builds eleven.

    The lookup count is `gkrlink.rs`'s own `residual_stream_look_count`: one
    `VAL` lookup per committed leaf, and five per add slot or two per multiply
    slot."""
    g = p["residual_stream"]
    sticky = p["round"] == "even"
    add, mul = ADD_SLOT + int(sticky), MUL_SLOT
    a, d, s = g["a"], g["d"], p["s"]
    nreal = d // a                       # `Tiling::new`: committed leaves per row
    lr1 = log_rows_for(a * s)            # `rows1 = tokens * a`
    lr2 = log_rows_for(s)
    out = []

    def looks(leaves, slots, kind):
        return leaves + (5 if kind == "add" else 2) * slots

    def square_tree(n):
        # `Prog::tree` under the square lookup: the norm's level-1 reduction
        slots = _tree_slots(n)
        return n * UN_LEAF + slots * add, looks(n, slots, "add")

    def norm_resid():
        # `Prog::norm_resid`: the squares' tree and the residual add on one row
        slots = _tree_slots(nreal) + nreal
        cols = nreal * UN_LEAF + nreal * LEAF + slots * add
        return cols, looks(2 * nreal, slots, "add")

    def tree_plain(n):
        # `Prog::tree` with no unary: the second-level reduction over `a` rows
        slots = _tree_slots(n)
        return n * LEAF + slots * add, looks(n, slots, "add")

    def norm_scale(n):
        # `Prog::norm_scale`: `h = (x * r) * w`, `r` one committed leaf per row
        # under the `NormR` lookup, `w` preprocessed
        return n * LEAF + UN_LEAF + 2 * n * mul, looks(n + 1, 2 * n, "mul")

    def pairs_silu(n):
        # `Prog::pairs`: the SwiGLU product, group 0 under the silu lookup
        return n * UN_LEAF + n * LEAF + n * mul, looks(2 * n, n, "mul")

    def plain_norm(tag, width):
        # `Prog`'s three components for a norm with no residual add: the output
        # head's final norm, and the norm a latent attention puts on each of its
        # two low-rank reads (`q_a_layernorm`, `kv_a_layernorm`).  The tiling is
        # `a` rows a token whatever the width, so only the columns move.
        n = width // a
        return [(f"{tag}", *square_tree(n), lr1, 2),
                (f"{tag}b", *tree_plain(a), lr2, 2),
                (f"{tag}t", *norm_scale(n), lr1, 3)]

    if g.get("head_norm"):
        out += plain_norm("gnf", d)
        return [(t, c, lr, b, lk) for t, c, lk, lr, b in out]

    # a layer: two norms, each a level-1 reduction folded with a residual add,
    # a level-2 reduction and a scale; then the SwiGLU and the two rotations
    for n1, n2 in (("gn1", "gn1b"), ("gn2", "gn2b")):
        # `In(0)`, `Out(0)`, `Out(1)`, and the `o` / `down` output claim
        out.append((n1, *norm_resid(), lr1, 4))
        out.append((n2, *tree_plain(a), lr2, 2))
    for tag in ("nt1", "nt2"):
        out.append((tag, *norm_scale(nreal), lr1, 3))
    t2_nreal = g["dff"] // a
    out.append(("gsw", *pairs_silu(t2_nreal), log_rows_for(a * s), 3))
    for i, width in enumerate(g.get("norms") or []):
        if width % a:
            raise ValueError(f"a norm of width {width} does not tile {a} rows")
        out += plain_norm(f"ln{i}", width)
    rope = p.get("rope")
    if rope:
        # a latent attention rotates the decoupled key alone, which is narrower
        # than a head and is one per key rather than one per head
        hd = rope.get("hd", p["hd"])
        for name, heads in (("q", p["nh"]), ("k", rope["nkv"])):
            lr = log_rows_for(s * heads)
            # `Prog::rope_mul`: `hd` leaves, `2 hd` multiply slots
            out.append((f"rp{name}m", hd * LEAF + 2 * hd * mul,
                        looks(hd, 2 * hd, "mul"), lr, 2))
            # `Prog::rope_add`: the multiplies' `2 hd` outputs, `hd` add slots
            out.append((f"rp{name}a", 2 * hd * LEAF + hd * add,
                        looks(2 * hd, hd, "add"), lr, 2))
    return [(t, c, lr, b, lk) for t, c, lk, lr, b in out]


# ------------------------------------------------------------- the lookup tables --
# The bf16 tables' row counts, in closed form (`bf16.rs`, `residual_stream.rs`,
# `requant.rs`, `rope.rs`).  Only `norm` and `frac` depend on anything the run
# calibrates; the rest are properties of the encoding.
TT_RANGE = 9
POW_ROWS = 13 + 260                      # `pow_table`: DMAX + 1 exponents, UMIN zeros
VAL_ROWS = 1 + 254 * 128 * 2             # `val_table`: every legal (sm, ef)
SMUL_ROWS = 257 * 257                    # `smul_table`: signed significand pairs
MUL_LOG_ROWS = 14                        # `bf16::mul_table`: 128 x 128, fixed


def norm_rows(kmax, sticky):
    """`bf16::norm_table`: the zero row, the 255 exactly-representable
    magnitudes, and 256 significands per binade (twice with the sticky bit)."""
    return 1 + 255 + (kmax + 1) * 256 * (1 + int(sticky))


def frac_rows(kmax):
    """`Tables::new`: one row per (ph, r) pair, `ph = 2^k` for `k <= kmax`."""
    return 1 + sum(1 << k for k in range(kmax + 1))


def _rq_table_rows(bf16=True):
    """`requant.rs`, `rq_table`: the division table's admissible rows."""
    n = 0
    for tt in range(TT_RANGE):
        for ms in (range(128) if bf16 else (0,)):
            sden = 128 + ms if bf16 else 1
            for m7 in range(128):
                m = 128 + m7
                q = (((m << tt) + sden) // (2 * sden) if bf16
                     else ((m << (tt + 1)) + 512) // 1024)
                if 1 <= q <= 127:
                    n += 1
    return n


def _qdiv_table_rows(bits):
    """`rope.rs`, `qdiv_table`: the same shape at the quantize side's width."""
    qmax = (1 << (bits - 1)) - 1
    n = 0
    for tt in range(TT_RANGE + bits - 8):
        for m7 in range(128):
            q = (((128 + m7) << (tt + 1)) + 512) // 1024
            if 1 <= q <= qmax:
                n += 1
    return n


def embed_layout(d, tokens, vocab, target):
    """`embed.rs`, `EmbLayout::choose`: the smallest power-of-two block count
    that brings a block at or under the target.  Returns (pe, k, log_use,
    log_tab)."""
    k = 1
    while d % k or d // k > target:
        k *= 2
        assert k <= d, f"no block count divides d = {d} at target {target}"
    log_v = log_rows_for(vocab)
    return d // k, k, log_rows_for(tokens * k), log_v + (k.bit_length() - 1)


def range_widths(p):
    """`compose.rs`, `Plan::new`: the range tables a segment registers.  Four of
    the eight base widths are the run's own calibration of its witness, which is
    where an architecture-only count stops being possible -- see
    `witness_terms`."""
    w = [p["k_bits"], p["q_bits"], 8, p["z_lo_bits"], p["z_hi_bits"], p["ao_rem_bits"],
         p["ao_acc_lo_bits"], p["ao_acc_hi_bits"]]
    # the product form's `ahn = ahi num` table, only in that form; the set below
    # takes it away again wherever another width is already `30 - lo`
    if p.get("ao_acc_prod"):
        w.append(ao_prod_bits(p))
    name, k = attn_variant(p)
    if name == "intsplit":
        w.append(k)
    if any(m["variant"] == "fold" for m in p["mats"]):
        w += [13, 16, p["fold_rem_bits"], 7, p["ao_q_bits"]]
    if p["ao_q_bits"] > 7:
        w.append(p["ao_q_bits"] + 1)
    if p.get("residual_stream"):
        w.append(15)
    return sorted(set(w))


def ao_prod_bits(p):
    """`compose.rs`, `ao_prod_bits`: the product form's range table, `ahn = ahi
    num` checked below `2^(30 - lo)`."""
    return max(30 - p["ao_acc_lo_bits"], 0)


def e_window(p):
    """`compose.rs`, `Plan::e_window`: the exponent window the attention output's
    `EPOW` table covers, `e = jv - jo` over the ranges the run measured."""
    return p["ao_jv_min"] - p["ao_jo_max"], p["ao_jv_max"] - p["ao_jo_min"]


def table_rows(p):
    """Every lookup table's row count, by name.  One base multiplicity column
    and one QM31 interaction column each, so a table of `2^r` rows costs
    `5 * 2^r` and nothing else."""
    sticky = p["round"] == "even"
    rows = {"exp": 1 << p["log_table"],
            "norm": 1 << log_rows_for(norm_rows(p["kmax"], sticky)),
            "frac": 1 << log_rows_for(frac_rows(p["kmax"])),
            "mul": 1 << MUL_LOG_ROWS,
            "qnz": 1 << p["ao_q_bits"]}
    for w in range_widths(p):
        rows[f"range{w}"] = 1 << w
    lo, hi = e_window(p)
    rows["epow"] = 1 << log_rows_for(hi - lo + 1)
    g = p.get("residual_stream")
    if g:
        # `residual_stream_table_kinds`: pow, smul, val, and one per unary kind
        rows["residual_stream0"] = 1 << log_rows_for(POW_ROWS)
        rows["residual_stream1"] = 1 << log_rows_for(SMUL_ROWS)
        log_val = log_rows_for(VAL_ROWS)
        rows["residual_stream2"] = 1 << log_val
        n_un = 2 if g.get("head_norm") else 3      # Square, NormR, and the silu
        for i in range(n_un):
            rows[f"residual_stream{3 + i}"] = 1 << log_val
    return rows


def requant_table_rows(p):
    """`compose.rs`'s requantize table block: per scale variant a division table
    and a scale table, then the sign table, then the quantize family's own
    division tables and its `(q, nz)` pair table."""
    rows = {}
    if not p.get("residual_stream"):
        return rows
    variants = sorted({"pow2" if _seam_variant(p, name) == "pow2" else "bf16"
                       for name in _seam_names(p)})
    for v in variants:
        rows[f"rq_{v}"] = 1 << log_rows_for(_rq_table_rows(v == "bf16"))
        rows[f"scl_{v}"] = 1 << 15
    if variants:
        rows["rqv"] = 1 << 9
    widths = _quant_widths(p)
    for w in widths:
        rows[f"qdiv{w}"] = 1 << log_rows_for(_qdiv_table_rows(w))
    if widths:
        rows["qnz_q"] = 1 << (max(widths) - 1)
    return rows


V_BITS = 8       # `rope.rs`, `V_BITS`: the value side's quantize width


def _quant_widths(p):
    return sorted({bits for *_, bits in _quant_sides(p)})


def _quant_sides(p):
    """`compose.rs`: q, k and v get a quantize component, except a `v` whose
    dequantize is the fold -- the fold IS the quantize.  Each is (name, values,
    tokens, width): the quantize width has to divide the head's own width, which
    a latent attention states separately for the two sides.

    A block may name the sides itself (`quant_sides`), which is what a latent
    attention does: its key side is one latent per key and not one key a head,
    and its absorbed form quantizes no value at all, because the value the grid
    reads IS that latent."""
    if not p.get("residual_stream"):
        return []
    if p.get("quant_sides") is not None:
        return [(q["name"], q["n_out"], q["tokens"], q["hd"],
                 p[q["bits"]] if isinstance(q["bits"], str) else q["bits"])
                for q in p["quant_sides"]]
    rope = p.get("rope")
    if not rope:
        return []
    out = []
    for name, heads, bits in (("q", p["nh"], p["q_bits"]),
                              ("k", rope["nkv"], p["k_bits"]),
                              ("v", rope["nkv"], V_BITS)):
        m = _mat(p, name)
        if m is None:
            continue
        if name == "v" and m["variant"] == "fold":
            continue
        out.append((name, m["n_out"], m["tokens"], p["hd"], bits))
    return out


def _mat(p, name):
    for m in p["mats"]:
        if m["name"] == name:
            return m
    return None


def _seam_names(p):
    """The requantize seams a segment builds, read off the groups it carries.

    A block may name them instead (`seams`), which is what a mixture layer whose
    feed-forward matmuls are all routed has to do: it requantizes the residual
    once before its experts and once before their down projections, and neither
    of those seams is a group of this segment's own name.  A latent attention
    names them too, and its own are narrower than the residual: a low-rank read
    is requantized at the rank it wrote."""
    if p.get("seams") is not None:
        return [s[0] if isinstance(s, (tuple, list)) else s for s in p["seams"]]
    g = p.get("residual_stream") or {}
    names = ["head"] if g.get("head_norm") else ["qkv", "gateup", "down"]
    return [n for n in names if any(g2 == n for g2, _ in _freivalds_groups(p))]


def _seam_members(p, name):
    """The matmuls one seam feeds.  A seam a block states carries its own list;
    a seam of a dense layer is the Freivalds group of the same name."""
    for s in (p.get("seams") or []):
        if isinstance(s, (tuple, list)) and s[0] == name:
            return tuple(s[2])
    return dict(FREIVALDS_GROUPS).get(name, ())


def _seam_variant(p, name):
    for mn in _seam_members(p, name):
        m = _mat(p, mn)
        if m is not None:
            return m["variant"]
    return "chain"


def _seam_width(p, name):
    for s in (p.get("seams") or []):
        if isinstance(s, (tuple, list)) and s[0] == name:
            return s[1]
    g = p["residual_stream"]
    return g["dff"] if name == "down" else g["d"]


def _freivalds_groups(p):
    """The groups this segment builds: those whose matmuls it carries.

    A matmul that names its own `group` is checked on its own and is not in the
    group its name would put it in.  That is what a routed expert of a mixture
    is: its matmul covers only the tokens routed to it, so it does not share a
    committed input with the rest and the check over it is its own."""
    out = []
    for name, members in FREIVALDS_GROUPS:
        idx = [m for m in p["mats"] if m["name"] in members and not m.get("group")]
        if idx:
            out.append((name, idx))
    own = {}
    for m in p["mats"]:
        if m.get("group"):
            own.setdefault(m["group"], []).append(m)
    for name in sorted(own):
        out.append((name, own[name]))
    return out


def _repeat(mats):
    """`repeat` on a matmul means that many matmuls of exactly this shape, each
    its own component: the routed experts of one mixture layer, which the
    uniform-routing assumption makes identical."""
    reps = {m.get("repeat", 1) for m in mats}
    assert len(reps) == 1, f"a group's matmuls repeat unequally: {sorted(reps)}"
    return reps.pop()


# ------------------------------------------------------------------ the count --
def exact_cells(p):
    """Every committed cell of one composed segment, from its `params` block.

    Returns a dict with `base`, `interaction`, `boundary` and `fresh` totals and
    a `components` map from family name to (base, interaction).  Compare it with
    a recorded chunk's `cells`: `base` is `cells.base`, `interaction` is
    `cells.interaction`, and `fresh` is `cells.fresh` -- base plus interaction
    plus the boundary trees, which no `family_cells` entry names."""
    comp = {}
    s, c, hd, nh = p["s"], p["c"], p["hd"], p["nh"]
    sticky = p["round"] == "even"
    head = bool((p.get("residual_stream") or {}).get("head_norm"))

    # ---- the attention key tiles
    comp["key_tiles"] = key_tile_cells(p)

    # ---- the dequantize components, one per matmul
    deq_b = deq_i = 0
    mat_lays = {}
    for j, m in enumerate(p["mats"]):
        rep_m = m.get("repeat", 1)
        pr, _, lr = mat_layout(m["n_out"], m["tokens"], p["deq_target"])
        mat_lays[j] = (pr, lr)
        deq_b += rep_m * (deq_columns(m["variant"], sticky, pr) << lr)
        # `mat_interaction`: the two Horner accumulators `z` and `h`, the output
        # claim the wiring family reads, and -- on a matmul a requantize seam
        # feeds, and only if it is not folded -- the x-side scale's fold
        qm = 2 + int(_has_output_claim(p, m["name"])) + int(_has_scale_claim(p, m))
        if not p.get("gkr_dequant"):
            qm += -(-deq_looks(m["variant"], pr) // 2)
        deq_i += rep_m * ((M31_PER_QM31 * qm) << lr)
    comp["dequant"] = (deq_b, deq_i)

    # ---- the attention output
    ao_pr, _, ao_lr = ao_layout(nh, s, hd_v(p), p["ao_target"])
    # `layer.rs`, `ao_component`: the row's `den` and `joff`, then per slot
    # acc, sg, qm, nz2, denp, d, num, pj, alo -- and `mag` tenth under `acc_mag`,
    # and `ahn` eleventh under `acc_prod`, which is built on the magnitude form
    # (`trace.rs`, `slot_cols`)
    ao_mag, ao_prod = bool(p.get("ao_acc_mag", False)), bool(p.get("ao_acc_prod", False))
    assert ao_mag or not ao_prod, "the product form is built on the magnitude form"
    ao_cols = 2 + ao_pr * (9 + int(ao_mag) + int(ao_prod))
    # seven lookups per slot, and the product form's `ahn` eighth (`slot_looks`)
    ao_qm = 5 if p.get("gkr_attnout") else 5 + -(-(7 + int(ao_prod)) * ao_pr // 2)
    comp["attnout"] = (ao_cols << ao_lr, (M31_PER_QM31 * ao_qm) << ao_lr)

    # ---- the Freivalds checks, one per group of matmuls sharing an input
    fv_b = fv_i = 0
    fv_lays = []
    for name, mats in _freivalds_groups(p):
        n_in = mats[0]["n_in"]
        rep_g = _repeat(mats)
        pk, pq, lr = freivalds_layout(n_in, max(m["n_out"] for m in mats),
                                      mats[0]["tokens"], p["freivalds_p"])
        fv_lays.append((name, pk, pq, lr, len(mats)))
        fv_b += rep_g * (pq << lr)
        # one accumulator pair per matmul, plus the group's own `z` and its
        # claim over the committed activations
        qm = 2 * len(mats) + 2
        if not p.get("gkr_freivalds"):
            qm += -(-pq // 2)
        fv_i += rep_g * ((M31_PER_QM31 * qm) << lr)
    comp["freivalds"] = (fv_b, fv_i)

    # ---- the residual stream
    rs_b = rs_i = 0
    rs_comps = residual_stream_components(p) if p.get("residual_stream") else []
    for _tag, cols, lr, bindings, looks in rs_comps:
        rs_b += cols << lr
        qm = bindings + (0 if p.get("gkr_residual_stream") else -(-looks // 2))
        rs_i += (M31_PER_QM31 * qm) << lr
    comp["residual_stream"] = (rs_b, rs_i)

    # ---- the rotation's quantize components
    q_b = q_i = 0
    for _name, n_out, toks, width, _bits in _quant_sides(p):
        pr, _, lr = q_layout(n_out, width, toks, p["deq_target"])
        # `rope.rs`, `Q_PER_SLOT`: ef, m7, q, nz, sm, sg, cl
        q_b += (7 * pr) << lr
        # five lookups per slot, batched in pairs, plus the two Horner claims
        q_i += (M31_PER_QM31 * (-(-5 * pr // 2) + 2)) << lr
    comp["quant"] = (q_b, q_i)

    # ---- the requantize seams
    rq_b = rq_i = 0
    for name in _seam_names(p):
        n = _seam_width(p, name)
        pr, _, lr = mat_layout(n, s, p["deq_target"])
        # `requant.rs`: the row's max pattern, the scale (one column for a
        # power-of-two scale, two for a bf16 one), seven columns per slot, and
        # the selector sum last
        n_scale = 1 if _seam_variant(p, name) == "pow2" else 2
        rq_b += (2 + n_scale + 7 * pr) << lr
        # `rq_interaction`: five lookups per slot and the per-token scale's own,
        # `5 * pr + 1` in all, batched in pairs (`plogup::paired`, `div_ceil`),
        # and three Horner claims -- the two bindings and the scale.  The odd
        # lookup has to be inside the `div_ceil` and not beside it: at an even
        # `pr` the two readings agree, and every recorded seam has one, so what
        # separates them is a feed-forward width whose layout puts an odd number
        # of slots in a row (13,824 at 27 a row, Qwen2.5-14B).
        rq_i += (M31_PER_QM31 * (-(-(5 * pr + 1) // 2) + 3)) << lr
    comp["requant"] = (rq_b, rq_i)

    # ---- the output head's argmax, which `family_cells` does not name (#159)
    am_b = am_i = 0
    if p.get("prove_argmax"):
        m = _mat(p, "lm_head")
        pr, _, lr = mat_layout(m["n_out"], m["tokens"], p["deq_target"])
        # `argmax.rs`, `component`: the row maximum and its token, then a key
        # and an indicator per slot, then the running selector sum
        am_b = (2 * pr + 3) << lr
        # the binding pair, and one range lookup per slot batched in pairs
        am_i = (M31_PER_QM31 * (2 + -(-pr // 2))) << lr
    comp["argmax"] = (am_b, am_i)

    # ---- the embedding gadget on the first segment of the stack
    em_b = em_i = 0
    if p.get("embed_vocab"):
        pe, k, log_use, log_tab = embed_layout(
            (p["residual_stream"] or {})["d"], s, p["embed_vocab"], p["embed_per_row"])
        # `embed.rs`: the token id and an (sm, ef) pair per channel on the use
        # side; one multiplicity column on the table side
        em_b = ((1 + 2 * pe) << log_use) + (1 << log_tab)
        # `1 + pe` lookups per use row batched in pairs, two Horner claims, and
        # the table's own side
        em_i = (M31_PER_QM31 * (-(-(1 + pe) // 2) + 2)) << log_use
        em_i += M31_PER_QM31 << log_tab
    comp["embed"] = (em_b, em_i)

    # ---- the tables: one base multiplicity column and one QM31 column each
    rows = sum(table_rows(p).values())
    comp["tables"] = (rows, M31_PER_QM31 * rows)
    rq_rows = sum(requant_table_rows(p).values())
    comp["requant_tables"] = (rq_rows, M31_PER_QM31 * rq_rows)

    # ---- the GKR link: 16 M31 columns per distinct height it carries
    heights = link_heights(p, mat_lays, fv_lays, rs_comps, ao_lr)
    comp["gkrlink"] = (0, 16 * sum(1 << h for h in heights))

    base = sum(b for b, _ in comp.values())
    inter = sum(i for _, i in comp.values())
    boundary = boundary_cells(nh, hd, (p.get("residual_stream") or {}).get("d", 0),
                              s, c, head=head, kv_per_key=cache_per_key(p))
    return {"components": comp, "base": base, "interaction": inter,
            "boundary": boundary, "fresh": base + inter + boundary,
            "table_rows": rows}


def _has_output_claim(p, name):
    """`compose.rs`, `Plan::has_output_claim`: the wiring family publishes one
    for every matmul whose output a component reads, and the head publishes the
    logits claim."""
    g = p.get("residual_stream")
    if g and not g.get("head_norm"):
        if name in ("o", "down", "gate", "up"):
            return True
        if p.get("rope") and name in ("q", "k", "v"):
            return True
        # a latent attention's five matmuls are read the same way a dense
        # layer's seven are: every one of them by the wiring family, a norm, a
        # rotation or the attention output
        if p.get("latent"):
            return True
    return bool(p.get("logits_claim")) and name == "lm_head"


def _has_scale_claim(p, m):
    """`compose.rs`: the x-side scale's fold, on the matmuls a requantize seam
    feeds -- and never on a folded one, whose second shared column is not the
    exponent the claim would compare."""
    if m["variant"] == "fold":
        return False
    for name in _seam_names(p):
        if m["name"] in _seam_members(p, name):
            return True
    return False


def link_heights(p, mat_lays, fv_lays, rs_comps, ao_lr):
    """`gkr_key_tile.rs`, `sites_of`: the distinct trace heights the batched
    reduction reads, which is what the link pays 16 committed columns for."""
    hs = set()
    if p.get("gkr_key_tiles") or p.get("gkr_bands"):
        hs.update(lay["log_rows"] for _n, lay, _acc, _look in tile_components(p))
        for insts, c_tile, rows_comp, _diag in _key_tile_shapes(p):
            if rows_comp:
                hs.add(row_layout(insts * p["s"], p["rw"])[1])
    if p.get("gkr_dequant"):
        hs.update(lr for _, lr in mat_lays.values())
    if p.get("gkr_residual_stream"):
        hs.update(lr for _, _, lr, _, _ in rs_comps)
    if p.get("gkr_freivalds"):
        hs.update(lr for _, _, _, lr, _ in fv_lays)
    if p.get("gkr_attnout"):
        hs.add(ao_lr)
    return sorted(hs)


def attn_variant(p):
    """`attn_witness.rs`, `Variant::parse`: `intexp`, `intsplit<k>` or `bf16x`.
    Returns (name, k) with `k` only for `intsplit`."""
    v = p["variant"]
    if v in ("intexp", "bf16x"):
        if v == "bf16x":
            raise SystemExit(
                "exact_cells: the `bf16x` attention variant commits no row maximum, "
                "so its per-token component is a different width; no recorded "
                "composed segment uses it and this module does not describe it.")
        return v, None
    if v.startswith("intsplit"):
        return "intsplit", int(v[len("intsplit"):])
    raise SystemExit(f"exact_cells: unknown attention variant {v!r}")


def key_tile_cells(p):
    """The key tile family of one segment, from the three grids themselves.

    `key_tile.py`'s closed form is this count where nothing is padded -- `hd`,
    `S` and `C` powers of two and both chunk counts dividing what they are read
    against.  Here the grids are built, so a context that is not a power of two
    (the output head's `C = 3584`, whose key grid carries 32,768 inert rows)
    counts right instead of counting `saturated`."""
    base = inter = 0
    gkr = p.get("gkr_key_tiles", False) or p.get("gkr_bands", False)
    for n, lay, acc, look in tile_components(p):
        base += n << lay["log_rows"]
        # the LogUp running sums, batched in pairs, where the reduction is off
        qm = acc + (0 if gkr else M31_PER_QM31 * -(-look // 2))
        inter += qm << lay["log_rows"]
    for insts, _c_tile, rows_comp, _diagonal in _key_tile_shapes(p):
        if rows_comp:
            # the per-token component: one row maximum, one denominator and its
            # two limbs per token, `rw` tokens per row
            rw, lr = row_layout(insts * p["s"], p["rw"])
            base += (4 * rw) << lr
            ri = 8 + (0 if gkr else M31_PER_QM31 * 2 * rw)
            inter += ri << lr
    return base, inter


def cache_per_key(p):
    """What one key costs the boundary trees, `2 nh hd` where a head holds a
    whole K and a whole V and the latent's own width where it does not."""
    lat = p.get("latent")
    return None if not lat else lat["cache_per_key"]


def cache_instances(p, insts):
    """How many copies of the key and the value cache one key tile of `insts`
    attention heads commits.

    The AIR commits one per HEAD: `Shape::chans()` and `Shape::keys()` both carry
    `insts` (#178).  An absorbed latent attention is the one shape where that is
    not what the model holds -- its cache is ONE latent for every head at once --
    so a key tile that holds it holds it once, and the columns that hold it are
    their own component at that instance count rather than a share of the
    component the queries and the scores are in."""
    lat = p.get("latent")
    return 1 if lat and lat.get("cache_shared") else insts


def tile_components(p):
    """The components of every key tile, as (committed columns, layout, Horner
    and claim columns, LogUp running sums).

    `attn_trace.rs`, `commit_base`: the channel component commits the key cache
    and the queries; the grid commits the score, its numerator, the `intsplit`
    variant's auxiliary column per score, and the value cache.
    `interaction_phase`: three Horner accumulator pairs on the channel component,
    six on the grid, and stream NN's root-equality claims -- Q on every bound key
    tile, K and V on the diagonal block.

    Where the cache is committed at its own instance count (an absorbed latent)
    the cache columns are a second channel component and a second grid, each
    built at that count and each carrying its own accumulators and the claim over
    its own columns.  That is the one place this module charges a component it
    has not seen: it is 12 and 24 columns over a tile whose rows are the cache's
    and not the heads', and `--selftest` prints what it comes to."""
    bind = p.get("rope") is not None
    split = attn_variant(p)[0] == "intsplit"
    q_claim = M31_PER_QM31 if bind else 0
    out = []
    for insts, c_tile, _rows_comp, diagonal in _key_tile_shapes(p):
        kvi = cache_instances(p, insts)
        kv_claim = M31_PER_QM31 if bind and diagonal else 0
        lc, lk = chan_layout(p, insts, c_tile), key_layout(p, insts, c_tile)
        n_aux = lk["q"] if split else 0
        if kvi == insts:
            out.append((lc["pk"] + lc["pq"], lc, 12 + q_claim + kv_claim,
                        lc["pk"] + lc["pq"]))
            out.append((2 * lk["q"] + n_aux + lk["pv"], lk, 24 + kv_claim,
                        lk["q"] + n_aux + lk["pv"]))
            continue
        lcc, lkc = chan_layout(p, kvi, c_tile), key_layout(p, kvi, c_tile)
        out.append((lc["pq"], lc, 12 + q_claim, lc["pq"]))
        out.append((lcc["pk"], lcc, 12 + kv_claim, lcc["pk"]))
        out.append((2 * lk["q"] + n_aux, lk, 24, lk["q"] + n_aux))
        out.append((lkc["pv"], lkc, 24 + kv_claim, lkc["pv"]))
    return out


def row_layout(toks, rw_target):
    """`layout.rs`, `Layouts::choose`: the per-token component tiles exactly, so
    the tokens per row come down until the row count lands on the SIMD floor
    rather than padding up to it.  Returns (rw, log_rows)."""
    floor = 1 << MIN_LOG_ROWS
    rw = max(1, min(rw_target, toks // floor))
    rows = max(floor, -(-toks // rw))
    assert rw * rows == toks, f"the per-token component does not tile: rw={rw}"
    return rw, max(MIN_LOG_ROWS, (rows - 1).bit_length())


def _grid_chunks(groups, log_rows, cap):
    """`layout.rs`, `Grid::new`: `chunks` rows per group, capped, with the slack
    carried as inert lead rows."""
    rows = 1 << log_rows
    if rows < groups or groups == 0:
        return None
    return min(rows // groups, max(cap, 1))


def hd_qk(p):
    """The channel width the score grid runs over: the head width, unless the
    attention states its two sides separately (a latent attention does)."""
    return p.get("hd_qk", p["hd"])


def hd_v(p):
    """The width the value cache and the attention output run over."""
    return p.get("hd_v", p["hd"])


def chan_layout(p, insts, c_tile):
    """`layout.rs`, `LayoutChan::choose`: one row group per (instance, channel),
    the height whose keys per row is nearest `chan_target`."""
    best = None
    for lr in range(MIN_LOG_ROWS, 27):
        chunks = _grid_chunks(hd_qk(p) * insts, lr, min(c_tile, p["s"]))
        if chunks is None:
            continue
        pk = -(-c_tile // chunks)
        key = (abs(pk - p["chan_target"]), 1 << lr)
        if best is None or key < best[0]:
            best = (key, {"log_rows": lr, "chunks": chunks, "pk": pk,
                          "pq": -(-p["s"] // chunks)})
    assert best is not None, "no channel layout"
    return best[1]


def key_layout(p, insts, c_tile):
    """`layout.rs`, `LayoutKey::choose`: one row group per (instance, key),
    scored on the chunk count, and the token axis has to tile exactly."""
    want = -(-p["s"] // p["grid_target"])
    best = None
    for lr in range(MIN_LOG_ROWS, 27):
        chunks = _grid_chunks(c_tile * insts, lr, min(p["s"], hd_v(p)))
        if chunks is None or p["s"] % chunks:
            continue
        key = (abs(chunks - want), 1 << lr)
        if best is None or key < best[0]:
            best = (key, {"log_rows": lr, "chunks": chunks,
                          "q": -(-p["s"] // chunks), "pv": -(-hd_v(p) // chunks)})
    assert best is not None, "no key layout"
    return best[1]


def _key_tile_shapes(p):
    """The segment's key tiles: `c / s` prefix tiles of `s` keys per head group
    (stream AA's `--kv-boundary` tiling) and the diagonal block, with the
    per-token component riding on the group's first tile."""
    groups = p.get("head_groups")
    if groups is None:
        raise SystemExit(
            "exact_cells: this params block carries no `head_groups`, and how the "
            "attention heads are grouped into key tiles decides the grids' heights "
            "and therefore what the GKR link pays for.  It is the prover's "
            "`--head-groups`, a run knob; run outputs before 2026-09-08 do not "
            "record it.")
    out = []
    for insts in groups:
        n_pre = 0 if not p["c"] else (p["c"] // p["s"] if p.get("kv_boundary") else 1)
        sub_c = p["s"] if p.get("kv_boundary") else p["c"]
        for k in range(n_pre):
            out.append((insts, sub_c, k == 0, False))
        if p.get("diagonal_block"):
            out.append((insts, p["s"], n_pre == 0, True))
    return out




# --------------------------------------------- what is not the architecture --
def witness_terms(p):
    """The terms of the count that a `params` block carries but a model and a
    context do not: the run measured them on its own witness.  Returns each
    one's cells and its share of `fresh`, so a caller who has to ASSUME them can
    see what the assumption is worth."""
    total = exact_cells(p)["fresh"]
    sticky = p["round"] == "even"
    out = {}
    # the master exponent table: sized to cover the scores this witness produced
    out["log_table"] = 5 * (1 << p["log_table"])
    # `kmax` sizes the normalisation table and the fraction table
    out["kmax"] = 5 * ((1 << log_rows_for(norm_rows(p["kmax"], sticky)))
                       + (1 << log_rows_for(frac_rows(p["kmax"]))))
    # the attention output's measured exponent ranges size three range tables
    # and the exponent window
    lo, hi = e_window(p)
    out["ao_ranges"] = 5 * ((1 << p["ao_acc_hi_bits"]) + (1 << p["ao_rem_bits"])
                            + (1 << log_rows_for(hi - lo + 1)))
    # the accumulator's low limb and the quantize width, one range table each,
    # and the pair table the quantize width keys.  The product form's `ahn`
    # table is `30 - lo` bits, so the low limb sizes that one too.
    out["ao_acc_lo_bits"] = 5 * ((1 << p["ao_acc_lo_bits"])
                                 + ((1 << ao_prod_bits(p)) if p.get("ao_acc_prod") else 0))
    out["ao_q_bits"] = 5 * ((1 << p["ao_q_bits"])
                            + ((1 << (p["ao_q_bits"] + 1)) if p["ao_q_bits"] > 7 else 0))
    # the fold's own remainder range table, on a segment that folds at all
    if any(m["variant"] == "fold" for m in p["mats"]):
        out["fold_rem_bits"] = 5 * (1 << p["fold_rem_bits"])
    return {k: (v, v / total) for k, v in out.items()}


# ------------------------------------ what a shape's segment looks like --
# A `params` block is what `exact_cells` counts, and a run writes one.  The
# three builders below write one from a model's shape instead, so that a
# projection at a shape no run has proved is counted by the same function that
# reproduces every recorded chunk.  What they cannot get from the shape is the
# witness-calibrated fields and the run knobs: both are arguments, and
# `frontier.py` names them and states what they are worth.


class Refused(Exception):
    """This shape's segment is not a `params` block: the prover has no layout for
    it.  Carries the reason, which a caller prints instead of a number."""


# The knobs the two headline runs were proved at, read off the fixture's own
# shared block rather than typed in again: the layout targets, the quantize
# widths and the attention variant.  They are the prover's flags, not the
# model's shape, so a projection carries the recorded statement's.
_KNOB_FIELDS = ("deq_target", "ao_target", "freivalds_p", "chan_target", "grid_target",
                "rw", "nmax", "z_min", "z_hi_bits", "z_lo_bits", "k_bits", "q_bits",
                "round", "variant", "layer_boundary", "diagonal_block")
# `residual_stream.a`: committed leaves per row of the wiring family, 32 on every
# recorded segment, and `embed_per_row`: the embedding gadget's target width.
RS_A = 32
EMBED_PER_ROW = 256


def statement_knobs():
    """The recorded statement's run knobs, from `exact_chunks.json`'s shared block."""
    d = json.loads(CHUNKS.read_text())["params"]
    return {k: d[k] for k in _KNOB_FIELDS if k in d}


def head_groups_for(nh, rule="half-quarter-quarter"):
    """How the attention heads are grouped into key tiles.  The prover's
    `--head-groups`, a run knob: the two headline runs proved 32 heads as 16, 8, 8
    and the micro-benchmark proved 4 as 2, 2.  The rule here reproduces the
    headline runs' grouping and is what a projection assumes."""
    if rule == "one-group":
        return [nh]
    if rule != "half-quarter-quarter":
        raise Refused(f"no such head-group rule: {rule!r}")
    if nh % 4:
        raise Refused(f"{nh} attention heads do not split into a half and two "
                      f"quarters, which is how the recorded runs grouped theirs")
    return [nh // 2, nh // 4, nh // 4]


def expected_distinct_experts(experts, topk, s):
    """`E * (1 - (1 - k/E)^S)`: how many of a mixture's experts a block of `s`
    tokens touches, when each token's top-k is uniform over the experts.  It is
    what the segment's preprocessed trace has to hold, and how many routed
    matmuls the segment builds."""
    if experts <= 1 or topk <= 0:
        return 0.0
    return experts * (1.0 - (1.0 - topk / experts) ** s)


def routed_blocks(s, topk, touched):
    """How one segment's routed matmuls are laid out, under uniform routing.

    `s * topk` token slots spread over `touched` experts.  A matmul's rows are
    `tokens * chunks` and `chunks` divides the output width, so a token count
    with an odd factor the width cannot cancel does not tile at any width
    (`MatLayout::choose_padded` refuses a lead of more than a quarter of the
    rows, and `5 * 2^j` is always three eighths short of the next power of two).
    So the block a routed matmul covers is the largest power of two at or below
    the uniform share, and the segment builds as many blocks as that takes --
    the token count is conserved, and every block tiles.  Returns (tokens per
    block, blocks)."""
    slots = s * topk
    share = int(slots / max(1.0, touched))
    te = 1 << max(0, share.bit_length() - 1)
    if slots % te:
        raise Refused(f"{slots} routed token slots do not divide into blocks of "
                      f"{te}, so the routed matmuls do not tile")
    return te, slots // te


def has_layout(n_out, tokens):
    """Whether `MatLayout::choose_or_padded` has a layout for `tokens` tokens of
    `n_out` outputs at all: an exact tiling, or a padded one whose lead is at most
    a quarter of the rows.  `_mat_layout_padded` asserts where this is false."""
    for pr in range(1, n_out + 1):
        if n_out % pr:
            continue
        active = tokens * (n_out // pr)
        log_rows = max(MIN_LOG_ROWS, (active - 1).bit_length())
        if ((1 << log_rows) - active) / (1 << log_rows) <= 0.25:
            return True
    return False


def measured_blocks(distribution, widths):
    """How a routing MEASURED on a real run is laid out, at the output widths a
    routed expert's matmuls are built at.

    `distribution` is `[(tokens, matmuls), ...]`: that many experts of the block
    took that many tokens each.  An expert is ONE matmul wherever the layout
    chooser has a layout for its own token count at every one of those widths --
    463 tokens of a 4,096-output `down` matmul are 512 rows with 49 of lead, which
    `choose_padded` takes -- and that is what the prover builds.  Where it has
    none, the count falls back to the rule `routed_blocks` lays the uniform share
    out by: the largest power of two at or below the count, then the remainder the
    same way.  268 tokens are such a count, because a `down` matmul's width has no
    divisor but a power of two, so its rows are the token count shifted and a lead
    of 244 in 512 is over the quarter; the count builds 256 + 8 + 4 for it, three
    matmuls over that expert's own weights.

    Both ends are stated: a single padded matmul commits rows for tokens that are
    not there, and a split pays the fixed side of a matmul once per block.  The
    token count is conserved in the split and rounded up in the padding, and how
    many experts the segment reads is neither: that is `touched`, and it is what
    the preprocessed trace holds the weights of."""
    out = {}
    for tokens, mats in distribution:
        t, n = int(tokens), int(mats)
        if t <= 0 or n <= 0:
            continue
        if all(has_layout(w, t) for w in widths):
            out[t] = out.get(t, 0) + n
            continue
        while t:
            b = 1 << (t.bit_length() - 1)
            out[b] = out.get(b, 0) + n
            t -= b
    return sorted(out.items(), reverse=True)


def layer_mats(d, nh, nkv, hd, dff, tokens):
    """`compose.rs`, the layer plan's seven matmuls: the three projections, the
    attention output, the two gates and the down projection.  `v` is folded --
    its dequantize IS the quantize the attention output reads."""
    return [dict(name="q", n_in=d, n_out=nh * hd, tokens=tokens, variant="chain"),
            dict(name="k", n_in=d, n_out=nkv * hd, tokens=tokens, variant="chain"),
            dict(name="v", n_in=d, n_out=nkv * hd, tokens=tokens, variant="fold"),
            dict(name="o", n_in=nh * hd, n_out=d, tokens=tokens, variant="chain"),
            dict(name="gate", n_in=d, n_out=dff, tokens=tokens, variant="chain"),
            dict(name="up", n_in=d, n_out=dff, tokens=tokens, variant="chain"),
            dict(name="down", n_in=dff, n_out=d, tokens=tokens, variant="chain")]


def head_deq_target(vocab):
    """The dequantize target the output head segment is planned at.

    `MatLayout::choose` needs the row count `tokens * chunks` to be a power of
    two, and `tokens` is one, so `chunks` is a power of two and the narrowest
    row the vocabulary admits is its own odd part.  The two headline runs' head
    segments were proved at exactly that (501 for a vocabulary of 128,256)."""
    odd = vocab
    while odd % 2 == 0:
        odd //= 2
    return odd


def segment_params(kind, d, nh, nkv, hd, dff, vocab, s, c, witness,
                   knobs=None, head_groups=None, gkr=REDUCTIONS_ON, moe=None,
                   latent=None):
    """One composed segment of a model, as the `params` block `exact_cells` counts.

    `kind` is `layer`, `first` (the segment of the stack that carries the
    embedding lookup) or `head` (the output head).  `dff` is the feed-forward
    width every token goes through.  `witness` is the block of witness-calibrated
    fields the caller assumes; `knobs` the run knobs, the recorded statement's by
    default.  Raises `Refused` where the shape is not a `params` block at all.

    `moe` makes it a mixture layer, and then the feed-forward side is THREE
    things rather than one: the shared experts, which are a dense matmul over
    every token and are in `dff`; the routed experts, each a matmul over the
    tokens routed to it alone; and the router's own matmul of `experts` logits.
    `{"experts", "topk", "moe_dff", "touched", "n_in", "router"}` -- `touched` is
    how many routed matmuls the segment builds (`expected_distinct_experts`) and
    `n_in` the width a routed expert reads, which is not always the model's
    (Kimi K3 projects to `routed_expert_hidden_size` first).  `distribution`
    overrides the uniform split with `[(tokens, matmuls), ...]` measured on a real
    run of the model, which is the one way to remove the routing assumption.

    `latent` makes it a multi-head latent attention (`attention/latent.py`): the
    widths a config states, at one of the two forms.  Then the attention side is
    five matmuls and not four, each low-rank read carries a norm, the rotation is
    over the decoupled key alone, and what a key costs the cache is that form's
    own width rather than `2 nkv hd`."""
    if kind not in ("layer", "first", "head"):
        raise Refused(f"no such segment kind: {kind!r}")
    lat = None if latent is None else dict(latent)
    if lat is None:
        if int(nkv) != nkv or nkv < 1:
            raise Refused(f"{nkv} key/value heads: the layer plan commits a whole "
                          f"K and V head per group, so a fractional count is not a "
                          f"`params` block.  Multi-head latent attention is the "
                          f"case this refuses, and `latent=` is how it is stated")
        nkv = int(nkv)
    else:
        # the cache a MODEL holds: one latent for every head under the absorbed
        # form, one K and one V a head under the materialised one
        nkv = 1 if lat["cache_shared"] else nh
    moe = dict(moe or {})
    active = dff + (moe.get("topk", 0) * moe.get("moe_dff", 0))
    for name, n in (("d", d), ("the active feed-forward width", active)):
        if kind != "head" and n % RS_A:
            raise Refused(f"{name} = {n} is not a multiple of {RS_A}, the wiring "
                          f"family's committed leaves per row")
    p = dict(statement_knobs() if knobs is None else knobs)
    p.update(witness)
    # `nkv` is on the block itself and not only on the rotation, because the
    # output head segment carries a key tile family and no rotation, and
    # `kv_cells` prices its cache like any other.  Nothing in `exact_cells`
    # reads it: what the count charges per key is `nh`, which is the point.
    p.update(s=s, c=c, hd=hd, nh=nh, nkv=nkv,
             head_groups=list(head_groups if head_groups is not None
                              else head_groups_for(nh)))
    for r in gkr:
        p[f"gkr_{r}"] = True
    if lat is None:
        mats, spec = layer_mats(d, nh, nkv, hd, dff, s), None
    else:
        spec = latent_attention.layer(lat, d, s, rope=lat.get("rope", True))
        p.update(hd_qk=lat["hd_qk"], hd_v=lat["hd_v"], latent=lat)
        mats = spec["mats"] + [m for m in layer_mats(d, nh, nkv, hd, dff, s)
                               if m["name"] in ("gate", "up", "down")]
    if not dff:
        # a layer whose feed-forward side is entirely routed builds no matmul
        # over all `s` tokens there
        mats = [m for m in mats if m["name"] not in ("gate", "up", "down")]
    if kind == "head":
        mats.append(dict(name="lm_head", n_in=d, n_out=vocab, tokens=s,
                         variant="pow2"))
        p.update(mats=mats, prove_argmax=True, logits_claim=True,
                 deq_target=head_deq_target(vocab),
                 residual_stream=dict(a=RS_A, d=d, dff=0, head_norm=True))
        return p
    if moe.get("experts", 0) > 1:
        w, n_in = moe["moe_dff"], moe.get("n_in") or d
        dist = moe.get("distribution")
        if dist:
            # a measured routing is a routing OF THIS BLOCK: the token slots it
            # spreads are `s` positions' top-k and not another block length's
            slots = sum(int(t) * int(n) for t, n in dist)
            if slots != s * moe["topk"]:
                raise Refused(f"the measured routing spreads {slots} token slots "
                              f"and a block of {s} positions at top-{moe['topk']} "
                              f"has {s * moe['topk']}; it was measured at another "
                              f"block length")
        else:
            te, blocks = routed_blocks(s, moe["topk"], moe["touched"])
            dist = [(te, blocks)]
        moe["distribution"] = dist = measured_blocks(dist, (w, n_in))
        for j, (te, blocks) in enumerate(dist):
            for name in ("gate", "up"):
                mats.append(dict(name=name, n_in=n_in, n_out=w, tokens=te,
                                 variant="chain", group=f"expert_gateup{j}",
                                 repeat=blocks))
            mats.append(dict(name="down", n_in=w, n_out=n_in, tokens=te,
                             variant="chain", group=f"expert_down{j}", repeat=blocks))
        if moe.get("router", True):
            # the gate logits: one matmul of `experts` outputs over every token.
            # The selection on top of them has no layout and is not here; the
            # caller prices it as a stated bound and says so.
            mats.append(dict(name="gate", n_in=d, n_out=moe["experts"], tokens=s,
                             variant="chain", group="router"))
        # both seams are built whatever the groups are called: the residual is
        # requantized once before the experts and once before their outputs
        p["seams"] = (spec["seams"] if spec else ["qkv"]) + ["gateup", "down"]
        p["moe"] = moe
    p.update(mats=mats, kv_boundary=True,
             residual_stream=dict(a=RS_A, d=d, dff=active), rope=dict(nkv=nkv))
    if spec:
        # the rotation, the quantize sides and the two low-rank norms are the
        # latent attention's own, and so are the seams its ranks requantize at
        p["rope"] = spec["rope"]
        p["quant_sides"] = spec["quant_sides"]
        p["residual_stream"]["norms"] = spec["norms"]
        p.setdefault("seams", spec["seams"] + ["gateup", "down"])
    if kind == "first":
        # The magnitude form is what the recorded first segments take, and it is
        # not a property of being first: `params_of` takes it exactly when the
        # offset window does not fit, which is a question about that segment's
        # own accumulator and exponents.  So a witness block that states it wins,
        # and the recorded runs' answer is what a block that does not gets.
        p.update(embed_vocab=vocab, embed_per_row=EMBED_PER_ROW,
                 ao_acc_mag=witness.get("ao_acc_mag", True))
    return p


def cells_of(p, bound=False, kv_shared=False):
    """`exact_cells`, with every layout assertion turned into a refusal that says
    what the shape did to the layout.  `bound=True` counts the same segment with
    every witness-calibrated field at its bound; `kv_shared=True` counts it with
    the keys and values committed once per key/value head (`kv_shared_cells`)."""
    if bound:
        p = dict(p, **witness_bounds())
    try:
        return kv_shared_cells(p) if kv_shared else exact_cells(p)
    except SystemExit as e:
        raise Refused(str(e).replace("exact_cells: ", "")) from None
    except (AssertionError, ZeroDivisionError, ValueError) as e:
        raise Refused(f"the layout does not tile at this shape: {e}") from None


# ------------------------------------------- the keys and values, both ways --
# The AIR commits one K and one V column set per ATTENTION head.  `LayoutChan`
# and `LayoutKey` are built from `Shape::chans()` and `Shape::keys()`, both of
# which carry `insts` = attention heads (`layout.rs`), and `boundary.rs`'s
# `kv_sides` hands on a cache of that same width.  A shape with `nkv < nh` reads
# each key and each value `nh / nkv` times, so a prover that committed them once
# per KEY/VALUE head -- a key tile's K and V columns shared by the score grids
# of the `nh / nkv` heads of the group -- would commit that many fewer cells.
# No AIR in this project builds that layout.  What the two functions below do is
# PRICE it: `kv_cells` counts the same three terms at either instance count, and
# `kv_sharing` reports the pair and their difference.
#
# What moves and what does not.  The score grid, its numerator, the queries the
# channel component re-commits per key tile and the per-token component are per
# QUERY head either way and are untouched.  The K columns of the channel
# component, the V columns of the key grid and the boundary trees that hand the
# cache on are the whole of the term, and each is counted from the layout at
# `insts * nkv / nh` key/value instances rather than divided by anything.
#
# What is NOT in the saving, so that the difference is a floor and not a guess:
# the interaction columns those same columns carry.  With the five reductions on
# -- the statement both headline runs prove -- the K and V LogUp running sums
# are in the batched reduction rather than in the components, and what the link
# pays is one price per distinct trace height, which is a property of a layout
# nobody has built.  With the reductions off those sums are committed and would
# fall too, which this does not count either.


def kv_heads(p):
    """This segment's key/value head count.

    A `params` block a shape builder wrote names it (`nkv`); a run's block names
    it on the rotation, which is the component whose width the sharing sets
    (`rope.nkv`).  A block that names neither is refused rather than assumed to
    share nothing: an output head segment written before this field existed
    carries a key tile family whose instance count no field of it states."""
    n = p.get("nkv")
    if n is None:
        n = (p.get("rope") or {}).get("nkv")
    if n is None:
        raise Refused("this params block names no key/value head count, and how "
                      "many attention heads share one key and one value is not "
                      "something the rest of the block says")
    return int(n)


def kv_instances(p):
    """Each head group's key/value instances, `insts * nkv / nh`.

    A key tile covers a group of attention heads; under the sharing it would
    cover that group's key/value heads, and a group that carries a fractional
    number of them is not a tile the layout can state."""
    nh, nkv = p["nh"], kv_heads(p)
    if nkv < 1 or nh % nkv:
        raise Refused(f"{nh} attention heads over {nkv} key/value heads: the heads "
                      f"of one key/value head are a group of the same size, so a "
                      f"count that does not divide is not this attention")
    out = []
    for insts in p["head_groups"]:
        if insts * nkv % nh:
            raise Refused(f"a head group of {insts} of the {nh} attention heads "
                          f"carries {insts * nkv / nh:g} of the {nkv} key/value "
                          f"heads, and a key tile commits whole ones")
        out.append(insts * nkv // nh)
    return out


def kv_cells(p, per_kv_head=False):
    """The cells this segment commits to hold its key and value cache, counted
    per attention head as the prover builds them and per KEY/VALUE head as the
    sharing allows.  `{"keys", "values", "boundary", "total"}`."""
    if p.get("latent"):
        raise Refused("this segment's cache is a latent and not a K and a V per "
                      "key/value head, so there is nothing here for a prover to "
                      "share: what the two latent-attention forms commit is "
                      "counted as they are, side by side")
    nh = p["nh"]
    n = kv_heads(p) if per_kv_head else nh
    kv_instances(p)                      # refuses a group the sharing cannot tile
    keys = values = 0
    for insts, c_tile, _rows_comp, _diagonal in _key_tile_shapes(p):
        i = insts * n // nh
        lc = chan_layout(p, i, c_tile)
        lk = key_layout(p, i, c_tile)
        # `attn_trace.rs`, `commit_base`: the key cache is the channel
        # component's `pk` columns and the value cache the key grid's `pv`
        keys += lc["pk"] << lc["log_rows"]
        values += lk["pv"] << lk["log_rows"]
    g = p.get("residual_stream") or {}
    head = bool(g.get("head_norm"))
    # `boundary.rs`, `kv_sides`: the keys and values a layer segment hands on,
    # `n * hd` wide over the blocks it publishes and the blocks it read.  Taken
    # as the boundary form at `n` heads less the same form at none, so that the
    # form stays `key_tile.py`'s one and is not written down twice here.  An
    # output head publishes no cache and pays none of this.
    bd = [boundary_cells(m, p["hd"], g.get("d", 0), p["s"], p["c"], head=head)
          for m in (n, 0)]
    return {"keys": keys, "values": values, "boundary": bd[0] - bd[1],
            "total": keys + values + bd[0] - bd[1]}


def cache_cells(p):
    """What this segment commits to HOLD its keys and its values, whatever the
    attention is: the key cache in the channel component, the value cache in the
    key grid, and the boundary trees that hand both on.

    Counted at the instance count the cache is committed at (`cache_instances`),
    so a latent attention that holds one cache for every head is counted holding
    one.  `kv_cells` is this at an attention that holds a K and a V per head, and
    `--selftest` checks the two agree there."""
    keys = values = 0
    for insts, c_tile, _rows_comp, _diagonal in _key_tile_shapes(p):
        kvi = cache_instances(p, insts)
        lc, lk = chan_layout(p, kvi, c_tile), key_layout(p, kvi, c_tile)
        keys += lc["pk"] << lc["log_rows"]
        values += lk["pv"] << lk["log_rows"]
    g = p.get("residual_stream") or {}
    head = bool(g.get("head_norm"))
    # `boundary.rs`, `kv_sides`, as the difference of the boundary form with the
    # cache and without it, so that form stays `key_tile.py`'s one
    bd = [boundary_cells(p["nh"], p["hd"], g.get("d", 0), p["s"], p["c"],
                         head=head, kv_per_key=k)
          for k in (cache_per_key(p), 0)]
    return {"keys": keys, "values": values, "boundary": bd[0] - bd[1],
            "total": keys + values + bd[0] - bd[1]}


def kv_sharing(p):
    """The K/V term both ways, and what the difference is worth on this segment.

    `as_built` is what the prover commits today; `shared` is the same three terms
    at the key/value head count; `saving` is the difference and
    `saving_of_fresh` its share of the segment's committed cells.  A shape with
    `nkv == nh` saves nothing at all, which is the control."""
    built, shared = kv_cells(p), kv_cells(p, per_kv_head=True)
    fresh = exact_cells(p)["fresh"]
    saving = built["total"] - shared["total"]
    return dict(nh=p["nh"], nkv=kv_heads(p), groups=list(p["head_groups"]),
                kv_groups=kv_instances(p), as_built=built, shared=shared,
                fresh=fresh, fresh_kv_shared=fresh - saving, saving=saving,
                share_of_fresh=built["total"] / fresh,
                saving_of_fresh=saving / fresh)


def kv_shared_cells(p):
    """The same segment with the keys and values committed once per key/value
    head: `exact_cells` less what the sharing saves, in the same shape, with the
    `kv` block beside it.  The key tile family and the boundary trees are the
    only two that move."""
    got = exact_cells(p)
    sav = kv_sharing(p)
    grids = ((sav["as_built"]["keys"] + sav["as_built"]["values"])
             - (sav["shared"]["keys"] + sav["shared"]["values"]))
    base, inter = got["components"]["key_tiles"]
    got = dict(got, components=dict(got["components"], key_tiles=(base - grids, inter)),
               base=got["base"] - grids,
               boundary=got["boundary"] - (sav["as_built"]["boundary"]
                                           - sav["shared"]["boundary"]),
               fresh=got["fresh"] - sav["saving"], kv=sav)
    return got


# ----------------------------------------- the assumption and its bound --
# `witness_terms` says which fields of a `params` block the run took off its own
# witness.  A caller with a model and a context but no witness has to assume
# them.  These two read the assumption's two ends off the recorded 8B headline
# runs: the MODE, which is the point estimate, and the widest recorded value,
# which is the bound.  Neither is typed in.
HEADLINE_RUNS = ("2026-09-09-headline-c4096-h100", "2026-09-08-headline-c16384-h100")

WITNESS_FIELDS = ("log_table", "kmax", "ao_acc_hi_bits", "ao_acc_lo_bits",
                  "ao_rem_bits", "ao_q_bits", "ao_jo_min", "ao_jo_max",
                  "ao_jv_min", "ao_jv_max", "fold_rem_bits")
def headline_chunks(kind=None, runs=HEADLINE_RUNS):
    """The recorded segments of the two headline runs, optionally of one kind or
    of several (`kind` is one name or a tuple of them)."""
    want = (kind,) if isinstance(kind, str) else kind
    return [p for p in chunks()
            if p["run"] in runs
            and (want is None or segment_kind(p["params"]) in want)]


def modal_witness(kind=None, runs=HEADLINE_RUNS):
    """Each witness-calibrated field's most common value over those segments, and
    the run and segment it is read from.  `{field: (value, run, segment, n, of)}`."""
    pts = headline_chunks(kind, runs)
    if not pts:
        raise SystemExit(f"no recorded {kind or 'headline'} segment to read a "
                         f"modal witness value off")
    out = {}
    for f in WITNESS_FIELDS:
        counts = {}
        for p in pts:
            counts[p["params"][f]] = counts.get(p["params"][f], 0) + 1
        value = max(sorted(counts), key=lambda v: counts[v])
        first = next(p for p in pts if p["params"][f] == value)
        out[f] = (value, first["run"], first["segment"], counts[value], len(pts))
    return out


def modal_assumption(kind="layer", runs=HEADLINE_RUNS):
    """The witness block a projection assumes, as `exact_cells` reads it: each
    field at the modal value of the recorded runs' segments of that kind.

    `kind="layer"` reads the layer and first segments together, which is the
    block a shape's layer segment is counted at; `kind="head"` reads the output
    head's own, which is a different plan and agrees on a narrower remainder."""
    kinds = ("output head",) if kind == "head" else ("layer", "first segment")
    return {f: v for f, (v, *_) in modal_witness(kinds, runs).items()}


def witness_range(runs=HEADLINE_RUNS):
    """Every value each witness-calibrated field takes over the recorded runs."""
    pts = headline_chunks(None, runs)
    return {f: sorted({p["params"][f] for p in pts}) for f in WITNESS_FIELDS}


_BOUND = {}


def witness_bounds(runs=HEADLINE_RUNS):
    """Each witness-calibrated field at the value that makes a segment DEAREST,
    out of the values the two headline runs carry.

    It is a bound over the evidence and not a theorem: the prover's own automatic
    path caps the exponent table at `2^22` (`attn_witness.rs`, `d_bits.clamp(4,
    22)`), and every value above that is the per-layer calibration the model
    manifest publishes, so the ceiling a projection can state is the widest one a
    run has published.  `--selftest` checks that it dominates every recorded chunk.

    The widest value is not always the dearest, which is why this maximises
    rather than taking a maximum: `Plan::new` registers the range tables as a
    SET, so widening one width can collapse it into a width the segment already
    carries and take a whole table away."""
    if runs in _BOUND:
        return dict(_BOUND[runs])
    rng = witness_range(runs)
    ref = headline_chunks("layer", runs)[0]["params"]
    best = {f: ref[f] for f in WITNESS_FIELDS}
    for _ in range(2):
        for f in WITNESS_FIELDS:
            best[f] = max(rng[f], key=lambda v, f=f: exact_cells(
                dict(ref, **dict(best, **{f: v})))["fresh"])
    _BOUND[runs] = dict(best)
    return dict(best)


def bound_cells(p):
    """The same segment counted with every witness-calibrated field at its bound:
    the largest this segment can be without the witness that sized it."""
    return exact_cells(dict(p, **witness_bounds()))


# ------------------------------------------------------------- the self-test --
def chunks():
    d = json.loads(CHUNKS.read_text())
    return [dict(p, params=dict(d["params"], **p["params"])) for p in d["points"]]


def selftest(argv_points=None):
    """The form against every recorded chunk, to the cell."""
    pts = chunks() if argv_points is None else argv_points
    bad = []
    for p in pts:
        got = exact_cells(p["params"])
        want = (p["base"], p["interaction"], p["fresh"])
        have = (got["base"], got["interaction"], got["fresh"])
        if have != want:
            bad.append((p, want, have, got))
    for p, want, have, got in bad[:12]:
        print(f"  {p['run']} {p['segment']}: base {have[0]:,} vs {want[0]:,} "
              f"({have[0] - want[0]:+,}), interaction {have[1]:,} vs {want[1]:,} "
              f"({have[1] - want[1]:+,})")
        for k, (b, i) in sorted(got["components"].items()):
            rec = p.get("family", {}).get(k)
            if rec is not None and (b, i) != tuple(rec):
                print(f"      {k}: base {b:,} vs {rec[0]:,}, "
                      f"interaction {i:,} vs {rec[1]:,}")
    if bad:
        print(f"exact cells: {len(bad)} of {len(pts)} recorded chunks do not reproduce")
        return 1
    # the negative controls: the comparison above has to reject each recorded
    # number moved by one cell, or "exact" means nothing
    for p in pts:
        got = exact_cells(p["params"])
        for field in ("base", "interaction", "fresh"):
            if got[field] == p[field] + 1:
                print(f"exact cells: the comparison accepts a one-cell difference "
                      f"in `{field}` at {p['segment']}")
                return 1
    def spread(f):
        v = sorted({f(p["params"]) for p in pts})
        return f"{v[0]:,} to {v[-1]:,}" if len(v) > 1 else f"{v[0]:,}"

    kinds = sorted({segment_kind(p["params"]) for p in pts})
    red = sorted({tuple(sorted(k for k in p["params"] if k.startswith("gkr_")
                               and p["params"][k])) for p in pts})
    print(f"exact cells: {len(pts)} recorded segments, exact to the cell")
    print(f"  C = {spread(lambda p: p['c'])}, S = {spread(lambda p: p['s'])}, "
          f"hd = {spread(lambda p: p['hd'])}, nh = {spread(lambda p: p['nh'])}")
    print(f"  segment kinds {', '.join(kinds)}; attention variants "
          f"{', '.join(sorted({p['params']['variant'] for p in pts}))}; roundings "
          f"{', '.join(sorted({p['params']['round'] for p in pts}))}")
    print(f"  {len(red)} reduction configurations, from none of the five to all five")
    print("  a one-cell difference is a failure: checked")
    return (bound_selftest(pts) or seam_selftest() or product_selftest()
            or builder_selftest() or moe_selftest() or gqa_selftest(pts)
            or latent_selftest())


def bound_selftest(pts):
    """The bound has to dominate every recorded chunk, and be reached by one."""
    bounds = witness_bounds()
    low = [p for p in pts if bound_cells(p["params"])["fresh"] < p["fresh"]]
    for p in low[:6]:
        print(f"exact cells: the bound is under the recorded count at {p['run']} "
              f"{p['segment']}: {bound_cells(p['params'])['fresh']:,} < {p['fresh']:,}")
    if low:
        return 1
    loose = [f for f, v in bounds.items()
             if all(p["params"][f] != v for p in headline_chunks())]
    if loose:
        print(f"exact cells: no recorded segment reaches the bound on {loose}, so it "
              f"is not read off the runs it claims to be")
        return 1
    print(f"  the witness bound ({', '.join(f'{f} {v}' for f, v in sorted(bounds.items()))})")
    print(f"  is at or above all {len(pts)} recorded counts and is reached by a "
          f"recorded segment: checked")
    return 0


def moe_selftest():
    """A mixture layer's routed matmuls: `repeat` is exactly writing them out.

    `repeat` says a matmul stands for that many of exactly its own shape, each
    its own component and its own check.  The only thing that makes it sound is
    that it counts what the same block with every matmul written out counts, so
    that is what this compares -- and it checks that the routed blocks conserve
    the token count, which is the one thing the uniform-routing assumption must
    not lose."""
    bad = 0
    wit = modal_assumption()
    shape = dict(d=4096, nh=32, nkv=8, hd=128, vocab=128256, s=512, c=4096)
    moe = dict(experts=128, topk=8, moe_dff=1536, touched=128.0)
    p = segment_params("layer", dff=0, witness=wit, moe=moe, **shape)
    dist = p["moe"]["distribution"]
    slots = sum(t * n for t, n in dist)
    if slots != shape["s"] * moe["topk"]:
        print(f"exact cells: the routed blocks {dist} are {slots} token slots, not "
              f"the {shape['s'] * moe['topk']} the segment routes")
        bad += 1
    te, blocks = dist[0]
    written = dict(p, mats=[])
    for m in p["mats"]:
        n = m.get("repeat", 1)
        one = {k: v for k, v in m.items() if k != "repeat"}
        for i in range(n):
            written["mats"].append(dict(one, group=f"{one['group']}_{i}")
                                   if one.get("group") and n > 1 else dict(one))
    a, b = exact_cells(p), exact_cells(written)
    for field in ("base", "interaction", "fresh"):
        if a[field] != b[field]:
            print(f"exact cells: a routed matmul at repeat {blocks} counts "
                  f"{a[field]:,} of `{field}` and the same matmuls written out "
                  f"count {b[field]:,} ({a[field] - b[field]:+,})")
            bad += 1
    if bad:
        return 1
    print(f"  a mixture layer's {blocks} routed matmuls of {te} tokens count what "
          f"the same matmuls")
    print(f"  written out one by one count, and conserve the {slots:,} token "
          f"slots: checked")
    return 0


def gqa_selftest(pts):
    """The keys and values counted per head and per key/value head.

    Three things.  On every recorded chunk that names a key/value head count the
    two counts stand in exactly the ratio `nkv / nh`: the grids tile there, so
    the cache laid out at the key/value head count is exactly the fraction of the
    cache it shares, and nothing is divided -- the layout is built at the other
    instance count and the integers come out.  On a shape with `nkv == nh` the
    saving is zero to the cell, and CodeLlama-13B is the one shape this
    repository registers with one key/value head per attention head (40 heads of
    128, hidden 5,120, inner 13,824, vocabulary 32,016).  And the segment the
    sharing leaves still adds up: base, interaction and boundary to `fresh`."""
    bad, shares = [], []
    for p in pts:
        q = p["params"]
        try:
            sav = kv_sharing(q)
        except Refused:
            continue                     # a block that names no key/value count
        shares.append((sav["saving_of_fresh"], p, sav))
        nh, nkv = sav["nh"], sav["nkv"]
        if sav["shared"]["total"] * nh != sav["as_built"]["total"] * nkv:
            bad.append(f"{p['run']} {p['segment']}: the K/V term is "
                       f"{sav['as_built']['total']:,} per head and "
                       f"{sav['shared']['total']:,} per key/value head, which is not "
                       f"the {nkv}/{nh} the shape shares")
        got = kv_shared_cells(q)
        if got["base"] + got["interaction"] + got["boundary"] != got["fresh"]:
            bad.append(f"{p['run']} {p['segment']}: the shared segment's trees do "
                       f"not add up to its fresh count")
    # the control: one key/value head per attention head saves nothing
    wit = modal_assumption()
    shape = dict(d=5120, nh=40, nkv=40, hd=128, dff=13824, vocab=32016)
    for c in (4096, 16384):
        p13 = segment_params("layer", s=512, c=c, witness=wit, **shape)
        s13 = kv_sharing(p13)
        if s13["saving"] or kv_shared_cells(p13)["fresh"] != exact_cells(p13)["fresh"]:
            bad.append(f"a shape with {shape['nkv']} key/value heads over "
                       f"{shape['nh']} attention heads saves {s13['saving']:,} cells "
                       f"at C={c:,}, and it shares nothing to save")
    for b in bad:
        print("exact cells:", b)
    if bad:
        return 1
    _, ref, sav = max(shares, key=lambda r: r[0])
    print(f"  the keys and values are {sav['share_of_fresh']:.2%} of the recorded "
          f"segment they cost most on ({ref['segment']}, {sav['nh']} heads over "
          f"{sav['nkv']},")
    print(f"  C={ref['params']['c']:,}) and committing them once per key/value head "
          f"would leave {sav['saving_of_fresh']:.2%}")
    print(f"  of it uncommitted; the two counts are in the ratio the shape shares on "
          f"every recorded")
    print(f"  chunk, and a shape with one key/value head per attention head saves "
          f"nothing: checked")
    return 0


def seam_selftest():
    """The requantize seam's interaction columns, where the recorded chunks cannot
    separate two readings of them.

    `requant.rs`'s `rq_interaction` looks up `5 * per_row + 1` fractions -- five a
    slot and the per-token scale's own -- batches them in pairs (`plogup::paired`,
    a `div_ceil`) and adds three Horner claims: the two bindings and the scale.
    Pairing the per-slot lookups alone and calling the scale a fourth claim gives
    the same number at every EVEN `per_row`, and every seam of every recorded
    segment is laid out at an even one, so the fixture is blind to the
    difference.  At an odd width the two part by one QM31 column, which is
    1,048,576 cells at Qwen2.5-14B's feed-forward width (13,824 at 27 slots a
    row) -- so this checks the arithmetic where no recorded chunk can.
    """
    def paired(n):
        return -(-n // 2)

    bad = []
    for pr in (1, 2, 3, 13, 14, 16, 20, 27, 28):
        counted = paired(5 * pr + 1) + 3
        other = paired(5 * pr) + 4
        if pr % 2 == 0 and counted != other:
            bad.append(f"the two readings part at an even per_row {pr}: "
                       f"{counted} against {other}")
        if pr % 2 and counted != other - 1:
            bad.append(f"at an odd per_row {pr} the readings differ by "
                       f"{other - counted} and not by one")
    odd = []
    for p in chunks():
        q = p["params"]
        for name in _seam_names(q):
            pr, _, _ = mat_layout(_seam_width(q, name), q["s"], q["deq_target"])
            if pr % 2:
                odd.append((p["segment"], name, pr))
    if odd:
        bad.append(f"a recorded segment lays a seam out at an odd width after all "
                   f"({odd[:3]}), so the fixture does separate the two readings and "
                   f"this check is not the one to make")
    for b in bad:
        print("exact cells:", b)
    if bad:
        return 1
    print("  the requantize seam batches `5 * per_row + 1` lookups in pairs and "
          "carries three claims,")
    print("  which no recorded segment separates from the other reading: every one "
          "of their seams is even")
    return 0


def product_selftest():
    """The attention output's product form, which no recorded segment is in.

    `Params::ao_acc_prod` adds three things to the magnitude form's segment: an
    eleventh column a slot (`ahn = ahi num`), an eighth lookup a slot, and a
    range table of `30 - lo` bits that `Plan::new` puts in its width SET.  No
    recorded chunk separates any of them from the magnitude form, so this checks
    each against the segment they are added to, on the Llama 3.1 8B dump the form
    was written for: block 0 layer 1 of a 512-token table of contents, whose
    22-bit accumulator splits 16 + 6 under `jv_max = -5`, `jo_min = -14`, so that
    the product table is 14 bits.  And where `30 - lo` is a width the segment
    already registers, the set takes the table away again.
    """
    bad = []
    ref = headline_chunks("layer")[0]["params"]
    mag = dict(ref, ao_acc_lo_bits=16, ao_acc_hi_bits=6, ao_jv_max=-5, ao_jo_min=-14,
               ao_acc_mag=True)
    prod = dict(mag, ao_acc_prod=True)
    if ao_prod_bits(prod) != 14:
        bad.append(f"the 16 + 6 split's product table is {ao_prod_bits(prod)} bits "
                   f"and not 30 - 16 = 14")
    pr, _, lr = ao_layout(ref["nh"], ref["s"], hd_v(ref), ref["ao_target"])
    table = 1 << 14
    if 14 in range_widths(mag) or 14 not in range_widths(prod):
        bad.append(f"the 14-bit table is not the product form's own: "
                   f"{range_widths(mag)} against {range_widths(prod)}")
    # with the attention output's lookups in the batched reduction, which is how
    # both headline runs prove it, the eighth lookup commits no column; without
    # it, `8 pr` fractions in pairs where there were `7 pr`
    for gkr in (True, False):
        m, q = dict(mag, gkr_attnout=gkr), dict(prod, gkr_attnout=gkr)
        a, b = exact_cells(m), exact_cells(q)
        pairs = 0 if gkr else -(-8 * pr // 2) - -(-7 * pr // 2)
        want = dict(base=(pr << lr) + table,
                    interaction=((M31_PER_QM31 * pairs) << lr) + M31_PER_QM31 * table,
                    boundary=0)
        for field, w in want.items():
            if b[field] - a[field] != w:
                bad.append(f"the product form adds {b[field] - a[field]:,} to `{field}` "
                           f"{'with' if gkr else 'without'} --gkr-attnout, and not "
                           f"{w:,}")
        if b["components"]["attnout"][0] - a["components"]["attnout"][0] != pr << lr:
            bad.append("the product form's column is not one a slot")
    # a split whose `30 - lo` the segment already carries (15, the requantize
    # seams' width) registers no table of its own
    m15 = dict(mag, ao_acc_lo_bits=15, ao_acc_hi_bits=7)
    if exact_cells(dict(m15, ao_acc_prod=True))["table_rows"] != exact_cells(m15)["table_rows"]:
        bad.append("a product table of a width already registered was counted twice")
    try:
        exact_cells(dict(prod, ao_acc_mag=False))
        bad.append("a product form not on the magnitude form was counted")
    except AssertionError:
        pass
    for b_ in bad:
        print("exact cells:", b_)
    if bad:
        return 1
    print(f"  the product form adds one column and one lookup a slot ({pr} slots at "
          f"2^{lr} rows) and a 14-bit table")
    print("  to the 8B table of contents' segment, and no table where `30 - lo` is "
          "already registered: checked")
    return 0


def builder_selftest():
    """`segment_params` against the recorded 8B segments: the block a shape writes
    has to be the block the run wrote, field for field, once the witness-calibrated
    fields are taken from that segment."""
    shape = dict(d=4096, nh=32, nkv=8, hd=128, dff=14336, vocab=128256)
    want = {"B8_L16": ("layer", 4096), "B8_L0": ("first", 4096), "B8": ("head", 4096),
            "B7": ("head", 3584), "B32_L16": ("layer", 16384), "B32": ("head", 16384)}
    bad = 0
    for p in headline_chunks():
        if p["segment"] not in want:
            continue
        kind, c = want[p["segment"]]
        rec = p["params"]
        built = segment_params(kind, s=rec["s"], c=c,
                               witness={f: rec[f] for f in WITNESS_FIELDS}, **shape)
        got, recorded = cells_of(built)["fresh"], p["fresh"]
        if got != recorded:
            print(f"exact cells: the block built from the 8B shape does not reproduce "
                  f"{p['run']} {p['segment']}: {got:,} against {recorded:,} "
                  f"({got - recorded:+,})")
            bad += 1
    if bad:
        return 1
    print(f"  a `params` block built from the shape alone reproduces the recorded "
          f"layer, first and output head segments at both contexts: checked")
    return 0


def latent_selftest():
    """Multi-head latent attention, at both forms (`attention/latent.py`).

    Four checks, and the first two are the ones that say the generalisation did
    not move the layout: a latent attention whose two widths are one head width
    counts exactly what the ordinary attention at that head width counts, and the
    two forms hold the same weights.  Then the two identities that separate them:
    what a key costs the cache, and how many copies of it a segment holds."""
    lat = latent_attention
    w = modal_assumption("layer")
    d, nh, s, c, dff, vocab = 4096, 32, 512, 4096, 14336, 128256

    def block(cfg, form, **kw):
        z = lat.dims(cfg, nh, form)
        return segment_params("layer", d=d, nh=nh, nkv=nh, hd=z["hd_v"], dff=dff,
                              vocab=vocab, s=s, c=c, witness=w, latent=z, **kw), z

    # (0) the cache term is the key/value term wherever a shape has one
    for q in chunks():
        try:
            want_kv = kv_cells(q["params"])
        except Refused:
            continue
        if cache_cells(q["params"]) != want_kv:
            print(f"exact cells: the cache term and the key/value term part on "
                  f"{q['run']} {q['segment']}")
            return 1

    # (1) a latent attention of one head width is the ordinary one at that width
    flat = dict(q_lora_rank=1536, kv_lora_rank=512, qk_nope_head_dim=64,
                qk_rope_head_dim=64, v_head_dim=128)
    p_lat, _z = block(flat, "materialised")
    p_gqa = segment_params("layer", d=d, nh=nh, nkv=nh, hd=128, dff=dff,
                           vocab=vocab, s=s, c=c, witness=w)
    a, b = key_tile_cells(p_lat), key_tile_cells(p_gqa)
    if a != b:
        print(f"exact cells: a latent attention of one head width does not count "
              f"what the ordinary attention at that width counts: {a} against {b}")
        return 1
    ga = boundary_cells(nh, 128, d, s, c, kv_per_key=cache_per_key(p_lat))
    gb = boundary_cells(nh, 128, d, s, c)
    if ga != gb:
        print(f"exact cells: its boundary trees do not agree either: {ga:,} "
              f"against {gb:,}")
        return 1

    # (2) the weights are the same either way: `W_UK` and `W_UV` ARE `kv_b`
    cfg = dict(q_lora_rank=1536, kv_lora_rank=512, qk_nope_head_dim=128,
               qk_rope_head_dim=64, v_head_dim=128)
    ws = {f: lat.attn_params(lat.dims(cfg, nh, f), d) for f in lat.FORMS}
    if len(set(ws.values())) != 1:
        print(f"exact cells: the two latent-attention forms do not hold the same "
              f"weights: {ws}")
        return 1

    # (3) what one key costs the cache, and (4) how many copies a segment holds
    out = {}
    for form in lat.FORMS:
        q, z = block(cfg, form)
        out[form] = (z, exact_cells(q)["fresh"], cache_cells(q), q)
    want = {"materialised": nh * (cfg["qk_nope_head_dim"] + cfg["qk_rope_head_dim"]
                                  + cfg["v_head_dim"]),
            "absorbed": cfg["kv_lora_rank"] + cfg["qk_rope_head_dim"]}
    for form, (z, _fresh, _cache, _q) in out.items():
        if z["cache_per_key"] != want[form]:
            print(f"exact cells: the {form} form caches {z['cache_per_key']} a key "
                  f"and not {want[form]}")
            return 1
    q = out["absorbed"][3]
    groups = len(q["head_groups"])
    tiles = {lay["log_rows"] for n, lay, _a, _l in tile_components(q)}
    if cache_instances(q, q["head_groups"][0]) != 1:
        print("exact cells: the absorbed form does not hold ONE cache a key tile")
        return 1
    ratio = out["materialised"][2]["total"] / out["absorbed"][2]["total"]
    fresh = {f: out[f][1] for f in lat.FORMS}
    print(f"  the cache term is the key/value term on every recorded chunk, a "
          f"latent attention of one head width counts what the")
    print(f"  ordinary one counts, the two forms hold the same weights "
          f"({ws['absorbed']:,}),")
    print(f"  and one key costs the cache {want['materialised']:,} cells "
          f"materialised against {want['absorbed']:,} absorbed, which is "
          f"{ratio:.1f}x")
    print(f"  of the segment's cache term over {groups} head groups and "
          f"{len(tiles)} distinct tile heights; the segment itself is")
    print(f"  {fresh['materialised']:,} cells materialised and "
          f"{fresh['absorbed']:,} absorbed: checked")
    return 0


def segment_kind(p):
    """What this segment is: the output head, the first segment of a stack (the
    one that carries the embedding lookup), or a layer."""
    if (p.get("residual_stream") or {}).get("head_norm"):
        return "output head"
    return "first segment" if p.get("embed_vocab") else "layer"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true",
                    help="check the count against the recorded chunks (the default)")
    ap.add_argument("--witness-share", action="store_true",
                    help="print what the witness-derived terms cost on each chunk")
    a = ap.parse_args(argv)
    if a.witness_share:
        for p in chunks():
            w = witness_terms(p["params"])
            print(f"{p['run']} {p['segment']}: " + ", ".join(
                f"{k} {v:,} ({f:.2%})" for k, (v, f) in w.items()))
        return 0
    return selftest()


if __name__ == "__main__":
    sys.exit(main())
