"""E4, section 7 steps 6-7: the measured layer chunks against the registered prediction.

Reads the prover's per-layer JSONs (one `compose` run per layer, reference backend)
and the committed registration, writes the per-chunk record the run-book asks
for and the comparison, in the run-book's order:

  (a) the key tiles integers, exact -- reported whatever rho does.  Against the
      registered integers if the chunk was proved under the statement the
      registration prices, and against the closed form at the chunk's own
      statement if it was not: the registration prices `--diagonal-block
      --gkr-key-tiles ...`, and a chunk proved without `--gkr-key-tiles`
      commits the LogUp running sums and reports a larger interaction for the
      same layout.  The base is the same integer either way;
  (a2) the boundary trees, exact -- the other closed form of the layout and the
      other term that grows with the context, `cells.fresh - cells.base -
      cells.interaction` against `2*d*S + 2*nh*hd*(2*C + S)`.  No `family_cells`
      entry names it.  Reported whatever rho does, and a deviation does not
      void the run;
  (b) rho = measured / predicted on layers-only fresh cells per decoded token,
      with section 3's protocol widening and no other adjustment;
  (c) the tier, and the consequence the bracket table fixed in advance.

The predicted figures are READ OUT OF `out/registration.json` (step 0c); nothing
here recomputes them.

    uv run python e4/compare.py --runs e4/out/h200 --shape qwen2p5_14b --ctx 4096
    uv run python e4/compare.py --runs e4/out/h100 --shape llama3p1_70b --ctx 4096

A shape E4 did not register is read out of the registration written for it
instead (step 0c is the same, the document is another one).
"""
import argparse
import glob
import json
import os
import re
import sys

from harness.costmodel.key_tile import boundary_cells, key_tile_family

HERE = os.path.dirname(os.path.abspath(__file__))
REG = os.path.join(HERE, "out", "registration.json")
SHAPES = {"qwen2p5_14b": "Qwen2.5-14B-Instruct",
          "qwen2p5_3b": "Qwen2.5-3B-Instruct",
          "qwen2p5_32b": "Qwen2.5-32B-Instruct",
          # zk-results #93's shape, registered on its own and not by an E4 run.
          # What it is measured against is a synthetic layer segment at this
          # width, which is the shape and not the model.
          "llama3p1_70b": "Llama-3.1-70B",
          # E4's third scale as it is actually measured: Qwen2.5-14B cannot be
          # proved (#172, #174) and this is the bias-free model of that class.
          "codellama_13b": "CodeLlama-13b-hf"}

# The document a shape is read out of.  E4's three are registered together in
# `REG`; a shape registered on its own is written beside it, under the name
# `predict.out_name` gives it -- the shape name lower-cased, with every run of
# characters that is not a letter or a digit turned into one dash.  Named here
# rather than imported: importing `predict` reads the cost model's inputs
# directory at import, and this script prices nothing and needs none of it.
REGISTRATION = {"llama3p1_70b": os.path.join(HERE, "out",
                                             "registration-llama-3-1-70b.json"),
                "codellama_13b": os.path.join(HERE, "out",
                                              "registration-codellama-13b-hf.json")}

# The shapes E4 registered, whatever document they are registered in: the
# counted registration of the 14B is its own file and its own selection, and it
# is still E4's shape and still graded by E4's acceptance range.
E4_REGISTERED = ("Qwen2.5-14B-Instruct", "Qwen2.5-3B-Instruct",
                 "Qwen2.5-32B-Instruct")


def load_run(path):
    r = json.load(open(path))
    p = r["result"]["point"]
    fc = p["family_cells"]
    ibf = fc["interaction_by_family"]
    pa = p["params"]
    k = int(re.search(r"L(\d+)\.json$", path).group(1))
    return dict(
        layer=k, file=os.path.basename(path),
        accepts=p["accepts"], prove_error=p.get("prove_error"),
        source=r["result"].get("source"),
        cells=dict(fresh=p["cells"]["fresh"], base=p["cells"]["base"],
                   interaction=p["cells"]["interaction"],
                   preprocessed=p["cells"]["preprocessed"]),
        columns=p["columns"],
        family_cells=dict(key_tiles=fc["key_tiles"], dequant=fc["dequant"],
                          residual_stream=fc["residual_stream"],
                          attnout=fc["attnout"], interaction=fc["interaction"]),
        interaction_by_family={f: ibf[f]["cells"] for f in ibf},
        # every lookup table's rows, which is what the whole-chunk identity is
        # exact in; `log_table` alone is only the master exponent table of them
        table_rows=sum(1 << t["log_rows"] for t in fc["tables"]),
        # issue #171: the prover renamed this statement key to `head_groups` in
        # 81a5ec9 and every chunk proved from that commit on carries the new
        # name.  The record's own field is renamed with it, because `head_groups`
        # is what the statement calls it; the fallback keeps the chunks proved
        # before the rename readable, and both headline runs are pre-rename.
        params=dict(log_table=pa["log_table"], ao_rem_bits=pa["ao_rem_bits"],
                    ao_q_bits=pa["ao_q_bits"],
                    head_groups=pa.get("head_groups", pa.get("key_tiles")),
                    ao_acc_lo_bits=pa.get("ao_acc_lo_bits"),
                    ao_acc_hi_bits=pa.get("ao_acc_hi_bits"),
                    ao_acc_mag=bool(pa.get("ao_acc_mag", False)),
                    # skipped in the statement unless it is set, as `ao_acc_mag` is
                    ao_acc_prod=bool(pa.get("ao_acc_prod", False)),
                    ao_jo_min=pa.get("ao_jo_min"), ao_jo_max=pa.get("ao_jo_max"),
                    ao_jv_min=pa.get("ao_jv_min"), ao_jv_max=pa.get("ao_jv_max"),
                    fold_rem_bits=pa.get("fold_rem_bits", 0),
                    nh=pa["nh"], hd=pa["hd"],
                    s=pa["s"], c=pa["c"], diagonal_block=pa.get("diagonal_block", False), layer_boundary=pa.get("layer_boundary", False),
                    kv_boundary=pa.get("kv_boundary"), rope=pa.get("rope"),
                    gkr_key_tiles=pa.get("gkr_key_tiles"),
                    grid_target=pa["grid_target"], chan_target=pa["chan_target"], rw=pa["rw"],
                    k_bits=pa["k_bits"], q_bits=pa["q_bits"], nmax=pa["nmax"],
                    kmax=pa["kmax"], round=pa["round"], variant=pa["variant"],
                    z_hi_bits=pa["z_hi_bits"], z_lo_bits=pa["z_lo_bits"],
                    residual_stream=pa.get("residual_stream"), mats=pa["mats"]),
        off_table=p.get("off_table"), ref_mismatch=p.get("ref_mismatch"),
        residual_stream_ties_ok=p.get("residual_stream_ties_ok"),
        peak_rss_bytes=p.get("peak_rss_bytes"),
        proof_sha256=p["proof_sha256"], proof_bytes=p["proof_bytes"],
        proof_ser_bytes=p["proof_ser_bytes"],
        statement_bytes=p.get("statement_bytes"),
        secs=dict(prove=p.get("prove_secs"), trace=p.get("trace_secs"),
                  interaction=p.get("interaction_secs"), verify=p.get("verify_secs"),
                  commit=(p.get("phases") or {}).get("commit_secs"),
                  stwo_wall=(p.get("phases") or {}).get("wall_secs"), fri=(p.get("phases") or {}).get("fri_secs")),
        env=r.get("env"),
    )


STATEMENT_KEYS = ("diagonal_block", "kv_boundary", "bind", "gkr")


def statement_of(p):
    """The key tile configuration this chunk was proved under, off its own `params`.

    The four settings the closed form prices different columns under: whether
    the prefix is tiled into key blocks, whether the chunk binds its K/Q/V to
    the boundary roots, and whether the key tiles' batched reduction proves the
    LogUp sums instead of committing them.  Read rather than assumed, for the
    same reason `predict.statement` reads it: the chunk is the only thing that
    says which statement it is.
    """
    return dict(diagonal_block=bool(p["diagonal_block"]),
                kv_boundary=bool(p["kv_boundary"]),
                bind=p["rope"] is not None,
                gkr=bool(p["gkr_key_tiles"]))


def key_tiles_at(p, st):
    """The closed form's (base, interaction) at this chunk's own shape and `st`."""
    b, i, _ = key_tile_family(p["nh"], p["hd"], p["s"], p["c"],
                              grid_target=p["grid_target"], chan_target=p["chan_target"],
                              rw=p["rw"], **st)
    return int(b), int(round(i))


def tier_of(rho, br):
    if br["prize"][0] <= rho <= br["prize"][1]:
        return "prize"
    if br["pass_"][0] <= rho <= br["pass_"][1]:
        return "pass"
    if br["marginal"][0] <= rho <= br["marginal"][1]:
        return "marginal"
    return "fail"


CONSEQUENCE = {
    "prize": "the model predicts a never-fitted scale to 3%. Licence claimed unqualified; "
             "section 6.3 keeps [projected]",
    "pass": "licence claimed with the measured residual quoted beside it",
    "marginal": "licence claimed only with an explicit +/-25% key tile printed on every "
                "frontier row",
    "fail": "the cost model is refuted as an extrapolator. Section 6.3 regraded "
            "[projected] -> [illustrative]; the section 7 'you didn't build it' mitigation "
            "falls back to publishing the model runnable; E4 written up as a negative "
            "result. No re-fit may be presented as if it were this prediction.",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="directory of the prover's L<k>.json files")
    ap.add_argument("--shape", default="qwen2p5_14b")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--registration", metavar="FILE",
                    help="the registration to compare against; default: the "
                         "document `out/registration.json`, which is where "
                         "zk-results' register driver copies the recorded one")
    ap.add_argument("--card", default="H200")
    ap.add_argument("--backend", default="ref (SimdBackend)")
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    a = ap.parse_args()

    reg_file = a.registration or REGISTRATION.get(a.shape, REG)
    reg = json.load(open(reg_file))
    shape = reg["shapes"][SHAPES[a.shape]]
    L = shape["cfg"][0]
    pc = shape["contexts"][str(a.ctx)]
    br = reg["bracket"]
    pred_cpt = pc["cells_per_token_layers"]
    pred_key_tiles_base = pc["key_tiles_family_base_per_chunk"]
    pred_key_tiles_inter = pc["key_tiles_family_interaction_per_chunk"]
    pred_chunk = pc["per_layer_chunk_fresh"]

    files = sorted(glob.glob(os.path.join(a.runs, "L*.json")),
                   key=lambda f: int(re.search(r"L(\d+)\.json$", f).group(1)))
    runs = [load_run(f) for f in files]
    if not runs:
        sys.exit(f"no L<k>.json under {a.runs}")

    # ---- step 5's two traps, per chunk ------------------------------------
    problems = []
    for r in runs:
        k = r["layer"]
        if not r["accepts"]:
            problems.append(f"L{k}: accepts != true")
        if not r["cells"]["fresh"] > 0:
            problems.append(f"L{k}: cells.fresh not > 0")
        if not r["family_cells"]["key_tiles"] > 0:
            problems.append(f"L{k}: family_cells empty")
        want = dict(q=(5120, 5120), k=(5120, 1024), v=(5120, 1024), o=(5120, 5120),
                    gate=(5120, 13824), up=(5120, 13824), down=(13824, 5120))
        if a.shape == "qwen2p5_14b":
            for m in r["params"]["mats"]:
                if (m["n_in"], m["n_out"]) != want[m["name"]] or m["variant"] != "chain" \
                        or m["tokens"] != 512:
                    problems.append(f"L{k}: mat {m}")
        p = r["params"]
        cfg_ok = (p["s"] == 512 and p["c"] == a.ctx and p["hd"] == 128 and p["diagonal_block"]
                  and p["layer_boundary"] and p["grid_target"] == 64 and p["chan_target"] == 128
                  and p["rw"] == 16 and p["k_bits"] == 11 and p["q_bits"] == 12
                  and p["round"] == "even" and p["nmax"] == 511)
        if not cfg_ok:
            problems.append(f"L{k}: configuration is not the registered one: {p}")
        if r["off_table"]:
            problems.append(f"L{k}: off_table {r['off_table']}")
        if r["ref_mismatch"]:
            problems.append(f"L{k}: ref_mismatch {r['ref_mismatch']}")
        if r["residual_stream_ties_ok"] is False:
            problems.append(f"L{k}: residual_stream_ties_ok false")

    # ---- (a) the key tiles integers, exact ------------------------------------
    # The registered integers price one statement (`reg["statement"]`).  A chunk
    # proved under another one is compared against the closed form at ITS
    # statement, at the chunk's own shape, and the row says so; comparing it
    # against the registered interaction would report a difference the layout
    # never had.  A registration with no statement block predates this and is
    # taken at its word.
    reg_st = reg.get("statement")
    registered_st = {k: reg_st[k] for k in STATEMENT_KEYS} if reg_st else None
    key_tiles_rows = []
    for r in runs:
        b, i = r["family_cells"]["key_tiles"], r["interaction_by_family"]["key_tiles"]
        st = statement_of(r["params"])
        as_registered = registered_st is None or st == registered_st
        if as_registered:
            eb, ei, against = pred_key_tiles_base, pred_key_tiles_inter, "the registered integers"
            if registered_st is not None and key_tiles_at(r["params"], st) != (eb, ei):
                problems.append(f"L{r['layer']}: the closed form at this chunk's own shape and "
                                f"the registered statement is not the registered pair "
                                f"{eb:,} / {ei:,}: the shape is not the registered one")
        else:
            eb, ei = key_tiles_at(r["params"], st)
            against = ("`key_tile_family` at this chunk's statement "
                       + " ".join(f"{k}={st[k]}" for k in STATEMENT_KEYS))
        key_tiles_rows.append(dict(layer=r["layer"], key_tiles_base=b, key_tiles_interaction=i,
                               base_delta=b - eb, interaction_delta=i - ei,
                               expected_base=eb, expected_interaction=ei,
                               statement=st, statement_as_registered=as_registered,
                               compared_against=against,
                               exact=(b == eb and i == ei)))
    key_tiles_exact = all(x["exact"] for x in key_tiles_rows)
    off_statement = sorted({x["layer"] for x in key_tiles_rows if not x["statement_as_registered"]})

    # ---- (a2) the boundary trees, exact ---------------------------------------
    # The other closed form of the layout, and the other term that grows with
    # the context.  Like the key tiles integers this is reported whatever rho
    # does and a deviation is a finding in its own right -- it means the layout
    # identity does not hold at a shape whose `nh` and `d` are not powers of two
    # -- but it does not VOID the run: a void is a run that did not reproduce
    # the registered configuration, and this is a run that did and disagreed.
    boundary_rows = []
    for r in runs:
        want = boundary_cells(r["params"]["nh"], r["params"]["hd"],
                              r["params"]["residual_stream"]["d"], r["params"]["s"],
                              r["params"]["c"])
        got = r["cells"]["fresh"] - r["cells"]["base"] - r["cells"]["interaction"]
        boundary_rows.append(dict(layer=r["layer"], boundary=got, expected=want,
                                  delta=got - want, exact=(got == want)))
    boundary_exact = all(x["exact"] for x in boundary_rows)

    # ---- (a3) the counted registration, layer by layer ------------------------
    # A registration made on the counted path states what EVERY segment costs,
    # not one number for the stack, and it states the witness fields it counted
    # each of them at.  Both halves are checked here:
    #
    #   * the count against the chunk's own `cells.fresh`, per layer.  There is
    #     no averaging in this and no protocol widening: the registration named
    #     one integer per counted segment before the run and the run reports one
    #     for each;
    #   * the FIELDS against the chunk's own `params`.  They were read off the
    #     dumps before anything was proved, so the proof has to agree with them
    #     -- and if it does not, the registration counted a statement this run
    #     did not prove, which voids the comparison rather than failing it.
    counted = pc.get("layers")
    layer_rows, field_rows = [], []
    if counted:
        by_layer = {int(k) if not isinstance(k, int) else k: v
                    for k, v in ((r["layer"], r) for r in counted)}
        for r in runs:
            k = r["layer"]
            pred = by_layer.get(k)
            if pred is None:
                problems.append(f"L{k}: the registration counts no layer {k}")
                continue
            got, want = r["cells"]["fresh"], pred["fresh"]
            layer_rows.append(dict(layer=k, predicted=want, measured=got,
                                   delta=got - want,
                                   residual_pct=100.0 * (got / want - 1.0),
                                   exact=(got == want), bound=pred["bound_fresh"]))
            bad_fields = {f: [pred["fields"][f], r["params"].get(f)]
                          for f in sorted(pred["fields"])
                          if r["params"].get(f) is not None
                          and r["params"][f] != pred["fields"][f]}
            field_rows.append(dict(layer=k, fields=pred["fields"],
                                   disagrees=bad_fields, agrees=not bad_fields))
            if bad_fields:
                problems.append(
                    f"L{k}: the registration counted this segment at fields the run "
                    f"did not prove it at: "
                    + ", ".join(f"{f} {v[0]} against {v[1]}" for f, v in bad_fields.items()))
    counted_exact = bool(layer_rows) and all(x["exact"] for x in layer_rows)

    # ---- (b) rho, with the protocol widening ------------------------------
    n = len(runs)
    layers = [r["layer"] for r in runs]
    # What the prediction is stated over, which is not always the stack.  A
    # counted registration of a model whose every layer can be dumped counts
    # `L` segments; one that names a layer absent counts the ones it has and
    # says nothing about the one it does not, so its `cells_per_token_layers`
    # is over those.  Scaling the measured side to `L` against that would put
    # `L` layers on one side and the counted set on the other -- at 39 of 40
    # a difference of 2.6%, none of it the model's.  So the basis is the
    # registration's own.
    absent = sorted((reg.get("frozen_inputs", {}).get("witness_fields")
                     or {}).get("absent") or ())
    basis = L - len(absent)
    if n == basis:
        protocol, widen_lo, widen_hi = "A", 0.0, 0.0
    elif n >= 8:
        gaps = sorted(set(b - a_ for a_, b in zip(layers, layers[1:])))
        protocol = "B" if len(gaps) == 1 else "B (not evenly spaced!)"
        widen_lo = widen_hi = br["protocol_widening"]["B_even_subset"]["widen_pct"]
    else:
        protocol = "C"
        widen_lo = br["protocol_widening"]["C_single_layer"]["widen_pct_lower"]
        widen_hi = br["protocol_widening"]["C_single_layer"]["widen_pct_upper"]
    fresh_sum = sum(r["cells"]["fresh"] for r in runs)
    fresh_mean = fresh_sum / n
    measured_cpt = fresh_mean * basis / 512      # layers only, S = 512 decoded tokens
    rho = measured_cpt / pred_cpt
    widened = {t: [lo * (1 - widen_lo / 100), hi * (1 + widen_hi / 100)]
               for t, (lo, hi) in (("prize", br["prize"]), ("pass_", br["pass_"]),
                                   ("marginal", br["marginal"]))}
    tier = tier_of(rho, widened)

    # ---- the per-chunk record (step 6) ------------------------------------
    total_proof = sum(r["proof_bytes"] for r in runs)
    total_ser = sum(r["proof_ser_bytes"] for r in runs)
    rss = [r["peak_rss_bytes"] for r in runs if r["peak_rss_bytes"]]
    table_cells = [r["interaction_by_family"]["table"] for r in runs]
    logt = sorted(set(r["params"]["log_table"] for r in runs))
    # The whole-chunk identity, at this shape.  On the 8B chunks
    # `cells.fresh - key tiles - boundary trees - 5 * table rows` is one integer
    # on every non-embedding layer at both contexts (`predict.chunk_identity`),
    # and this is the same subtraction here.  The boundary trees are the term no
    # `family_cells` entry names -- what `fresh` has above `base + interaction`
    # -- and leaving them out of it, as this line used to, misses a term that
    # grows with the context: 105.2 M cells a chunk at C=4,096 on the 8B and
    # 306.6 M at C=16,384.
    rest = [r["cells"]["fresh"] - r["family_cells"]["key_tiles"]
            - r["interaction_by_family"]["key_tiles"]
            - boundary_cells(r["params"]["nh"], r["params"]["hd"],
                             r["params"]["residual_stream"]["d"], r["params"]["s"],
                             r["params"]["c"])
            - 5 * r["table_rows"]
            for r in runs]

    measured = dict(
        schema="e4-measured/1", shape=SHAPES[a.shape], hf=shape["hf"], cfg=shape["cfg"],
        ctx=a.ctx, s=512, spec="b10-final", card=a.card, backend=a.backend,
        boundary="cell count (card-independent); bytes; peak host RSS",
        layers_proved=layers, n_layers_proved=n, n_layers_model=L,
        chunks=runs,
        summary=dict(
            fresh_sum=fresh_sum, fresh_mean_per_chunk=fresh_mean,
            fresh_min=min(r["cells"]["fresh"] for r in runs),
            fresh_max=max(r["cells"]["fresh"] for r in runs),
            cells_per_token_layers_measured=measured_cpt,
            log_table_values=logt,
            log_table_histogram={str(t): sum(1 for r in runs if r["params"]["log_table"] == t)
                                 for t in logt},
            table_cells_min=min(table_cells), table_cells_max=max(table_cells),
            ao_rem_bits_values=sorted(set(r["params"]["ao_rem_bits"] for r in runs)),
            kmax_values=sorted(set(r["params"]["kmax"] for r in runs)),
            rest_after_key_tiles_boundary_and_tables=dict(
                values=sorted(set(rest)), constant=len(set(rest)) == 1,
                form="cells.fresh - key tiles - boundary trees - 5 * table rows"),
            # which families move across layers (at 8B every family but the table is
            # identical to the cell on all 32 chunks)
            family_spread={f: dict(min=min(r["family_cells"][f] for r in runs),
                                   max=max(r["family_cells"][f] for r in runs))
                           for f in ("key_tiles", "dequant", "residual_stream", "attnout", "interaction")},
            interaction_spread={f: dict(min=min(r["interaction_by_family"][f] for r in runs),
                                        max=max(r["interaction_by_family"][f] for r in runs))
                                for f in runs[0]["interaction_by_family"]},
            base_spread=dict(min=min(r["cells"]["base"] for r in runs),
                             max=max(r["cells"]["base"] for r in runs)),
            preprocessed_spread=dict(min=min(r["cells"]["preprocessed"] for r in runs),
                                     max=max(r["cells"]["preprocessed"] for r in runs)),
            proof_bytes_sum=total_proof, proof_ser_bytes_sum=total_ser,
            proof_bytes_per_layer_chunk_mean=total_proof / n,
            proof_ser_bytes_per_layer_chunk_mean=total_ser / n,
            proof_bytes_per_token_layers=total_proof / n * L / 512,
            proof_ser_bytes_per_token_layers=total_ser / n * L / 512,
            peak_rss_bytes_max=max(rss) if rss else None,
            peak_rss_bytes_min=min(rss) if rss else None,
            prove_secs_sum=sum(r["secs"]["prove"] or 0 for r in runs),
            trace_secs_sum=sum(r["secs"]["trace"] or 0 for r in runs),
        ),
        problems=problems,
    )
    comparison = dict(
        schema="e4-comparison/1", shape=SHAPES[a.shape], ctx=a.ctx, s=512, spec="b10-final",
        card=a.card, backend=a.backend,
        registration=dict(file=os.path.relpath(reg_file, HERE), prediction_sha256=reg["prediction_sha256"],
                          prediction_block_sha256=reg["prediction_block_sha256"]),
        predicted=dict(cells_per_token_layers=pred_cpt, per_layer_chunk_fresh=pred_chunk,
                       key_tiles_family_base_per_chunk=pred_key_tiles_base,
                       key_tiles_family_interaction_per_chunk=pred_key_tiles_inter,
                       cells_per_token_layers_key_tiles=pc["cells_per_token_layers_key_tiles"],
                       cells_per_token_layers_rest=pc["cells_per_token_layers_rest"]),
        a2_boundary_exact=dict(
            exact_on_every_chunk=boundary_exact,
            form="cells.fresh - cells.base - cells.interaction == 2*d*S + 2*nh*hd*(2*C + S)",
            rows=boundary_rows),
        a_key_tiles_exact=dict(exact_on_every_chunk=key_tiles_exact,
                               registered_statement=registered_st,
                               layers_not_at_the_registered_statement=off_statement,
                               rows=key_tiles_rows),
        a3_counted_per_layer=(None if not counted else dict(
            exact_on_every_layer=counted_exact,
            layers_counted=len(counted), layers_proved=len(layer_rows),
            worst_residual_pct=max((abs(x["residual_pct"]) for x in layer_rows),
                                   default=None),
            sum_predicted=sum(x["predicted"] for x in layer_rows),
            sum_measured=sum(x["measured"] for x in layer_rows),
            fields_agree_on_every_layer=all(x["agrees"] for x in field_rows),
            rows=layer_rows, fields=field_rows,
            note=(f"the registration counted every segment from the prover's trace "
                  f"layout at the witness fields its own dump carries, so these are "
                  f"{len(counted or ())} integers named before the run and "
                  f"{len(layer_rows)} reported by it, compared one for one and with "
                  f"no widening"))),
        b_rho=dict(protocol=protocol, layers_proved=n, layers_counted=basis,
                   layers_in_stack=L, layers_absent=absent, widen_pct_lower=widen_lo,
                   widen_pct_upper=widen_hi, measured_cells_per_token_layers=measured_cpt,
                   predicted_cells_per_token_layers=pred_cpt, rho=rho,
                   residual_pct=(rho - 1) * 100,
                   chunk_ratio_min=min(r["cells"]["fresh"] for r in runs) / pred_chunk,
                   chunk_ratio_max=max(r["cells"]["fresh"] for r in runs) / pred_chunk,
                   brackets_widened=widened),
        c_tier=dict(tier=tier, consequence=CONSEQUENCE[tier]),
        void=bool(problems), problems=problems,
    )
    os.makedirs(a.out, exist_ok=True)
    mp = os.path.join(a.out, f"measured_{a.shape}_c{a.ctx}.json")
    cp = os.path.join(a.out, f"comparison_{a.shape}_c{a.ctx}.json")
    json.dump(measured, open(mp, "w"), indent=1, sort_keys=True)
    json.dump(comparison, open(cp, "w"), indent=1, sort_keys=True)

    # ---- the table ----------------------------------------------------------
    print(f"E4 -- {SHAPES[a.shape]} at C={a.ctx}, S=512, spec b10-final, {a.card}, {a.backend}")
    print(f"registration: prediction_sha256 {reg['prediction_sha256'][:12]}...  "
          f"predicted cells/token (layers) {pred_cpt:,.1f}, per chunk {pred_chunk:,.1f}")
    print()
    print(f"{'L':>3} {'fresh':>14} {'key_tiles':>12} {'i.key tiles':>12} {'table':>12} {'logT':>4} "
          f"{'aorem':>5} {'kmax':>4} {'proof_B':>10} {'ser_B':>11} {'rss_GiB':>7} {'prove_s':>7} digest")
    for r in runs:
        print(f"{r['layer']:>3} {r['cells']['fresh']:>14,} {r['family_cells']['key_tiles']:>12,} "
              f"{r['interaction_by_family']['key_tiles']:>12,} {r['interaction_by_family']['table']:>12,} "
              f"{r['params']['log_table']:>4} {r['params']['ao_rem_bits']:>5} {r['params']['kmax']:>4} "
              f"{r['proof_bytes']:>10,} {r['proof_ser_bytes']:>11,} "
              f"{(r['peak_rss_bytes'] or 0) / 2**30:>7.1f} {(r['secs']['prove'] or 0):>7.1f} "
              f"{r['proof_sha256'][:16]}")
    print()
    print(f"(a) key tiles exact on every chunk: {key_tiles_exact}  "
          f"(registered {pred_key_tiles_base:,} / {pred_key_tiles_inter:,})")
    if registered_st is None:
        print("    the registration names no statement: the registered integers are "
              "taken as they stand")
    elif off_statement:
        other = next(x for x in key_tiles_rows if not x["statement_as_registered"])
        print(f"    proved under another statement than the registration prices, so "
              f"compared against the closed form at their own: L{off_statement} "
              f"({other['compared_against']})")
    else:
        print("    every chunk was proved under the statement the registration prices: "
              + " ".join(f"{k}={registered_st[k]}" for k in STATEMENT_KEYS))
    if not key_tiles_exact:
        for x in key_tiles_rows:
            if not x["exact"]:
                print(f"    L{x['layer']}: base {x['key_tiles_base']:,} ({x['base_delta']:+,}), "
                      f"interaction {x['key_tiles_interaction']:,} ({x['interaction_delta']:+,}) "
                      f"against {x['expected_base']:,} / {x['expected_interaction']:,}, "
                      f"{x['compared_against']}")
    print(f"(a2) boundary trees exact on every chunk: {boundary_exact}  "
          f"(the form 2*d*S + 2*nh*hd*(2*C + S) = "
          f"{boundary_rows[0]['expected']:,} at this shape)")
    if not boundary_exact:
        for x in boundary_rows:
            if not x["exact"]:
                print(f"    L{x['layer']}: {x['boundary']:,} ({x['delta']:+,}) against "
                      f"{x['expected']:,} -- a finding in its own right, reported "
                      f"whatever rho does, and not a void")
    if counted:
        worst = max(layer_rows, key=lambda x: abs(x["residual_pct"]), default=None)
        print(f"(a3) counted per layer, exact on every layer: {counted_exact}  "
              f"({len(layer_rows)} of {len(counted)} counted layers proved)")
        if worst is not None and not counted_exact:
            print(f"     worst layer L{worst['layer']}: measured "
                  f"{worst['measured']:,} against {worst['predicted']:,} counted "
                  f"({worst['delta']:+,}, {worst['residual_pct']:+.4f}%)")
        agree = all(x["agrees"] for x in field_rows)
        print(f"     the fields the registration counted at are the fields the run "
              f"proved at: {agree}")
        if not agree:
            for x in field_rows:
                if x["disagrees"]:
                    print(f"     L{x['layer']}: "
                          + ", ".join(f"{f} registered {v[0]}, proved {v[1]}"
                                      for f, v in x["disagrees"].items()))
    gap = ("" if not absent else
           f" (the model has {L}; layer{'s' if len(absent) > 1 else ''} "
           f"{', '.join(str(k) for k in absent)} "
           f"{'are' if len(absent) > 1 else 'is'} in no dump, so neither side counts "
           f"{'them' if len(absent) > 1 else 'it'})")
    print(f"(b) protocol {protocol}: {n}/{basis} of the segments the registration "
          f"counts{gap}; measured {measured_cpt:,.1f} cells/token "
          f"(layers), predicted {pred_cpt:,.1f}; rho = {rho:.5f} ({(rho - 1) * 100:+.3f}%)")
    print(f"    per-chunk ratio range {comparison['b_rho']['chunk_ratio_min']:.4f} .. "
          f"{comparison['b_rho']['chunk_ratio_max']:.4f}; log_table {logt}; "
          f"rest after key tiles, boundary trees and tables constant: "
          f"{len(set(rest)) == 1} {sorted(set(rest))[:4]}")
    print(f"(c) tier: {tier.upper()} -- {CONSEQUENCE[tier]}")
    # E4's three shapes are graded by E4's range wherever their registration
    # lives; a shape E4 did not register is not, and the sentence is for that
    # case and not for "the document is not the default file"
    if SHAPES[a.shape] not in E4_REGISTERED:
        print("    the tier and the consequence above are E4's acceptance range, applied "
              "to a shape E4 did not register: what is measured against this "
              "registration is in its own document, and whether that run is graded this "
              "way is not settled here")
    sp = measured["summary"]
    moving = {f: v for f, v in {**{"family." + k: v for k, v in sp["family_spread"].items()},
                                **{"interaction." + k: v for k, v in sp["interaction_spread"].items()}}.items()
              if v["min"] != v["max"]}
    print(f"families that move across layers: "
          f"{ {f: v['max'] - v['min'] for f, v in moving.items()} or 'none'}")
    print(f"proof bytes: {total_proof:,} over {n} chunks ({total_proof / n:,.0f}/chunk, "
          f"{total_proof / n * L / 512:,.0f} B/token layers-only); ser {total_ser:,} "
          f"({total_ser / n:,.0f}/chunk, {total_ser / n * L / 512:,.0f} B/token)")
    if rss:
        print(f"peak RSS: {max(rss) / 2**30:.1f} GiB max, {min(rss) / 2**30:.1f} GiB min")
    if problems:
        print("VOID -- the run does not reproduce the registered configuration:")
        for p in problems:
            print("   ", p)
    print(f"-> {mp}\n-> {cp}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
