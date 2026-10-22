//! The command line: one subcommand per gadget and per composed segment, JSON on
//! stdout and progress on stderr.
//!
//! Every subcommand reads its line left to right and the last occurrence of a
//! flag wins; `prover/README.md` says why. The modules the subcommands reach
//! for are the library's, imported below so a bare `compose::` reads here as it
//! read when this file was the crate root.

use serde_json::{Value, json};

use crate::{
    bf16, boundary, chunks, common, compose, cudagen, device, exprprobe, gkr_key_tile, layer,
    logup_run, plogup, shaperoot, spans, witread,
};
use crate::gadgets::{argmax, attention, dequant, embed, freivalds, requant, residual_stream, rope};

use crate::gadgets::attention::witness::{Attn, Corrupt, Src};
use crate::bf16::Round;
use crate::gadgets::dequant::{Spec, Tamper, Variant, Witness};
use crate::gadgets::attention::layout::Shape;
use crate::util::emit;

/// The value of `key` on a command line: **the last occurrence wins** (issue #30).
///
/// A command line is read left to right and a later `--key value` overrides an
/// earlier one, so a shared prefix is written first and what one segment decides
/// for itself after it. `compose $SHARED $line` and the plan line
/// `compose-chunks` composes out of the same two pieces therefore resolve every
/// repeated flag to the same value, which is what makes a one-shot rerun of a
/// plan line reproduce the loop's digest. A bare flag carries no value and
/// `flag` only asks whether it is present, so repeating one is harmless.
fn arg_pair<'a>(args: &'a [String], key: &str) -> Option<&'a str> {
    args.windows(2).rev().find(|w| w[0] == key).map(|w| w[1].as_str())
}

pub fn arg_scalar<T: std::str::FromStr>(args: &[String], key: &str, default: T) -> T {
    arg_pair(args, key).and_then(|v| v.parse().ok()).unwrap_or(default)
}

pub fn arg_list<T: std::str::FromStr + Copy>(args: &[String], key: &str, default: &[T]) -> Vec<T> {
    match arg_pair(args, key) {
        Some(v) => v.split(',').filter_map(|s| s.trim().parse().ok()).collect(),
        None => default.to_vec(),
    }
}

pub fn arg_strs(args: &[String], key: &str, default: &[&str]) -> Vec<String> {
    match arg_pair(args, key) {
        Some(v) => v.split(',').map(|s| s.trim().to_string()).collect(),
        None => default.iter().map(|s| s.to_string()).collect(),
    }
}

pub fn arg_str(args: &[String], key: &str) -> Option<String> {
    arg_pair(args, key).map(|v| v.to_string())
}

fn spec_of(args: &[String], variant: &str, round: &str, per_row: usize) -> Spec {
    Spec {
        variant: Variant::parse(variant),
        round: Round::parse(round),
        // 21 is the largest sound value: 512 * 2^21 = 2^30 < p (bf16::KMAX_SOUND)
        kmax: arg_scalar(args, "--kmax", 21u32),
        per_row,
        limb_bits: (arg_scalar(args, "--limb-lo", 16u32), arg_scalar(args, "--limb-hi", 14u32)),
        // 7 bits on the quotient is B1's clamp at 127; 19 on the error covers
        // `2 den` for the denominators B5 measured (Z max 38,848, blind bound 2^22)
        q_bits: arg_scalar(args, "--q-bits", 7u32),
        rem_bits: arg_scalar(args, "--rem-bits", 19u32),
        fold_alpha: 0,
    }
}

pub fn flag(args: &[String], key: &str) -> bool {
    args.iter().any(|a| a == key)
}

/// The soundness switches.  Every binding gadget is on by default wherever the
/// witness lets it be stated, and a witness that cannot state one is refused;
/// `--unsound-skip-<name>` turns one off, loudly: the name lands in
/// `provenance.unsound_skips` and the binding table marks the edge unbound.
///
/// - `residual-stream`: prove a chunk without its residual-stream family (the
///   norms, the residual adds, the requantize seams), or one whose dump has no
///   such columns;
/// - `layer-boundary`: commit no boundary trees over the residual stream;
/// - `kv-boundary`: commit no boundary trees over the keys and values, or
///   prove a chunk that has no diagonal block to commit them from;
/// - `argmax`: prove a head chunk without tying the emitted token to the
///   logits, or one whose dump has no `key_head.u16`;
/// - `embedding`: prove a layer-0 chunk without tying its residual stream to
///   the embedding rows of its ids, or one whose dump lacks them.
pub const UNSOUND_SKIPS: [&str; 5] =
    ["residual-stream", "layer-boundary", "kv-boundary", "argmax", "embedding"];

/// The one switch that weakens a gadget instead of turning it off:
/// `--unsound-argmax-lax` proves "the emitted token attains the maximum of the
/// logits" where the default proves "the emitted token is the lowest-indexed
/// maximum", which is the rule the model itself decodes by.  On a row where two
/// logits are co-maximal -- 15 of 512 rows at `C = 4096` on the real
/// continuations, so not a theoretical case -- the lax form admits either of
/// them and the strict form admits only the first.  It is a control and not a
/// statement about a forward pass: the name lands in
/// `provenance.unsound_skips` as `argmax-lax` and the binding table marks edge
/// 16 unbound with that reason.
pub const ARGMAX_LAX: &str = "--unsound-argmax-lax";

pub fn skip(args: &[String], name: &str) -> bool {
    debug_assert!(UNSOUND_SKIPS.contains(&name), "not a skip: {name}");
    flag(args, &format!("--unsound-skip-{name}"))
}

/// An unknown skip name is an error: a typo must not prove a bound statement
/// where an unbound one was meant, or the other way round.  The flags the
/// skips replaced are reported once so an old driver line does not pass in
/// silence.
pub fn check_skips(args: &[String]) {
    for a in args {
        if let Some(n) = a.strip_prefix("--unsound-skip-") {
            assert!(UNSOUND_SKIPS.contains(&n),
                    "{a}: not a skip; the skips are --unsound-skip-{}",
                    UNSOUND_SKIPS.join(", --unsound-skip-"));
        }
    }
    for old in ["--bounds", "--kv-bind", "--embed", "--head-argmax", "--head-key",
                "--norm-tails", "--swiglu", "--deq-ties", "--no-glue"] {
        if flag(args, old) {
            eprintln!("compose: {old} is retired -- what it switched is decided by default now, \
                       and an --unsound-skip-<name> flag is the only way off it");
        }
    }
    // The lax argmax was `--argmax-lax`, a name that said nothing about what it
    // gives up.  Ignoring the old name would prove the *strict* statement under
    // a command line that asked for the weaker one, which is a different
    // measurement reported as the same run, so this is an error and not a note.
    assert!(!flag(args, "--argmax-lax"),
            "--argmax-lax is renamed {ARGMAX_LAX}: it proves the weaker statement \
             (the emitted token attains the maximum, not that it is the lowest-indexed \
             maximum), so it is a control and says so");
}

pub fn src_of(args: &[String]) -> Src {
    Src {
        dir: arg_str(args, "--witness"),
        head: arg_scalar(args, "--head", 0usize),
        q_bits: arg_scalar(args, "--qbits", 12u32),
        k_bits: arg_scalar(args, "--kbits", 11u32),
        nmax: (1u32 << arg_scalar(args, "--numbits", 8u32)) - 1,
        // 0 = as many rows as the witness needs, i.e. a complete table
        log_table: arg_scalar(args, "--table-bits", 0u32),
        seed: arg_scalar(args, "--seed", 7u64),
        // B10 step 0: empty = the real witness's own `deq_variants`, B6.5's
        // mixed layout for a synthetic one.  `--layout b10` is the chain on all
        // seven, which is the shape the emit chain has to generate for B10.
        layout: arg_strs(args, "--layout", &[""])[0].clone(),
        // B10 step 3: prove the key tile as the diagonal block instead of the prefix
        // rectangle.  `--ctx` must equal `--s`; the mask is `key <= token`.
        diagonal_block: args.iter().any(|a| a == "--diagonal-block"),
        // Stream AA (E11): `--kv-boundary` also tiles the unchecked prefix into
        // blocks of S keys, which is a witness-side decision as much as a plan
        // one -- see `briefs/aa-kvbind.md`.
        // On by default wherever the chunk has its own block (`--diagonal-block`): a
        // chunk without one has no keys of its own to commit, and `params_of`
        // refuses it unless the K/V boundary is skipped.
        tile: args.iter().any(|a| a == "--diagonal-block") && !skip(args, "kv-boundary"),
    }
}

pub fn attn_cfg(args: &[String], grid_target: usize) -> attention::trace::Cfg {
    let nmax = (1u32 << arg_scalar(args, "--numbits", 8u32)) - 1;
    attention::trace::Cfg {
        chan_target: arg_scalar(args, "--keys-per-row", 128usize),
        grid_target,
        rw: arg_scalar(args, "--tokens-per-row", 16usize),
        range_check: !flag(args, "--no-range"),
        // 16 + 6 = 22 bits on the denominator: B5's parameter, which is exactly
        // what puts |A.V| inside the dequant's 2^30 window
        z_lo_bits: arg_scalar(args, "--z-lo-bits", 16u32),
        z_hi_bits: arg_scalar(args, "--z-hi-bits", 6u32),
        z_min: arg_scalar(args, "--z-min", nmax),
    }
}

/// Median of `reps` timings, with the spread recorded.
fn median(mut runs: Vec<(f64, Value)>) -> Value {
    let n = runs.len();
    runs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let (lo, hi) = (runs[0].0, runs[n - 1].0);
    let mut j = runs.swap_remove(n / 2).1;
    j["reps"] = json!(n);
    j["prove_secs_min"] = json!(lo);
    j["prove_secs_max"] = json!(hi);
    j
}

// ------------------------------------------------------------------ tables ----
/// The two tables, checked against an independent single f64 rounding.
fn tables_cmd(args: &[String]) {
    let mut out = vec![];
    for k in arg_list(args, "--kmax", &[12u32, 16, 21]) {
        for r in arg_strs(args, "--round", &["even", "away", "zero"]) {
            eprintln!("tables kmax={k} round={r}");
            out.push(dequant::check_tables(k, Round::parse(&r),
                                           arg_scalar(args, "--samples", 200_000usize)));
        }
    }
    emit("tables", json!({"checks": out}));
}

// ----------------------------------------------------------------- dequant ----
/// The ladder: what each way of scaling and rounding an accumulator costs.
fn dequant_cmd(args: &[String]) {
    let dir = arg_str(args, "--witness");
    let n: usize = arg_scalar(args, "--n", 1 << 20);
    let per_row: usize = arg_scalar(args, "--per-row", 16usize);
    let reps: usize = arg_scalar(args, "--reps", 1usize);
    let mat = arg_str(args, "--mat").unwrap_or_else(|| "gate".into());
    let k: usize = arg_scalar(args, "--k", 896usize);
    let mut points = vec![];
    for vname in arg_strs(args, "--variants", &dequant::LADDER) {
        for rname in arg_strs(args, "--round", &["even"]) {
            let sp = spec_of(args, &vname, &rname, per_row);
            let w = Witness::get(&dir, &mat, n, per_row, k);
            let mut runs = vec![];
            for rep in 0..reps.max(1) {
                eprintln!("dequant {} n={n} per_row={per_row} (rep {})", sp.label(), rep + 1);
                let o = dequant::run(&w, &sp, None);
                if !o.accepts() {
                    eprintln!("  REJECTED: stark={} sums={} off_table={} err={:?}",
                              o.proven.stark_ok, o.sums.cancels(), o.off_table,
                              o.proven.prove_error);
                }
                assert!(o.accepts(), "honest dequant proof rejected ({})", sp.label());
                assert_eq!(o.ref_n - o.ref_ok, 0,
                           "gadget disagrees with the f64 reference ({})", sp.label());
                runs.push((o.proven.prove_secs, o.json(&w.source)));
            }
            points.push(median(runs));
        }
    }
    emit("dequant", json!({"points": points, "n": n, "per_row": per_row}));
}

// ------------------------------------------------------------------- width ----
/// What a wider row buys, B3's trade-off on a third component.
fn width_cmd(args: &[String]) {
    let dir = arg_str(args, "--witness");
    let n: usize = arg_scalar(args, "--n", 1 << 20);
    let reps: usize = arg_scalar(args, "--reps", 1usize);
    let mat = arg_str(args, "--mat").unwrap_or_else(|| "gate".into());
    let v = arg_strs(args, "--variants", &["pow2"])[0].clone();
    let mut points = vec![];
    for pr in arg_list(args, "--widths", &[4usize, 8, 16, 32, 64, 128]) {
        let sp = spec_of(args, &v, &arg_strs(args, "--round", &["even"])[0], pr);
        let w = Witness::get(&dir, &mat, n, pr, arg_scalar(args, "--k", 896usize));
        let mut runs = vec![];
        for _ in 0..reps.max(1) {
            eprintln!("width {} per_row={pr}", sp.label());
            let o = dequant::run(&w, &sp, None);
            assert!(o.accepts(), "honest proof rejected at per_row={pr}");
            runs.push((o.proven.prove_secs, o.json(&w.source)));
        }
        points.push(median(runs));
    }
    emit("dequant_width", json!({"points": points, "n": n, "variant": v}));
}

// ------------------------------------------------------------------- kmax -----
/// `kmax` is a completeness knob: how many table rows buy how much accumulator.
fn kmax_cmd(args: &[String]) {
    let dir = arg_str(args, "--witness");
    let n: usize = arg_scalar(args, "--n", 1 << 18);
    let per_row: usize = arg_scalar(args, "--per-row", 16usize);
    let mat = arg_str(args, "--mat").unwrap_or_else(|| "gate".into());
    let mut points = vec![];
    for k in arg_list(args, "--kmax-sweep", &[10u32, 12, 14, 16, 18, 21]) {
        let mut sp = spec_of(args, "pow2", "even", per_row);
        sp.kmax = k;
        let w = Witness::get(&dir, &mat, n, per_row, arg_scalar(args, "--k", 896usize));
        eprintln!("kmax {k}");
        let o = dequant::run(&w, &sp, None);
        let mut j = o.json(&w.source);
        j["accepts"] = json!(o.accepts());
        points.push(j);
    }
    emit("dequant_kmax", json!({"points": points, "n": n, "per_row": per_row}));
}

// --------------------------------------------------------------- freivalds ----
/// What the weight Freivalds check costs per weight, at a sweep of weights per row.
///
/// Cells per weight is a layout quantity, and the sweep is what says so rather than
/// asserting it. The check's three QM31 accumulator columns are 12 M31 cells a
/// trace row, a row carries `p` weights of each matmul in the group, and nothing
/// else in the component grows with the weights -- the committed activations are
/// the same `tokens x n_in` at every `p`, and so is their range check. So each
/// point proves one weight block at two output widths and reports the difference
/// over the weights the two differ by, which leaves the activations, their range
/// check and the shared range table out of the figure the cost model multiplies by
/// a parameter count.
fn freivalds_cmd(args: &[String]) {
    assert!(arg_str(args, "--witness").is_none(),
            "the marginal needs one weight block at two output widths, which a dump \
             has not got: drop --witness and sweep a synthetic block");
    let group = arg_strs(args, "--group", &["o"])[0].clone();
    let n_out: usize = arg_scalar(args, "--nout", 64usize);
    let names = layer::FREIVALDS_GROUPS.iter().find(|(g, _)| *g == group)
        .unwrap_or_else(|| panic!("no Freivalds group {group}")).1;
    let build = |n: usize| -> layer::Layer {
        // the last occurrence of a flag wins, so this overrides whatever the line said
        let mut a = args.to_vec();
        a.push("--nout".into());
        a.push(n.to_string());
        layer_of(&a).0
    };
    let (l1, l2) = (build(n_out), build(2 * n_out));
    let mats: Vec<usize> = l1.mats.iter().enumerate()
        .filter(|(_, m)| names.contains(&m.name.as_str())).map(|(i, _)| i).collect();
    assert!(!mats.is_empty(), "the layer has no matmul of the {group} group");
    let mut points = vec![];
    for p in arg_list(args, "--widths", &[8usize, 16, 32, 64]) {
        eprintln!("freivalds {group} weights_per_row={p}");
        let small = freivalds::run::run(&l1, &group, &mats, p);
        let large = freivalds::run::run(&l2, &group, &mats, p);
        for (o, n) in [(&small, n_out), (&large, 2 * n_out)] {
            assert!(o.accepts(),
                    "honest Freivalds proof rejected at p={p}, n_out={n}: stark={} sums={} \
                     off_table={} reference_mismatch={} err={:?}",
                    o.proven.stark_ok, o.sums.cancels(), o.off_table, o.ref_n - o.ref_ok,
                    o.proven.prove_error);
        }
        points.push(freivalds::run::marginal(&small, &large));
    }
    emit("freivalds", json!({"points": points, "group": group,
                             "n_out": [n_out, 2 * n_out], "source": l2.source}));
}

// ------------------------------------------------------------------- logup ----
/// What one lookup costs in committed cells, at each batch size the crate uses.
///
/// The cost model charges a lookup by a marginal figure -- the cells it adds
/// beyond the values a witness commits anyway -- and until now nothing in the tree
/// measured one: the other cell-cost subcommands each report a whole gadget with
/// its own lookups inside. Every point here proves the crate's own use and table
/// sides at two lookup counts and reports the difference, for the range check
/// every committed integer takes and for a general two-column table lookup, at
/// batch 1 (`finalize_logup`, what a table side finalizes with) and batch 2
/// (`finalize_logup_in_pairs`, what every use side finalizes with).
fn logup_cmd(args: &[String]) {
    let bits: u32 = arg_scalar(args, "--table-bits", 12u32);
    let log_rows: u32 = arg_scalar(args, "--log-rows", 10u32);
    let n: usize = arg_scalar(args, "--lookups-per-row", 4usize);
    let seed: u64 = arg_scalar(args, "--seed", 7u64);
    let batches = arg_list(args, "--batches", &[1usize, 2]);
    let mut points = vec![];
    // the range check first and the general lookup last, so a reader that keeps
    // one figure per batch size keeps the one whose gadget commits a key and a
    // value -- which is the shape `cost_model.py` takes two cells off
    for kind in [logup_run::Kind::Range, logup_run::Kind::Pair] {
        for batch in &batches {
            eprintln!("logup {} batch={batch}", kind.name());
            let small = logup_run::run(kind, bits, n, log_rows, *batch, seed);
            let large = logup_run::run(kind, bits, 2 * n, log_rows, *batch, seed);
            for o in [&small, &large] {
                assert!(o.accepts(),
                        "honest lookup proof rejected ({} batch {batch}, {} a row): stark={} \
                         sums={} off_table={} reference_mismatch={} err={:?}",
                        kind.name(), o.per_row, o.proven.stark_ok, o.sums.cancels(),
                        o.off_table, o.ref_n - o.ref_ok, o.proven.prove_error);
            }
            points.push(logup_run::marginal(&small, &large));
        }
    }
    emit("logup", json!({"points": points, "table_bits": bits, "log_rows": log_rows,
                         "lookups_per_row": [n, 2 * n],
                         "closed_form": {
                             "column_cells": plogup::COLUMN_CELLS,
                             "marginal_cells_per_lookup":
                                 batches.iter().map(|b| plogup::marginal_cells(*b))
                                     .collect::<Vec<_>>()}}));
}

// ------------------------------------------------------------------ probes ----
/// Everything that has to be rejected, and the one thing that must not be.
fn probes_cmd(args: &[String]) {
    let dir = arg_str(args, "--witness");
    let n: usize = arg_scalar(args, "--n", 1 << 12);
    let per_row: usize = arg_scalar(args, "--per-row", 8usize);
    let mat = arg_str(args, "--mat").unwrap_or_else(|| "gate".into());
    let sp = spec_of(args, "pow2", "even", per_row);
    let w = Witness::get(&dir, &mat, n, per_row, arg_scalar(args, "--k", 896usize));
    let mut out = vec![];

    let honest = dequant::run(&w, &sp, None);
    assert!(honest.accepts(), "honest dequant proof rejected");
    let honest_root = honest.proven.roots.first().cloned().unwrap_or_default();
    out.push(json!({"probe": "honest", "note": "the reference run",
                    "result": honest.json(&w.source)}));

    let mut probe = |name: &str, note: &str, t: Tamper| {
        eprintln!("probe {name}");
        let o = dequant::run(&w, &sp, Some(t));
        let root = o.proven.roots.first().cloned().unwrap_or_default();
        out.push(json!({"probe": name, "note": note, "result": o.json(&w.source),
                        "root_differs_from_honest": root != honest_root}));
        (o.accepts(), o.proven.stark_ok, o.sums.cancels(), o.off_table)
    };

    // The wrap probe is the one that matters: it is B4 §1's negative control, and
    // it passes here *because* the normalisation's own bound is the integer reading.
    let (ok, ..) = probe(
        "wrap: decompose acc + p",
        "the same field element, a different integer.  B4's two 2^16 limbs accept \
         this and decode an accumulator off by exactly p; the normalisation cannot, \
         because M2 < 512 and ph <= 2^kmax bound the reconstruction below 2^30",
        Tamper::Wrap(11));
    assert!(!ok, "a wrap-around decomposition must be rejected");

    let (ok, stark, cancels, _) = probe(
        "wrong mantissa",
        "the committed bf16 mantissa is a different, perfectly legal value: every \
         constraint still holds and only the claimed sums object",
        Tamper::Mantissa(13));
    assert!(!ok && stark && !cancels,
            "a wrong mantissa must be caught by the claimed sums, not by verify()");

    let (ok, ..) = probe(
        "non-normalised decomposition",
        "M2 halved and ph doubled, which keeps mag = M2 ph + r satisfied: the \
         normalisation table's domain is what rejects it, since rows with M2 < 256 \
         exist only at ph = 1",
        Tamper::Denorm(17));
    assert!(!ok, "a non-normalised decomposition must be rejected");

    let (ok, ..) = probe(
        "remainder above its bound",
        "r >= ph with M2 lowered to compensate, so the reconstruction is still \
         satisfied: the (ph, r) pair table is the only thing that can see it",
        Tamper::Borrow(19));
    assert!(!ok, "r >= ph must be rejected");

    let (ok, ..) = probe(
        "sticky flag flipped",
        "the flag round-to-nearest-even reads, and the (ph, r, st) table pins it \
         from r -- so flipping it leaves no valid row",
        Tamper::Sticky(23));
    assert!(!ok, "a flipped sticky flag must be rejected");

    let (ok, _, cancels, _) = probe(
        "corrupted table",
        "one row of the normalisation table changed and the whole witness rebuilt \
         to match: internally consistent, claimed sums cancel, and only the \
         preprocessed root differs (B4 section 3, third gadget)",
        Tamper::Table(4321));
    assert!(ok && cancels, "a corrupted table with a matching witness is consistent");
    out.last_mut().unwrap()["extra"] =
        json!({"preprocessed_root_honest": honest_root,
               "note": "the root is the only difference, so every table root is \
                        part of the statement"});

    // the int8 fold's two closed edges (B6 review, open gap 7): both must reject
    // now, and both verified under B6's gadget
    let sp8 = spec_of(args, "int8", "even", per_row);
    let w8 = Witness::get(&dir, &mat, n, per_row, arg_scalar(args, "--k", 896usize));
    let honest8 = dequant::run(&w8, &sp8, None);
    assert!(honest8.accepts(), "honest int8 proof rejected");
    // an element with qm >= 1 for the binding shift, one with qm = 0, acc < 0 for the sign
    let pick = |want_zero: bool| -> Option<usize> {
        (0..w8.acc.len()).find(|i| {
            let den = w8.den[(i / per_row) % w8.den.len()];
            let num = w8.num[(i % per_row) % w8.num.len()];
            let qm = (2 * w8.acc[*i].abs() * num + den) / (2 * den);
            if want_zero { qm == 0 && w8.acc[*i] < 0 } else { qm >= 1 }
        })
    };
    let mut out8 = vec![];
    if let Some(i) = pick(false) {
        eprintln!("probe int8 binding shift");
        let o = dequant::run(&w8, &sp8, Some(Tamper::TieShift(i)));
        assert!(!o.accepts(), "the binding's second witness must be rejected");
        out8.push(json!({"probe": "int8: the binding's other witness (qm - 1, d + 2 den)",
                         "note": "B6's `2 den - d` check accepted this whenever d = 0; \
                                  the -1 excludes it for every d",
                         "accepted": o.accepts(), "stark_ok": o.proven.stark_ok,
                         "sums_cancel": o.sums.cancels(), "off_table": o.off_table}));
    }
    if let Some(i) = pick(true) {
        eprintln!("probe int8 sign flip");
        let o = dequant::run(&w8, &sp8, Some(Tamper::SignFlip(i)));
        assert!(!o.accepts(), "a flipped sub-half sign must be rejected");
        out8.push(json!({"probe": "int8: flipped sign on a sub-half quotient",
                         "note": "accepted before the (q, nz) pin; the constraint \
                                  sg (1 - nz) now refuses it in the prover",
                         "accepted": o.accepts(), "stark_ok": o.proven.stark_ok,
                         "sums_cancel": o.sums.cancels(), "off_table": o.off_table,
                         "prove_error": o.proven.prove_error}));
    } else {
        out8.push(json!({"probe": "int8: flipped sub-half sign",
                         "note": "no qm = 0, acc < 0 element in this witness"}));
    }
    emit("dequant_probes", json!({"n": n, "spec": sp.json(), "probes": out,
                                  "int8_edges": out8}));
}

// -------------------------------------------------------------------- pack ----
/// The packed key tile at `sh`, with an exp table that holds every key it looks up.
///
/// The table is sized from the witness and capped at 2^22 rows, past which the
/// table dominates a single-key-tile benchmark.  One master exponent serves the
/// whole key tile, so every instance below it has its shift scaled up to that
/// exponent by a public power of two, and the wider the key tile the wider that
/// spread: at 8 and 16 instances of this sweep's shape the scaled index is one
/// bit past the cap.  The keys past it were left off the table, where only the
/// rejected proof at the end of the run reported them (#89).
///
/// `--table-bits N` bounds the size where the line gives one, which is what it
/// does for a synthetic layer's table; it never lowers the bound below the 2^22
/// rows the sizing has always been capped at, so the counts at the instance
/// counts that already fit do not move under the flag.  Where the bound cannot
/// hold the keys the sweep refuses, before the first commitment, with the number
/// to pass.
fn packed_key_tile(src: &Src, sh: Shape, variant: attention::witness::Variant) -> Attn {
    let ceiling = src.log_table.max(22);
    // pass 1: the witness's own sizing, capped as it always was
    let a = Attn::get(&Src { log_table: 0, ..src.clone() }, sh, variant);
    let need = a.key_bits();
    if need <= a.log_table {
        return a;
    }
    assert!(need <= ceiling,
            "the packed key tile at insts={} looks up an exp table key of {need} bits \
and --table-bits gives at most {ceiling}: pass --table-bits {need}", sh.insts);
    // pass 2: the keys are counted and the table is sized to hold them.  A key is the
    // shift scaled by a public power of two, and neither of those depends on how many
    // rows the table has, so one repeat is enough.
    Attn::get(&Src { log_table: need, ..src.clone() }, sh, variant)
}

/// The column budget: what packing `insts` heads onto one column set buys.
///
/// This is the question B6 exists to answer.  B5 proved one head at ~1,200 columns
/// and 600 KiB-1 MiB of proof; B1's model has 24 x 14 = 336 head-instances, so
/// per-head instantiation extrapolates to a 200-340 MB proof, two orders over
/// a single-digit MiB proof.  Packing shares one column set across the key tile, and
/// the sweep below is what that is worth.
fn pack_cmd(args: &[String]) {
    let src = src_of(args);
    let s: usize = arg_scalar(args, "--s", 512usize);
    let c: usize = arg_scalar(args, "--ctx", 1024usize);
    let hd: usize = arg_scalar(args, "--hd", 64usize);
    let reps: usize = arg_scalar(args, "--reps", 1usize);
    let cfg = attn_cfg(args, arg_scalar(args, "--scores-per-row", 64usize));
    let v = attention::witness::Variant::parse(&arg_strs(args, "--variants", &["intexp"])[0]);
    let mut points = vec![];
    for insts in arg_list(args, "--insts", &[1usize, 2, 4, 8, 16]) {
        assert!(insts.is_power_of_two(), "insts must be a power of two (see layout.rs)");
        let sh = Shape { s, c, hd, insts };
        let a = packed_key_tile(&src, sh, v);
        let mut runs = vec![];
        for rep in 0..reps.max(1) {
            eprintln!("pack {} {} (rep {}, exp table 2^{} rows)",
                      sh.label(), v.name(), rep + 1, a.log_table);
            let o = attention::run::run(&a, &cfg);
            if !o.accepts() {
                eprintln!("  REJECTED: stark={} qk={} av={} den={} sums={} off={}",
                          o.proven.stark_ok, o.qk_ok, o.av_ok, o.den_ok, o.sums.cancels(),
                          o.off_table);
            }
            assert!(o.accepts(), "honest packed proof rejected at insts={insts}");
            runs.push((o.proven.prove_secs, o.json(&a, &cfg)));
        }
        let mut p = median(runs);
        p["insts"] = json!(insts);
        points.push(p);
    }
    emit("pack", json!({"points": points, "s": s, "ctx": c, "hd": hd}));
}

/// Everything the packed attention has to reject, at a key tile of instances.
fn attn_probes_cmd(args: &[String]) {
    let src = src_of(args);
    let sh = Shape {
        s: arg_scalar(args, "--s", 64usize),
        c: arg_scalar(args, "--ctx", 256usize),
        hd: arg_scalar(args, "--hd", 64usize),
        insts: arg_scalar(args, "--insts", 4usize),
    };
    let cfg = attn_cfg(args, arg_scalar(args, "--scores-per-row", 16usize));
    let v = attention::witness::Variant::IntExp;
    let honest = attention::run::run(&Attn::get(&src, sh, v), &cfg);
    assert!(honest.accepts(), "honest packed proof rejected");
    let mut out = vec![json!({"probe": "honest",
                              "result": honest.json(&Attn::get(&src, sh, v), &cfg)})];
    let mut cases: Vec<(&str, Corrupt)> = vec![
        ("corrupted key in one instance", Corrupt::Key(1234, 3)),
        ("corrupted value in one instance", Corrupt::Value(999, 5)),
        ("wrong numerator in one instance", Corrupt::Numerator(555)),
        ("shift below the maximum in one instance", Corrupt::LowMax(3)),
        ("corrupted denominator in one instance", Corrupt::Denominator(7, 1)),
    ];
    if src.diagonal_block {
        // the diagonal block's own control: a token that attends to a future key
        cases.push(("attends to one future key", Corrupt::Unmask(11)));
        cases.push(("attends to a second future key", Corrupt::Unmask(4242)));
    }
    for (name, bad) in cases {
        eprintln!("attn probe {name}");
        let a = Attn::corrupted(&src, sh, v, bad);
        let o = attention::run::run(&a, &cfg);
        assert!(!o.accepts(), "{name} must be rejected");
        out.push(json!({"probe": name, "result": o.json(&a, &cfg)}));
    }
    emit("attn_probes", json!({"shape": sh.label(), "probes": out}));
}

// ------------------------------------------------------------------ argmax ----
/// B10 step 4: the token binding, at both strengths, so the difference is measured.
fn argmax_cmd(args: &[String]) {
    let tokens: usize = arg_scalar(args, "--tokens", 8usize);
    let vocab: usize = arg_scalar(args, "--vocab", 4096usize);
    let bindings: f64 = arg_scalar(args, "--bindings", 0.25f64);
    let pr: usize = arg_scalar(args, "--per-row", 64usize);
    let seed: u64 = arg_scalar(args, "--seed", 7u64);
    // `--witness DIR` proves the argmax on the head's own keys: `key_head.u16`
    // and `tok_head.i32` out of an `extract.py --head` dump, with the shape read
    // from that dump's manifest rather than from the flags.
    let w = match arg_str(args, "--witness") {
        Some(dir) => {
            let man: serde_json::Value =
                witread::read_manifest(&dir).expect("manifest.json");
            let tk = man["head"]["tokens"].as_u64().expect("manifest: head.tokens") as usize;
            let vc = man["head"]["vocab"].as_u64().expect("manifest: head.vocab") as usize;
            eprintln!("argmax on the dump's own keys: {tk} rows of {vc}");
            argmax::AWitness::from_dump(&dir, tk, vc).expect("the head dump's argmax columns")
        }
        None => argmax::AWitness::synthetic(tokens, vocab, bindings, seed),
    };
    let (tokens, vocab) = (w.tokens, w.vocab);
    let mut points = vec![];
    for lax in [false, true] {
        eprintln!("argmax {}x{} lax={lax}", tokens, vocab);
        let (o, lay) = argmax::run(&w, lax, pr, None);
        if !o.accepts() {
            eprintln!("  REJECTED: stark={} sums={} off={}", o.proven.stark_ok,
                      o.sums.cancels(), o.off_table);
        }
        assert!(o.accepts(), "honest argmax proof rejected (lax={lax})");
        points.push(o.json(&w, &lay));
    }
    emit("argmax", json!({"points": points, "co_max_rows_in_witness": w.co_max_rows, "tokens": tokens,
                          "vocab": vocab}));
}

/// What each statement rejects -- and, for the binding, what only the strict one does.
fn argmax_probes_cmd(args: &[String]) {
    let tokens: usize = arg_scalar(args, "--tokens", 8usize);
    let vocab: usize = arg_scalar(args, "--vocab", 4096usize);
    let pr: usize = arg_scalar(args, "--per-row", 64usize);
    // every row bound, so the binding probe always has a second maximum to move to
    let w = argmax::AWitness::synthetic(tokens, vocab, 1.0, arg_scalar(args, "--seed", 7u64));
    let cases: Vec<(&str, argmax::Tamper)> = vec![
        ("the token is a *later* co-maximal entry", argmax::Tamper::CoMax(3)),
        ("the token does not attain the maximum", argmax::Tamper::NotMax(5)),
        ("the claimed maximum is above every entry", argmax::Tamper::PmHigh(2)),
        ("the claimed maximum is below the true one", argmax::Tamper::PmLow(4)),
        // the strict close's own probe: `tok = vocab` is the all-ones indicator,
        // killed only by the final-slot constraint; the lax sum rejects it too
        ("the token equals the vocabulary size", argmax::Tamper::TokVocab(1)),
    ];
    let mut out = vec![];
    for lax in [false, true] {
        for (name, bad) in &cases {
            eprintln!("argmax probe [{}] {name}", if lax { "lax" } else { "strict" });
            let (o, lay) = argmax::run(&w, lax, pr, Some(*bad));
            out.push(json!({"probe": name, "lax": lax, "rejected": !o.accepts(),
                            "result": o.json(&w, &lay)}));
            // The binding probe is exactly the difference between the two statements:
            // the lax one cannot see it, and saying so is the point of running it.
            let is_binding = name.contains("co-maximal");
            if !lax {
                assert!(!o.accepts(), "strict: {name} must be rejected");
            } else if !is_binding {
                assert!(!o.accepts(), "lax: {name} must be rejected");
            }
        }
    }
    emit("argmax_probes", json!({"probes": out, "tokens": tokens, "vocab": vocab,
                                 "co_max_rows_in_witness": w.co_max_rows}));
}


// ------------------------------------------------------------------ embed ----

/// Stream BB: the embedding gadget on its own, and its cells price.
///
/// `--witness DIR` proves it on the dump's own `embed.u16`, `tok_in.i32` and
/// the wiring family's `xin`; without one it runs a synthetic block, which is
/// a shape check and nothing more -- the record says so wherever a synthetic
/// number appears.
fn embed_cmd(args: &[String]) {
    let target: usize = arg_scalar(args, "--embed-per-row", 256usize);
    let w = match arg_str(args, "--witness") {
        Some(dir) => {
            let (l, _) = {
                let mut a = args.to_vec();
                a.push("--witness".into());
                a.push(dir.clone());
                layer_of(&a)
            };
            let g = l.residual_stream.as_ref()
                .expect("embed needs a witness extracted with --residual-stream");
            let tab = l.embed_tab.as_ref().expect("embed needs embed.u16 in the dump");
            let tok = l.tok_in.as_ref().expect("embed needs tok_in.i32 in the dump");
            let vocab = tab.len() / g.d;
            eprintln!("embed on the dump's own table: {} positions of {} channels, \
                       vocabulary {vocab}", g.tokens, g.d);
            embed::EWitness::from_cols(g.tokens, g.d, vocab, tab.clone(), tok.clone(),
                                       g.xin.clone())
                .expect("the dump's embedding columns")
        }
        None => embed::EWitness::synthetic(
            arg_scalar(args, "--tokens", 16usize),
            arg_scalar(args, "--d", 128usize),
            arg_scalar(args, "--vocab", 512usize),
            arg_scalar(args, "--seed", 11u64)),
    };
    let (o, lay) = embed::run(&w, target, None);
    if !o.accepts() {
        eprintln!("  REJECTED: stark={} sums={} off={} binding={}", o.proven.stark_ok,
                  o.sums.cancels(), o.off_table, o.binding_ok);
    }
    assert!(o.accepts(), "honest embedding proof rejected");
    // the forecast at the two real vocabularies, beside the measured point, so
    // the record never has to re-derive the arithmetic
    let price = json!({
        "llama8b": embed::price(4096, 512, 128_256, target),
        "qwen05b": embed::price(896, 512, 151_936, target),
    });
    // What the emit chain will see, at the *real* shapes: an emit is a box slot,
    // so the two kernels' sizes are worth knowing before one is booked.  This
    // lowers the two evaluators and reports the IR; it proves nothing and runs
    // no prover.
    let shapes: Vec<serde_json::Value> = [4096usize, 896]
        .iter()
        .flat_map(|d| {
            let vocab = if *d == 4096 { 128_256usize } else { 151_936 };
            let l = embed::EmbLayout::choose(*d, 512, vocab, target);
            let rel = {
                let mut ch = stwo::core::channel::Blake2sM31Channel::default();
                stwo::core::channel::Channel::mix_u64(&mut ch, 0xB65_BB01);
                embed::EmbRel::draw(&mut ch, l.arity())
            };
            let rk = {
                let mut ch = stwo::core::channel::Blake2sM31Channel::default();
                stwo::core::channel::Channel::mix_u64(&mut ch, 0xB65_BB02);
                crate::relations::RangeRel::draw(&mut ch)
            };
            let felt = |a: u32| stwo::core::fields::qm31::SecureField::from_m31_array(
                [common::m31_u32(a), common::m31_u32(a + 1),
                 common::m31_u32(a + 2), common::m31_u32(a + 3)]);
            let mut binding = embed::binding_w(felt(7), felt(101), *d, 512, &l);
            binding.claimed = felt(1_000_003);
            binding.claimed_tok = felt(2_000_003);
            let use_eval = embed::EmbedEval { log_rows: l.log_use, per_row: l.pe,
                                              rel: rel.clone(), rel_ef: rk, binding };
            let tab_eval = embed::EmbTableEval { log_rows: l.log_tab, per_row: l.pe, rel };
            let one = |name: &str, ir: cudagen::Ir| json!({
                "family": name, "d": d, "vocab": vocab,
                "layout": l.json(512, vocab),
                "fingerprint": format!("{:#010x}", ir.fingerprint()),
                "instructions": ir.ops.len(), "constraints": ir.constraints.len(),
                "scalars": ir.scalars.len(), "m31_muls_per_row": ir.m31_muls(),
                "loads_per_row": ir.loads(),
            });
            vec![
                one("zk_embed_eval",
                    cudagen::lower(&use_eval, stwo::core::fields::qm31::SecureField::default(),
                                   l.log_use, l.log_use + 1)),
                one("zk_emb_table_eval",
                    cudagen::lower(&tab_eval, stwo::core::fields::qm31::SecureField::default(),
                                   l.log_tab, l.log_tab + 1)),
            ]
        })
        .collect();
    emit("embed", json!({"point": o.json(&w, &lay), "price_forecast": price,
                         "emit_shapes": shapes}));
}

/// What the gadget rejects, and which half of the argument each probe aims at.
fn embed_probes_cmd(args: &[String]) {
    let target: usize = arg_scalar(args, "--embed-per-row", 256usize);
    let w = embed::EWitness::synthetic(
        arg_scalar(args, "--tokens", 16usize),
        arg_scalar(args, "--d", 128usize),
        arg_scalar(args, "--vocab", 512usize),
        arg_scalar(args, "--seed", 11u64));
    let cases: Vec<(&str, &str, embed::Tamper)> = vec![
        ("one token id moved, the committed row left alone", "the embedding lookup",
         embed::Tamper::Tok(3)),
        ("one token id moved *and* its row recomputed from the table",
         "the binding to the layer's residual-stream input", embed::Tamper::TokRow(5)),
        ("one committed channel of the stream changed", "the embedding lookup",
         embed::Tamper::Val(777)),
    ];
    let mut out = vec![];
    for (name, by, bad) in &cases {
        eprintln!("embed probe {name}");
        let (o, lay) = embed::run(&w, target, Some(*bad));
        out.push(json!({"probe": name, "expect_reject_by": by, "rejected": !o.accepts(),
                        "result": o.json(&w, &lay)}));
        assert!(!o.accepts(), "{name} must be rejected");
    }
    // The table probe is the pinned-root probe and the standalone harness has no
    // pinned root to check against, so it is *expected to accept here* and the
    // composed run is where it bites. Running it anyway is what says so.
    // Stream KK: the cell is chosen in a row *no token uses*, so the rebuilt
    // witness is the honest one value for value and the pinned root is the only
    // thing that can see the change.  A cell in a used row makes the stream
    // differ from the layer's, which the binding catches first -- a rejection for
    // the wrong reason, and one that came and went with the witness's draw.
    let unused = (0..w.vocab as u32).find(|v| !w.tok.contains(v)).unwrap_or(0) as usize;
    let (o, lay) = embed::run(&w, target, Some(embed::Tamper::Table(unused * w.d + 1)));
    out.push(json!({"probe": "one cell of the committed table changed, witness rebuilt",
                    "expect_reject_by": "the pinned preprocessed root (composed only)",
                    "rejected": !o.accepts(), "accepts_standalone": o.accepts(),
                    "result": o.json(&w, &lay)}));
    assert!(o.accepts(),
            "the table probe must be internally consistent -- only the pinned root sees it");
    // Stream KK, `notes/witness-slack.md` section 1a, both arms of the same
    // witness: a position whose committed id is a *padded* one, outside the
    // vocabulary and inside the table, with the all-zero stream row that padded
    // table row held.  Nothing here is corrupted -- every constraint the component
    // carries holds -- so before the sentinel this accepted, and that is the
    // whole item.  Both runs are made, because "it rejects now" is only half
    // the claim.
    let lay0 = embed::EmbLayout::choose(w.d, w.tokens, w.vocab, target);
    if lay0.vpad == w.vocab {
        out.push(json!({"probe": "a token id outside the vocabulary",
                        "expect_reject_by": "the embedding lookup",
                        "skipped": "this vocabulary is a power of two, so the table \
                                    has no padded row and section 1a is vacuous here"}));
        emit("embed_probes", json!({"probes": out, "tokens": w.tokens, "d": w.d,
                                    "vocab": w.vocab, "vocab_padded": lay0.vpad}));
        return;
    }
    let wp = embed::EWitness::padded(w.tokens, w.d, w.vocab, target, 23, 3);
    let (o_open, lay) = embed::run(&wp, target, Some(embed::Tamper::LegacyPad));
    out.push(json!({"probe": "a token id outside the vocabulary, with a zero stream \
                              row, against the pre-KK table",
                    "expect_reject_by": "nothing -- this is the open statement, and it accepts",
                    "rejected": !o_open.accepts(), "accepts_before_the_fix": o_open.accepts(),
                    "result": o_open.json(&wp, &lay)}));
    let (o_shut, lay) = embed::run(&wp, target, None);
    out.push(json!({"probe": "the same witness against the table this crate builds",
                    "expect_reject_by": "the embedding lookup: a padded row carries the \
                                         out-of-range block index k, is the tuple of no use \
                                         row, and its multiplicity cannot cancel",
                    "rejected": !o_shut.accepts(), "result": o_shut.json(&wp, &lay)}));
    assert!(o_open.accepts(),
            "the pre-KK table must accept the padded id -- that is what section 1a records");
    assert!(!o_shut.accepts(), "the padded id must be rejected now");
    emit("embed_probes", json!({"probes": out, "tokens": w.tokens, "d": w.d,
                                "vocab": w.vocab, "vocab_padded": lay0.vpad}));
}

// ----------------------------------------------------------------- compose ----

/// Build the layer witness: real from `--witness`, else a small synthetic one.
///
/// A `--witness` that does not load is fatal (exit 2), never a fallback.  It
/// used to print one stderr line and prove a synthetic layer instead, and a
/// synthetic layer has `residual_stream: None`, so a gate pointed at a dump the box did
/// not hold came back `rc=0` with the whole residual-stream battery skipped -- a pass that
/// measured nothing (open item 17).
/// Stream KK: the chunk's unchecked coordinate, `--layer N` / `--block N` over
/// the manifest's own.
///
/// Not a switch on a check: the coordinate goes into the statement either way
/// and `boundary-grid` compares it either way.  It is an override because one
/// chunk's manifest cannot know where the grid puts it -- the `lm_head` chunk
/// is extracted at some layer and proved *after* the last one, so its plan
/// line says `--layer 32` and its block is the block it reads.
fn coord_of(l: &mut layer::Layer, args: &[String]) {
    if let Some(v) = arg_str(args, "--layer") {
        l.layer = Some(v.parse().expect("--layer takes a layer index"));
    }
    if let Some(v) = arg_str(args, "--block") {
        l.block = Some(v.parse().expect("--block takes a block index"));
    }
}

/// Issue #112: the per-layer calibration a parameter-only load takes, and the
/// switch that turns that load on.
///
/// Two forms, and each of them is a thing this repository already publishes:
///
/// * `--frac-per-head a,b,c` -- the per-layer calibration, one exponent per
///   head. It is what each `zk-prover model-roots` row carries and what
///   `extract.py` writes into a dump's manifest, and `Layer::calibration_of`
///   cycles it into the key tiles exactly as the model-only load does.
/// * `--calibration FILE` -- a JSON document carrying either that same
///   `frac_per_head` (a dump's `manifest.json`, or one row of a model
///   manifest) or the per-key-tile `calibration` block every run publishes as
///   `result.shape` (issue #18). The second form is the general one: a segment
///   whose key tiles are not one exponent per head -- a synthetic segment is
///   such a segment -- has no `frac_per_head` and this names its calibration
///   exactly.
///
/// Either flag is both the switch and the input: with one of them on the line
/// the segment is built from the statement, and the only files opened are the
/// two the line names.
fn calibration_of_args(args: &[String], p: &compose::Params)
                       -> Option<shaperoot::Calibration> {
    let fracs = arg_str(args, "--frac-per-head").map(|_| {
        let v = arg_list::<f64>(args, "--frac-per-head", &[]);
        assert!(!v.is_empty(), "--frac-per-head wants a comma-separated list of exponents");
        v
    });
    let doc = arg_str(args, "--calibration").map(|f| {
        let bytes = std::fs::read(&f).unwrap_or_else(|e| panic!("--calibration {f}: {e}"));
        let v: Value = serde_json::from_slice(&bytes)
            .unwrap_or_else(|e| panic!("--calibration {f}: not JSON: {e}"));
        (f, v)
    });
    assert!(fracs.is_none() || doc.is_none(),
            "--frac-per-head and --calibration are two names for one input: pass one");
    if let Some(v) = fracs {
        return Some(layer::Layer::calibration_of(p, &v));
    }
    let (f, v) = doc?;
    // the document forms are `shaperoot`'s to read: the verifier reads the same
    // ones off `--verify-calibration` (issue #95), and two readers of one
    // document would be two calibrations to drift apart
    Some(shaperoot::calibration_of_doc(&f, &v, p).unwrap_or_else(|e| panic!("--calibration {e}")))
}

/// The table width the line or the manifest states, in bits; 0 when neither names one.
fn stated_table_bits(src: &Src) -> u32 {
    match (src.log_table, &src.dir) {
        (0, Some(dir)) => witread::read_manifest(dir)
            .ok()
            .and_then(|m| m["master_table_bits"].as_u64())
            .unwrap_or(0) as u32,
        (n, _) => n,
    }
}

/// The commands that build a layer only to read its `Params`.
fn lowering(args: &[String]) -> bool {
    matches!(args.first().map(String::as_str), Some("cudaemit") | Some("genemit"))
}

/// Under `--tight-table`, a lowering command loads once at a 16-row table to
/// learn the keys' width and once more at that width; every row of the table
/// is a function of its own index, so this is the layer `tighten_of` would
/// cut the wide one into.
pub fn layer_of(args: &[String]) -> (layer::Layer, String) {
    if lowering(args)
        && flag(args, "--tight-table")
        && !flag(args, "--weights-only")
        && !flag(args, "--calibration")
        && !flag(args, "--frac-per-head")
        && !flag(args, "--synth-residual-stream") // its rotation sizes the table off the key tiles
    {
        let slack: i32 = arg_scalar(args, "--tight-table", 0i32);
        let stated = stated_table_bits(&src_of(args));
        if slack >= 0 && stated > 0 {
            eprintln!("tight-table: probe load at a 16-row table");
            let (probe, _) = layer_at(args, Some(4), false);
            let (need, key_tiles) = (probe.key_bits(), probe.key_tiles.len());
            drop(probe);
            let width = ((need as i32 + slack).max(1) as u32).min(stated);
            eprintln!("tight-table: {key_tiles} key tiles, the keys need 2^{need} rows; \
loading at 2^{width}, stated 2^{stated}");
            return layer_at(args, Some(width), true);
        }
    }
    layer_at(args, None, true)
}

/// The layer a command line names, at `table_bits` rows when given; `finish`
/// applies `--tight-table` and the coordinate.
fn layer_at(args: &[String], table_bits: Option<u32>, finish: bool) -> (layer::Layer, String) {
    let vname = arg_strs(args, "--variants", &["intsplit6"])[0].clone();
    let variant = attention::witness::Variant::parse(&vname);
    let head_groups = arg_list(args, "--head-groups", &[8usize, 4, 2]);
    let src = match table_bits {
        Some(b) => Src { log_table: b, ..src_of(args) },
        None => src_of(args),
    };
    // Issue #112: the parameter-only load, before the dump's -- the statement
    // and the calibration, and no dump on disk. It derives the **shape** root:
    // it holds no model, so the model tree it would commit is a tree of
    // stand-ins and `--shape-only` is what stops before it.
    if flag(args, "--calibration") || flag(args, "--frac-per-head") {
        assert!(compose::SHAPE_ONLY.load(std::sync::atomic::Ordering::Relaxed),
                "the parameter-only load holds no model, so it can derive the shape root \
                 and not the model root: pass `expect-root --shape-only`");
        assert!(src.dir.is_none(),
                "--calibration builds the segment from the statement alone: drop --witness, \
                 or drop --calibration and let the model-only load read the dump");
        assert!(arg_str(args, "--params").is_some(),
                "the parameter-only load needs --params FILE (the segment's own \
                 `Fixed.params`): every layout, every table and every selector comes off \
                 the statement");
        let (_, p) = plan_params(args);
        let calib = calibration_of_args(args, &p).expect("the calibration");
        let mut l = layer::Layer::from_params(&p, &calib);
        // the statement's own coordinate, when `--params` is a `Fixed` and not
        // a bare `Params`; `--layer` / `--block` override it as they do a
        // manifest's, through `coord_of` below
        if let Some(f) = arg_str(args, "--params") {
            if let Ok(v) = serde_json::from_slice::<Value>(&std::fs::read(&f).unwrap_or_default()) {
                l.layer = v["layer"].as_u64().map(|x| x as usize);
                l.block = v["block"].as_u64().map(|x| x as usize);
            }
        }
        eprintln!("expect-root: the segment from the statement and the calibration \
(no dump: {} key tiles, c={} s={} log_table={})",
                  l.key_tiles.len(), p.c, p.s, p.log_table);
        coord_of(&mut l, args);
        let vname = p.variant.clone();
        return (l, vname);
    }
    if let Some(dir) = &src.dir {
        // Stream LL (E11): the verifier's load.  Only the model half of the dump
        // is read; see `layer::Layer::load_weights`.  `--tight-table` sizes the
        // exp table off the witness's own shifts, so it is refused here rather
        // than silently sizing a table from stand-in zeros.
        let weights_only = flag(args, "--weights-only");
        assert!(!weights_only || !args.iter().any(|a| a == "--tight-table"),
                "--weights-only and --tight-table are exclusive: --tight-table \
                 sizes the exp table off the witness's own shifts, and the \
                 model-only load has no witness -- the size comes from the \
                 statement's `log_table` instead");
        // Under `--weights-only` **every** loader knob comes off the statement,
        // not off the command line: the variant, the prefix key tile list, whether
        // the dump is read with its diagonal block and whether the prefix is tiled, the
        // master table's size and `nmax`.  A verifier holds `Fixed`, not the
        // plan line that produced it -- and a chunk proved with `--tight-table`
        // carries a `log_table` that neither the manifest nor any flag names.
        // So `zk-prover expect-root --weights-only --params <tag>.fixed.json --witness
        // DIR` is the whole command, and there is no line to get wrong.
        let (variant, vname, head_groups, src) = if weights_only {
            let f = arg_str(args, "--params").expect(
                "--weights-only needs --params FILE (the chunk's own `Fixed.params`)");
            let bytes = std::fs::read(&f).unwrap_or_else(|e| panic!("--params {f}: {e}"));
            let v: serde_json::Value = serde_json::from_slice(&bytes)
                .unwrap_or_else(|e| panic!("--params {f}: not JSON: {e}"));
            let p: compose::Params =
                serde_json::from_value(v.get("params").cloned().unwrap_or(v))
                    .unwrap_or_else(|e| panic!("--params {f}: not a statement's params: {e}"));
            eprintln!("expect-root: model-only load from the statement \
(variant={} head_groups={:?} diagonal_block={} kv_boundary={} log_table={} nmax={})",
                      p.variant, p.head_groups, p.diagonal_block, p.kv_boundary, p.log_table, p.nmax);
            (attention::witness::Variant::parse(&p.variant), p.variant.clone(), p.head_groups.clone(),
             attention::witness::Src { log_table: p.log_table, nmax: p.nmax,
                                 diagonal_block: p.diagonal_block, tile: p.kv_boundary, ..src.clone() })
        } else {
            (variant, vname, head_groups, src.clone())
        };
        let loaded = if weights_only {
            layer::Layer::load_weights(dir, &src, &head_groups, variant)
        } else {
            // Issue #53: `--round` is not a free choice on a dump -- it is the
            // mode the dump was quantized under, and the loader compares the two.
            layer::Layer::load(dir, &src, &head_groups, variant,
                               Round::parse(&arg_strs(args, "--round", &["even"])[0]))
        };
        match loaded {
            Ok(mut l) => {
                // Stream LL: the one stand-in the statement has to choose.
                // `jo` is the attention output's per-token scale exponent --
                // committed data, not model data, so the model-only load has
                // none -- and the AO component admits it only inside the stated
                // window (`joff = jo_max - jo >= 0`, `layer.rs`'s "attnout: jo
                // above the window's top").  The window's *floor* is the safe
                // end: it makes `e = jv - jo` non-negative, so the fold's `pj`
                // is 1 and `denp` stays small, where the ceiling would put
                // `2 denp` within a factor of two of the field.  No
                // preprocessed column reads `jo`, so this moves no tree-0 cell.
                if weights_only {
                    let (_, p) = plan_params(args);
                    for x in l.ao.jo.iter_mut() {
                        *x = p.ao_jo_min;
                    }
                }
                if finish {
                    tighten_of(&mut l, args);
                    coord_of(&mut l, args);
                }
                return (l, vname);
            }
            Err(e) => {
                eprintln!("--witness {dir}: the layer witness did not load: {e}");
                eprintln!("refusing to fall back to a synthetic layer; drop --witness to prove one");
                std::process::exit(2);
            }
        }
    }
    let head_groups = arg_list(args, "--head-groups", &[2usize, 1]);
    let mut l = layer::Layer::synthetic(
        arg_scalar(args, "--s", 32usize), arg_scalar(args, "--ctx", 128usize),
        arg_scalar(args, "--hd", 64usize), &head_groups, variant, &src,
        arg_scalar(args, "--nin", 64usize), arg_scalar(args, "--nout", 48usize), src.seed);
    // Stream Z: exercise the widened attention output on a synthetic layer --
    // a real dump declares its width in the manifest, a synthetic one takes it
    // here.  The int8-calibrated synthetic data stays complete at any wider
    // width, so this widens only the table domain and the o Freivalds check.
    l.attn_out_bits = arg_scalar(args, "--synth-ao-bits", l.attn_out_bits);
    // Stream II: the wiring family on a synthetic layer.  A witness-construction
    // switch and not a statement one -- without it a synthetic layer has no
    // residual-stream columns and composes exactly as it always did, which is
    // what keeps every other stream's recorded synthetic digest where it is.
    if flag(args, "--synth-residual-stream") {
        let round = Round::parse(&arg_strs(args, "--round", &["even"])[0]);
        l.synth_residual_stream(src.seed, arg_scalar(args, "--residual-stream-a", 32usize), round);
    }
    // Stream II (E11): the head chunk on a synthetic layer -- an `lm_head`
    // matmul whose committed input is the final norm's output, and the wiring
    // family that proves it.  A witness-construction switch like
    // `--synth-residual-stream`: without it a synthetic layer has no `lm_head` and no
    // family, and composes exactly as it always did.
    if flag(args, "--synth-head") {
        let round = Round::parse(&arg_strs(args, "--round", &["even"])[0]);
        l.synth_head(src.seed, arg_scalar(args, "--residual-stream-a", 32usize), round,
                     arg_scalar(args, "--synth-vocab", 256usize));
    }
    if finish {
        tighten_of(&mut l, args);
        coord_of(&mut l, args);
    }
    (l, vname)
}

/// Stream J, off by default: `--tight-table [slack]` sizes the master exp table by
/// the largest key the key tiles actually look up rather than by the manifest's
/// `d_bits + spread`.  A negative slack is the tamper -- it takes the table *below*
/// what the witness needs, leaving off-table keys with the numerators they already
/// have, which no honest lookup argument can satisfy.
fn tighten_of(l: &mut layer::Layer, args: &[String]) {
    if !args.iter().any(|a| a == "--tight-table") {
        return;
    }
    let slack: i32 = arg_scalar(args, "--tight-table", 0i32);
    let need = l.key_bits() as i32;
    let (before, after) = l.set_master_table((need + slack).max(1) as u32);
    let off: usize = l.key_tiles.iter().map(|b| b.off_table).sum();
    eprintln!("tight-table: 2^{before} -> 2^{after} (key max {}, {} bits, slack {slack}, \
off_table {off})",
              l.key_tiles.iter().map(|b| b.key_max()).max().unwrap_or(0), need);
}

/// The design point a lowering command works on: a witness dump, or -- with
/// `--params FILE` -- the `params` object of a serialized statement.
///
/// Stream EE (open item 21).  `Fixed` carries every field `params_of` derives
/// from a dump, so a `<tag>.fixed.json` a run left on disk is a complete
/// description of that chunk's plan.  Reading it back is how a machine with no
/// witness -- the coordinating box -- lowers a real 8B plan line and compares
/// the fingerprints against what the run asked for.  It is a lowering path
/// only: nothing here can prove, because a statement carries no trace.
pub fn plan_params(args: &[String]) -> (String, compose::Params) {
    if let Some(f) = arg_str(args, "--params") {
        let bytes = std::fs::read(&f)
            .unwrap_or_else(|e| panic!("--params {f}: {e}"));
        let v: serde_json::Value = serde_json::from_slice(&bytes)
            .unwrap_or_else(|e| panic!("--params {f}: not JSON: {e}"));
        // a `Fixed` (what `--ser-dir` writes) or a bare `Params`
        let p = v.get("params").cloned().unwrap_or(v);
        let params: compose::Params = serde_json::from_value(p)
            .unwrap_or_else(|e| panic!("--params {f}: not a statement's params: {e}"));
        eprintln!("params: {f} (s={} c={} key_tiles={:?} kv_boundary={})",
                  params.s, params.c, params.head_groups, params.kv_boundary);
        return (format!("--params {f}"), params);
    }
    let (l, vname) = layer_of(args);
    let params = params_of(&l, args, &vname);
    (l.source.clone(), params)
}

/// Stream LL (E11): where a derivation's `Params` come from.
///
/// `zk-prover compose` derives them from the witness, as it always has. `zk-prover
/// expect-root` may instead be handed `--params FILE` -- a `Fixed` or a bare
/// `Params`, `plan_params`' own two shapes -- and then the plan is built from
/// *the statement's* own shape rather than from a re-derivation of it.
/// That is the shape the check wants: the verifier holds the statement, and
/// four of the fields (`ao_jo_min` / `ao_jo_max` and `ao_acc_lo_bits` /
/// `ao_acc_hi_bits`) are read off activations a verifier does not have, so
/// `--weights-only` cannot derive them at all and requires the file.
///
/// Gated on `PRE_ONLY` so that `compose`'s own behaviour is untouched: nothing
/// but `expect-root` reads `--params` here.
fn expect_root_params(l: &layer::Layer, args: &[String], vname: &str) -> compose::Params {
    if !compose::PRE_ONLY.load(std::sync::atomic::Ordering::Relaxed) {
        assert!(!flag(args, "--weights-only"),
                "--weights-only is `expect-root`'s: it loads no activations, so \
                 there is nothing to prove");
        return params_of(l, args, vname);
    }
    match arg_str(args, "--params") {
        Some(_) => {
            let (src, p) = plan_params(args);
            eprintln!("expect-root: params from the statement ({src})");
            p
        }
        None => {
            assert!(!flag(args, "--weights-only"),
                    "--weights-only needs --params FILE (the chunk's own \
                     `Fixed.params`): `ao_jo_min`/`ao_jo_max` and \
                     `ao_acc_lo_bits`/`ao_acc_hi_bits` are read off activations \
                     this load does not have");
            params_of(l, args, vname)
        }
    }
}

pub fn params_of(l: &layer::Layer, args: &[String], vname: &str) -> compose::Params {
    let b0 = &l.key_tiles[0];
    let kmax = l.mats.iter().map(|m| m.kmax_blind()).max().unwrap()
        .min(bf16::KMAX_SOUND);
    // B10 step 3: `key_tiles` names the *prefix* key tiles only.  The diagonal block rides as one
    // more key tile per prefix key tile, with the same `insts` and head offset, and the
    // plan rebuilds them from the flag rather than from the list.
    let n_pre = l.key_tiles.iter().filter(|b| !b.diagonal_block).count();
    let diagonal_block = l.key_tiles.len() > n_pre;

    // The sound defaults.  Each gadget is on wherever the witness can state
    // it; a witness that cannot is refused, and only the named skip gets past
    // the refusal.  See `UNSOUND_SKIPS`.
    check_skips(args);
    let skip = |name: &str| skip(args, name);
    let head = l.residual_stream.as_ref().is_some_and(|g| g.head);
    let has_head_mat = l.mats.iter().any(|m| m.name == "lm_head");
    let family = l.residual_stream.is_some() && !skip("residual-stream");
    assert!(l.residual_stream.is_some() || skip("residual-stream"),
            "compose: this witness carries no residual-stream columns (the norms, the \
             residual adds and the requantize seams), so nothing between two matmuls would \
             be proved; extract it with --residual-stream, or pass --unsound-skip-residual-stream to \
             prove the unbound statement");
    let layer_boundary = family && !skip("layer-boundary");
    assert!(diagonal_block || head || skip("kv-boundary"),
            "compose: this chunk has no diagonal block (extracted or built without \
             --diagonal-block), so its own keys and values are not in the statement and the K/V \
             boundary cannot be committed; pass --diagonal-block, or --unsound-skip-kv-boundary \
             to prove the unbound statement");
    let kv_boundary = diagonal_block && !head && !skip("kv-boundary");
    assert!(!has_head_mat || l.head_keys.is_some() || skip("argmax"),
            "compose: the head chunk carries no key_head.u16, so the emitted token cannot \
             be bound to the logits; pass --unsound-skip-argmax to prove the head without \
             the argmax");
    let argmax = has_head_mat && l.head_keys.is_some() && !skip("argmax");
    // The lax form weakens the argmax; it has nothing to weaken where the
    // argmax is not stated, and a command line that asks for a control it does
    // not get is an error rather than a run that quietly proves something else.
    let argmax_lax = flag(args, ARGMAX_LAX);
    assert!(!argmax_lax || argmax,
            "{ARGMAX_LAX}: this statement states no argmax to weaken -- it is a head chunk's \
             edge, and this segment either carries no lm_head matmul or has the argmax \
             skipped");
    let has_embed = family && l.embed_tab.is_some() && l.tok_in.is_some();
    assert!(l.layer != Some(0) || has_embed || skip("embedding"),
            "compose: a layer-0 chunk without embed.u16 and tok_in.i32 (extract it with \
             --embed), so the residual stream it consumes cannot be bound to the embedding \
             rows of its ids; pass --unsound-skip-embedding to prove the unbound statement");
    let embed = has_embed && !skip("embedding");
    // Stream AA (E11): with the K/V boundary the unchecked prefix is `l.nsub` key tiles of
    // `S` keys per head key tile group, so the chunk's context is the sub-key-tile's
    // times `nsub` and the head key tile list takes every `nsub`-th entry.  Both are
    // the identity when `nsub == 1`, which is every run without the flag.
    // Block 0 (open item 1) has no prefix key tile at all: `nsub == 0`, `c == 0`,
    // and the head key tile list is the diagonal block key tiles', one per group.
    let (c, head_groups) = if l.nsub == 0 {
        (0, l.key_tile_sizes[n_pre..].to_vec())
    } else {
        (b0.shape.c * l.nsub, l.key_tile_sizes[..n_pre].iter().step_by(l.nsub).copied().collect())
    };
    // Stream JJ (E11): data-derived and overridable exactly as `ao_rem_bits`
    // is, and for the same reason -- a driver pins the model-wide worst case
    // and proves every layer as one shape.  Below the witness's own value an
    // `EPOW` lookup or an accumulator limb would land off its table, so that is
    // an error and not a smaller proof.
    //
    // Stream TT (E11), log row 73: these four come out of the struct literal
    // because the two limb widths now depend on the exponent window -- which
    // form of the magnitude bound fits is a question about `e_hi`.
    let ao_jv_min = {
        let want = arg_scalar(args, "--ao-jv-min", l.ao.jv_range().0);
        assert!(want <= l.ao.jv_range().0,
                "--ao-jv-min {want} is above this witness's {}", l.ao.jv_range().0);
        want
    };
    let ao_jv_max = {
        let want = arg_scalar(args, "--ao-jv-max", l.ao.jv_range().1);
        assert!(want >= l.ao.jv_range().1,
                "--ao-jv-max {want} is below this witness's {}", l.ao.jv_range().1);
        want
    };
    let ao_jo_min = {
        let want = arg_scalar(args, "--ao-jo-min", l.ao.jo_range().0);
        assert!(want <= l.ao.jo_range().0,
                "--ao-jo-min {want} is above this witness's {}", l.ao.jo_range().0);
        want
    };
    let ao_jo_max = {
        let want = arg_scalar(args, "--ao-jo-max", l.ao.jo_range().1);
        assert!(want >= l.ao.jo_range().1,
                "--ao-jo-max {want} is below this witness's {}", l.ao.jo_range().1);
        want
    };
    // Stream JJ: `|acc| <= 2^(lo + hi - 1)` under the offset form, so the two
    // limbs together carry one bit more than the accumulator needs.  16 is the
    // widest range table the layer already builds, which is why the low limb
    // stops there; neither limb ever goes below `MIN_LOG_ROWS`, since a range
    // table *is* its own `1 << bits` rows and the LogUp builder has no shorter
    // column.
    //
    // Stream TT (E11), log row 73: and the magnitude form is the same split one
    // bit narrower.  There the limbs range-check the committed `mag = |acc|`
    // rather than `acc + 2^(lo+hi-1)`, so they need `acc_bits()` and not
    // `acc_bits() + 1` -- the sign is read off the `sg` column the component
    // already commits.  The wider form is kept wherever it fits, so that every
    // chunk that composed before this existed keeps its columns and its digest;
    // the narrow one is taken only when `Plan::new`'s window would refuse.
    // `--ao-acc-mag` forces it, which is what the controls and the gate use.
    let e_hi = (ao_jv_max - ao_jo_min).max(0) as u32;
    let split = |need: u32| -> (u32, u32) {
        let lo = arg_scalar(args, "--ao-acc-lo-bits",
                            need.clamp(crate::common::MIN_LOG_ROWS, 16));
        let hi = arg_scalar(args, "--ao-acc-hi-bits",
                            need.saturating_sub(lo).max(crate::common::MIN_LOG_ROWS));
        (lo, hi)
    };
    let need_off = l.ao.acc_bits() + 1;
    let (lo_off, hi_off) = split(need_off);
    let (lo_mag, hi_mag) = split(l.ao.acc_bits());
    let ao_acc_mag =
        flag(args, "--ao-acc-mag") || 1 + lo_off + hi_off + e_hi > 31;
    let (ao_acc_lo_bits, ao_acc_hi_bits, need) = if ao_acc_mag {
        (lo_mag, hi_mag, l.ao.acc_bits())
    } else {
        (lo_off, hi_off, need_off)
    };
    assert!(ao_acc_lo_bits + ao_acc_hi_bits >= need,
            "--ao-acc-*-bits {ao_acc_lo_bits}+{ao_acc_hi_bits} is below this witness's {need}");
    assert!(ao_acc_lo_bits >= crate::common::MIN_LOG_ROWS
                && ao_acc_hi_bits >= crate::common::MIN_LOG_ROWS,
            "--ao-acc-*-bits {ao_acc_lo_bits}+{ao_acc_hi_bits}: a limb is below \
             MIN_LOG_ROWS");
    if ao_acc_mag && !flag(args, "--ao-acc-mag") {
        eprintln!("compose: the attention output takes the magnitude form -- the \
                   offset window is 1 + {lo_off} + {hi_off} + {e_hi} > 31 (log row 73)");
    }
    // Stream TT (E11), log row 73: the sibling bound, decided the same way.
    // `ao_q_bits` is `attn_out_bits - 1` and `ao_rem_bits` is the witness's own
    // (a driver may only widen it), so `11 + 20 = 31` is not a pin anything can
    // choose its way out of either -- the `denp2` form is what admits it.
    let ao_q_bits = l.attn_out_bits - 1;
    let ao_rem_bits = {
        let want = arg_scalar(args, "--ao-rem-bits", l.ao.rem_bits());
        assert!(want >= l.ao.rem_bits(),
                "--ao-rem-bits {want} is below this witness's {}", l.ao.rem_bits());
        want
    };
    let ao_denp2 = flag(args, "--ao-denp2") || ao_q_bits + ao_rem_bits > 30;
    if ao_denp2 && !flag(args, "--ao-denp2") {
        eprintln!("compose: the attention output takes the denp2 form -- the quotient \
                   term is {ao_q_bits} + {ao_rem_bits} > 30 (log row 73)");
    }
    compose::Params {
        s: l.ao.s, c, hd: l.ao.hd, nh: l.ao.nh,
        head_groups,
        diagonal_block,
        // Stream E: the head's order-key claim.  Not under `--no-glue`'s gate,
        // because a head dump has no wiring columns to switch off.
        // the argmax implies the logits claim: it is the counterparty
        // -- and the head publishes it exactly when the argmax reads it.
        logits_claim: argmax,
        prove_argmax: argmax,
        prove_argmax_lax: argmax_lax,
        variant: vname.to_string(),
        log_table: b0.log_table,
        q_bits: b0.q_bits, k_bits: b0.k_bits, nmax: b0.nmax,
        chan_target: arg_scalar(args, "--keys-per-row", 128usize),
        grid_target: arg_scalar(args, "--scores-per-row", 64usize),
        rw: arg_scalar(args, "--tokens-per-row", 16usize),
        z_lo_bits: arg_scalar(args, "--z-lo-bits", 16u32),
        z_hi_bits: arg_scalar(args, "--z-hi-bits", 6u32),
        z_min: b0.nmax,
        mats: l.mats.iter().map(|m| compose::MatParams {
            name: m.name.clone(), variant: m.variant.name().into(),
            n_in: m.n_in, n_out: m.n_out, tokens: m.tokens,
        }).collect(),
        deq_target: arg_scalar(args, "--deq-per-row", 16usize),
        kmax: arg_scalar(args, "--kmax", kmax),
        round: arg_strs(args, "--round", &["even"])[0].clone(),
        ao_target: arg_scalar(args, "--ao-per-row", 16usize),
        // Stream Z: the witness is the knob.  A dump extracted at a wider
        // attention output declares it in the manifest, and a dump that
        // predates the knob reads back 8 -- so every pre-change statement is
        // parameter-identical and the standing digests stand.
        ao_q_bits,
        // B9 step 1: `ao_rem_bits` and `kmax` are data-derived, so two real layers
        // of one model are two shapes and the chunk loop rejects the pair.  Both
        // are overridable now, so a driver can pin them at the model-wide worst
        // case and prove every layer of a model as one shape -- a few extra table
        // rows against a second emit chain.  Below the witness's own value the
        // range check would just push rows into `off_table`, so that is an error.
        // (Both are derived above, with the `denp2` decision that reads them.)
        ao_rem_bits,
        ao_jv_min, ao_jv_max, ao_jo_min, ao_jo_max,
        ao_acc_lo_bits, ao_acc_hi_bits, ao_acc_mag, ao_denp2,
        // Stream Z: the fold's two pins, data-derived exactly as `ao_rem_bits`
        // and overridable the same way so a driver can pin the model-wide worst
        // case.  Zero -- absent from the serialized statement -- on a witness
        // with no fold matmul.
        fold_rem_bits: match l.fold_pins() {
            Some((_, rem)) => {
                let want = arg_scalar(args, "--fold-rem-bits", rem);
                assert!(want >= rem,
                        "--fold-rem-bits {want} is below this witness's {rem}");
                want
            }
            None => 0,
        },
        fold_alpha: match l.fold_pins() {
            Some((alpha, _)) => arg_scalar(args, "--fold-alpha", alpha),
            None => 0,
        },
        freivalds_p: arg_scalar(args, "--p", 136usize),
        // B10 step 2: the wiring family rides on a witness `extract.py --residual-stream`
        // wrote, and on nothing else -- a dump without those columns composes
        // exactly as it did before.  `--residual-stream-a` is how many rows one token's
        // stream takes at the reduction's first level; it has to divide `d`.
        // Stream E's two components are opt-in: `--norm-tails` wires `h = (x r) w`
        // and `--swiglu` the `silu(gate) * up` product.  Off, the composition is
        // the pre-change one cell for cell, which is what the bit-for-bit control
        // checks.
        // The family is carried whenever the witness has it, unless
        // `--unsound-skip-residual-stream` drops it.
        residual_stream: l.residual_stream.as_ref().filter(|_| family).map(|g| {
            compose::ResidualStreamParams {
                d: g.d,
                a: arg_scalar(args, "--residual-stream-a", 32usize),
                // Stream II: the two additive components are unconditional, so
                // `dff` is always carried.
                dff: g.dff,
                eps: g.eps,
                inv_d: g.inv_d,
                // stream II: which family this chunk carries is the witness's
                // to say, not the command line's
                head_norm: g.head,
            }
        }),
        // Stream H: the boundary commitment, off unless asked for either way --
        // by the flag on the command line or by `ZK_LAYER_BOUNDARY` in the environment.
        // Off is what makes the recorded digest the bit-for-bit line.
        // On by default wherever the family is; `--unsound-skip-layer-boundary`
        // turns the trees off, and off is skipped in the serialization.
        layer_boundary,
        // Stream AA (E11): the K/V boundary trees, and the prefix tiling they
        // need.  Opt-in and skipped in the serialization, so a statement
        // written without it is byte identical to one from a crate that never
        // had the feature.
        // On by default wherever the chunk has its own block;
        // `--unsound-skip-kv-boundary` turns the trees off.
        kv_boundary: kv_boundary,
        // B11 stream Q: opt in to the LogUp-GKR backend for the key tiles. A flag and
        // not an env mode, because it changes the proof.
        gkr_key_tiles: flag(args, "--gkr-key-tiles"),
        gkr_dequant: flag(args, "--gkr-dequant"),
        gkr_residual_stream: flag(args, "--gkr-residual-stream"),
        gkr_freivalds: flag(args, "--gkr-freivalds"),
        gkr_attnout: flag(args, "--gkr-attnout"),
        // Stream BB (E11): the embedding gadget.  `--embed` needs the wiring
        // columns (its binding's counterparty is their `stream_in` claim) and the
        // dump's `embed.u16` / `tok_in.i32`; it panics on a dump without them
        // rather than standing anything in.  Zero -- absent from the serialized
        // statement -- everywhere else.
        embed_vocab: if embed {
            let g = l.residual_stream.as_ref().expect("the embedding gadget bindings into the family");
            let tab = l.embed_tab.as_ref().expect("the embedding gadget needs embed.u16");
            assert_eq!(tab.len() % g.d, 0,
                       "embed.u16 holds {} values, which is not a multiple of d = {}",
                       tab.len(), g.d);
            tab.len() / g.d
        } else {
            0
        },
        embed_per_row: if embed {
            arg_scalar(args, "--embed-per-row", 256usize)
        } else {
            0
        },
        // Stream NN: the rotation and the q/k/v quantize, present exactly when
        // the witness carries what they need.  Not a flag: on a real dump the
        // columns are there or they are not, and on a synthetic layer
        // `--synth-residual-stream` builds them exactly when the widths line up.  The
        // wiring family is a precondition -- the input side bindings into it.
        rope: l.residual_stream.as_ref().filter(|g| !g.head).and(l.rope.as_ref()).map(|r| {
            compose::RopeParams {
                nkv: r.nkv,
                pos0: r.pos0,
                theta: r.theta.to_bits(),
                rope_type: r.rope_type.clone(),
                rope_scale: r.rope_scale.iter().map(|v| v.to_bits()).collect(),
            }
        }),
    }
}

/// Run the whole serialized round trip: prove, write the statement and the
/// proof as bytes, verify from the bytes alone.
///
/// Stream I: `ser_to` is the sequence assembly's hook. The chain check that
/// joins two chunks reads four byte strings off disk and nothing else, so a
/// chunk that is going to be chained has to leave its statement and its proof
/// behind as files. Passing `None` is the old behaviour exactly -- the bytes
/// are still built, still verified from, and then dropped.
fn compose_once(l: &layer::Layer, params: &compose::Params, tamper: &compose::ProverTamper,
                ser_to: Option<(&str, &str)>) -> (serde_json::Value, bool) {
    let out = compose::prove_layer(l, params, tamper);
    let ser = compose_ser(&out);
    if let Some((dir, tag)) = ser_to {
        ser_write(dir, tag, &ser);
    }
    compose_finish(out, ser, params)
}

/// Leave one chunk's statement and proof on disk under `tag`.
///
/// Stream O: the chunk loop has to write exactly these two files, per chunk
/// and under the chunk's own tag, because `boundary-chain` joins two chunks by
/// reading them back off disk and nothing else. It was a branch inside
/// `compose_once` and is a function so that the loop can call it too.
pub fn ser_write(dir: &str, tag: &str, ser: &Ser) {
    std::fs::create_dir_all(dir).expect("the serialization directory");
    std::fs::write(format!("{dir}/{tag}.fixed.json"), &ser.fixed_bytes)
        .expect("writing the statement");
    // Stream KK: the claims, which `verify_composed` needs and which nothing
    // wrote before -- so `boundary-grid --verify` could not have verified a
    // serialized chunk however hard it tried.  They are small (tens of KB
    // beside a 50 MB proof) and they are a third of the document the verifier
    // reads, so they are written beside the other two.
    std::fs::write(format!("{dir}/{tag}.claims.json"), &ser.claims_bytes)
        .expect("writing the claims");
    if let Some(b) = &ser.proof_bytes {
        std::fs::write(format!("{dir}/{tag}.proof.json"), b).expect("writing the proof");
    }
    eprintln!("compose: statement, claims and proof written to \
               {dir}/{tag}.{{fixed,claims,proof}}.json");
}

/// The statement and the proof as bytes.
pub struct Ser {
    pub claims_bytes: Vec<u8>,
    pub fixed_bytes: Vec<u8>,
    pub proof_bytes: Option<Vec<u8>>,
}

/// B7a milestone 4 (18.2): section 5 puts serialization inside the whole wall
/// and no harness has ever timed it. B7b (26): on its own, so the chunk loop
/// serializes inside the wall and verifies after it.
pub fn compose_ser(out: &compose::Composed) -> Ser {
    let phase_t = std::time::Instant::now();
    let claims_bytes = serde_json::to_vec(&out.claims).unwrap();
    let fixed_bytes = serde_json::to_vec(&out.fixed).unwrap();
    crate::util::add_phase("wall/ser_statement", phase_t.elapsed().as_secs_f64());
    let proof_bytes = out.proof.as_ref().map(|pf| {
        let phase_t = std::time::Instant::now();
        let b = serde_json::to_vec(pf).unwrap();
        crate::util::add_phase("wall/ser_proof", phase_t.elapsed().as_secs_f64());
        b
    });
    Ser { claims_bytes, fixed_bytes, proof_bytes }
}

/// Verify from the bytes alone and assemble the report.
pub fn compose_finish(out: compose::Composed, ser: Ser, params: &compose::Params)
                      -> (serde_json::Value, bool) {
    let Ser { claims_bytes, fixed_bytes, proof_bytes } = ser;
    let mut proof_sha256 = String::new();
    let mut proof_ser_bytes = 0usize;
    let (verdict, proof_bytes_len, verify_json) = match (&out.proof, proof_bytes) {
        (Some(pf), Some(proof_bytes)) => {
            // section 5 rung 1: the GPU proof must be byte-identical to the
            // SIMD proof built with the same options. This is the digest.
            proof_sha256 = crate::device::sha256_hex(&proof_bytes);
            proof_ser_bytes = proof_bytes.len();
            let v = compose::verify_composed(&fixed_bytes, &claims_bytes, &proof_bytes);
            let j = serde_json::to_value(&v).unwrap();
            (Some(v), pf.size_estimate(), j)
        }
        _ => (None, 0, json!({"prove_error": out.prove_error})),
    };
    let accepts = verdict.as_ref().map(|v| v.accepts()).unwrap_or(false);
    // the binding table of this statement, from the plan the verifier rebuilds
    let plan_of_statement = compose::Plan::new(&out.fixed.params);
    let binding = plan_of_statement.binding_json(out.fixed.layer);
    // issue #96: and the census of published claims from the same plan -- every
    // claim the residual stream publishes and which mechanism reads it.  A
    // report: a shape whose matmul group is absent loses the seam that read its
    // producer's output, so `unread` is empty on the shapes we record and not
    // by a blanket assertion.
    let claims_census = plan_of_statement.claims_json();
    if !proof_sha256.is_empty() {
        crate::util::prov_note_proof(&proof_sha256, binding.clone());
    }
    let j = json!({
        "accepts": accepts,
        "binding": binding,
        "verdict": verify_json,
        "prove_error": out.prove_error,
        "proof_bytes": proof_bytes_len,
        "statement_bytes": {"fixed": fixed_bytes.len(), "claims": claims_bytes.len()},
        // issue #96: the census of published claims -- `unread` is the claims no
        // mechanism of this statement reads, `readers` names them claim by claim
        "claims": claims_census,
        "pre_root": out.fixed.pre_root,
        // stream PP: the per-layer model commitment this chunk proves against
        "model_root": out.fixed.model_root,
        "columns": {"preprocessed": out.n_columns[0], "base": out.n_columns[1],
                    "interaction": out.n_columns[2],
                    "total": out.n_columns.iter().sum::<usize>()},
        // stream PP: `preprocessed` is the shape tree and the model tree
        // together, so the number stays comparable with the pre-split records.
        "cells": {"preprocessed": out.stats.pre_cells(),
                  "shape": out.stats.tree_cells.first().copied().unwrap_or(0),
                  "model": out.stats.model_cells(),
                  "base": out.stats.tree_cells.get(crate::common::BASE_TREE)
                      .copied().unwrap_or(0),
                  "interaction": out.stats.tree_cells.get(crate::common::INTER_TREE)
                      .copied().unwrap_or(0),
                  "fresh": out.stats.fresh_cells()},
        "family_cells": out.family_cells,
        "ref_mismatch": out.ref_bad, "off_table": out.off_table,
        "stream_bindings_ok": out.stream_bindings_ok,
        // Stream H: the boundary trees' own cells and columns, and the two roots
        // the chain check compares.  Null on a chunk proved with the flag off.
        "layer_boundary": out.layer_boundary,
        // Stream AA: the same for the K/V boundary trees, and every boundary
        // root in the prover's order (stream H's two, then the K/V sides).
        "kv_boundary": out.kv_boundary,
        // Stream EE: the residual pair is found by its label, not by tree index,
        // so this JSON keeps H's shape while the statement may carry any number
        // of sides; `boundary_sides` is every side with its label.
        //
        // Issue #119: the pair went in under `boundary_roots` too -- the rename
        // of `bound_roots` gave it the name the ordered list already had and
        // `json!` keeps the last, so the pair never reached the document. The
        // three keys are three views of one set of roots, and the one a reader
        // trusts is `boundary_sides`, the labelled one: `boundary_roots` is the
        // same roots in the prover's commit order, which is the plan's order and
        // not a fixed one, and `layer_boundary_roots` is the pair the chain
        // check compares, pulled out by label.
        "layer_boundary_roots": verdict.as_ref().map(|v| json!({
            "in": v.side_root(boundary::Axis::Layer, "in", None),
            "out": v.side_root(boundary::Axis::Layer, "out", None)})),
        "boundary_roots": verdict.as_ref().map(|v| v.boundary_roots.clone()),
        "boundary_sides": verdict.as_ref().map(|v| serde_json::to_value(&v.boundary_sides).unwrap()),
        "trace_secs": out.trace_secs, "interaction_secs": out.interaction_secs,
        "prove_secs": out.stats.prove_secs,
        "verify_secs": verdict.as_ref().map(|v| v.verify_secs).unwrap_or(0.0),
        "peak_rss_bytes": out.stats.peak_rss,
        "preprocessed_is_widest": out.stats.pre_ok,
        "phases": out.stats.phases,
        "host_phases": crate::util::phases_report(),
        // B11 stream F: the bytes the named phases moved, so a phase's GB/s is
        // read off the same dump as its seconds
        "host_bytes": crate::util::bytes_report(),
        "proof_sha256": proof_sha256,
        "proof_ser_bytes": proof_ser_bytes,
        // milestone 2a: what the host->device crossing cost (README 10.2)
        "upload": crate::device::report(),
        // milestone 3: what the constraint lowering cost, and where
        "blob": crate::cudagen::blob_report(),
        "params": serde_json::to_value(params).unwrap(),
        // issue #18: the per-layer calibration the shape root is a function of
        // beside the statement, and what the shape-root cache did with it
        "shape": crate::shaperoot::shape_json(),
        // issue #19: the attention output's no-wrap window and what is left of
        // it, per segment -- at layer 0 under the paper's quantization the
        // margin is zero bits and this is where that is visible
        "ao_wrap": compose::ao_wrap_json(&out.fixed.params),
    });
    (j, accepts)
}

/// Issue #119: one key per value in the run JSON, pinned where the object is
/// written. `json!` keeps the last of two equal keys, so a rename that lands on
/// a name the object already carries drops a value and reports nothing -- twice
/// now, in the grid report (#39) and here. The object is a literal, so the check
/// reads the source it is written in.
#[cfg(test)]
mod run_json_keys {
    /// Every key of the run JSON's own object, in the order it is written.
    fn keys() -> Vec<&'static str> {
        let src = include_str!("cli.rs");
        let mut out = vec![];
        let mut inside = false;
        for line in src.lines() {
            if !inside {
                inside = line == "    let j = json!({";
                continue;
            }
            if line == "    });" {
                break;
            }
            // A key of the object itself starts at the object's own
            // indentation. A nested object opens on the line of the key it
            // belongs to, so on such a line only the first key is this
            // object's; a line that opens none carries this object's keys and
            // nothing else.
            if !line.starts_with("        \"") {
                continue;
            }
            let flat = !line.contains('{');
            let mut rest = &line[8..];
            while let Some(i) = rest.find('"') {
                let after = &rest[i + 1..];
                let Some(j) = after.find('"') else { break };
                if after[j + 1..].starts_with(':') {
                    out.push(&after[..j]);
                    if !flat {
                        break;
                    }
                }
                rest = &after[j + 1..];
            }
        }
        out
    }

    #[test]
    fn the_run_json_writes_every_key_once() {
        let ks = keys();
        // the last of the four shares its line with the key before it
        for k in ["boundary_roots", "layer_boundary_roots", "boundary_sides", "interaction_secs"] {
            assert!(ks.contains(&k), "the scan found the run JSON's object; it read {ks:?}");
        }
        let mut seen = std::collections::BTreeSet::new();
        for k in &ks {
            assert!(seen.insert(*k),
                    "the run JSON writes {k:?} twice and `json!` keeps the last");
        }
    }
}

/// Stream LL (E11), `notes/witness-slack.md` section 5 item 1, deliverable 1:
/// what tree 0 is made of, from `Params` alone.
///
/// ```text
///     zk-prover pre-split --params FILE
/// ```
///
/// `FILE` is a JSON `Params` -- `result.point.params` of any `compose` run, so
/// the 8B reference chunk's tree 0 can be priced without the dump. The rollup
/// answers the two questions the design fork turns on: how much of tree 0 a
/// verifier could rebuild from the statement alone (`needs_params`), and how
/// much of it is the same on every token block of the layer (`varies_layer`),
/// which is what a published per-layer root would cover.
fn pre_split_cmd(args: &[String]) {
    let f = arg_str(args, "--params").expect("pre-split needs --params FILE");
    let bytes = std::fs::read(&f).unwrap_or_else(|e| panic!("--params {f}: {e}"));
    let p: compose::Params = serde_json::from_slice(&bytes)
        .unwrap_or_else(|e| panic!("--params {f}: not a Params document: {e}"));
    let plan = compose::Plan::new(&p);
    let mut report = plan.pre_split_report();
    report["params_file"] = json!(f);
    report["c"] = json!(p.c);
    report["s"] = json!(p.s);
    let cols: Vec<Value> = plan.pre_split().iter()
        .map(|c| json!({"id": c.id.id, "family": c.family, "log_size": c.log_size,
                        "needs": c.needs, "varies": c.varies}))
        .collect();
    report["column_list"] = json!(cols);
    eprintln!("pre-split: {} columns, {} cells; needs_model {:.2}%, varies_block {:.2}%",
              report["columns"], report["cells"],
              100.0 * report["needs_model"]["share"].as_f64().unwrap(),
              100.0 * report["varies_block"]["share"].as_f64().unwrap());
    emit("pre_split", report);
}

/// Stream LL (E11), `notes/witness-slack.md` section 5 item 1, deliverable 2:
/// the expected tree-0 root of one chunk, derived from the model.
///
/// ```text
///     zk-prover expect-root --witness DIR [--layer k] [--block b] <the chunk's flags>
///     zk-prover expect-root --weights-only --params <tag>.fixed.json --witness DIR ...
/// ```
///
/// It takes exactly the arguments the chunk's `compose` line takes -- the plan
/// line, minus `--ser-tag` -- builds the chunk's `Plan` and its tree 0, and
/// stops there: the root, the model hash and the classification, and no proof.
/// The derived root is the prover's own by construction, because it is the
/// prover's own code up to `commit_pre` and there is no second implementation
/// of tree 0 to drift from it.
///
/// **`--weights-only` is the verifier's form.** Without it `Layer::load` reads
/// the whole dump, activations included, and recomputes `Q.K^T` on the way; tree
/// 0 needs only the model half (the weights, the weight scales, `jv`, the norm
/// weights, the embedding table, this stream's rope exponents, and the
/// manifest's `frac_per_head` / `master_table_bits`).  With it,
/// `Layer::load_weights` reads exactly those files and nothing else, and the
/// derived root must be the same -- which is what `ll_box.sh`'s `root` step
/// checks on a real dump, three ways: the full load's, the weights-only load's
/// and the recorded proof's own `pre_root`.
///
/// `--weights-only` requires `--params FILE`, the chunk's own `Fixed.params`:
/// `ao_jo_min`/`ao_jo_max` and `ao_acc_lo_bits`/`ao_acc_hi_bits` are read off
/// activations, so they cannot be re-derived here -- and taking them from the
/// statement is what the check wants anyway, since it is the statement's
/// own shape a verifier holds.
///
/// **Stream PP (E11): `--shape-only` is the committed-model verifier's form.**
/// It stops one tree earlier -- after the shape tree and before the model tree
/// -- and emits `pre_root` alone.  The shape tree is a function of
/// `Fixed.params` and of the per-layer calibration the model manifest
/// publishes, so the root it derives is one no weight can move: the desk gate
/// derives it at two seeds, and the two agree while the model roots differ.
///
/// **Issue #112: `--calibration` is the form that opens no dump at all.**
///
/// ```text
///     zk-prover expect-root --shape-only --params <tag>.fixed.json \
///                     --frac-per-head <the layer's exponents>
///     zk-prover expect-root --shape-only --params <tag>.fixed.json \
///                     --calibration <a manifest, or a run's own result.shape>
/// ```
///
/// `--shape-only` above still loads the dump -- under it the model files are
/// not opened, but `manifest.json` is, and the loader builds the exponent table
/// and every stand-in array before the derivation starts. With `--calibration`
/// or `--frac-per-head` the segment is `Layer::from_params` instead: the
/// statement gives every layout and every table, the calibration gives the two
/// shape columns the statement does not, and there is no dump on disk. A
/// verifier holds both (issue #95), so this is the derivation it could make.
/// It refuses `--witness` and needs `--params`, and it derives the shape root
/// only: a segment built from parameters carries no model.
///
/// **Issue #18: `--shape-cache DIR` is the derivation, done once.** The shape
/// tree is a function of `Fixed.params` and of the per-layer calibration and of
/// nothing else, so `shaperoot` keys the root on exactly those two and stores
/// it under `DIR`. A second segment whose statement and calibration agree
/// reports the stored root and builds no column; a segment that differs in one
/// parameter has a different key and derives its own. What each root was keyed
/// on is published beside it as `result.shape`, so a reader of the row can see
/// what it is a function of. `--shape-cache-verify` derives anyway and asserts
/// the fresh root against the stored one.
fn expect_root_cmd(args: &[String]) {
    if flag(args, "--shape-only") {
        compose::SHAPE_ONLY.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    compose::PRE_ONLY.store(true, std::sync::atomic::Ordering::Relaxed);
    compose_cmd(args);
    unreachable!("expect-root: the prover did not stop after the preprocessed trees");
}

/// Stream LL (E11), deliverable 2: the model manifest KK's
/// `boundary-grid --expect-pre-roots` consumes.
///
/// ```text
///     zk-prover model-roots --plan PLAN [--pins "..."] [--out FILE] [--limit N]
///                     [--weights-only --params-dir DIR]
///                     [--from-params --params-dir DIR --calibration MANIFEST]
///                     [--shape-cache DIR] [--shape-cache-verify]
/// ```
///
/// Issue #112: `--from-params --params-dir DIR --calibration MANIFEST` derives
/// every row's shape root from the statements and the calibration a model
/// manifest already published, with no dump on disk: each line's `--witness` is
/// dropped, its statement is `DIR/<ser-tag>.fixed.json` and its calibration is
/// the manifest row of the same tag. It implies `--shape-only` -- a segment
/// built from parameters holds no model -- so the rows it writes carry a shape
/// root and no model root.
///
/// `PLAN` is an `o_plan.py` plan file: one `compose` argument line per chunk.
/// A plan line carries only the chunk's *own* arguments, so `--pins` supplies
/// the driver's shared prefix (`--round`, `--numbits`, `--variants`, `--head-groups`,
/// `--kmax` and the fold's two -- `ss_box.sh`'s `SEQPT`).
///
/// `--weights-only --params-dir DIR` is the verifier's form: each row is derived
/// from the model half of the dump and from `DIR/<ser-tag>.fixed.json`, the
/// statement `compose --ser-dir` wrote.  Without it the rows are derived the
/// prover's way, from the whole dump, which is what a manifest built before any
/// chunk has been proved has to do.
///
/// Three things it checks and one it reports: two chunks at one `(layer, c)`
/// must derive the same root and name the same model; one layer must name one
/// model; and **`master_table_bits` must be one value per layer across blocks**,
/// or the layer is a named rejection and no manifest is written (the desk
/// record's open item 3 -- `table/exp` is `1 << log_table` rows and `log_table`
/// comes off each extraction's own manifest, so this is a requirement on the
/// plan that nothing else states).
/// Each line is run through `expect-root` in its own process -- one chunk's
/// tree 0 at a time, which is what keeps the peak resident set to one chunk's
/// -- and the rows are collected into
///
/// ```text
///     [{"layer": k, "block": b, "c": n, "pre_root": "...", "model": "..."}]
/// ```
///
/// The rows a grid needs are keyed by `(layer, c)`, which is what
/// `grid_cmd` looks them up by; `block` and `model` ride along so a reader can
/// see which dump each row came from and so the grid can compare model hashes.
///
/// Issue #18: `--shape-cache DIR` is passed to every child, which is the only
/// way a cache can serve this loop -- each line is its own process. `(layer, c)`
/// deduplicates the lines this command runs at all; the cache is what makes two
/// *different* cells whose statement and calibration agree cost one derivation
/// between them, and `shape_key` on each row says which cells those were.
fn model_roots_cmd(args: &[String]) {
    let plan = arg_str(args, "--plan").expect("model-roots needs --plan FILE");
    let text = std::fs::read_to_string(&plan)
        .unwrap_or_else(|e| panic!("--plan {plan}: {e}"));
    let limit = arg_scalar::<usize>(args, "--limit", usize::MAX);
    // Stream LL box half.  A plan file holds a chunk's *own* arguments; the
    // shared pins (`--round`, `--numbits`, `--variants`, `--head-groups`, `--kmax`,
    // the fold's two) live in the driver that runs it (`ss_box.sh`'s `SEQPT`),
    // so a plan line alone does not compose.  `--pins` is that prefix.
    let pins: Vec<String> = arg_str(args, "--pins").unwrap_or_default()
        .split_whitespace().map(|s| s.to_string()).collect();
    // The verifier's mode: derive each row from the model half of the dump and
    // the *statement's* own params, `<tag>.fixed.json` in `--params-dir` (what
    // `compose --ser-dir` writes).  Without it the rows are derived the
    // prover's way, from the whole dump, which is what a manifest built ahead
    // of any proof has to do.
    let weights_only = flag(args, "--weights-only");
    let params_dir = arg_str(args, "--params-dir");
    assert!(!weights_only || params_dir.is_some(),
            "model-roots --weights-only needs --params-dir DIR (the serialized \
             statements: DIR/<ser-tag>.fixed.json)");
    // Issue #112: the rows derived from the statements and the published
    // calibration, with no dump on disk. `--calibration FILE` is a model
    // manifest -- the rows `--out` wrote on a run that had the dumps -- and
    // every line takes its own row's `frac_per_head` by `--ser-tag`. It implies
    // `--shape-only`, because a segment built from parameters holds no model
    // and the model root it would derive would be a root of stand-ins.
    let from_params = flag(args, "--from-params");
    let calibration: Option<Vec<Value>> = arg_str(args, "--calibration").map(|f| {
        let bytes = std::fs::read(&f).unwrap_or_else(|e| panic!("--calibration {f}: {e}"));
        let v: Value = serde_json::from_slice(&bytes)
            .unwrap_or_else(|e| panic!("--calibration {f}: not JSON: {e}"));
        let rows = if v.is_array() { v } else { v["rows"].clone() };
        rows.as_array()
            .unwrap_or_else(|| panic!("--calibration {f}: not a model manifest's rows"))
            .clone()
    });
    assert!(!from_params || params_dir.is_some(),
            "model-roots --from-params needs --params-dir DIR (the serialized statements: \
             DIR/<ser-tag>.fixed.json)");
    assert!(!from_params || calibration.is_some(),
            "model-roots --from-params needs --calibration FILE (a model manifest, whose \
             rows carry each segment's own `frac_per_head`)");
    assert!(!from_params || !weights_only,
            "--from-params and --weights-only are two loads: the first opens no dump, the \
             second reads the dump's model half");
    // Issue #18: the shape-root cache the children share.
    let shape_cache = arg_str(args, "--shape-cache");
    let shape_cache_verify = flag(args, "--shape-cache-verify");
    let exe = std::env::current_exe().expect("model-roots: no path to this binary");
    let mut rows: Vec<Value> = vec![];
    let mut seen: std::collections::BTreeSet<(i64, u64)> = Default::default();
    let mut skipped = 0usize;
    let mut missing: Vec<String> = vec![];
    let t0 = std::time::Instant::now();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if rows.len() >= limit {
            break;
        }
        // The shared pins **first**, then the plan line's own arguments, minus
        // the two that name the artifact rather than the chunk.  The order is
        // required and not merely tidy: `arg_str` and `arg_scalar` take the
        // *last* match, so a line's own `--table-bits` or `--ao-rem-bits` has
        // to follow the shared one or the shared one silently wins (stream SS's
        // rider 1, `o_plan.py --ao-rem-per-line`).  It is the order
        // `compose $SHARED $line` writes and the order `compose-chunks`
        // composes a plan line in (issue #30).
        let mut a: Vec<String> = pins.clone();
        a.push("--plan-line".into());
        let mut tag = String::new();
        let mut it = line.split_whitespace();
        while let Some(w) = it.next() {
            if w == "--ser-dir" {
                it.next();
                continue;
            }
            if w == "--ser-tag" {
                tag = it.next().unwrap_or("").to_string();
                continue;
            }
            // issue #112: the parameter-only load opens no dump, so the line's
            // own `--witness` is dropped rather than passed to a child that
            // would refuse it
            if from_params && w == "--witness" {
                it.next();
                continue;
            }
            a.push(w.to_string());
        }
        // Issue #112: the statement and the published calibration, in place of
        // the dump. `--shape-only` is not optional here and is added rather
        // than required of the caller: a segment built from parameters has no
        // model, so the shape root is the whole of what it can derive.
        let mut fracs: Vec<f64> = vec![];
        if from_params {
            let d = params_dir.as_ref().unwrap();
            let f = format!("{d}/{tag}.fixed.json");
            if !std::path::Path::new(&f).exists() {
                missing.push(tag.clone());
                continue;
            }
            fracs = calibration.as_ref().unwrap().iter()
                .find(|r| r["tag"].as_str() == Some(tag.as_str()))
                .and_then(|r| r["frac_per_head"].as_array().cloned())
                .map(|v| v.iter().filter_map(|x| x.as_f64()).collect())
                .unwrap_or_else(|| panic!("model-roots --from-params: the calibration names \
                                           no frac_per_head for {tag}"));
            a.push("--params".into());
            a.push(f);
            a.push("--shape-only".into());
            a.push("--frac-per-head".into());
            a.push(fracs.iter().map(|x| x.to_string()).collect::<Vec<String>>().join(","));
        }
        // Issue #18: each line runs in a process of its own, so an in-process
        // cache cannot cross them and the directory the caller named is what
        // does.  Passed through here rather than folded into `--pins`, so a
        // plan line cannot name a different one.
        if let Some(d) = shape_cache.as_ref() {
            a.push("--shape-cache".into());
            a.push(d.clone());
        }
        if shape_cache_verify {
            a.push("--shape-cache-verify".into());
        }
        if weights_only {
            let d = params_dir.as_ref().unwrap();
            let f = format!("{d}/{tag}.fixed.json");
            if !std::path::Path::new(&f).exists() {
                missing.push(tag.clone());
                continue;
            }
            a.push("--weights-only".into());
            a.push("--params".into());
            a.push(f);
        }
        let out = std::process::Command::new(&exe)
            .arg("expect-root")
            .args(&a)
            .output()
            .unwrap_or_else(|e| panic!("model-roots: cannot run {exe:?}: {e}"));
        if !out.status.success() {
            eprintln!("model-roots: {line}\n{}", String::from_utf8_lossy(&out.stderr));
            panic!("model-roots: expect-root failed on a plan line");
        }
        let doc: Value = serde_json::from_slice(&out.stdout)
            .unwrap_or_else(|e| panic!("model-roots: expect-root wrote no JSON: {e}"));
        let r = &doc["result"];
        let layer = r["layer"].as_i64().unwrap_or(-1);
        let c = r["c"].as_u64().unwrap_or(0);
        // one row per (layer, c): every chunk of a grid at that cell has the
        // same tree 0, and a second run of the same cell must agree with the
        // first or the derivation is not a function of what it claims to be
        if let Some(prev) = rows.iter().find(|p| {
            p["layer"].as_i64() == Some(layer) && p["c"].as_u64() == Some(c)
        }) {
            assert_eq!(prev["pre_root"], r["pre_root"],
                       "model-roots: two chunks at (layer {layer}, c {c}) derive \
                        different shape roots");
            assert_eq!(prev["model_root"], r["model_root"],
                       "model-roots: two chunks at (layer {layer}, c {c}) derive \
                        different model roots");
            assert_eq!(prev["model"], r["model"],
                       "model-roots: two chunks at (layer {layer}, c {c}) name \
                        different models");
            skipped += 1;
            continue;
        }
        seen.insert((layer, c));
        // Stream PP (E11), the desk record's open item 2: the shape tree holds
        // two columns that are functions of the *calibration* and of no weight
        // -- `key_tile/dinv` and the exp table's value column, both classified
        // `PreNeeds::Manifest` -- and the calibration they need is
        // `frac_per_head` and `master_table_bits`.  `model-roots` published
        // `log_table` and not `frac_per_head`, so a verifier deriving the shape
        // root on a real dump had to take `frac_per_head` off the dump's own
        // manifest.  Publishing it here is what makes it public input: the
        // manifest carries the model root the verifier compares AND the
        // calibration the verifier derives with, and the dump is needed for
        // neither.
        let calib = a.iter().rposition(|w| w == "--witness")
            .and_then(|i| a.get(i + 1))
            .and_then(|d| witread::read_manifest(d).ok());
        let field = |k: &str| calib.as_ref().map(|m| m[k].clone()).unwrap_or(Value::Null);
        rows.push(json!({
            "layer": layer,
            "block": r["block"],
            "c": c,
            "pre_root": r["pre_root"],
            // issue #18: the key that row's shape root was cached under -- the
            // digest of the statement and of the calibration below, so two rows
            // that name one key name one shape tree
            "shape_key": r["shape"]["key"],
            // stream PP: the per-layer calibration the shape derivation needs,
            // published beside the root rather than read off the dump
            // issue #112: and with no dump to read it off, it is the one this
            // loop passed the child -- the manifest's own, republished
            "frac_per_head": if from_params { json!(fracs) } else { field("frac_per_head") },
            "frac_shared_per_head": field("frac_shared_per_head"),
            "master_table_bits": field("master_table_bits"),
            // stream PP: the per-layer model commitment.  The shape root is
            // keyed by `(layer, c)` because the shape moves with the chunk;
            // this one is keyed by the layer alone, and the rejection below is
            // what says so.
            "model_root": r["model_root"],
            "model": r["model"],
            "source": r["source"],
            "tag": tag,
            "log_table": r["log_table"],
            "weights_only": r["weights_only"],
        }));
    }
    // the model hash is a per-layer constant: two rows at one layer that name
    // two models is exactly the case section 5 item 1's last sentence names
    let mut by_layer: std::collections::BTreeMap<i64, &Value> = Default::default();
    for r in &rows {
        let l = r["layer"].as_i64().unwrap_or(-1);
        if let Some(prev) = by_layer.get(&l) {
            assert_eq!(prev["model"], r["model"],
                       "model-roots: layer {l} appears with two different model hashes");
        } else {
            by_layer.insert(l, r);
        }
    }
    // Stream LL box half, the desk record's open item 3: `master_table_bits` had
    // to be pinned per layer across blocks, or the model part of tree 0 was not
    // a per-layer constant -- `table/exp` is `1 << log_table` rows, it was a
    // model column, and `log_table` comes off each chunk's own statement, so two
    // blocks of one layer can legitimately carry two table sizes and neither is
    // wrong.  A layer with two sizes was a named rejection and no manifest was
    // written.
    //
    // **Stream PP (E11) retires that rejection.** The exp table's index column
    // is `PreNeeds::Params` and its value column `PreNeeds::Manifest`, so the
    // whole table is in the SHAPE tree and no model-tree column's height depends
    // on the width any more (`Plan::pre_split`).  The shape root is derived per
    // chunk from that chunk's own `Fixed.params`, which is where the tightened
    // width lives, so a per-block width costs the derivation nothing.  Two
    // widths in one layer is now a **report**, not a rejection -- and it is the
    // report that says `--tight-table` is admissible again, which is what
    // design (a)'s per-layer `--table-bits` map (+0.496% preprocessed, log row
    // 75) existed to avoid.  What still has to hold is the model root itself,
    // and that is the check below.
    let mut tb: std::collections::BTreeMap<i64, std::collections::BTreeMap<u64, Vec<String>>> =
        Default::default();
    for r in &rows {
        let l = r["layer"].as_i64().unwrap_or(-1);
        let n = r["log_table"].as_u64().unwrap_or(0);
        tb.entry(l).or_default().entry(n).or_default()
            .push(r["tag"].as_str().unwrap_or("").to_string());
    }
    let widths_vary: Vec<Value> = tb.iter().filter(|(_, m)| m.len() > 1)
        .map(|(l, m)| json!({
            "layer": l,
            "table_bits": m.iter().map(|(n, tags)| json!({"log_table": n, "chunks": tags}))
                .collect::<Vec<Value>>(),
        }))
        .collect();
    for w in &widths_vary {
        eprintln!("model-roots: layer {} carries two exp-table widths across its \
                   blocks; the exp table is a shape column, so the model root is \
                   unaffected", w["layer"]);
    }
    let mut rejections: Vec<Value> = vec![];
    // Stream PP (E11): the model root is the *published commitment* for a
    // layer, so it has to be one value across that layer's blocks.  The
    // classification is what should make that true -- every model-tree column
    // is `varies: Layer`, asserted in `Plan::pre_ids_split` -- and this is the
    // check that says it held on a real plan.  A layer with two model roots is
    // a named rejection and no manifest is written: a manifest whose rows
    // disagree about a layer's model commits nothing.
    let mut mr: std::collections::BTreeMap<i64, std::collections::BTreeMap<String, Vec<String>>> =
        Default::default();
    for r in &rows {
        let l = r["layer"].as_i64().unwrap_or(-1);
        let root = r["model_root"].as_str().unwrap_or("").to_string();
        mr.entry(l).or_default().entry(root).or_default()
            .push(r["tag"].as_str().unwrap_or("").to_string());
    }
    rejections.extend(mr.iter().filter(|(_, m)| m.len() > 1).map(|(l, m)| json!({
        "layer": l,
        "why": "the model root is not one value across this layer's blocks",
        "model_roots": m.iter().map(|(root, tags)| json!({"model_root": root, "chunks": tags}))
            .collect::<Vec<Value>>(),
    })));
    let body = json!({
        "plan": plan,
        "rows": rows,
        "distinct_cells": rows.len(),
        "repeat_cells": skipped,
        "layers": by_layer.len(),
        "models": by_layer.values()
            .map(|r| r["model"].as_str().unwrap_or("").to_string())
            .collect::<std::collections::BTreeSet<String>>().len(),
        "weights_only": weights_only,
        // issue #112: whether these rows were derived from the statements and
        // the published calibration alone, with no dump on disk
        "from_params": from_params,
        "no_statement": missing,
        "table_bits_by_layer": tb.iter()
            .map(|(l, m)| json!({"layer": l,
                                 "log_table": m.keys().collect::<Vec<&u64>>()}))
            .collect::<Vec<Value>>(),
        // stream PP: the layers whose blocks tightened to two different exp-table
        // widths.  A report, not a rejection: the exp table is a shape column, so
        // the width reaches the shape root (per chunk) and not the model root.
        "table_bits_vary": widths_vary,
        // stream PP: the published model commitment, one per layer
        "model_roots_by_layer": mr.iter()
            .map(|(l, m)| json!({"layer": l,
                                 "model_root": m.keys().collect::<Vec<&String>>()}))
            .collect::<Vec<Value>>(),
        "rejections": rejections,
        "secs": t0.elapsed().as_secs_f64(),
    });
    if !rejections.is_empty() {
        for r in &rejections {
            eprintln!("model-roots: REJECT layer {}: {}",
                      r["layer"], r["why"].as_str().unwrap_or(""));
        }
        emit("model_roots", body);
        eprintln!("model-roots: {} layer(s) rejected; no manifest written",
                  rejections.len());
        std::process::exit(1);
    }
    if let Some(f) = arg_str(args, "--out") {
        std::fs::write(&f, serde_json::to_string_pretty(&body["rows"]).unwrap())
            .unwrap_or_else(|e| panic!("--out {f}: {e}"));
        eprintln!("model-roots: {} rows -> {f}", rows.len());
    }
    emit("model_roots", body);
}

fn compose_cmd(args: &[String]) {
    // Stream EE (open item 21): before anything, because `prove_gen` reads it.
    cudagen::fp_out_set(args);
    // Issue #18: `--shape-cache DIR` and `--shape-cache-verify` are read in
    // `main` before any subcommand, which is before the load here and before
    // `compose::generate` consults the cache.
    let (mut l, vname) = layer_of(args);
    // B7a milestone 4 (section 3 row 1): rung 3's sixth probe on its own, so the
    // `nopatch` control has a witness whose exp table is *not* the closed form
    // to bite on (section 5 rules 5 and 8). The proof still accepts here -- the
    // table and the witness agree; it is the pinned preprocessed root, which
    // only `compose-probes` checks, that rejects it.
    // Stream PP (E11), the model-tree control: bend one cell of the first
    // matmul's quantized weight array.  It is a *weight*, so only the model
    // tree moves -- the shape root must not.  `--corrupt-exp` is the mirror
    // control: the exp table is a shape column now, so it moves the shape root
    // and leaves the model root alone.
    if let Some(i) = arg_str(args, "--corrupt-weight") {
        let idx: usize = i.parse().expect("--corrupt-weight wants a weight index");
        let m = l.mats.first_mut().expect("--corrupt-weight: the layer has no matmul");
        assert!(idx < m.qw.len(),
                "--corrupt-weight {idx}: the first matmul has {} weights", m.qw.len());
        m.qw[idx] += 1;
    }
    if let Some(i) = arg_str(args, "--corrupt-exp") {
        l.corrupt_exp_table(i.parse().expect("--corrupt-exp wants a table row"));
        eprintln!("compose: exp table row {i} flipped, witness rebuilt");
    }
    // B7a milestone 4 (section 3 row 1): rung 3's *seventh* probe on its own,
    // so the `nopatch` control has a run whose normalisation table is not the
    // closed form to bite on (section 5 rules 5 and 8). As with `--corrupt-exp`
    // the proof still accepts here -- the table and the witness agree; it is
    // the pinned preprocessed root, which only `compose-probes` checks, that
    // rejects it.
    let mut tamper = compose::ProverTamper::default();
    if let Some(i) = arg_str(args, "--corrupt-norm") {
        tamper.norm_row = Some(i.parse().expect("--corrupt-norm wants a table row"));
        eprintln!("compose: norm table row {i} flipped, witness rebuilt");
    }
    // Stream Z: the wired sites' bend controls.  Each bends one committed cell
    // after the honest derivation; a rejected proof (a non-zero exit through the
    // `accepts` assert below) is the pass.
    for (key, slot) in [("--corrupt-fold-qm", &mut tamper.fold_qm),
                        ("--corrupt-fold-dent", &mut tamper.fold_dent),
                        ("--corrupt-ao-clamp", &mut tamper.ao_clamp),
                        ("--corrupt-ao-den", &mut tamper.ao_den),
                        ("--corrupt-ao-jo", &mut tamper.ao_jo),
                        ("--corrupt-ao-num", &mut tamper.ao_num),
                        ("--corrupt-ao-acc", &mut tamper.ao_acc),
                        ("--corrupt-ao-jo-tok", &mut tamper.ao_jo_tok),
                        // stream TT (log row 73): the magnitude window's own
                        ("--corrupt-ao-mag", &mut tamper.ao_mag),
                        ("--corrupt-freivalds-act", &mut tamper.freivalds_over)] {
        if let Some(i) = arg_str(args, key) {
            *slot = Some(i.parse().unwrap_or_else(|_| panic!("{key} wants an index")));
            eprintln!("compose: bend {key} at {i} armed (a rejection is the pass)");
        }
    }
    let params = expect_root_params(&l, args, &vname);
    eprintln!("compose: {} key_tiles={:?} variant={} kmax={} rem={} p={} jv=[{},{}] jo=[{},{}] \
acc={}+{}",
              l.source, l.key_tile_sizes, vname, params.kmax, params.ao_rem_bits, params.freivalds_p,
              params.ao_jv_min, params.ao_jv_max, params.ao_jo_min, params.ao_jo_max,
              params.ao_acc_lo_bits, params.ao_acc_hi_bits);
    if params.ao_acc_mag {
        eprintln!("compose: attnout accumulator in the magnitude form (log row 73)");
    }
    // Stream I: `--ser-dir DIR [--ser-tag TAG]` leaves this chunk's statement and
    // proof on disk under that tag, which is what `boundary-chain` reads back.
    let ser_dir = arg_str(args, "--ser-dir");
    let ser_tag = arg_str(args, "--ser-tag").unwrap_or_else(|| "chunk".into());
    let (mut j, accepts) = compose_once(&l, &params, &tamper,
                                        ser_dir.as_deref().map(|d| (d, ser_tag.as_str())));
    // B12 stream GG: the `Composed` is dropped and the proof serialized by
    // now, so this is what the process still holds; then every mark of the
    // chunk goes into the point (the cuda backend only -- the reference backend has
    // none and the field stays absent).
    device::mem_mark("after_drop");
    if let Some(m) = device::mem_marks_json() {
        j["mem_marks"] = m;
    }
    // Issue #172: the point below carries every field, and the two numbers that
    // say *why* a rejected segment was rejected sit deep inside it.  Name them in
    // the panic as well, so the failure reads as a diagnosis and not as a dump.
    assert!(accepts, "honest composed proof rejected{}: {j}", quant_vs_key_tile(&j));
    if tamper.norm_row.is_none() {
        assert_eq!(j["ref_mismatch"], 0,
                   "composed witness disagrees with the f64 reference");
    }
    assert_eq!(j["off_table"], 0, "honest composed witness has off-table values");
    emit("compose", json!({"source": l.source, "point": j}));
}

/// Issue #172: the quantize components' integers against the key tiles', as one
/// clause for the rejection message.  Empty where the two agree, which is every
/// segment whose rejection is something else.
fn quant_vs_key_tile(j: &Value) -> String {
    let Some(qs) = j["family_cells"]["quant"].as_array() else { return String::new() };
    let bad: Vec<String> = qs.iter()
        .filter(|q| q["mismatch"].as_u64().unwrap_or(0) > 0)
        .map(|q| format!("{} {}/{}", q["component"].as_str().unwrap_or("?"),
                         q["mismatch"], q["checked"]))
        .collect();
    if bad.is_empty() {
        return String::new();
    }
    format!(" -- the quantize components derived integers the key tiles do not hold \
({}), which is the layer the witness carries and not the layer the statement can state",
            bad.join(", "))
}

/// Stream J: the master exp table at three sizes on one witness.
///
/// `sized` is what the driver picks today (`d_bits + spread`, or `--table-bits`);
/// `tight` is the largest key the key tiles actually look up; `tamper` is one bit
/// below that, which leaves keys off the table holding the numerators they already
/// have -- an internally consistent witness that only the lookup can object to.
fn oversize_probes_cmd(args: &[String]) {
    let (l, vname) = layer_of(args);
    let mut points = vec![];
    let mut probe = |name: &str, l: &layer::Layer| -> bool {
        let params = params_of(l, args, &vname);
        let off: usize = l.key_tiles.iter().map(|b| b.off_table).sum();
        eprintln!("oversize probe {name}: table 2^{} off_table {off}", params.log_table);
        let (j, accepts) = compose_once(l, &params, &Default::default(), None);
        points.push(json!({
            "probe": name, "log_table": params.log_table,
            "table_rows": 1usize << params.log_table,
            "witness_off_table": off,
            "accepts": accepts,
            "off_table": j["off_table"],
            "cells": j["cells"], "pre_root": j["pre_root"],
            "prove_error": j["prove_error"],
        }));
        accepts
    };
    assert!(probe("sized", &l), "the honest layer at the driver's size was rejected");
    let need = l.key_bits();

    let mut tight = layer_clone(&l);
    tight.set_master_table(need);
    assert!(probe("tight", &tight), "the honest layer at the tight size was rejected");

    let mut bad = layer_clone(&l);
    bad.set_master_table(need - 1);
    assert!(bad.key_tiles.iter().map(|b| b.off_table).sum::<usize>() > 0,
            "the tamper probe has no off-table key -- nothing to reject");
    assert!(!probe("tamper", &bad),
            "a witness with keys past the table produced an accepting proof");

    emit("oversize-probes", json!({"source": l.source, "key_bits": need, "probes": points}));
}

fn psweep_cmd(args: &[String]) {
    let (l, vname) = layer_of(args);
    let mut points = vec![];
    for p in arg_list(args, "--p-sweep", &[32usize, 64, 136, 272, 541]) {
        let mut params = params_of(&l, args, &vname);
        params.freivalds_p = p;
        eprintln!("psweep p={p}");
        let (mut j, accepts) = compose_once(&l, &params, &Default::default(), None);
        assert!(accepts, "honest composed proof rejected at p={p}");
        j["p"] = json!(p);
        points.push(j);
    }
    emit("psweep", json!({"source": l.source, "points": points}));
}

/// Stream II, the requantize seams' output-side control.  One committed
/// activation of a Freivalds group is moved to another legal value and the layer's
/// exact accumulators move with it, so the chunk proves an honest matmul of
/// the changed input: no tamper, no inconsistent component, just a prover that
/// multiplied activations of its own choosing -- which is what
/// `witness-slack.md` section 1d says the Freivalds check used to allow.
///
/// The element is the one whose bend disturbs the fewest *dequantized* outputs
/// of its token, because those are what the rest of the chunk consumes: one
/// activation's worth of accumulator is below a bf16 ulp on most channels, so
/// the search usually finds an element the consumers cannot see at all and the
/// seam's own output binding is then the only thing that can reject.  Returns the
/// element and how many dequantized outputs it did move.
fn bend_freivalds_qx(l: &mut layer::Layer, specs: &[dequant::Spec], mats: &[usize],
                     start: usize) -> Option<(usize, usize)> {
    let m0 = &l.mats[mats[0]];
    let (n_in, tokens) = (m0.n_in, m0.tokens);
    if n_in * tokens == 0 {
        return None;
    }
    let noise = |l: &layer::Layer, idx: usize, d: i64| -> usize {
        let (t, k) = (idx / n_in, idx % n_in);
        mats.iter().map(|mi| {
            let m = &l.mats[*mi];
            (0..m.n_out).filter(|c| {
                let a0 = m.acc[t * m.n_out + c];
                let a1 = a0 + d * m.qw[k * m.n_out + c] as i64;
                // stream SS: a folded matmul's output is an integer, not a bf16
                // pattern -- `deq_ref_a` has no answer for it and used to panic
                // here the moment a `y-delay-v` witness reached a probe.
                if m.variant == dequant::Variant::Fold {
                    return layer::fold_out_a(a1, m, t, *c) != layer::fold_out_a(a0, m, t, *c);
                }
                layer::deq_ref_a(a1, m, &specs[*mi], t, *c)
                    != layer::deq_ref_a(a0, m, &specs[*mi], t, *c)
            }).count()
        }).sum()
    };
    // any other legal activation: the Freivalds checks range check `qx + 2^(bits-1)`, and
    // both neighbours of an interior value are inside that window
    let dir = |l: &layer::Layer, idx: usize| -> i64 {
        if l.mats[mats[0]].qx[idx] < 126 { 1 } else { -1 }
    };
    let mut best = (usize::MAX, 0usize, 0i64);
    for step in 0..n_in * tokens {
        let idx = (start + step) % (n_in * tokens);
        let d = dir(l, idx);
        let s = noise(l, idx, d);
        if s < best.0 {
            best = (s, idx, d);
        }
        if s == 0 {
            break;
        }
    }
    let (moved, idx, d) = best;
    let (t, k) = (idx / n_in, idx % n_in);
    for mi in mats {
        let m = &mut l.mats[*mi];
        assert_eq!(m.n_in, n_in, "Freivalds group mixes input widths");
        m.qx[idx] = (m.qx[idx] as i64 + d) as i32;
        let n_out = m.n_out;
        for c in 0..n_out {
            m.acc[t * n_out + c] += d * m.qw[k * n_out + c] as i64;
        }
    }
    Some((idx, moved))
}

/// Issue #17: the model tree's control, named once -- the probe, the skip and
/// the test in `prover/tests/model_tree_control.rs` all read it by this name.
pub const MODEL_SWAP: &str = "model tree: this segment under another layer's weights";

/// Issue #24: the `o` seam's control, named once for the same reason -- the
/// probe and the skip read it by this name.
pub const O_SEAM: &str = "o seam: one activation the attention output did not produce";

/// Issue #134: the control suite's record, written as it is made rather than
/// when the suite finishes.
///
/// `compose-probes` emitted its record at the end, so a control that aborted the
/// process took every refusal before it with it and the run had nothing to keep
/// -- on the 8B segment, 41 designed rejections lost to the 42nd.  Each
/// control's result lands here as it happens, and a panic emits what is there
/// before the process leaves, so a crash costs one control and not the suite.
static PROBE_RECORD: std::sync::Mutex<Vec<Value>> = std::sync::Mutex::new(Vec::new());

fn record_probe(v: Value) {
    PROBE_RECORD.lock().unwrap_or_else(|e| e.into_inner()).push(v);
}

/// Everything the suite has established, and nothing twice: the record is taken
/// out, so the panic hook and the ordinary end of the run cannot both write it.
fn recorded_probes() -> Vec<Value> {
    std::mem::take(&mut *PROBE_RECORD.lock().unwrap_or_else(|e| e.into_inner()))
}

fn compose_probes_cmd(args: &[String]) {
    let (l, vname) = layer_of(args);
    // Issue #134: what the suite established, when it cannot finish.  The
    // default hook still prints the panic and the exit code is still the
    // panic's; what is new is that the controls already refused reach the
    // record, which says it was cut short and by what.
    {
        let source = l.source.clone();
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            previous(info);
            static EMITTED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if EMITTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                return;
            }
            emit("compose_probes", json!({"source": source, "probes": recorded_probes(),
                                          "crashed": info.to_string()}));
        }));
    }
    let params = params_of(&l, args, &vname);
    let (hj, ok) = compose_once(&l, &params, &Default::default(), None);
    assert!(ok, "honest composed proof rejected");
    let honest_root = hj["pre_root"].as_str().unwrap().to_string();
    // The preprocessed trace is two trees, so a control that moves a
    // preprocessed value moves whichever of the two carries its column: the
    // shape tree for a table, a selector or a rotation angle, the model tree
    // for a weight or a weight scale.  Both honest roots are held, and the
    // probe pins both, or a control aimed at the model tree is compared with
    // nothing and is accepted.
    let honest_model_root = hj["model_root"].as_str().unwrap().to_string();
    let honest_v = hj["verdict"].clone();
    record_probe(json!({"probe": "honest", "result": hj.clone()}));
    // Stream NN: which verdict fields a control actually turned off.  "The
    // control rejects" is not the whole claim a record should make -- "and by
    // exactly this one field" is, because a probe that trips three checks at
    // once has not isolated the binding it was written for.  Fields that are true
    // (or all-true) on the honest run and not on this one, and nothing else.
    let falsified = |got: &serde_json::Value| -> Vec<String> {
        let all_true = |v: &serde_json::Value| -> Option<bool> {
            match v {
                serde_json::Value::Bool(b) => Some(*b),
                serde_json::Value::Array(a)
                    if !a.is_empty() && a.iter().all(|x| x.is_boolean()) =>
                    Some(a.iter().all(|x| x.as_bool().unwrap())),
                _ => None,
            }
        };
        let mut f = vec![];
        if let Some(o) = honest_v.as_object() {
            for (k, hv) in o {
                if all_true(hv) == Some(true)
                    && all_true(got.get(k).unwrap_or(&serde_json::Value::Null)) == Some(false)
                {
                    f.push(k.clone());
                }
            }
        }
        f
    };

    // Stream NN-3: `--probe <substring>[,<substring>...]` runs only the controls
    // whose name contains one of them (`arg_strs` splits on commas and reads
    // one flag), so a control can have a process of its own.  The reason is paid
    // for on the H200's real 8B chunk, where the suite is 46 probes and about two
    // hours.  With no `--probe` the list is what it always was.
    let only = arg_strs(args, "--probe", &[]);
    let selected = |name: &str| only.is_empty() || only.iter().any(|w| name.contains(w.as_str()));
    // Stream JJ (E11): the two `jo` controls move `jo` by one and have to stay
    // inside the statement's own windows, or the `EPOW` lookup would catch the
    // move instead of the pin the control exists to test.  A shape whose
    // windows admit one `jo` and no other -- the synthetic layer's, whose `jv`
    // spans the exponent window and whose every token carries the same `jo` --
    // can state neither, and the prover's own step choice panics on it.  A
    // control the shape cannot state is skipped here by name, with its reason
    // in its own record: a crash is never a verdict, and it would take every
    // probe after it and the suite's JSON with it.
    let (jo_row_site, jo_tok_site) = {
        let lay = layer::AoLayout::choose(params.nh, params.s, params.hd, params.ao_target);
        let (e_lo, e_hi) = compose::Plan::new(&params).e_window();
        (l.ao.jo_bend_site(&lay, params.ao_jo_max, e_lo, e_hi),
         l.ao.jo_tok_site(params.ao_jo_max, e_lo, e_hi))
    };
    let mut unstatable: Vec<(&str, String)> = [("attnout: one row's jo", jo_row_site.is_none(), "row"),
                                           ("attnout: a whole token's jo", jo_tok_site.is_none(),
                                            "token")]
        .iter()
        .filter(|(_, none, _)| *none)
        .map(|(w, _, what)| (*w, format!(
            "this statement's windows admit one jo for every {what} of this witness and no \
             other, so a one-step move leaves the EPOW window and is refused by that lookup \
             before the joff pin it tests can see it; a run with --ao-jo-max above the \
             witness's own jo states it")))
        .collect();
    // Issue #17: the model tree's control rebuilds this segment's activations
    // around a second draw of the model half, which is the synthetic
    // construction's to do -- a dump's model half is the dump's, and a rebuild
    // of it here would be a witness this repository invented.  A shape with no
    // residual-stream family has nothing to rebuild with either.
    if arg_str(args, "--witness").is_some() {
        unstatable.push((MODEL_SWAP,
            "a dump carries its own model half, and rebuilding this segment's activations \
             around a second draw is the synthetic construction's; on a dump the swap would \
             be a witness nothing wrote".into()));
    } else if l.residual_stream.is_none() {
        unstatable.push((MODEL_SWAP,
            "this shape carries no residual-stream family, so there is nothing to rebuild \
             the segment around the swapped weights with".into()));
    }
    // Issue #24: the `o` seam's control rebuilds this segment around a second
    // draw of the residual stream -- the same generator the three key-tile
    // controls use, and those are skipped on a dump for the reason that decides
    // this one too.  A dump's activations are B1's; a second draw is another
    // layer's, and on a real segment it lands outside the windows this statement
    // pins from the honest witness: the master exp table's width and the
    // attention output's accumulator, on the 8B chunk both at once.  A witness
    // refused for the draw says nothing about the seam, so this is a skip by name
    // and not a rejection the control did not earn.
    if arg_str(args, "--witness").is_some() {
        unstatable.push((O_SEAM,
            "the seam's control rebuilds this segment around a second draw of the residual \
             stream, which is the synthetic construction's; a dump's activations are the \
             dump's, and a second draw lands outside the windows this statement pins from \
             the honest witness -- the master exp table's width and the attention output's \
             accumulator -- so what refuses such a witness is the draw and not the seam".into()));
    } else if !l.residual_stream.as_ref().is_some_and(|g| !g.head)
        || !l.mats.iter().any(|m| m.name == "o" && m.n_in == l.ao.nh * l.ao.hd)
    {
        unstatable.push((O_SEAM,
            "this shape has no residual-stream family of a layer segment, or its family is \
             the head's final norm, or its o Freivalds check is not nh*hd wide, so the two \
             folds are over different sequences and ao_q_ok is not statable here".into()));
    }
    let unstatable = unstatable;
    let statable = |name: &str| !unstatable.iter().any(|(w, _)| name.contains(*w));
    // Issue #151: a control's own copy of the segment, and how many were built.
    // Six of the controls below bend the witness rather than the prover, so
    // each needs a segment of its own, and on a dump that is another read of
    // the whole witness.  They were built before the filter was read, so a run
    // that asked for one control paid a reload for every control it was never
    // going to run: 3,144 s on the 8B chunk, for a segment whose honest proof
    // takes 137 s.  Every caller asks `selected` first now, and the count goes
    // to stderr at the end of the run, which is what a filtered run can be read
    // for.
    let rebuilds = std::cell::Cell::new(0usize);
    let rebuild = |what: &str| -> layer::Layer {
        rebuilds.set(rebuilds.get() + 1);
        eprintln!("  segment rebuilt for {what}");
        layer_of(args).0
    };
    let probe = |name: &str, l2: &layer::Layer, t: &compose::ProverTamper,
                 expect: &str| -> bool {
        if !selected(name) {
            record_probe(json!({"probe": name, "skipped": "not selected by --probe"}));
            return false;
        }
        if let Some((_, why)) = unstatable.iter().find(|(w, _)| name.contains(*w)) {
            eprintln!("compose probe SKIP {name}: {why}");
            record_probe(json!({"probe": name, "skipped": why}));
            return false;
        }
        eprintln!("compose probe {name}");
        // verify against the HONEST fixed statement: the trusted half comes from
        // setup, never from the prover under test
        let out2 = compose::prove_layer(l2, &params, t);
        // a tamper that changed nothing proved the honest witness, and calling
        // that "accepted" is how `drop-big` at layer 0 read as a soundness gap
        // (its aimed slot's operands shared an exponent); it is a broken probe
        assert_ne!(out2.residual_stream_tamper_fired, Some(0),
                   "probe {name}: the residual-stream tamper did not fire, so nothing was measured");
        let claims_b = serde_json::to_vec(&out2.claims).unwrap();
        let mut fixed = out2.fixed.clone();
        fixed.pre_root = honest_root.clone();
        // both preprocessed roots come from the honest run: `model_root` left
        // as the tamper's own is a declaration of what the tamper committed,
        // and `model_root_ok` would compare the bent tree with itself.
        fixed.model_root = honest_model_root.clone();
        let fixed_b = serde_json::to_vec(&fixed).unwrap();
        let (verdict, root_differs, model_root_differs) = match &out2.proof {
            Some(pf) => {
                let proof_b = serde_json::to_vec(pf).unwrap();
                let v = compose::verify_composed(&fixed_b, &claims_b, &proof_b);
                let rd = format!("{:?}", pf.commitments[0]) != honest_root;
                let md = format!("{:?}", pf.commitments[1]) != honest_model_root;
                (Some(v), rd, md)
            }
            None => (None, false, false),
        };
        let accepted = verdict.as_ref().map(|v| v.accepts()).unwrap_or(false);
        let flipped = verdict.as_ref()
            .map(|v| falsified(&serde_json::to_value(v).unwrap()))
            .unwrap_or_default();
        record_probe(json!({
            "probe": name, "expect_reject_by": expect,
            "accepted": accepted,
            "falsified": flipped,
            "tamper_fired": out2.residual_stream_tamper_fired,
            // Stream NN-3: the `s_v` binding is an equality between two tree-0
            // column families, so no verdict field can carry it -- a verifier
            // holds it through `pre_root`.  The probe record carries the
            // build's own reading, which is how a control on it is visible.
            "sv_binding": out2.family_cells.get("sv_binding"),
            // Issue #134: how many committed values no table of this statement can
            // absorb.  A control that moves a key past the master exp table --
            // which `--tight-table` leaves no slack against -- is refused by that
            // lookup, and this is where the record says so; the verdict alone
            // would only show the sum failing to cancel.
            "off_table": out2.off_table,
            "prove_error": out2.prove_error,
            "root_differs_from_honest": root_differs,
            "model_root_differs_from_honest": model_root_differs,
            "verdict": verdict.as_ref().map(|v| serde_json::to_value(v).unwrap()),
        }));
        accepted
    };

    // Stream OO: every `layer_clone` below is dropped as soon as its probe has
    // run.  A shadowed `let mut l2` is not freed until the function returns, so
    // the six of them were held for the whole run and the 8B suite was
    // OOM-killed on the box at probe 31 of 46 (log row 57, rc 137).  Nothing
    // about what a probe proves changes.
    // 1. a corrupted accumulator: the dequant component stays self-consistent and the
    //    Y = W X identity is what rejects
    let mut l2 = layer_clone(&l);
    let dn = l2.mats.iter().position(|m| m.name == "down").unwrap();
    let n = l2.mats[dn].acc.len();
    l2.mats[dn].acc[123 % n] += 1;
    assert!(!probe("corrupted accumulator (down)", &l2, &Default::default(),
                   "the Y = W X identity"), "corrupted acc accepted");
    drop(l2);

    // 2. a corrupted activation: the same identity, other side
    let mut l2 = layer_clone(&l);
    for name in ["q", "k", "v"] {
        let mi = l2.mats.iter().position(|m| m.name == name).unwrap();
        let n = l2.mats[mi].qx.len();
        let x = &mut l2.mats[mi].qx[77 % n];
        *x = if *x < 127 { *x + 1 } else { *x - 1 };
    }
    assert!(!probe("corrupted activation (qkv)", &l2, &Default::default(),
                   "the Y = W X identity"), "corrupted qx accepted");
    drop(l2);

    // 3. a corrupted key in key tile 0: the Q.K^T identity
    let mut l2 = layer_clone(&l);
    let n = l2.key_tiles[0].kk.len();
    l2.key_tiles[0].kk[1234 % n] += 1;
    assert!(!probe("corrupted key (key tile 0)", &l2, &Default::default(), "Q.K^T"),
            "corrupted key accepted");
    drop(l2);

    // 4. a corrupted attention output: the A.V link
    let mut l2 = layer_clone(&l);
    let n = l2.ao.acc.len();
    l2.ao.acc[999 % n] += 1;
    assert!(!probe("corrupted attention output", &l2, &Default::default(), "A.V"),
            "corrupted attnout accepted");
    drop(l2);

    // 5. a corrupted attnout denominator: the denominator link
    let mut l2 = layer_clone(&l);
    let n = l2.ao.den.len();
    l2.ao.den[7 % n] += 1;
    assert!(!probe("corrupted attnout denominator", &l2, &Default::default(),
                   "the denominator link"), "corrupted den accepted");
    drop(l2);

    // 5b. B10 step 3: a token that attends to a key the causal mask removes.  The
    //     diagonal block's `sn` and the prefix's `sn` are what the per-token component's `az`
    //     is read against, and the AIR's fold drops the masked slot, so the two no
    //     longer add up to the denominator the witness claims.
    if l.key_tiles.iter().any(|b| b.diagonal_block) {
        for (k, i) in [(0usize, 11usize), (1, 4242)] {
            let mut l2 = layer_clone(&l);
            l2.unmask_diagonal_block(i);
            let name: &'static str = if k == 0 {
                "attends to one future key"
            } else {
                "attends to a second future key"
            };
            assert!(!probe(name, &l2, &Default::default(), "the denominator link"),
                    "an unmasked future key accepted");
        }
    }

    // 6. the corrupted exp table, witness rebuilt to match: every internal check
    //    passes and the PINNED ROOT is what rejects -- B6.5's acceptance flip
    let mut l2 = layer_clone(&l);
    l2.corrupt_exp_table(4321);
    assert!(!probe("corrupted exp table, witness rebuilt", &l2, &Default::default(),
                   "the pinned preprocessed root"), "corrupted exp table accepted");
    drop(l2);

    // 7. the corrupted normalisation table, same discipline
    assert!(!probe("corrupted norm table, witness rebuilt", &l,
                   &compose::ProverTamper { norm_row: Some(1000), ..Default::default() },
                   "the pinned preprocessed root"), "corrupted norm table accepted");

    // 9-11. B10 step 2, rule 13: the wiring family's own probes, inside the
    //    composed path.  The first is the one the bindings exist for -- a component's
    //    copy of the residual stream is changed and everything *inside* that
    //    component stays consistent, so only the equality between two components'
    //    claims can catch it.  The other two are the add's own argument, run
    //    here rather than only in `residual-stream-probes`, on the real layer's columns.
    if l.residual_stream.is_some() {
        // stream II: the head chunk's family has one reduction and no add, so
        // there is no second component's copy of the stream to break -- that probe
        // is the layer's, and the head chunk's equivalent is the norm scale's own
        // copy, which probe 12 below runs on whichever family this chunk has.
        if !l.residual_stream.as_ref().is_some_and(|g| g.head) {
            assert!(!probe("residual stream: gn2's copy of the residual stream", &l,
                           &compose::ProverTamper { residual_stream_copy: Some((2, 4321)),
                                                    ..Default::default() },
                           "the binding gn1.xmid == gn2.in"),
                    "a broken residual thread accepted");
        }
        assert!(!probe("residual stream: one add's result, one step up", &l,
                       &compose::ProverTamper {
                           residual_stream: Some((0, residual_stream::Tamper::Result(97))),
                           ..Default::default() },
                       "the add's output constraint"),
                "a corrupted residual-stream result accepted");
        assert!(!probe("residual stream: the anchor at the smaller exponent", &l,
                       &compose::ProverTamper {
                           residual_stream: Some((0, residual_stream::Tamper::DropBig(1234))),
                           ..Default::default() },
                       "the POW lookup (the dropped operand is the smaller one)"),
                "drop-big accepted");
        assert!(!probe("residual stream: a leaf that is not a bf16 value", &l,
                       &compose::ProverTamper {
                           residual_stream: Some((1, residual_stream::Tamper::NotBf16(55))),
                           ..Default::default() },
                       "the VAL lookup"),
                "an illegal residual-stream leaf accepted");
    }

    // 12-15. stream E's components, on whichever of them this invocation built.
    //    The indices come from the plan rather than being counted by hand, so a
    //    later component cannot silently shift what these probes point at.
    {
        let gplan = compose::Plan::new(&params);
        if let Some(g) = gplan.residual_stream.as_ref() {
            // `Tamper::Stream` counts leaves across every group of the row and
            // fires only inside group 0, so the index has to be derived from the
            // layout rather than picked: a norm scale row is `[nreal x, 1 ms]` since
            // stream II moved the weight into tree 0, and anything at or past
            // `nreal` corrupts nothing at all.
            let in0 = g.t.nreal / 2;
            // stream JJ: how far into the SwiGLU component's value list `up` starts
            let swiglu_n = g.tff.map(|t| t.tokens * t.d).unwrap_or(0);
            for (k, i) in g.norm_scales.iter().copied().enumerate() {
                // stream II: the component's own tag, so a head chunk's single
                // norm scale is named `ntf` and the layer's two keep the names the
                // record already carries
                let n = g.components[i].tag.clone();
                let _ = k;
                // the binding is the whole point of the norm scale: its copy of the
                // stream is changed and the component stays internally consistent,
                // so only `nt.In(0) == gn.In(0)` can catch it
                assert!(!probe(&format!("norm_scale {n}: its own copy of the stream"), &l,
                               &compose::ProverTamper {
                                   residual_stream: Some((i, residual_stream::Tamper::Stream(in0))),
                                   ..Default::default() },
                               &format!("the binding {n}.In(0) == its reduction's In(0)")),
                        "a norm_scale reading a different stream accepted");
                assert!(!probe(&format!("norm_scale {n}: one multiply's result"), &l,
                               &compose::ProverTamper {
                                   residual_stream: Some((i, residual_stream::Tamper::Result(97))),
                                   ..Default::default() },
                               "the multiply's output constraint"),
                        "a corrupted norm_scale result accepted");
                // stream II's three controls on the norm scale.  The first two
                // change what `r` is without breaking anything inside the
                // component: the `NormR` lookup still finds a row and the two
                // multiplies still hold, so the only thing that can reject is
                // the binding to the reduction's published sum of squares.
                assert!(!probe(&format!("norm_scale {n}: a wrong r on one row"), &l,
                               &compose::ProverTamper {
                                   residual_stream: Some((i, residual_stream::Tamper::Ms(37))),
                                   ..Default::default() },
                               &format!("the binding {n}.In(1) == G the reduction's Out(0)")),
                        "a norm_scale with a wrong r accepted");
                assert!(!probe(&format!("norm_scale {n}: two rows of one token, two rs"), &l,
                               &compose::ProverTamper {
                                   residual_stream: Some((i, residual_stream::Tamper::MsRow(37))),
                                   ..Default::default() },
                               &format!("the binding {n}.In(1) == G the reduction's Out(0)")),
                        "a token whose rows disagree about r accepted");
            }
            // and the third: one norm-weight cell bent.  It is preprocessed
            // now, so every constraint still holds and the *root* is what
            // moves -- a different kind of rejection from every probe above,
            // and the one that says `w` is the model's and not the prover's.
            // The weight is a model column, so the root that moves is the
            // model tree's and `model_root_ok` is the field that goes false.
            let wtag = g.components[g.norm_scales[0]].tag.clone();
            assert!(!probe(&format!("norm_scale {wtag}: one norm-weight cell"), &l,
                           &compose::ProverTamper { residual_stream_w: Some(4321),
                                                    ..Default::default() },
                           "the pinned model root"),
                    "a bent norm weight accepted");
            // the dequant output bindings: the wiring family's copy of `o_out` (and
            // of `d_out`) is changed, the component's own add is recomputed from it,
            // and only the binding against the dequant component's published output can
            // tell.  This is the probe those bindings exist for.
            // Stream JJ (E11): one per binding that has a counterparty.  For the
            // SwiGLU seam the bent value is `gate`'s copy or `up`'s -- the
            // component recomputes `silu(g) * u` from it either way, so nothing
            // inside the component can tell and only the binding against the gate (or
            // up) dequant component's published output rejects.
            for t in g.output_bindings.iter() {
                let Some((ri, ti)) = t.at else { continue };
                let name = t.name;
                let vi = if name == "up" { 4321 + swiglu_n } else { 4321 };
                let _ = ti;
                assert!(!probe(&format!("deq binding: the wiring copy of {name}'s output"), &l,
                               &compose::ProverTamper {
                                   residual_stream_add: Some((ri, vi)), ..Default::default() },
                               &format!("the binding {name}_out == the {name} component's output")),
                        "a corrupted dequant output copy accepted");
            }
            if let Some(i) = g.swiglu {
                assert!(!probe("swiglu: the product, one step up", &l,
                               &compose::ProverTamper {
                                   residual_stream: Some((i, residual_stream::Tamper::Result(4097))),
                                   ..Default::default() },
                               "the multiply's output constraint"),
                        "a corrupted swiglu product accepted");
                assert!(!probe("swiglu: a leaf that is not a bf16 value", &l,
                               &compose::ProverTamper {
                                   residual_stream: Some((i, residual_stream::Tamper::NotBf16(1234))),
                                   ..Default::default() },
                               "the UN / VAL lookup"),
                        "an illegal swiglu leaf accepted");
            }
        }
    }

    // stream II: the three requantize seams.  Each gets the three controls the
    //     brief names, and each has to reject by exactly one verdict field:
    //     the input probe by `rq_in_ok`, the output probe by `rq_out_ok`, and
    //     the scale probe by `sums_cancel` (the committed scale no longer
    //     divides the attained maximum, so its `(pm, scale)` lookup is off the
    //     table).  The indices are derived from the plan, never counted.
    {
        let rplan = compose::Plan::new(&params);
        for (k, rp) in rplan.rqs.iter().enumerate() {
            let n = rp.name.clone();
            assert!(!probe(&format!("requant {n}: one input value, replaced"), &l,
                           &compose::ProverTamper {
                               rq: Some((k, requant::RqTamper::In(37))),
                               ..Default::default() },
                           &format!("the binding rq_in[{k}] == the producer's Out(0)")),
                    "a requantize of values the producer did not write accepted");
            // the output side: no tamper at all.  The layer itself carries one
            // activation the requantize did not derive, and its own
            // accumulators move with it, so every constraint and every other
            // equality in the chunk still holds -- an honest matmul of the
            // prover's own choice of input, which is exactly what
            // `witness-slack.md` section 1d says the Freivalds check used to allow.
            // Issue #151: the name is known before the segment is, so the
            // filter decides first and a control this run will not reach costs
            // no rebuild.  `probe` writes the same skip either way.
            let out_side = format!("requant {n}: one activation the requantize did not derive");
            let out_expect = format!("the binding rq_out[{k}] == freivalds_qx[{}]", rp.freivalds);
            if !selected(&out_side) {
                probe(&out_side, &l, &Default::default(), &out_expect);
            } else {
                let mut lb = rebuild(&out_side);
                let bent = bend_freivalds_qx(&mut lb, &rplan.specs,
                                             &rplan.freivalds[rp.freivalds].mats, 41)
                    .unwrap_or_else(|| panic!("requant {n}: the output-side control bent nothing"));
                eprintln!("  {n}: activation {} bent, {} dequantized outputs moved with it",
                          bent.0, bent.1);
                assert!(!probe(&out_side, &lb, &Default::default(), &out_expect),
                        "an activation the requantize did not derive accepted");
            }
            assert!(!probe(&format!("requant {n}: the per-token scale, one step up"), &l,
                           &compose::ProverTamper {
                               rq: Some((k, requant::RqTamper::Scale(3))),
                               ..Default::default() },
                           "the (pm, scale) lookup"),
                    "a scale that is not the ceil-divide of the attained max accepted");
        }
        // Stream II (E11), the scale binding.  A *consumer's* per-token scale moved
        // by one, with its dequant component rederived under it, so every
        // constraint that component carries still holds -- an honest dequant by a
        // scale the requantize did not pin.  The bend has to be on a matmul
        // that is not its group's first, because the seam reads its scale off
        // that one and moving it would move both sides together; `down`'s group
        // is `[down]` alone, so its own pair is not separable by a witness bend
        // and would need a committed-column tamper instead.
        for (k, rp) in rplan.rqs.iter().enumerate() {
            let mats = &rplan.freivalds[rp.freivalds].mats;
            let Some(mi) = mats.get(1).copied() else { continue };
            // Issue #151: the matmul's name is the segment's, and the honest
            // segment already in hand carries it -- so the probe's name, and
            // with it the filter, is read before anything is rebuilt.
            let name = l.mats[mi].name.clone();
            let sc_side = format!("requant {}: {name} dequantized by another scale", rp.name);
            let sc_expect = format!("the binding rq_sc[{k}] == deq_sc[{name}]");
            if !selected(&sc_side) {
                probe(&sc_side, &l, &Default::default(), &sc_expect);
                continue;
            }
            let mut lb = rebuild(&sc_side);
            match rp.v {
                requant::SVariant::Pow2 => lb.mats[mi].jx[0] += 1,
                requant::SVariant::Bf16 => lb.mats[mi].sx[0].1 += 1,
            }
            let at = mats.iter().position(|x| *x == mi).unwrap();
            let base: usize = rplan.rqs[..k].iter().map(|q| rplan.freivalds[q.freivalds].mats.len()).sum();
            assert!(!probe(&sc_side, &lb, &Default::default(), &sc_expect),
                    "a dequant under a scale the requantize did not pin accepted");
            let _ = base + at;
        }

        // Stream II (E11), the `o` seam.  `o` has no requantize component --
        // `AttnOutEval` is its requantize -- so its control is the same shape as
        // the three above's output side and no tamper either: one `o`
        // activation the attention output did not produce, with the `o` matmul's
        // own accumulator moved to match, so the matmul is honestly proved and
        // only `claims.ao_q` against the Freivalds check's `qX` can see it.
        if let Some(wi) = rplan.freivalds.iter().position(|w| w.name == "o") {
            let expect = format!("the binding ao_q == freivalds_qx[{wi}]");
            // Issue #24: the rebuild draws the whole segment again, which on the
            // 8B chunk is another load of the dump and most of an hour -- and
            // where the suite is not going to run the control, a witness nobody
            // proves.  So it is built only where the control is the suite's to
            // run, and `probe` writes the skip and its reason into the record for
            // every other shape.
            if !(selected(O_SEAM) && statable(O_SEAM)) {
                probe(O_SEAM, &l, &Default::default(), &expect);
            } else {
                let mut lb = rebuild(O_SEAM);
                let round = Round::parse(&arg_strs(args, "--round", &["even"])[0]);
                match lb.synth_residual_stream_bent(
                    src_of(args).seed, arg_scalar(args, "--residual-stream-a", 32usize), round, 41)
                {
                    Some(at) => {
                        eprintln!("  o seam: activation {at} bent, the whole layer rebuilt \
around it");
                        assert!(!probe(O_SEAM, &lb, &Default::default(), &expect),
                                "an o activation the attnout component did not requantize \
accepted");
                    }
                    // the skip above reads the same two conditions off the honest
                    // segment, so this arm is the generator disagreeing with it --
                    // a record either way, never a control that quietly did not run
                    None => record_probe(json!({"probe": O_SEAM, "skipped":
                        "the generator built no bend for this shape, though the segment's own \
                         widths said it could"})),
                }
            }
        }

        // ---- stream NN: the rotation and the quantize -----------------------
        //
        // Two bindings per side and one preprocessed pin each, and every one of
        // them has to reject by a *different* verdict field -- which is what
        // `falsified` in the probe record says, rather than prose:
        //
        //   producer side, q and k   one dequant output copy      deq_out_ok
        //   producer side, all three one committed input          nn_in_ok
        //   consumer side, all three one key tile cell                nn_out_ok
        //   the rotation's table     one angle cell               root_ok
        //   the quantize's table     one scale exponent           root_ok
        //   the division table       one quotient                 sums_cancel
        if !rplan.quants.is_empty() {
            let g = rplan.residual_stream.as_ref().expect("the quantize components need the wiring family");
            let round = Round::parse(&arg_strs(args, "--round", &["even"])[0]);
            let (seed, ga) = (src_of(args).seed, arg_scalar(args, "--residual-stream-a", 32usize));
            // the input side of each rotation: one copy of the matmul's dequant
            // output bent, the whole rotation and the quantize recomputed from
            // it, so every constraint in three components still holds and only the
            // binding against the matmul's own published output can tell
            for rr in &g.ropes {
                assert!(!probe(&format!("rope {}: one dequant output copy", rr.name), &l,
                               &compose::ProverTamper {
                                   residual_stream_add: Some((rr.mul, 4321)), ..Default::default() },
                               &format!("the binding deq_out[{}] == rp{}m.In(0)", rr.name, rr.name)),
                        "a rotation of values the matmul did not write accepted");
            }
            // one preprocessed angle: the products follow it, so every
            // constraint holds and the *root* is what moves.  This is the probe
            // that says `cos` and `sin` are the statement's and not the
            // prover's -- and, because they are a function of the block, that
            // tree 0 is pinned per block.
            assert!(!probe("rope q: one cos cell of the angle table", &l,
                           &compose::ProverTamper { rope_w: Some(97),
                                                    ..Default::default() },
                           "the pinned preprocessed root"),
                    "a bent rotation angle accepted");
            for (k, qp) in rplan.quants.iter().enumerate() {
                let n = qp.name;
                // the quantize's own input: another legal bf16 pattern, the
                // quotient rebuilt around it.  For q and k the counterparty is
                // the rotation's `Out(0)`; for v it is the matmul's `deq_out`.
                assert!(!probe(&format!("quant {n}: one input value, replaced"), &l,
                               &compose::ProverTamper {
                                   quant: Some((k, rope::QTamper::In(37))),
                                   ..Default::default() },
                               &format!("the binding nn_in[{k}] == the producer's output")),
                        "a quantize of values the producer did not write accepted");
                assert!(!probe(&format!("quant {n}: one quotient off its own table row"), &l,
                               &compose::ProverTamper {
                                   quant: Some((k, rope::QTamper::Out(41))),
                                   ..Default::default() },
                               "the (tt, m7, q) division table"),
                        "a quotient the division table does not carry accepted");
                assert!(!probe(&format!("quant {n}: one preprocessed scale exponent"), &l,
                               &compose::ProverTamper {
                                   quant: Some((k, rope::QTamper::Scale(5))),
                                   ..Default::default() },
                               "the pinned preprocessed root"),
                        "a bent quantize exponent accepted");
            }
            // the consumer side: one *key tile* cell moved and the key tiles rebuilt
            // around it, so the attention this chunk proves is an honest
            // attention over the bent column and nothing but the key tile's own
            // claim against the quantize component's output can see it.  The
            // generator is the witness's, so this needs a synthetic layer.
            for (which, name) in [(0usize, "Q"), (1, "K"), (2, "V")] {
                // the generator rebuilds the whole layer, so it only makes
                // sense on a synthetic one -- on a dump it would replace the
                // witness under test with a different layer and the probe would
                // measure nothing about this chunk
                if arg_str(args, "--witness").is_some() {
                    eprintln!("  key_tile {name}: SKIP -- the consumer-side control needs the \
synthetic generator (Layer::synth_residual_stream_key_tile_bent); on a dump the key_tiles are B1's");
                    continue;
                }
                // Issue #151: whether this shape has a wiring family is the
                // honest segment's answer, so the skip is decided without a
                // rebuild -- as is the filter below it.
                if l.residual_stream.is_none() {
                    eprintln!("  key tile {name}: SKIP -- no synthetic wiring family here");
                    continue;
                }
                // stream SS: with a folded `v` there is no quantize component on
                // that side, so the falsified field is the dequant output binding
                // and not `nn_out_ok` -- the fold publishes the int8 cell
                // itself and the key tiles' V is what reads it
                let binding = if which == 2 && rplan.output_claim_key_tile("v") {
                    "the binding deq_out[v] == the sum of the key tiles' own V claims"
                } else {
                    "the binding nn_out == the sum of the key tiles' own claims"
                };
                let kt = format!("key tile {name}: one committed cell the matmuls did not produce");
                if !selected(&kt) {
                    probe(&kt, &l, &Default::default(), binding);
                    continue;
                }
                let mut lb = rebuild(&kt);
                lb.synth_residual_stream_key_tile_bent(seed, ga, round, which, 41);
                assert!(!probe(&kt, &lb, &Default::default(), binding),
                        "a key tile column the layer's matmuls did not produce accepted");
            }
            // Issue #151: both skips below and the filter are read off the
            // honest segment, which is what this shape's family and its key
            // tiles are; nothing is rebuilt for a control that will not run.
            //
            // stream NN (NN-2): and the prefix key tile's own copy of Q, whose
            // counterparty is its group's diagonal block rather than a quantize component.
            // The bend moves one prefix cell and the key tile's whole attention is
            // rebuilt around it, so the chunk is an honest attention over the
            // cache with queries the `q` matmul did not produce -- and only
            // `nn_q_ok` can see that.
            if arg_str(args, "--witness").is_some() {
                eprintln!("  prefix Q: SKIP -- the consumer-side control needs the \
synthetic generator (Layer::synth_residual_stream_prefix_q_bent); on a dump the key_tiles are B1's");
            } else if l.residual_stream.is_none() {
                eprintln!("  prefix Q: SKIP -- no synthetic wiring family here");
            } else if l.key_tiles.iter().all(|b| b.diagonal_block) {
                eprintln!("  prefix Q: SKIP -- this shape has no prefix key tile");
            } else {
                let pq = "prefix key tile Q: one query the diagonal block does not carry";
                let pq_expect = "the binding prefix.Q == diagonal_block.Q, per key tile group";
                if !selected(pq) {
                    probe(pq, &l, &Default::default(), pq_expect);
                } else {
                    let mut lp = rebuild(pq);
                    lp.synth_residual_stream_prefix_q_bent(seed, ga, round, 41);
                    assert!(!probe(pq, &lp, &Default::default(), pq_expect),
                            "a prefix key tile's queries that are not the diagonal block's accepted");
                }
            }
        }
    }

    // Issue #17: the model tree's own control -- a whole-layer swap, not one
    //     cell.  The half of the model tree that hangs off the segment's second
    //     norm (the `gate`, `up` and `down` weights, their weight scales and
    //     that norm's own weight; on a head chunk the final norm's weight and
    //     `lm_head`) is taken from a second draw and the whole layer is rebuilt
    //     around it.  What is proved is an honest segment under another layer's
    //     weights -- every constraint, every equality and both boundaries hold
    //     -- so the pinned model root is the only thing that can reject, and
    //     the record says which of the two preprocessed roots moved.
    // Issue #151: the swap rebuilds the whole segment, so the filter is read
    // here rather than inside `probe`, as it is for the `o` seam above.
    if selected(MODEL_SWAP) && statable(MODEL_SWAP) {
        let mut lm = rebuild(MODEL_SWAP);
        let round = Round::parse(&arg_strs(args, "--round", &["even"])[0]);
        let (seed, ga) = (src_of(args).seed, arg_scalar(args, "--residual-stream-a", 32usize));
        lm.synth_model_swap(seed, ga, round, seed ^ 0x0000_0017);
        assert!(!probe(MODEL_SWAP, &lm, &Default::default(), "the pinned model root"),
                "a segment proved under another layer's weights accepted");
        drop(lm);
    } else {
        // `probe` writes the skip -- the shape's reason, or the filter -- into
        // the record itself
        probe(MODEL_SWAP, &l, &Default::default(), "the pinned model root");
    }

    // 16. stream E: the head binding.  One committed key of the argmax component is
    //     changed and the component is made consistent with it, so every constraint
    //     it carries still holds and only the equality with the head dequant's
    //     own key claim can reject.
    if compose::Plan::new(&params).argmax_mat().is_some() {
        assert!(!probe("head binding: one of the argmax's committed keys", &l,
                       &compose::ProverTamper {
                           argmax: Some(argmax::Tamper::Key(1234567)),
                           ..Default::default() },
                       "the binding am_key == logits_claim"),
                "an argmax on keys that are not the head's output accepted");
    }

    // 11c. Stream UU (E11): one cell of a *residual* boundary tree moved alone,
    //     the residual-stream component it is bound to left exactly as it was.  Stream H's
    //     `bounds` probe made this assertion for the pair of sides a layer
    //     chunk carries, but only on real witness dirs; here it runs on every
    //     shape the probe command is given, which is what puts the head chunk's
    //     single input side (`boundary::sides`, one side, tree 3) under it.
    //     Nothing else in the chunk moves, so the only thing that can reject is
    //     the plain equality constraint that binds the tree to the component --
    //     which is the whole statement that the root is bound to the proof and
    //     not merely appended to it.
    if params.layer_boundary {
        let n_residual_stream = compose::Plan::new(&params).layer_sides.len();
        assert!(n_residual_stream > 0,
                "the layer boundary produced no residual side: the chunk has no residual-stream components");
        for k in 0..n_residual_stream {
            assert!(!probe(&format!("residual boundary side {k}, its residual-stream component untouched"),
                           &l,
                           &compose::ProverTamper { layer_boundary: Some((k, 4321)),
                                                    ..Default::default() },
                           "the residual boundary equality constraint"),
                    "a residual boundary column that disagrees with its component accepted");
        }
    }

    // 12. Stream AA (E11): one cell of a K/V boundary tree moved *alone*, the
    //     component it is bound to left exactly as it was.  Nothing else in the
    //     chunk changes, so the only thing that can reject is `ChanEval`'s /
    //     `KeyEval`'s equality constraint -- and that is precisely the statement
    //     that the boundary root is *bound to the proof* and not merely appended
    //     to it.  A stream that cannot make this reject has not bound the root.
    //
    //     Every side is probed, not just the first: an out side and an in side
    //     are written by different sets of key tiles, and a binding that held on one
    //     and not the other would chain a chunk to a prefix it never proved.
    if params.kv_boundary {
        let n_sides = compose::Plan::new(&params).kv.len();
        assert!(n_sides > 0, "--kv-boundary produced no K/V boundary sides");
        for k in 0..n_sides {
            let side = compose::Plan::new(&params).layer_sides.len() + k;
            assert!(!probe(&format!("kv: one cell of K/V boundary side {k}, component untouched"),
                           &l,
                           &compose::ProverTamper { layer_boundary: Some((side, 4321)),
                                                    ..Default::default() },
                           "the K/V boundary equality constraint"),
                    "a K/V boundary column that disagrees with its component accepted");
        }
    }
    // 7b. Stream BB: the embedding gadget's three probes, when the run asked for
    //     it.  The second is the aimed corruption -- the id moved to a different
    //     valid entry with the committed row recomputed to match, so the component
    //     is internally consistent and every constraint inside it holds; the binding
    //     to the layer's residual-stream input is the only thing that can see it.
    if params.embed_vocab > 0 {
        for (name, expect, t) in [
            ("embedding: one token id moved, the committed row left alone",
             "the embedding lookup (sums do not cancel)", embed::Tamper::Tok(3)),
            ("embedding: one token id moved to another valid entry, its row \
              recomputed to match",
             "the binding to the layer's residual-stream input", embed::Tamper::TokRow(5)),
            ("embedding: one committed channel of the stream changed",
             "the embedding lookup (sums do not cancel)", embed::Tamper::Val(777)),
        ] {
            assert!(!probe(name, &l,
                           &compose::ProverTamper { embed: Some(t), ..Default::default() },
                           expect),
                    "{name} accepted");
        }
        // and the table probe, which is the pinned-root probe: the component
        // rebuilds the stream from the corrupted table, so every internal check
        // passes and only tree 0's root disagrees
        assert!(!probe("embedding: one cell of the committed table changed, witness rebuilt",
                       &l,
                       &compose::ProverTamper { embed: Some(embed::Tamper::Table(4321)),
                                                ..Default::default() },
                       "the pinned preprocessed root"),
                "a corrupted embedding table accepted");
    }

    // 17. Stream JJ (E11): the attention output's denominator on a row the `dz`
    //     fold never publishes.  `den` is one committed cell per row and it
    //     entered exactly two places -- `denp = den pj` per slot, and `den *
    //     last` in the `dz` fold -- so on every row of a token but the last it
    //     was free, and with it `denp`.  This probe rounds one such row honestly
    //     against `den - 1`: `denp`, `qm`, `sg` and `d` are all rederived from
    //     it, every lookup lands on a real table row and every constraint of the
    //     pre-change AIR holds.  It is the control for `witness-slack.md`
    //     section 7 item 1 and for the row-`den` hole found beside it.
    {
        let lay = layer::AoLayout::choose(params.nh, params.s, params.hd, params.ao_target);
        if lay.chunks > 1 {
            // `b = 1` is a channel block the `dz` fold does not close on
            let idx = params.hd + lay.per_row;
            assert!(!probe("attnout: one non-last row rounded against den - 1", &l,
                           &compose::ProverTamper { ao_den: Some(idx),
                                                    ..Default::default() },
                           "the `den` pin across a token's rows"),
                    "a rounding by an unpinned denominator accepted");
        }
    }

    // 18. Stream JJ (E11), `witness-slack.md` section 1f: the three controls the
    //     exponent window and the accumulator window exist for.  Each bends one
    //     committed cell and rederives everything the honest builder would from
    //     it, so every other check of the chunk still passes and exactly one
    //     thing can reject:
    //
    //       * `--corrupt-ao-jo` moves one *row*'s `jo` by one and leaves the
    //         token's other rows alone.  Every `EPOW` lookup still lands on a
    //         real table row and the row's own arithmetic is self-consistent, so
    //         only the `nofirst` pin on `joff` can see it.  (A whole token's
    //         `jo` moved by one is a different matter and is *accepted*: `jo` is
    //         the prover's choice inside the window, and what pins it to the
    //         consumer's scale is the binding on `claims.ao_q`, which is stream
    //         II's.  See the record.)
    //       * `--corrupt-ao-num` doubles one slot's committed `num` and derives
    //         `qm`, `sg`, `d` and the reference from it.  The triple
    //         `(jvp + joff, num, pj)` is then not a row of the `EPOW` table, so
    //         only that lookup can reject it.
    //       * `--corrupt-ao-acc` moves one slot's committed low accumulator limb
    //         by one inside its own table, so both range lookups land on real
    //         rows and only the high limb's *value* -- no longer an integer --
    //         parts company with the multiplicity the table was bumped at.
    {
        let lay = layer::AoLayout::choose(params.nh, params.s, params.hd, params.ao_target);
        if lay.chunks > 1 {
            // a row the `nofirst` pin compares against its predecessor and whose
            // channels have room for the move -- `jo_bend_site` picks it, and
            // `None` is the shape that cannot state this control at all, which
            // is skipped by name and asserted on by nothing
            let jo_row = "attnout: one row's jo moved by one";
            let rejected = !probe(jo_row, &l,
                                  &compose::ProverTamper { ao_jo: jo_row_site,
                                                           ..Default::default() },
                                  "the `joff` pin across a token's rows");
            assert!(!statable(jo_row) || rejected, "a per-row output exponent accepted");
        }
        assert!(!probe("attnout: one slot's num a doubling out of the window", &l,
                       &compose::ProverTamper { ao_num: Some(params.hd + 1),
                                                ..Default::default() },
                       "the `EPOW` window lookup"),
                "an exponent pair outside the window accepted");
        assert!(!probe("attnout: one accumulator limb off its own decomposition", &l,
                       &compose::ProverTamper { ao_acc: Some(params.hd + 2),
                                                ..Default::default() },
                       "the accumulator window's high limb"),
                "an unbounded accumulator accepted");
        // Stream TT (E11), log row 73: the magnitude form's own.  `amag` is
        // pinned by `(amag - acc)(amag + acc) = 0`, which has two roots; this
        // commits the other one.  Every constraint of the component holds and
        // every other lookup lands, and the committed magnitude is about
        // `2^31`, so `2 mag num` would run past the field -- which is the
        // forgery the window exists to refuse.  Only the range check on the
        // magnitude's limbs can see it.  Present only in the magnitude form,
        // because the offset form commits no such column.
        if params.ao_acc_mag {
            assert!(!probe("attnout: the magnitude committed as the negative root", &l,
                           &compose::ProverTamper { ao_mag: Some(params.hd + 3),
                                                    ..Default::default() },
                           "the magnitude window's limbs"),
                    "a magnitude past the no-wrap window accepted");
        } else {
            eprintln!("compose probe SKIP attnout magnitude root: this statement is in \
                       the offset form (--ao-acc-mag forces the other)");
        }
        // And the counterpart, whose verdict is now the *chunk's* to decide.  A
        // whole token's `jo` moved by one inside the window, with the component
        // rederived, changes every one of that token's requantized outputs.
        //
        // Stream JJ ran this and it accepted, and its record said what would
        // pin it: the consumer-side binding on `claims.ao_q`, which was stream II's
        // to add.  With that binding live it *does* pin it -- the `o` Freivalds check's
        // committed activations are the pre-bend integers and the attnout
        // component's fold is over the post-bend ones, so `ao_q_ok` alone rejects.
        //
        // On a chunk with no wiring family the `o` Freivalds check publishes no `qX` at
        // all, so there is no consumer and this still accepts.  That is not a
        // hole this control can close: `jo` is the prover's choice and a chunk
        // whose attention output nothing downstream reads has nothing to
        // compare it against.  Both cases are run, and the assert is the
        // chunk's own shape rather than a constant.
        // The head chunk's family builds no `o` seam, so the seam is not live
        // there either (issue #58) -- one more chunk with no consumer for `jo`.
        let seam_live = compose::Plan::new(&params).residual_stream
                .as_ref().is_some_and(|g| !g.head_norm)
            && l.mats.iter().any(|m| m.name == "o" && m.n_in == l.ao.nh * l.ao.hd);
        let jo_name = "attnout: a whole token's jo moved by one";
        let jo_tok = probe(jo_name, &l,
                           &compose::ProverTamper { ao_jo_tok: jo_tok_site,
                                                    ..Default::default() },
                           if seam_live { "the o seam, `ao_q` against the Freivalds check's qX" }
                           else { "nothing -- this chunk has no consumer for it" });
        // this is the one control whose pass is an acceptance on some shapes, so
        // it reads the probe's own answer rather than a rejection; a `--probe`
        // that leaves it out, and a shape that cannot state the move at all,
        // both return the same `false` a rejection does, and the assert has to
        // know the difference or a run fails on a control it never ran.
        assert!(!selected(jo_name) || !statable(jo_name) || !jo_tok == seam_live,
                "a whole token's jo must reject exactly when the o seam is live: \
                 seam_live={seam_live}, accepted={jo_tok}");
    }

    // 8. a tampered claim in the serialized statement: the replayed transcript
    //    diverges and stwo's verify rejects
    {
        eprintln!("compose probe tampered claim");
        let out2 = compose::prove_layer(&l, &params, &Default::default());
        let mut claims = out2.claims.clone();
        claims.mat_h[0][0] ^= 1;
        let claims_b = serde_json::to_vec(&claims).unwrap();
        let fixed_b = serde_json::to_vec(&out2.fixed).unwrap();
        let proof_b = serde_json::to_vec(out2.proof.as_ref().unwrap()).unwrap();
        let v = compose::verify_composed(&fixed_b, &claims_b, &proof_b);
        assert!(!v.accepts(), "tampered claim accepted");
        record_probe(json!({"probe": "tampered claim in the statement",
                            "expect_reject_by": "transcript divergence (stwo verify)",
                            "accepted": v.accepts(),
                            "verdict": serde_json::to_value(&v).unwrap()}));
    }
    // Issue #151: what the run's own filter cost, in segments.  A control that
    // holds a copy of the witness builds it only where it runs, so this is the
    // number of controls that ran and needed one -- never the number the suite
    // has.  The record does not carry it: a control is not a proof of record
    // and neither is the work a run skipped.
    eprintln!("compose probes: {} segment rebuilds", rebuilds.get());
    emit("compose_probes", json!({"source": l.source, "probes": recorded_probes(),
                                  "crashed": Value::Null}));
}

/// Issue #134: a control's rebuild keeps the exp table the statement carries.
///
/// Every control in the suite rebuilds the segment around its own bend and is
/// verified against the honest statement, so the master exp table it looks up is
/// that statement's.  Where the rebuilt key tiles want a wider key than the
/// statement takes -- which is what `--tight-table` makes of a real segment,
/// where the table is the honest witness's own with no slack -- the keys that do
/// not fit stay off the table, no lookup argument can absorb them, and the proof
/// is refused.  Before this the rebuild aborted the process, and the suite lost
/// every control it had already refused along with it.
///
/// The shape is the smallest one that reaches the case: its q, k and v matmuls
/// line up with the key tiles, so a control rebuilds the key tiles through the
/// rotation, and its keys are wide enough that the 2^22 rows a rebuild may
/// always grow to are not what stops it.
#[cfg(test)]
mod pinned_exp_table {
    use super::{Round, layer, layer_of};

    /// How many bits this shape's keys need, which is what `--table-bits` gives
    /// it: an honest construction sizes its own table and refuses a line that
    /// gives it a narrower one.
    const TABLE_BITS: u32 = 24;

    fn args(table_bits: u32) -> Vec<String> {
        format!("compose-probes --round away --variants intsplit6 --table-bits {table_bits} \
--s 32 --ctx 32 --hd 64 --head-groups 1,1 --nin 128 --nout 128 --qbits 15 --kbits 14 \
--residual-stream-a 32 --kmax 16 --synth-residual-stream --diagonal-block")
            .split_whitespace().map(str::to_string).collect()
    }

    /// The honest segment, and how wide its keys are.
    fn honest() -> (layer::Layer, u32) {
        let l = layer_of(&args(TABLE_BITS)).0;
        assert!(l.rope.is_some(),
                "this shape wires no rotation, so no control rebuilds the key tiles and \
                 the test measures nothing");
        let need = l.key_bits();
        assert_eq!(need, TABLE_BITS, "this shape's keys are {need} bits wide");
        assert!(need > 22,
                "the keys must be wider than the 2^22 rows a rebuild may always grow to, \
                 or the statement's own width is not what stops it");
        (l, need)
    }

    /// The `o` seam's control on a segment whose table is one bit under what the
    /// rebuilt key tiles look up -- a real segment's tight table, as the control's
    /// own rebuild finds it.  The rebuild leaves the table where the statement put
    /// it and the keys that do not fit off it, so the proof is refused; it used to
    /// abort the process instead.
    #[test]
    fn a_control_leaves_a_key_past_the_statements_table_off_it() {
        let (mut l, need) = honest();
        l.set_master_table(need - 1);
        assert_eq!(l.key_tiles[0].log_table, need - 1);
        let _bent = l.synth_residual_stream_bent(7, 32, Round::Away, 41)
            .expect("the o seam's control is statable on this shape");
        assert_eq!(l.key_tiles[0].log_table, need - 1,
                   "the control moved the exp table, so it proves another statement");
        assert_eq!(l.key_bits(), need,
                   "the rebuilt key tiles no longer look up a key past the table");
        assert!(l.key_tiles.iter().map(|b| b.off_table).sum::<usize>() > 0,
                "every rebuilt key fits the table, so nothing refuses this witness");
    }

    /// The other half of the same choice (issue #72): an honest construction sizes
    /// its own table, so a line that gives it a narrower one than its data needs is
    /// stopped before anything is proved, and told what to pass.
    #[test]
    fn an_honest_construction_still_refuses_a_line_whose_table_is_too_narrow() {
        let narrow = std::panic::catch_unwind(|| {
            layer_of(&args(TABLE_BITS - 1));
        });
        let Err(why) = narrow else {
            panic!("a table narrower than the honest data was accepted");
        };
        let msg = *why.downcast::<String>()
            .expect("the refusal says which --table-bits to pass");
        assert!(msg.contains(&format!("pass --table-bits {TABLE_BITS}")), "{msg}");
    }
}

/// The two-load path of `layer_of`.
#[cfg(test)]
mod lowering_load {
    use super::{layer, layer_of, params_of};

    fn line(cmd: &str, shape: &str) -> Vec<String> {
        format!("{cmd} --tight-table {shape}").split_whitespace().map(str::to_string).collect()
    }
    fn widths(l: &layer::Layer) -> Vec<u32> {
        l.key_tiles.iter().map(|t| t.log_table).collect()
    }

    /// A lowering line lands on the layer a proving line lands on: the same
    /// table width on every key tile and the same `Params`.
    #[test]
    fn a_lowering_line_lands_on_the_proving_lines_layer() {
        let shape = "--round away --variants intsplit6 --table-bits 22 --s 32 --ctx 128 --hd 32 \
--diagonal-block --unsound-skip-residual-stream";
        let (a, va) = layer_of(&line("cudaemit", shape));
        let (b, vb) = layer_of(&line("compose", shape));
        assert!(a.key_bits() < 22, "the keys want the whole stated table, so the probe decides nothing");
        assert_eq!(widths(&a), widths(&b));
        let pa = serde_json::to_string(&params_of(&a, &line("cudaemit", shape), &va)).unwrap();
        let pb = serde_json::to_string(&params_of(&b, &line("compose", shape), &vb)).unwrap();
        assert_eq!(pa, pb);
    }

    /// A synthetic residual stream rebuilds the key tiles under a rotation that
    /// sizes the table off them, so a lowering line keeps the single load and
    /// lands at the width it states.
    #[test]
    fn a_synthetic_residual_stream_line_keeps_the_single_load() {
        let shape = "--round away --variants intsplit6 --table-bits 24 --s 32 --ctx 32 --hd 64 \
--head-groups 1,1 --nin 128 --nout 128 --qbits 15 --kbits 14 --residual-stream-a 32 --kmax 16 \
--synth-residual-stream --diagonal-block";
        let (l, _) = layer_of(&line("cudaemit", shape));
        assert!(l.rope.is_some(), "this shape wires no rotation, so the test measures nothing");
        assert_eq!(l.key_bits(), 24);
        assert!(widths(&l).iter().all(|w| *w == 24), "{:?}", widths(&l));
    }
}

/// The Layer is plain data; clone by rebuilding the pieces (Attn is not Clone).
fn layer_clone(l: &layer::Layer) -> layer::Layer {
    layer::Layer {
        key_tiles: l.key_tiles.iter().map(attn_clone).collect(),
        key_tile_sizes: l.key_tile_sizes.clone(),
        head_offsets: l.head_offsets.clone(),
        nsub: l.nsub,
        mats: l.mats.iter().map(|m| layer::Mat {
            name: m.name.clone(), variant: m.variant, n_in: m.n_in, n_out: m.n_out,
            tokens: m.tokens, acc: m.acc.clone(), qx: m.qx.clone(), qw: m.qw.clone(),
            jx: m.jx.clone(), jw: m.jw.clone(), sx: m.sx.clone(), sw: m.sw.clone(),
            jv2: m.jv2.clone(),
        }).collect(),
        ao: layer::AttnOut { nh: l.ao.nh, s: l.ao.s, hd: l.ao.hd, acc: l.ao.acc.clone(),
                             den: l.ao.den.clone(), jo: l.ao.jo.clone(), jv: l.ao.jv.clone() },
        residual_stream: l.residual_stream.as_ref().map(|g| layer::ResidualStreamCols {
            d: g.d, dff: g.dff, tokens: g.tokens,
            xin: g.xin.clone(), o_out: g.o_out.clone(), xmid: g.xmid.clone(),
            d_out: g.d_out.clone(), xout: g.xout.clone(),
            r1: g.r1.clone(), r2: g.r2.clone(),
            w1: g.w1.clone(), w2: g.w2.clone(),
            h1: g.h1.clone(), h2: g.h2.clone(),
            silu_in: g.silu_in.clone(), silu_out: g.silu_out.clone(),
            x_down: g.x_down.clone(),
            eps: g.eps, inv_d: g.inv_d, reduce: g.reduce.clone(), head: g.head,
        }),
        head_keys: l.head_keys.clone(),
        head_tok: l.head_tok.clone(),
        attn_out_bits: l.attn_out_bits,
        embed_tab: l.embed_tab.clone(),
        tok_in: l.tok_in.clone(),
        layer: l.layer,
        block: l.block,
        rope: l.rope.clone(),
        source: l.source.clone(),
    }
}

fn attn_clone(a: &attention::witness::Attn) -> attention::witness::Attn {
    attention::witness::Attn {
        shape: a.shape, variant: a.variant, qq: a.qq.clone(), kk: a.kk.clone(),
        vv: a.vv.clone(), sc: a.sc.clone(), mm: a.mm.clone(), dd: a.dd.clone(),
        xx: a.xx.clone(), num: a.num.clone(), zz: a.zz.clone(), out: a.out.clone(),
        table: a.table.clone(), log_table: a.log_table, nmax: a.nmax, frac: a.frac,
        frac_per_inst: a.frac_per_inst.clone(), delta: a.delta.clone(),
        q_bits: a.q_bits, k_bits: a.k_bits, source: a.source.clone(),
        off_table: a.off_table, torch_mismatch: a.torch_mismatch, diagonal_block: a.diagonal_block,
        zz_total: a.zz_total.clone(),
    }
}

// ----------------------------------------------------------------- requant ----

fn rwitness_of(args: &[String], v: requant::SVariant) -> requant::RWitness {
    let default_mat = if v == requant::SVariant::Pow2 { "q" } else { "gate" };
    let mat = arg_str(args, "--mat").unwrap_or_else(|| default_mat.into());
    let n: usize = arg_scalar(args, "--n", 0usize);
    if let Some(dir) = arg_str(args, "--witness") {
        match requant::RWitness::load(&dir, &mat, n) {
            Ok(w) => return w,
            Err(e) => eprintln!("  (no real requant witness: {e})"),
        }
    }
    requant::RWitness::synthetic(arg_scalar(args, "--tokens", 64usize),
                                 arg_scalar(args, "--width", 64usize), v, 7)
}

/// The requantize + scale-argument gadget: cells per value, both scale shapes.
fn requant_cmd(args: &[String]) {
    let mut points = vec![];
    for vname in arg_strs(args, "--variants", &["pow2", "bf16"]) {
        let v = requant::SVariant::parse(&vname);
        let w = rwitness_of(args, v);
        for with_max in [false, true] {
            eprintln!("requant {} with_max={} ({})", vname, with_max, w.source);
            let o = requant::run(&w, v, with_max, arg_scalar(args, "--per-row", 16usize), None);
            if !o.accepts() {
                eprintln!("  REJECTED: stark={} sums={} off={} err={:?}", o.proven.stark_ok,
                          o.sums.cancels(), o.off_table, o.proven.prove_error);
            }
            assert!(o.accepts(), "honest requant proof rejected ({vname}, max={with_max})");
            assert_eq!(o.ref_n - o.ref_ok, 0,
                       "requant disagrees with B1's committed qx ({vname})");
            points.push(o.json(&w.source));
        }
    }
    let checks: Vec<_> = ["pow2", "bf16"].iter()
        .map(|v| requant::check_tables(requant::SVariant::parse(v), 20000)).collect();
    emit("requant", json!({"points": points, "tables": checks}));
}

fn requant_probes_cmd(args: &[String]) {
    let v = requant::SVariant::parse(&arg_strs(args, "--variants", &["bf16"])[0]);
    let w = rwitness_of(args, v);
    let honest = requant::run(&w, v, true, 8, None);
    assert!(honest.accepts(), "honest requant proof rejected");
    let mut out = vec![json!({"probe": "honest", "result": honest.json(&w.source)})];
    let cases: Vec<(&str, requant::Tamper)> = vec![
        ("wrong quotient", requant::Tamper::WrongQ(11)),
        ("nz flipped", requant::Tamper::FlipNz(13)),
        ("scale one step too big", requant::Tamper::ScaleUp(3)),
        ("max below an attained value", requant::Tamper::PmLow(5)),
        ("max above every value", requant::Tamper::PmHigh(5)),
    ];
    for (name, t) in cases {
        eprintln!("requant probe {name}");
        let o = requant::run(&w, v, true, 8, Some(t));
        assert!(!o.accepts(), "{name} must be rejected");
        out.push(json!({"probe": name, "accepted": o.accepts(),
                        "stark_ok": o.proven.stark_ok,
                        "sums_cancel": o.sums.cancels(), "off_table": o.off_table,
                        "prove_error": o.proven.prove_error}));
    }
    emit("requant_probes", json!({"variant": v.name(), "probes": out}));
}

// ------------------------------------------------------------- B10 residual stream ----
// The wiring between the composed layer's components (B6.5 open gap 1), priced per
// gadget and checked against B1's own committed columns where there is one.
fn residual_stream_cmd(args: &[String]) {
    use residual_stream::{ResidualStreamWitness, Kind, Prog, Tiling, UnKind};
    let dir = arg_str(args, "--witness");
    let kmax: u32 = arg_scalar(args, "--kmax", 18u32);
    let round = Round::parse(&arg_strs(args, "--round", &["even"])[0]);
    let leaves: usize = arg_scalar(args, "--leaves", 32usize);
    let per_row: usize = arg_scalar(args, "--per-row", 16usize);
    let rows: usize = arg_scalar(args, "--rows", 4096usize);
    let tokens: usize = arg_scalar(args, "--tokens", 512usize);
    let a: usize = arg_scalar(args, "--residual-stream-a", 32usize);
    let mut points = vec![];

    let check = |o: &residual_stream::Outcome, what: &str| {
        if !o.accepts() {
            eprintln!("  {what}: stark={} sums={} off={} binding={} err={:?}", o.proven.stark_ok,
                      o.sums.cancels(), o.off_table, o.binding_ok, o.proven.prove_error);
        }
        assert!(o.accepts(), "honest residual-stream proof rejected ({what})");
        assert_eq!(o.ref_n - o.ref_ok, 0, "residual stream disagrees with its own reference ({what})");
        assert_eq!(o.bad_exp, 0, "{what}: {} results leave bf16's normal range", o.bad_exp);
    };

    // the three primitives, on synthetic values with a wide exponent spread so
    // the dropping rows and the exact zero are both exercised
    let cases: Vec<(&str, Prog, u64)> = vec![
        ("add", Prog::tree(leaves, leaves, UnKind::None), 7),
        ("unary+add", Prog::tree(leaves, leaves, UnKind::Square), 11),
        ("mul", Prog::pairs(per_row, Kind::Mul, UnKind::None, UnKind::None), 13),
        ("swiglu", Prog::pairs(per_row, Kind::Mul, UnKind::Silu, UnKind::None), 17),
        // Stream II: the norm scale as it is composed -- the reduction's answer
        // through `UnKind::NormR` and a preprocessed weight.  Here it runs on
        // synthetic values, which is the isolation the composed rung has not
        // got: a failure in this case is the component, never the wiring.
        ("normscale", Prog::norm_scale(per_row, UnKind::NormR {
            inv_d: bf16::from_f64(1.0 / per_row as f64), eps: bf16::from_f64(1e-5) }), 19),
    ];
    for (name, prog, seed) in &cases {
        let w = ResidualStreamWitness::synthetic(rows, prog, round, *seed);
        let o = residual_stream::run(&w, prog, round, kmax, None);
        check(&o, name);
        points.push(o.json(&w.source, name));
    }

    // the composed layer's own four components, on B1's own committed columns:
    // `xin + o_out` has to reproduce `xmid` and `xmid + d_out` has to reproduce
    // `xout`, and the two levels of each norm have to be `tree_sum`'s own order
    let mut drift = vec![];
    let mut b1 = vec![];
    if let Some(d) = &dir {
        let man: Value = witread::read_manifest(d).expect("manifest");
        let dm = man["residual_stream"]["d"].as_u64()
            .expect("manifest: residual_stream.d -- extract with --residual-stream") as usize;
        let rd = |n: &str| -> Vec<u16> {
            residual_stream::read_u16(d, n).unwrap_or_else(|e| panic!("{e}"))
        };
        let t = Tiling::new(tokens, dm, a);
        let l1 = Prog::norm_resid(t.nreal, t.b);
        let l2 = Prog::tree(t.a, t.a, UnKind::None);
        let mut stream = t.lay(&rd("xin")[..tokens * dm]);
        for (k, (add, want, wname)) in
            [("o_out", "xmid", "n1"), ("d_out", "xout", "n2")].iter().enumerate()
        {
            let w1 = ResidualStreamWitness::new(t.rows1(), vec![stream.clone(), t.lay(&rd(add))],
                                                &l1, round, format!("B1 {add} [{tokens} x {dm}]"));
            let o1 = residual_stream::run(&w1, &l1, round, kmax, None);
            check(&o1, wname);
            let mut j = o1.json(&w1.source, &format!("norm{}+resid{}", k + 1, k + 1));
            // the component's own outputs, value by value, against B1's column
            let mut tab = residual_stream::ResidualStreamTables::new(kmax, round, &l1.un_kinds());
            let r1 = residual_stream::residual_stream_component(&w1, &l1, round, "r", &mut tab, None);
            let got = t.unlay(&r1.outs[1]);
            let wv = rd(want);
            let bad = (0..tokens * dm)
                .filter(|i| crate::encoding::canon(got[*i]) != crate::encoding::canon(wv[*i]))
                .count();
            let neg0 = wv[..tokens * dm].iter().filter(|q| **q == 0x8000).count();
            j["b1_values_checked"] = json!(tokens * dm);
            j["b1_mismatch"] = json!(bad);
            j["b1_negative_zeros"] = json!(neg0);
            assert_eq!(bad, 0, "{wname}: the bf16 add disagrees with B1 on {bad}");
            points.push(j);
            b1.push(json!({"column": want, "values": tokens * dm, "mismatch": bad}));

            let w2 = ResidualStreamWitness::new(tokens, vec![t.roots_to_l2(&r1.outs[0])], &l2, round,
                                                format!("the level-1 roots [{tokens} x {}]", t.a));
            let o2 = residual_stream::run(&w2, &l2, round, kmax, None);
            check(&o2, &format!("{wname} level 2"));
            points.push(o2.json(&w2.source, &format!("norm{}-l2", k + 1)));
            let mut tab2 = residual_stream::ResidualStreamTables::new(kmax, round, &[]);
            let r2 = residual_stream::residual_stream_component(&w2, &l2, round, "r", &mut tab2, None);
            drift.push(residual_stream::norm_drift(d, dm, tokens, round, wname, Some(&r2.outs[0]))
                       .expect("norm drift"));
            stream = t.lay(&got);
        }
    }
    // the three primitives, isolated -- the numbers `b2/cost_model.py` reads.
    // A bf16 value is two committed columns here (`sm`, `ef`), so a unary op's
    // marginal is its two output columns plus the lookup's paired 2 cells, and
    // the producer's own two columns are not its cost.
    let pick = |g: &str| -> f64 {
        points.iter().find(|p| p["gadget"] == g)
            .and_then(|p| p["cells_per_op_marginal"].as_f64()).unwrap_or(0.0)
    };
    let prim = json!({
        "add": pick("add"), "mul": pick("mul"),
        // a committed bf16 value is one column per part of the `(sm, ef)` pair,
        // and a unary op commits its output as one and pays its lookup's share of
        // an interaction column: `plogup::marginal_cells`, which `zk-prover logup`
        // measures rather than assuming (issue #140)
        "unary": residual_stream::VAL_COLS.len() as f64
            + plogup::marginal_cells(plogup::PAIRED),
        "value": residual_stream::VAL_COLS.len() as f64,
        "note": "add and mul measured on a power-of-two tiled component, each output range-checked against VAL; unary is its two output columns plus what one lookup costs at pairwise batching, which `logup` measures; a committed bf16 value is 2 cells",
    });
    emit("residual_stream", json!({"points": points, "primitives": prim,
                                   "tables": residual_stream::check_tables(round, 20000),
                                   "dmax": residual_stream::DMAX, "against_b1": b1,
                                   "norm_reduction_drift": drift}));
}

fn residual_stream_probes_cmd(args: &[String]) {
    use residual_stream::{ResidualStreamWitness, Kind, Prog, Tamper, UnKind};
    let kmax: u32 = arg_scalar(args, "--kmax", 18u32);
    let round = Round::parse(&arg_strs(args, "--round", &["even"])[0]);
    let rows: usize = arg_scalar(args, "--rows", 256usize);
    let mut out = vec![];
    let cases: Vec<(&str, Prog, u64)> = vec![
        ("add tree", Prog::tree(8, 8, UnKind::None), 7),
        ("square + add tree", Prog::tree(8, 8, UnKind::Square), 11),
        ("multiply", Prog::pairs(8, Kind::Mul, UnKind::None, UnKind::None), 13),
        ("norm + resid", Prog::norm_resid(8, 8), 19),
    ];
    for (label, prog, seed) in cases {
        let w = ResidualStreamWitness::synthetic(rows, &prog, round, seed);
        let honest = residual_stream::run(&w, &prog, round, kmax, None);
        if !honest.accepts() {
            eprintln!("  {label}: stark={} sums={} off={} binding={} refbad={} err={:?}",
                      honest.proven.stark_ok, honest.sums.cancels(), honest.off_table,
                      honest.binding_ok, honest.ref_n - honest.ref_ok, honest.proven.prove_error);
        }
        assert!(honest.accepts(), "honest {label} rejected");
        assert_eq!(honest.ref_n - honest.ref_ok, 0, "{label}: reference mismatch");
        out.push(json!({"component": label, "probe": "honest",
                        "accepted": honest.accepts(),
                        "result": honest.json(&w.source, label)}));
        let tampers: Vec<Tamper> = if prog.kind == Kind::Mul {
            vec![Tamper::Result(3), Tamper::NotBf16(5)]
        } else {
            vec![Tamper::Result(3), Tamper::NormOut(5), Tamper::Anchor(7),
                 Tamper::DropBig(9), Tamper::Sign(11), Tamper::NotBf16(5)]
        };
        for t in tampers {
            eprintln!("residual-stream probe {label}: {}", t.name());
            let o = residual_stream::run(&w, &prog, round, kmax, Some(t));
            assert!(o.fired > 0, "{label}: {} did not fire, so nothing was measured", t.name());
            assert!(!o.accepts(), "{label}: {} must be rejected", t.name());
            out.push(json!({"component": label, "probe": t.name(),
                            "accepted": o.accepts(),
                            "tamper_fired": o.fired,
                            "stark_ok": o.proven.stark_ok,
                            "sums_cancel": o.sums.cancels(),
                            "off_table": o.off_table,
                            "binding_ok": o.binding_ok,
                            "prove_error": o.proven.prove_error}));
        }
    }
    emit("residual_stream_probes", json!({"probes": out, "dmax": residual_stream::DMAX}));
}

// ---------------------------------------------------------------- selftest ----
fn selftest(args: &[String]) {
    let mut runs = vec![];
    let n = arg_scalar(args, "--n", 1 << 10);
    for v in dequant::LADDER {
        for r in ["even", "away"] {
            if v == "limbs" && r == "away" {
                continue; // the baseline has no rounding mode
            }
            let sp = spec_of(args, v, r, 8);
            let w = Witness::get(&arg_str(args, "--witness"), "gate", n, 8, 896);
            let o = dequant::run(&w, &sp, None);
            if !o.accepts() {
                eprintln!("  {}: stark={} sums={} off={} err={:?}", sp.label(),
                          o.proven.stark_ok, o.sums.cancels(), o.off_table,
                          o.proven.prove_error);
            }
            assert!(o.accepts(), "selftest failed ({})", sp.label());
            assert!(o.proven.pre_ok, "preprocessed tree is not the widest ({})", sp.label());
            assert_eq!(o.ref_n - o.ref_ok, 0, "reference mismatch ({})", sp.label());
            runs.push(json!({"spec": sp.label(), "point": o.json(&w.source)}));
        }
    }
    // The packed attention, at a key tile of instances.  `intexp` is left out on
    // purpose: its table is indexed by the integer shift, so it needs 2^22-2^23 rows
    // on this witness and ~4.5 GiB per run -- nine of those in one process is how the
    // first version of this selftest got OOM-killed.  `intsplit6` exercises the same
    // packing at 2^16 rows, which is also what the sweep runs.
    let mut packed = vec![];
    for insts in [1usize, 2, 4] {
        let sh = Shape { s: 32, c: 128, hd: 64, insts };
        let cfg = attn_cfg(args, 8);
        for name in ["bf16x", "intsplit6"] {
            let v = attention::witness::Variant::parse(name);
            let a = Attn::get(&src_of(args), sh, v);
            let o = attention::run::run(&a, &cfg);
            if !o.accepts() {
                eprintln!("  attn {name} insts={insts}: stark={} qk={} av={} den={} sums={} \
                           off={} err={:?}", o.proven.stark_ok, o.qk_ok, o.av_ok, o.den_ok,
                          o.sums.cancels(), o.off_table, o.proven.prove_error);
            }
            assert!(o.accepts(), "packed attention selftest failed ({name}, insts={insts})");
            packed.push(json!({"variant": name, "insts": insts, "point": o.json(&a, &cfg)}));
        }
    }
    // the tables themselves, against a single f64 rounding
    let tbl = dequant::check_tables(16, Round::Even, 50_000);
    assert!(tbl["norm_mismatch"].as_array().unwrap().is_empty(), "norm table disagrees with f64");
    assert!(tbl["mul_mismatch"].as_array().unwrap().is_empty(), "mul table disagrees with f64");
    // the requantize / max-argument gadget, both scale shapes
    let mut rq = vec![];
    for v in ["pow2", "bf16"] {
        let sv = requant::SVariant::parse(v);
        let w = requant::RWitness::synthetic(32, 48, sv, 7);
        let o = requant::run(&w, sv, true, 8, None);
        assert!(o.accepts(), "requant selftest failed ({v})");
        assert_eq!(o.ref_n - o.ref_ok, 0, "requant reference mismatch ({v})");
        rq.push(o.json(&w.source));
    }
    // the composed proof, synthetic, through the serialized statement
    let (lc, vname) = layer_of(&["--table-bits".into(), "0".into()]);
    let pc = params_of(&lc, args, &vname);
    let (cj, ok) = compose_once(&lc, &pc, &Default::default(), None);
    assert!(ok, "composed selftest rejected: {cj}");
    emit("selftest", json!({"dequant": runs, "attention": packed, "tables": tbl,
                            "requant": rq, "compose": cj}));
}

/// Stream EE (open item 21): `fpcompare EMIT.json ASK.json` -- does the plan
/// line's emitted kernel set cover everything a chunk on that line looks up?
///
/// The rule is containment, not equality: one emit line installs the shapes of
/// every component of its design point, and a chunk asks for the subset its own
/// components use, so `ask ⊆ emit` is the property that keeps a run off the
/// scalar route. Any fingerprint on the ask side and not on the emit side is
/// the defect item 21 recorded, and it is named here with its family and its
/// instruction count.
fn fpcompare_cmd(args: &[String]) {
    let files: Vec<&String> = args[1..].iter().filter(|a| !a.starts_with("--")).collect();
    assert_eq!(files.len(), 2, "fpcompare wants EMIT.json and ASK.json");
    let read = |p: &str| -> serde_json::Value {
        serde_json::from_slice(&std::fs::read(p).unwrap_or_else(|e| panic!("{p}: {e}"))).unwrap()
    };
    let (emitted, asked) = (read(files[0]), read(files[1]));
    let mut missing = vec![];
    let mut n_ask = 0usize;
    for (family, fps) in asked["families"].as_object().expect("ask: no families") {
        for (fp, instr) in fps.as_object().unwrap() {
            n_ask += 1;
            if emitted["families"][family].get(fp).is_none() {
                missing.push(json!({"family": family, "fingerprint": fp,
                                    "instructions": instr}));
            }
        }
    }
    let n_emit: usize = emitted["families"].as_object().map(|o| {
        o.values().map(|v| v.as_object().unwrap().len()).sum()
    }).unwrap_or(0);
    for m in &missing {
        eprintln!("fpcompare: {} {} ({} instructions) is asked for and never emitted",
                  m["family"].as_str().unwrap(), m["fingerprint"].as_str().unwrap(),
                  m["instructions"]);
    }
    let covered = missing.is_empty();
    eprintln!("fpcompare: {n_ask} shape(s) asked, {n_emit} emitted, {} missing -- {}",
              missing.len(), if covered { "COVERED" } else { "NOT COVERED" });
    emit("fpcompare", json!({
        "emit": files[0], "ask": files[1],
        "emit_shapes": n_emit, "ask_shapes": n_ask,
        "missing": missing, "covered": covered,
    }));
    if !covered {
        std::process::exit(1);
    }
}

/// Issue #30: the rule for a repeated flag, pinned where it is implemented.
/// The digest the rule decides is pinned in `tests/synthetic_digest.rs`.
#[cfg(test)]
mod repeated_flags {
    use super::{arg_list, arg_scalar, arg_str, arg_strs, flag};

    fn args(words: &str) -> Vec<String> {
        words.split_whitespace().map(str::to_string).collect()
    }

    #[test]
    fn the_last_occurrence_wins() {
        let a = args("--table-bits 17 --variants intexp --table-bits 16 --variants intsplit6 \
                      --head-groups 16,8 --head-groups 2,1");
        assert_eq!(arg_scalar(&a, "--table-bits", 25u32), 16);
        assert_eq!(arg_str(&a, "--variants").as_deref(), Some("intsplit6"));
        assert_eq!(arg_strs(&a, "--variants", &["intexp"]), vec!["intsplit6".to_string()]);
        assert_eq!(arg_list(&a, "--head-groups", &[16usize]), vec![2usize, 1]);
    }

    #[test]
    fn one_occurrence_and_none_are_unchanged() {
        let a = args("--table-bits 16 --diagonal-block");
        assert_eq!(arg_scalar(&a, "--table-bits", 25u32), 16);
        assert_eq!(arg_scalar(&a, "--ao-rem-bits", 19u32), 19);
        assert_eq!(arg_str(&a, "--witness"), None);
        // a bare flag carries no value, so repeating it says nothing new
        assert!(flag(&a, "--diagonal-block"));
        assert!(flag(&args("--diagonal-block --diagonal-block"), "--diagonal-block"));
    }

    /// A value that is not a number leaves the default in place, as it did when
    /// the first occurrence won; what moved is only which occurrence is read.
    #[test]
    fn an_unparsable_value_falls_back_to_the_default() {
        assert_eq!(arg_scalar(&args("--table-bits wide"), "--table-bits", 25u32), 25);
    }
}

pub fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().cloned().unwrap_or_default();
    spans::install();
    // B12 stream GG: the fork's three marks inside `prove_ex` (a no-op on
    // the SIMD backend).
    device::install_mem_hook();
    // Issue #18's cache and issue #95's calibration, before any subcommand:
    // every command that proves consults the first and every command that
    // verifies reads the second, and a flag read in one command and ignored in
    // another is a switch that silently does nothing.
    shaperoot::configure(&args);
    match cmd.as_str() {
        "tables" => tables_cmd(&args),
        "dequant" => dequant_cmd(&args),
        "width" => width_cmd(&args),
        "kmax" => kmax_cmd(&args),
        "freivalds" => freivalds_cmd(&args),
        "logup" => logup_cmd(&args),
        "probes" => probes_cmd(&args),
        "pack" => pack_cmd(&args),
        "attn-probes" => attn_probes_cmd(&args),
        "compose" => compose_cmd(&args),
        "pre-split" => pre_split_cmd(&args),
        "expect-root" => expect_root_cmd(&args),
        "model-roots" => model_roots_cmd(&args),
        "compose-chunks" => chunks::chunks_cmd(&args),
        "exprprobe" => exprprobe::cmd(&args),
        "cudacheck" => cudagen::cmd(&args),
        "cudaemit" => cudagen::emit_cmd(&args),
        "logupcheck" => cudagen::logup_cmd(&args),
        "genemit" => cudagen::gen_emit_cmd(&args),
        "compose-probes" => compose_probes_cmd(&args),
        "oversize-probes" => oversize_probes_cmd(&args),
        "psweep" => psweep_cmd(&args),
        "requant" => requant_cmd(&args),
        "requant-probes" => requant_probes_cmd(&args),
        "argmax" => argmax_cmd(&args),
        "argmax-probes" => argmax_probes_cmd(&args),
        "embed" => embed_cmd(&args),
        "embed-probes" => embed_probes_cmd(&args),
        "residual-stream" => residual_stream_cmd(&args),
        "boundary" => boundary::cmd(&args),
        "boundary-chain" => boundary::chain_cmd(&args),
        "boundary-grid" => boundary::grid_cmd(&args),
        "residual-stream-probes" => residual_stream_probes_cmd(&args),
        "selftest" => selftest(&args),
        "fpcompare" => fpcompare_cmd(&args),
        "gkr" => gkr_key_tile::cmd(&args),
        other => {
            eprintln!("unknown subcommand {other:?}; \
                       expected tables|dequant|width|kmax|probes|pack|attn-probes|exprprobe|cudacheck|cudaemit|logupcheck|genemit|\
                       argmax|argmax-probes|embed|embed-probes|\
                       compose|pre-split|expect-root|model-roots|compose-probes|boundary-chain|boundary-grid|psweep|requant|requant-probes|residual-stream|residual-stream-probes|selftest");
            std::process::exit(2);
        }
    }
}

// ------------------------------------------------------------------- tests ----

#[cfg(test)]
mod tests {
    use super::*;

    /// The sweep's own shape, and the flags `src_of` reads off a `pack` line.
    fn packed(extra: &[&str], insts: usize) -> Attn {
        let argv: Vec<String> = ["pack", "--s", "512", "--ctx", "1024", "--hd", "64",
                                 "--variants", "intexp"]
            .iter().chain(extra.iter()).map(|s| s.to_string()).collect();
        packed_key_tile(&src_of(&argv), Shape { s: 512, c: 1024, hd: 64, insts },
                        attention::witness::Variant::IntExp)
    }

    /// Issue #89: `--table-bits` bounds the packed key tile's exp table.
    ///
    /// One master exponent serves the whole key tile, so the wider the key tile
    /// the further the lower instances' shifts are scaled up to reach it.  At 8
    /// instances of the sweep's shape that puts the largest key one bit past the
    /// 2^22 rows the sizing is capped at, and the keys past it were left off the
    /// table, where only a rejected proof reported them.  At 1 instance the keys
    /// sit exactly at the cap, and the flag must not move that.
    #[test]
    fn table_bits_sizes_the_packed_key_tile() {
        let one = packed(&[], 1);
        assert_eq!(one.log_table, 22, "the table at 1 instance is no longer 2^22 rows");
        assert_eq!(one.off_table, 0, "{} keys off the table at 1 instance", one.off_table);
        assert_eq!(packed(&["--table-bits", "23"], 1).log_table, one.log_table,
                   "--table-bits moved the table at 1 instance");

        let eight = packed(&["--table-bits", "23"], 8);
        assert_eq!(eight.log_table, 23, "the table at 8 instances was not grown");
        assert!(eight.key_bits() <= eight.log_table,
                "keys of {} bits against a table of 2^{} rows",
                eight.key_bits(), eight.log_table);
        assert_eq!(eight.off_table, 0, "{} keys off the table at 8 instances",
                   eight.off_table);
    }

    /// The same shape with no bound on the line: the keys do not fit, and the
    /// refusal comes before the first commitment and names the number to pass.
    #[test]
    #[should_panic(expected = "pass --table-bits 23")]
    fn the_packed_key_tile_refuses_keys_past_the_bound() {
        packed(&[], 8);
    }
}

