//! The attention-output component's constraints, and the preprocessed columns
//! they read.

use num_traits::{One, Zero};
use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::SecureField;
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;
use stwo_constraint_framework::{EvalAtRow, FrameworkEval, INTERACTION_TRACE_IDX,
                                ORIGINAL_TRACE_IDX, RelationEntry};

use crate::relations::{FracRel, PairRel, RangeRel, id};

pub const AO_SEL: [&str; 6] = ["nofirst", "last", "m1", "m2", "m3", "end"];
pub fn ao_sel(name: &str) -> PreProcessedColumnId {
    id(format!("zk_ao_{name}"))
}
pub fn ao_chan(kind: &str, slot: usize) -> PreProcessedColumnId {
    id(format!("zk_ao_{kind}{slot:04}"))
}
/// Stream JJ (E11): the exponent-window table's three preprocessed columns,
/// `(e - e_min, 2^max(e,0), 2^max(-e,0))` row for row over the admissible
/// `e = jv - jo`.  One `FracRel` lookup per attnout slot lands on one of its
/// rows at the key `jvp + joff`, which is what pins the committed `(num, pj)`
/// to a real exponent -- and so bounds `pj`, and with it `denp`, inside the
/// AIR rather than in tree 0.
pub const EPOW_COLS: [&str; 3] = ["zk_epow_e", "zk_epow_n", "zk_epow_p"];

#[derive(Clone)]
pub struct AttnOutEval {
    pub log_rows: u32,
    /// channel slots per row
    pub per_row: usize,
    /// rv^0..rv^{per_row-1} and rv^per_row
    pub rv_pow: Vec<SecureField>,
    pub rv_step: SecureField,
    pub s: SecureField,
    /// rv^hd * s^-(S-1): the head-boundary step of `h`
    pub mu_out: SecureField,
    /// r^C * s^-(S-1): the head-boundary step of `dz`
    pub mu_den: SecureField,
    pub claimed_out: SecureField,
    pub claimed_den: SecureField,
    /// Stream JJ (E11), for stream II: the component's *own output* -- the signed
    /// requantized attention output `q = qm (1 - 2 sg)`, folded under exactly
    /// the weights `claimed_out` uses.  `AttnOutEval` is the requantization of
    /// the attention output, so this is what the `o` Freivalds check's `qX` has to equal
    /// and there is no requant component between them; the `eqs` entry that says
    /// so is II's.
    pub claimed_q: SecureField,
    pub qnz: PairRel,
    pub rel_rem: RangeRel,
    /// Stream JJ (E11): the exponent window, as a table
    /// `(e - e_min, 2^max(e,0), 2^max(-e,0))`.  One lookup per slot at the key
    /// `jvp + joff` pins the committed `(num, pj)` and bounds both, which is
    /// what takes `jo` out of tree 0 and gives `pj` an in-AIR bound.
    pub epow: FracRel,
    /// the two limbs of the range check on `acc + acc_off`, the other half of
    /// the no-wrap window
    pub rel_acc_lo: RangeRel,
    pub rel_acc_hi: RangeRel,
    pub acc_lo_bits: u32,
    pub acc_hi_bits: u32,
    /// Stream TT (E11), log row 73: the magnitude form.  The slot commits
    /// `amag` as a tenth column, pinned to `|acc|` by `(amag - acc)(amag + acc)
    /// = 0` together with its own range check, and the two limbs read *it*
    /// rather than `acc + 2^(lo+hi-1)`.  The window then needs `acc_bits()`
    /// bits and not `acc_bits() + 1`, because the sign is the pin's and not the
    /// offset's -- which is the bit `y-delay-vw12`'s layer 0 is short.
    pub acc_mag: bool,
    /// The product form (`compose::Params::ao_acc_prod`), on top of the
    /// magnitude form: the slot commits an eleventh column `ahn`, pinned to
    /// `ahi num` by a degree-2 constraint and range-checked in this table of
    /// `30 - acc_lo_bits` bits.  `None` in the other two forms.
    pub rel_acc_prod: Option<RangeRel>,
    /// Stream TT (E11), log row 73: the sibling bound's half.  `denp`'s own
    /// range check is on `2 denp` instead, which states `denp <= 2^(rem-1) - 1`
    /// -- the width `AttnOut::rem_bits` was derived from -- and buys the bit
    /// `Plan::new`'s quotient assert is short at `attn_out_bits` 12.
    pub denp2: bool,
    /// B11 stream V, lever 4: as `FreivaldsEval::gkr`.
    pub gkr: bool,
}

impl FrameworkEval for AttnOutEval {
    /// milestone 2b: the flat parameter block for the generated
    /// constraint kernel (`gpu/m2b/cudagen.rs`).
    fn cuda_eval_blob(&self) -> Option<Vec<u8>> {
        crate::cudagen::blob_for(self)
    }

    /// milestone 2a: no kernel is registered under this name, so the
    /// fork routes this component to `accumulate_pointwise_cpu`.
    fn cuda_eval_name(&self) -> &'static str { "zk_attn_out_eval" }

    fn log_size(&self) -> u32 {
        self.log_rows
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_rows + 1
    }
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let two = M31::from_u32_unchecked(2);
        // Stream JJ (E11): the per-slot `jvp` comes first, because a lowered
        // evaluator numbers its preprocessed loads in `get_preprocessed_column`
        // call order (`cudagen::Ir::pre_ids`) and the GKR backend's `fam_site`
        // indexes the component's own `pre` vector with that number.  `ao_component`
        // builds `pre` as the slots' `jvp` then `AO_SEL`, so the reads have to
        // run in that order or the backend's fractions read a selector for a `jv`.
        let jvps: Vec<E::F> = (0..self.per_row)
            .map(|slot| eval.get_preprocessed_column(ao_chan("jv", slot)))
            .collect();
        let sel: Vec<E::F> =
            AO_SEL.iter().map(|n| eval.get_preprocessed_column(ao_sel(*n))).collect();
        let (nofirst, last, m1, m2, m3, end) = (sel[0].clone(), sel[1].clone(), sel[2].clone(),
                                                sel[3].clone(), sel[4].clone(), sel[5].clone());
        // stream JJ's fourth claim reads the same six selectors
        let (nofirst2, last2) = (nofirst.clone(), last.clone());
        let (m1q, m2q, m3q) = (m1.clone(), m2.clone(), m3.clone());
        // `den` is the token's denominator Z, one committed cell per row, and
        // before this it entered exactly two places: `denp = den pj` per slot,
        // and the `dz` fold -- which takes it only on the token's *last* row
        // (`den * last`).  On every other row of a token nothing constrained it,
        // so `denp` was free (pick `den = denp / pj`) and the slot's rounding
        // was a rounding by a denominator of the prover's choosing.  Reading
        // `den` at [-1, 0] and pinning it across the token's rows -- `nofirst`
        // is exactly "not the first row of this token's group", the selector the
        // `z` fold already chains on -- makes every row's `den` the row the `dz`
        // fold publishes, and so the key tile's `az` claim.
        let [den_prev, den] = eval.next_interaction_mask(ORIGINAL_TRACE_IDX, [-1, 0]);
        eval.add_constraint((den.clone() - den_prev) * nofirst.clone());
        // Stream JJ (E11), `witness-slack.md` section 1f: the per-token output
        // exponent leaves tree 0.  `jo` is prover data, so what tree 0 may hold
        // is `jvp = jv - jv_min`, the per-(head, channel) model constant, and
        // `joff = jo_max - jo` is committed -- one cell per row, pinned across
        // the token's rows by the same `nofirst` the `den` pin uses.  The pair
        // `(num, pj) = (2^max(e,0), 2^max(-e,0))` for `e = jv - jo` is committed
        // too and pinned by one lookup per slot into the preprocessed `EPOW`
        // window at the key `jvp + joff`, which is `e - e_min`.  Tree 0 is then
        // a function of the model, the shape and the params alone.
        // The offsets are `[0, -1]` and not `den`'s `[-1, 0]`, and that order is
        // load-bearing: `joff` is the first column of this component a *lookup*
        // reads at a mask, and `gkrlink::LinkTie` resolves a referenced column
        // as `mask[tree][col][0]` -- the value at the column's *first* sampled
        // offset.  With `[-1, 0]` the GKR backend's binding would read the previous
        // row's `joff`, which agrees with this row's on every witness whose
        // `jo` is constant and on no other; `--gkr-attnout` then accepts or
        // rejects by luck of the data.  Any future lookup over a masked column
        // in this family has to put offset 0 first for the same reason.
        let [joff, joff_prev] = eval.next_interaction_mask(ORIGINAL_TRACE_IDX, [0, -1]);
        eval.add_constraint((joff.clone() - joff_prev) * nofirst.clone());
        // stream TT: zero in the magnitude form -- the limbs read the committed
        // `amag`, so nothing is offset
        let acc_off = E::F::from(M31::from_u32_unchecked(
            if self.acc_mag { 0 } else { 1 << (self.acc_lo_bits + self.acc_hi_bits - 1) }));
        let inv_lo = M31::from_u32_unchecked(1 << self.acc_lo_bits).inverse();
        let mut accs: Vec<E::F> = Vec::with_capacity(self.per_row);
        let mut qouts: Vec<E::F> = Vec::with_capacity(self.per_row);
        for slot in 0..self.per_row {
            let jvp = jvps[slot].clone();
            let acc = eval.next_trace_mask();
            let sg = eval.next_trace_mask();
            let qm = eval.next_trace_mask();
            let nz2 = eval.next_trace_mask();
            let denp = eval.next_trace_mask();
            let d = eval.next_trace_mask();
            let num = eval.next_trace_mask();
            let pj = eval.next_trace_mask();
            let alo = eval.next_trace_mask();
            // stream TT (E11), log row 73: the magnitude column, last of the
            // slot so every column before it keeps its index.  Two things pin
            // it: this product, which says it is `acc` or `-acc` and nothing
            // else, and the two limbs below, which say it is in
            // `[0, 2^(lo+hi))`.  Together they say `amag = |acc|` and
            // `|acc| <= 2^(lo+hi) - 1`, which is the `mag` half of
            // `2 mag num < p` at one bit less window than the offset form --
            // the offset spends a bit on the sign and this reads it off the
            // product.  Degree 2, and the limb arguments stay degree 1.
            let amag = self.acc_mag.then(|| eval.next_trace_mask());
            if let Some(m) = &amag {
                eval.add_constraint((m.clone() - acc.clone()) * (m.clone() + acc.clone()));
            }
            // The product form: `ahn = ahi num`, eleventh and last of the slot.
            // `ahi` is the high limb the range check below reads, degree 1, and
            // `num` is the `EPOW` lookup's `2^max(e,0)`, so this is degree 2.
            // With `ahi < 2^hi` and `num <= 2^e_hi` it cannot wrap
            // (`hi + e_hi <= 30`, `Plan::new`), so it holds over the integers,
            // and `ahn` below `2^(30 - lo)` then puts `mag num` below `2^30`.
            let ahn = self.rel_acc_prod.as_ref().map(|_| eval.next_trace_mask());
            if let (Some(m), Some(hn)) = (&amag, &ahn) {
                let ahi = (m.clone() - alo.clone()) * inv_lo;
                eval.add_constraint(hn.clone() - ahi * num.clone());
            }
            accs.push(acc.clone());
            eval.add_constraint(sg.clone() * sg.clone() - sg.clone());
            // the sign pin (B6 review, gap 7), as in the standalone gadget
            eval.add_constraint(sg.clone() * (E::F::one() - nz2.clone()));
            // denp = den pj = den 2^max(-e, 0), committed so the error identity
            // and the binding-excluding bound stay degree <= 3 / degree 1
            eval.add_constraint(denp.clone() - den.clone() * pj.clone());
            let mag = acc.clone() - (acc.clone() * sg.clone()) * two;
            eval.add_constraint(
                d.clone() - (mag.clone() * num.clone()) * two
                    + (qm.clone() * two - E::F::one()) * denp.clone(),
            );
            // stream JJ: the component's output value, degree 2, never a column
            qouts.push(qm.clone() - (qm.clone() * sg.clone()) * two);
            let dp = denp.clone() * two - E::F::one() - d.clone();
            if !self.gkr {
                // Stream JJ (E11), `witness-slack.md` section 1f: the exponent
                // window.  `num` and `pj` are committed -- `jo` is prover data,
                // so `e = jv - jo` cannot be preprocessed -- and this lookup is
                // what says the pair is `(2^max(e,0), 2^max(-e,0))` for the `e`
                // the row's own `joff` and the slot's model-only `jvp` name.  It
                // bounds `pj`, which is what section 1(c) had no answer for, and
                // it is why tree 0 no longer depends on this chunk's tokens.
                eval.add_to_relation(RelationEntry::new(
                    &self.epow, E::EF::one(),
                    &[jvp + joff.clone(), num.clone(), pj]));
                eval.add_to_relation(RelationEntry::new(&self.qnz, E::EF::one(),
                                                        &[qm, nz2]));
                eval.add_to_relation(RelationEntry::new(
                    &self.rel_rem, E::EF::one(), core::slice::from_ref(&d)));
                eval.add_to_relation(RelationEntry::new(
                    &self.rel_rem, E::EF::one(), core::slice::from_ref(&dp)));
                // witness-slack section 7 item 1.  `d` and `dp` alone do NOT
                // bound `denp`: they give `2 denp = d + dp + 1` over the field,
                // and halving is a bijection mod an odd prime, so the branch
                // where `d + dp + 1` is odd puts `denp` at `(d + dp + 1 + p)/2`
                // -- about 2^30, where `0 <= d <= 2 denp - 1` is not an integer
                // reading and `d = 2 denp0` (the exact binding's second witness the
                // `-1` exists to exclude) is back.  Range-checking `denp` in the
                // same `ao_rem_bits` table costs one lookup per slot, is
                // satisfied by every honest row -- `rem_bits` is derived as the
                // bit length of `max 2 denp` (`AttnOut::rem_bits`), so honest
                // `denp < 2^(rem_bits-1)` -- and makes the error identity an
                // integer identity.  Before it, the only guard was the host-side
                // assert at `layer.rs`'s `ao_component`.
                let denp_look = if self.denp2 { denp.clone() * two } else { denp.clone() };
                eval.add_to_relation(RelationEntry::new(
                    &self.rel_rem, E::EF::one(), core::slice::from_ref(&denp_look)));
                // Stream JJ (E11): the other half of the no-wrap window,
                // which until now was a host-side assert alone.  `2 mag num`
                // needs both factors bounded; the lookup above bounds `num` by
                // `2^max(ao_e_max,0)`, and this bounds `mag`.  `|acc|` is up to
                // 29 bits at 8B, far past any range table, so it goes in two
                // limbs on the *offset* accumulator, which keeps every argument
                // degree 1: `acc + acc_off = alo + 2^acc_lo_bits ahi` with `alo`
                // in the `ao_acc_lo_bits` table and `ahi` in the
                // `ao_acc_hi_bits` one, so `|acc| < 2^(lo + hi - 1)`.
                // `Plan::new` asserts
                // `1 + ao_acc_lo_bits + ao_acc_hi_bits + max(ao_e_max, 0) <= 31`.
                //
                // Stream TT (E11): in the magnitude form the value decomposed
                // is the committed `amag`, not the offset accumulator, and the
                // window is `[0, 2^(lo+hi))` rather than `[-2^(lo+hi-1),
                // 2^(lo+hi-1))`.
                let av = match &amag {
                    Some(m) => m.clone(),
                    None => acc.clone() + acc_off.clone(),
                };
                let ahi = (av - alo.clone()) * inv_lo;
                eval.add_to_relation(RelationEntry::new(
                    &self.rel_acc_lo, E::EF::one(), core::slice::from_ref(&alo)));
                eval.add_to_relation(RelationEntry::new(
                    &self.rel_acc_hi, E::EF::one(), core::slice::from_ref(&ahi)));
                // the product form's per-slot bound, `ahn < 2^(30 - lo)`
                if let (Some(rel), Some(hn)) = (&self.rel_acc_prod, &ahn) {
                    eval.add_to_relation(RelationEntry::new(
                        rel, E::EF::one(), core::slice::from_ref(hn)));
                }
            }
        }

        let [z_prev, z] = eval.next_extension_interaction_mask(INTERACTION_TRACE_IDX, [-1, 0]);
        let [h_prev, h] = eval.next_extension_interaction_mask(INTERACTION_TRACE_IDX, [-1, 0]);
        let [dz_prev, dz] = eval.next_extension_interaction_mask(INTERACTION_TRACE_IDX, [-1, 0]);
        let mut zf = E::EF::zero();
        for (i, a) in accs.iter().enumerate() {
            zf += a.clone() * self.rv_pow[i];
        }
        eval.add_constraint(z.clone() - (z_prev * self.rv_step) * nofirst - zf);
        let mut mult: E::EF = E::EF::from(m1.clone());
        mult += m2.clone() * self.s;
        mult += m3.clone() * self.mu_out;
        eval.add_constraint(h.clone() - h_prev * mult - z * last.clone());
        let mut dmult: E::EF = E::EF::from(m1);
        dmult += m2 * self.s;
        dmult += m3 * self.mu_den;
        eval.add_constraint(dz.clone() - dz_prev * dmult - E::EF::from(den * last));
        let mut multq: E::EF = E::EF::from(m1q);
        multq += m2q * self.s;
        multq += m3q * self.mu_out;
        let claim_out = eval.claim(self.claimed_out);
        eval.add_constraint((h - claim_out) * end.clone());
        let claim_den = eval.claim(self.claimed_den);
        eval.add_constraint((dz - claim_den) * end.clone());
        // stream JJ: the fourth claim, `z`/`h`'s shape over the component's own
        // output instead of over `acc`.  Same slot weights, same row step, same
        // token and head boundaries, so a consumer that folds the attention
        // output under `claimed_out`'s convention compares against this one.
        let [zq_prev, zq] = eval.next_extension_interaction_mask(INTERACTION_TRACE_IDX, [-1, 0]);
        let [hq_prev, hq] = eval.next_extension_interaction_mask(INTERACTION_TRACE_IDX, [-1, 0]);
        let mut qf = E::EF::zero();
        for (i, q) in qouts.iter().enumerate() {
            qf += q.clone() * self.rv_pow[i];
        }
        eval.add_constraint(zq.clone() - (zq_prev * self.rv_step) * nofirst2 - qf);
        eval.add_constraint(hq.clone() - hq_prev * multq - zq * last2);
        let claim_q = eval.claim(self.claimed_q);
        eval.add_constraint((hq - claim_q) * end);
        if !self.gkr {
            eval.finalize_logup_in_pairs();
        }
        eval
    }
}
