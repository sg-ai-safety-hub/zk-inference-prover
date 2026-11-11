//! B11 stream S, lever 3: the link that makes the GKR backend sound.
//!
//! `gkr_key_tile` proves the key tiles' fractional sums, but its `GkrArtifact` ends in
//! *claimed evaluations of the input layers at a random point* -- nothing bindings
//! those to the committed base columns. This module closes that gap along the
//! reduction of `experiments/b11/gkr/memo.md` section 4: with the input layer
//! laid out `(look, row)` and every look's numerator and denominator *affine*
//! in committed columns, an instance's input-layer claim collapses to
//!
//! ```text
//!     claim + (verifier scalar)  =  sum_c w_c * col_c~(x_row)
//! ```
//!
//! -- one affine combination of committed-column MLEs at the row part of the
//! OOD point, with weights `w_c` the verifier computes from `ood_point` and
//! the relation elements alone. Instances of equal height share `x_row`
//! (staggered instances take the *last* coordinates of the batch OOD point),
//! so they fold into **one univariate `mle_eval` IOP per distinct height**
//! under one post-GKR challenge.
//!
//! The univariate IOP itself (eq-column + shifted-prefix-sum constraints, the
//! succinct eq carry quotients) is promoted from the fork's examples crate,
//! `nitroo/crates/examples/src/xor/gkr_lookups/mle_eval.rs`, function for
//! function. Two deviations from the upstream component, both forced by the
//! composed prover running its constraint evaluation through generated CUDA
//! kernels rather than upstream's SIMD-only `ComponentProver`:
//!
//! * the carry-quotient column and the MLE coefficient column are *committed*
//!   (16 M31 columns per height instead of upstream's 8), so the whole
//!   constraint set is an ordinary `FrameworkEval` (`LinkEval`) that rides the
//!   existing kernel machinery;
//! * what upstream's `MleCoeffColumnOracle` did at the OODS point -- read the
//!   coefficient polynomial off the committed masks -- becomes two explicit
//!   *binding* constraints per height in a bespoke component (`LinkTie`): the
//!   committed coefficient column equals the `w_c` combination of the base
//!   columns' own OODS values, and the committed carry-quotient column equals
//!   its succinct analytic evaluation. Both are identically zero for an
//!   honest prover, so `LinkTie`'s prover side contributes zero quotients and
//!   only consumes its accumulator coefficients.

use std::collections::BTreeMap;

use num_traits::{One, Zero};
use rayon::prelude::*;
use stwo::core::circle::{CirclePoint, Coset};
use stwo::core::constraints::{coset_vanishing, point_vanishing};
use stwo::core::fields::{Field, FieldExpOps};
use stwo::core::fields::m31::{BaseField, M31};
use stwo::core::fields::qm31::SecureField;
use stwo::core::pcs::TreeVec;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::utils::{bit_reverse_index, coset_index_to_circle_domain_index};
use stwo::core::ColumnVec;
use stwo::core::air::Component;
use stwo::core::air::accumulation::PointEvaluationAccumulator;
use stwo::prover::{DomainEvaluationAccumulator, Trace};
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::simd::column::SecureColumn;
use stwo::prover::backend::simd::m31::LOG_N_LANES;
use stwo::prover::backend::simd::prefix_sum::inclusive_prefix_sum;
use stwo::prover::backend::simd::qm31::PackedSecureField;
use stwo::prover::backend::{Col, Column};
use stwo::prover::lookups::gkr_prover::GkrOps;
use stwo::prover::lookups::gkr_verifier::GkrArtifact;
use stwo::prover::lookups::utils::eq;
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::secure_column::SecureColumnByCoords;
use stwo::prover::ComponentProver;
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;
use stwo_constraint_framework::{EvalAtRow, FrameworkEval, INTERACTION_TRACE_IDX};

use crate::gadgets::attention::witness::Variant as AttnVariant;
use crate::common::Eval;
use crate::gkr_key_tile::Inst;

// ------------------------------------------------------------------------
// The univariate IOP, promoted from the fork's examples crate
// (`xor/gkr_lookups/mle_eval.rs`). The math is unchanged; only visibility
// and the generic bounds needed by this crate's evaluators differ.
// ------------------------------------------------------------------------

/// Metadata of one MLE evaluation point (upstream `MleEvalPoint`).
#[derive(Debug, Clone)]
pub struct MleEvalPoint {
    /// `eq({0}^n, p)`
    eq_0_p: SecureField,
    /// `eq({1}^n, p)`
    eq_1_p: SecureField,
    /// index `i` stores `eq(({1}^i, 0), p[0..i+1]) / eq(({0}^i, 1), p[0..i+1])`
    eq_carry_quotients: Vec<SecureField>,
    p: Vec<SecureField>,
}

impl MleEvalPoint {
    /// Panics if the point is empty or has a coordinate that is zero or one
    /// (upstream's completeness caveat; the GKR's OOD point is drawn from the
    /// channel, so hitting one is negligible and loud).
    pub fn new(p: &[SecureField]) -> Self {
        assert!(!p.is_empty());
        let n_variables = p.len();
        let zero = SecureField::zero();
        let one = SecureField::one();
        Self {
            eq_0_p: eq(&vec![zero; n_variables], p),
            eq_1_p: eq(&vec![one; n_variables], p),
            eq_carry_quotients: (0..n_variables)
                .map(|i| {
                    let mut num = vec![one; i + 1];
                    num[i] = zero;
                    let mut den = vec![zero; i + 1];
                    den[i] = one;
                    eq(&num, &p[..i + 1]) / eq(&den, &p[..i + 1])
                })
                .collect(),
            p: p.to_vec(),
        }
    }

    pub fn n_variables(&self) -> usize {
        self.p.len()
    }
}

/// Evaluates the EqEvals-column constraints; returns the eq column at offset 0
/// (upstream `eval_eq_constraints`).
fn eval_eq_constraints<E: EvalAtRow>(
    eq_interaction: usize,
    eval: &mut E,
    mle_eval_point: &MleEvalPoint,
    carry_quotients_col_eval: E::EF,
    is_first: E::F,
    is_second: E::F,
) -> E::EF {
    let [curr, next_next] = eval.next_extension_interaction_mask(eq_interaction, [0, 2]);

    // initial value on half_coset0 and final value on half_coset1; `is_first`
    // and `is_second` are never nonzero at the same row
    let half_coset0_initial_check = (curr.clone() - mle_eval_point.eq_0_p) * is_first;
    let half_coset1_final_check = (curr.clone() - mle_eval_point.eq_1_p) * is_second;
    eval.add_constraint(half_coset0_initial_check + half_coset1_final_check);

    // all the steps
    eval.add_constraint(curr.clone() - next_next * carry_quotients_col_eval);

    curr
}

/// Inclusive-prefix-sum constraints over a column whose values are shifted by
/// `cumulative_sum_shift = cumulative_sum / column_size` (upstream
/// `eval_prefix_sum_constraints`).
fn eval_prefix_sum_constraints<E: EvalAtRow>(
    interaction: usize,
    eval: &mut E,
    row_diff: E::EF,
    cumulative_sum_shift: E::EF,
) {
    let [curr, prev] = eval.next_extension_interaction_mask(interaction, [0, -1]);
    eval.add_constraint(curr - prev - row_diff + cumulative_sum_shift);
}

/// `(half_coset0_carry_quotients, half_coset1_carry_quotients)` (upstream).
fn gen_half_coset_carry_quotients(
    eval_point: &MleEvalPoint,
) -> (Vec<SecureField>, Vec<SecureField>) {
    let last_variable = *eval_point.p.last().unwrap();
    let one = SecureField::one();
    let zero = SecureField::zero();
    let mut half_coset0_carry_quotients = eval_point.eq_carry_quotients.clone();
    *half_coset0_carry_quotients.last_mut().unwrap() *=
        eq(&[one], &[last_variable]) / eq(&[zero], &[last_variable]);
    let half_coset1_carry_quotients =
        half_coset0_carry_quotients.iter().map(|v| v.inverse()).collect();
    (half_coset0_carry_quotients, half_coset1_carry_quotients)
}

/// The succinct eq carry-quotients column as committed values (upstream
/// `gen_carry_quotient_col`).
fn gen_carry_quotient_col(eval_point: &MleEvalPoint) -> SecureColumnByCoords<SimdBackend> {
    let (half0, half1) = gen_half_coset_carry_quotients(eval_point);

    let log_size = eval_point.n_variables() as u32;
    let size = 1usize << log_size;
    let half_coset_size = size / 2;

    // B11 stream T: upstream scatters straight into the packed
    // `SecureColumnByCoords`, and `set` on that is four strided lane writes
    // behind a divide -- 80 ns a row, which at the wall point's 3.5 M link rows
    // was 0.142 s, 40% of the whole link build. The scatter goes into a flat
    // vector instead and the packing is one linear pass afterwards. Every index
    // is still written exactly once with the value it was written with before
    // (`i * 2` and `i * 2 + 1` over the half coset cover the domain once), so
    // this is the same column; the digest is what says so.
    let mut vals = vec![SecureField::zero(); size];
    for i in 0..half_coset_size {
        let half_coset0_index = coset_index_to_circle_domain_index(i * 2, log_size);
        let half_coset1_index = coset_index_to_circle_domain_index(i * 2 + 1, log_size);
        let half_coset0_index_bit_rev = bit_reverse_index(half_coset0_index, log_size);
        let half_coset1_index_bit_rev = bit_reverse_index(half_coset1_index, log_size);

        let n_trailing_ones = i.trailing_ones() as usize;
        vals[half_coset0_index_bit_rev] = half0[n_trailing_ones];
        vals[half_coset1_index_bit_rev] = half1[n_trailing_ones];
    }
    vals.into_iter().collect()
}

/// Evaluates the succinct carry-quotients column at `p` (upstream
/// `eval_carry_quotient_col`).
pub fn eval_carry_quotient_col(eval_point: &MleEvalPoint, p: CirclePoint<SecureField>)
                               -> SecureField {
    let n_variables = eval_point.n_variables();
    let log_size = n_variables as u32;
    let coset = CanonicCoset::new(log_size).coset();

    let (half0, half1) = gen_half_coset_carry_quotients(eval_point);

    let mut out = SecureField::zero();
    for variable_i in 0..n_variables.saturating_sub(1) {
        let log_step = variable_i as u32 + 2;
        let offset = (1 << (log_step - 1)) - 2;
        let half_coset0_selector = eval_step_selector_with_offset(coset, offset, log_step, p);
        let half_coset1_selector = eval_step_selector_with_offset(coset, offset + 1, log_step, p);
        out += half_coset0_selector * half0[variable_i];
        out += half_coset1_selector * half1[variable_i];
    }
    let half_coset0_last = eval_is_first(coset, p + coset.step.double().into_ef());
    let half_coset1_first = eval_is_first(coset, p + coset.step.into_ef());
    out += *half0.last().unwrap() * half_coset0_last;
    out += *half1.last().unwrap() * half_coset1_first;
    out
}

/// A polynomial that is `1` every `2^log_step` coset points shifted by
/// `offset` and `0` elsewhere on the coset (upstream).
fn eval_step_selector_with_offset(
    coset: Coset,
    offset: usize,
    log_step: u32,
    p: CirclePoint<SecureField>,
) -> SecureField {
    let offset_step = coset.step.mul(offset as u128);
    eval_step_selector(coset, log_step, p - offset_step.into_ef())
}

/// A polynomial that is `1` every `2^log_step` coset points and `0` elsewhere
/// on the coset (upstream).
fn eval_step_selector(coset: Coset, log_step: u32, p: CirclePoint<SecureField>) -> SecureField {
    if log_step == 0 {
        return SecureField::one();
    }
    // rotate the coset so its first point is the identity element
    let p = p - coset.initial.into_ef();
    let mut vanish_at_log_step: Vec<SecureField> = (0..coset.log_size)
        .scan(p, |p, _| {
            let res = *p;
            *p = p.double();
            Some(res.y)
        })
        .collect();
    vanish_at_log_step.reverse();
    // only the first `log_step` many values are needed
    vanish_at_log_step.truncate(log_step as usize);
    let vanish_at_log_step_inv = SecureField::batch_inverse(&vanish_at_log_step);

    let half_coset_selector_dbl = (vanish_at_log_step[0] * vanish_at_log_step_inv[1]).square();
    let vanish_substep_inv_sum = vanish_at_log_step_inv[1..].iter().sum::<SecureField>();
    (half_coset_selector_dbl + vanish_at_log_step[0] * vanish_substep_inv_sum.double())
        / BaseField::from(1 << (log_step + 1))
}

/// `1` at the first coset point, `0` elsewhere, as a polynomial evaluation
/// (upstream `eval_is_first`).
pub fn eval_is_first(coset: Coset, p: CirclePoint<SecureField>) -> SecureField {
    coset_vanishing(coset, p)
        / (point_vanishing(coset.initial, p) * BaseField::from(1 << coset.log_size))
}

// ------------------------------------------------------------------------
// The affine looks: every key tile lookup's numerator and denominator as an
// affine form over committed columns, in mask-relative coordinates.
// ------------------------------------------------------------------------

/// One committed (or preprocessed) column, named the way the AIR reads it:
/// by its mask index within the component's component. The committed order and
/// the mask order are the same order -- that is what lets the prover resolve
/// these against `com.base[at[which] + i]` and the verifier against the
/// component's trace location span, and the two cannot disagree without the
/// binding constraint rejecting.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum RCol {
    /// the channel component's `i`-th trace mask (`ks` then `qs`)
    Chan(usize),
    /// the grid component's `i`-th trace mask (`idx`, `num`, `aux`, `vals`)
    Key(usize),
    /// the per-token component's `i`-th trace mask (`ms`, `zs`, `los`, `los2`)
    Row(usize),
    /// the diagonal block's `i`-th causal-mask column, preprocessed tree
    Msk(usize),
    /// B11 stream U: a dequant component's `i`-th trace mask. The index is
    /// `dequant::Map`'s column index, which *is* the mask index -- the map
    /// exists so that "the AIR, the trace generator and the LogUp generator
    /// agree by construction instead of by inspection", and this is a fourth
    /// reader of the same agreement.
    Deq(usize),
    /// B11 stream U: a dequant component's weight-scale mantissa column for slot
    /// `i` (`chan_col(tag, "m", i)`), preprocessed tree. The only preprocessed
    /// column any dequant lookup reads: the `chain` variant's second scale is
    /// offline, so its mantissa enters the `mul` lookup from the preprocessed
    /// side while everything else in the lookup is committed.
    DeqPre(usize),
    /// B11 stream V: a residual-stream component's `i`-th trace mask. The index is the
    /// component's own committed column index, which *is* the mask index: the
    /// evaluator reads its masks in `Prog` order and the witness generation writes the
    /// columns in the same order, which is what lets `residual_stream_interaction`'s
    /// builder index `r.base` by what `Prog` returns.
    ResidualStream(usize),
    /// Stream OO (E11): a residual-stream component's preprocessed value `i`, half `k` --
    /// the `(sm, ef)` pair of a `Src::Pre` operand. Stream II's norm scales read
    /// the model's norm weight, which is the statement's and so lives in tree 0
    /// under `pre_root`; the pair sits at `Prog::pre_at(i, k)` in the component's
    /// own `pre` list and is named by `residual_stream::residual_stream_pre_col(tag, i, k)`. The only
    /// preprocessed column a residual-stream lookup reads.
    ResidualStreamPre(usize, usize),
    /// B11 stream V: a Freivalds group's `i`-th trace mask. The activations are the
    /// component's first `pq` committed columns, which is the slice
    /// `freivalds_interaction`'s builder range-checks.
    Freivalds(usize),
    /// B11 stream V: the attention-output component's `i`-th trace mask.
    Ao(usize),
    /// Stream JJ (E11): the attention-output component's slot-`i` preprocessed
    /// column, `jvp = jv - jv_min`.  The only preprocessed column an attnout
    /// lookup reads: `jv` is a model constant, so the exponent window's key is
    /// half preprocessed and half committed.
    AoPre(usize),
}

/// Which component a site's looks and a folded weight's column belong to: the
/// family, and the component's index within it.
///
/// B11 stream U, lever 3: stream S keyed the fold's weights by key tile index
/// alone, because the key tiles were the only family with a backend. The dequant
/// family's components index into the plan's matmul list, and the two index spaces
/// are unrelated -- so the key is the pair, and `link_params`' `BTreeMap` is
/// keyed by it. Nothing else about the fold changes: a weight is still one
/// `(component, column)` pair's coefficient in one height's affine combination.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Comp {
    /// key tile `b`, in `Plan::diagonal_block`'s order (the prefix key tiles, then the diagonal blocks)
    KeyTile(usize),
    /// dequant component `m`, in the plan's matmul order
    Deq(usize),
    /// B11 stream V: residual-stream component `g`, in the plan's component order
    ResidualStream(usize),
    /// B11 stream V: Freivalds group `w`, in the plan's Freivalds order
    Freivalds(usize),
    /// B11 stream V: the attention-output component, of which there is one
    Ao,
}

impl Default for Comp {
    /// the collector's field needs one; key tile 0 is what stream S's single-family
    /// collector always meant
    fn default() -> Self {
        Comp::KeyTile(0)
    }
}

impl Comp {
    pub fn tag(&self) -> String {
        match self {
            Comp::KeyTile(b) => format!("key_tile{b}"),
            Comp::Deq(m) => format!("deq{m}"),
            Comp::ResidualStream(g) => format!("residual_stream{g}"),
            Comp::Freivalds(w) => format!("freivalds{w}"),
            Comp::Ao => "attnout".to_string(),
        }
    }
}

/// An affine form `sum coeff * col + k` over one key tile's columns.
#[derive(Clone, Debug, Default)]
pub struct Affine {
    pub terms: Vec<(RCol, SecureField)>,
    pub k: SecureField,
}

impl Affine {
    fn col(c: RCol, w: SecureField) -> Self {
        Affine { terms: vec![(c, w)], k: SecureField::zero() }
    }
}

/// One lookup: its numerator (`None` = the constant one) and denominator.
#[derive(Clone, Debug)]
pub struct LinkLook {
    pub num: Option<Affine>,
    pub den: Affine,
}

/// One site's looks, in the exact order the interaction pass pushes them.
pub struct SiteLooks {
    pub reg: Comp,
    pub which: usize,
    pub log_rows: u32,
    pub looks: Vec<LinkLook>,
}

fn range_den(rel: &crate::relations::RangeRel, slot: Affine) -> Affine {
    let (z, al) = rel.parts();
    let mut den = Affine {
        terms: slot.terms.iter().map(|(c, w)| (*c, *w * al[0])).collect(),
        k: slot.k * al[0] - z,
    };
    den.terms.retain(|(_, w)| !w.is_zero());
    den
}

/// The key tiles' lookup sites as affine forms, mirroring `gkr_key_tile::sites_of`'s
/// site order and `attn_trace::interaction_phase`'s look order exactly. This
/// is the verifier's half of "one definition of the fraction": the collector
/// reads the committed columns, this names them, and the binding constraint is
/// what makes the two the same function.
pub fn link_looks(plan: &crate::compose::Plan, rels: &crate::compose::Rels) -> Vec<SiteLooks> {
    let p = &plan.p;
    let cfg = &plan.cfg;
    let mut out = vec![];
    for (b, diagonal_block) in plan.diagonal_block.iter().enumerate() {
        let l = &plan.lays[b];
        let (lc, lk) = (&l.chan, &l.key);

        // ---- channel component: the key and query range checks, all unit
        if cfg.range_check {
            let mut looks = vec![];
            for pp in 0..lc.pk {
                looks.push(LinkLook {
                    num: None,
                    den: range_den(rels.range(p.k_bits),
                                   Affine::col(RCol::Chan(pp), SecureField::one())),
                });
            }
            for pp in 0..lc.pq {
                looks.push(LinkLook {
                    num: None,
                    den: range_den(rels.range(p.q_bits),
                                   Affine::col(RCol::Chan(lc.pk + pp), SecureField::one())),
                });
            }
            out.push(SiteLooks { reg: Comp::KeyTile(b), which: 0, log_rows: lc.g.log_rows, looks });
        }

        // ---- grid component: exp lookups, the variant's remainder checks, the
        // V range checks
        let n_aux = match plan.attn_variant {
            AttnVariant::IntExp => 0,
            _ => lk.q,
        };
        let mut looks = vec![];
        let (ez, eal) = rels.exp.parts();
        for i in 0..lk.q {
            let key = match plan.attn_variant {
                AttnVariant::Bf16x => RCol::Key(2 * lk.q + i),
                _ => RCol::Key(i),
            };
            let den = Affine {
                terms: vec![(key, eal[0]), (RCol::Key(lk.q + i), eal[1])],
                k: -ez,
            };
            let num = diagonal_block.then(|| Affine {
                terms: vec![(RCol::Msk(i), -SecureField::one())],
                k: SecureField::one(),
            });
            looks.push(LinkLook { num, den });
        }
        if let AttnVariant::IntSplit(kk) = plan.attn_variant {
            for i in 0..lk.q {
                looks.push(LinkLook {
                    num: None,
                    den: range_den(rels.range(kk),
                                   Affine::col(RCol::Key(2 * lk.q + i), SecureField::one())),
                });
            }
        }
        if cfg.range_check {
            for pp in 0..lk.pv {
                looks.push(LinkLook {
                    num: None,
                    den: range_den(rels.range(8),
                                   Affine::col(RCol::Key(2 * lk.q + n_aux + pp),
                                               SecureField::one())),
                });
            }
        }
        out.push(SiteLooks { reg: Comp::KeyTile(b), which: 1, log_rows: lk.g.log_rows, looks });

        // ---- per-token component: the two denominator bounds' limbs. The high
        // limbs are not committed -- `hi = (Z - lo) * 2^-z_lo_bits` -- so they
        // fold into weights on the committed `zs` and `los` columns, which is
        // memo section 4's first consequence.
        //
        // Gated on `plan.rows_key_tile[b]` and not on `!diagonal_block` for the reason
        // `gkr_key_tile::sites_of` gives at the same site: the component belongs to the
        // head key tile group, and with `--kv-boundary` a group has `c/s` prefix
        // sub-key-tiles of which only the first carries it. `rows_key_tile` is
        // `!diagonal_block` with the flag off.
        if plan.rows_key_tile[b] {
            let with_max = plan.attn_variant != AttnVariant::Bf16x;
            let ms = if with_max { l.rw } else { 0 };
            let inv = SecureField::from(M31::from_u32_unchecked(1 << cfg.z_lo_bits).inverse());
            let zmin = SecureField::from(M31::from_u32_unchecked(cfg.z_min));
            let mut looks = vec![];
            for i in 0..l.rw {
                let zs = RCol::Row(ms + i);
                let lo = RCol::Row(ms + l.rw + i);
                let lo2 = RCol::Row(ms + 2 * l.rw + i);
                looks.push(LinkLook {
                    num: None,
                    den: range_den(rels.range(cfg.z_lo_bits),
                                   Affine::col(lo, SecureField::one())),
                });
                looks.push(LinkLook {
                    num: None,
                    den: range_den(rels.range(cfg.z_hi_bits), Affine {
                        terms: vec![(zs, inv), (lo, -inv)],
                        k: SecureField::zero(),
                    }),
                });
                looks.push(LinkLook {
                    num: None,
                    den: range_den(rels.range(cfg.z_lo_bits),
                                   Affine::col(lo2, SecureField::one())),
                });
                looks.push(LinkLook {
                    num: None,
                    den: range_den(rels.range(cfg.z_hi_bits), Affine {
                        terms: vec![(zs, inv), (lo2, -inv)],
                        k: -zmin * inv,
                    }),
                });
            }
            out.push(SiteLooks { reg: Comp::KeyTile(b), which: 3, log_rows: l.log_row, looks });
        }
    }
    if plan.p.gkr_dequant {
        out.extend(deq_looks(plan, rels));
    }
    // B11 stream V, lever 3: the residual-stream family, after dequant, which is memo
    // section 8's order and therefore the order `sites_of` derives and the
    // interaction pass collects in.
    if plan.p.gkr_residual_stream {
        out.extend(residual_stream_looks(plan, rels));
    }
    if plan.p.gkr_freivalds {
        out.extend(freivalds_looks(plan, rels));
    }
    if plan.p.gkr_attnout {
        out.extend(ao_looks(plan, rels));
    }
    out
}

/// The dequant family's lookup sites as affine forms.
///
/// B11 stream U, lever 3. This mirrors `layer::mat_logup`'s hand-written
/// builder look for look and column for column -- that builder is the ground
/// truth for the *order*, because `plogup::paired` batches in it and the
/// generated kernel lowers `DequantEval::evaluate` in it, so all three agree or
/// `ZK_GENERATED_LOGUP=2` fails. Every value a dequant lookup reads is a committed
/// trace mask, one preprocessed column, or the constant zero, so every
/// denominator is affine in committed columns exactly as memo section 4 needs;
/// the `mul` lookups' numerator is the committed `nz`, which is affine too.
///
/// Only `pow2` and `chain` are reachable: the hand builder answers
/// `unreachable!()` for the others, and so does this.
pub fn deq_looks(plan: &crate::compose::Plan, rels: &crate::compose::Rels)
                 -> Vec<SiteLooks> {
    let one = SecureField::one();
    let mut out = vec![];
    for (m, sp) in plan.specs.iter().enumerate() {
        let map = crate::gadgets::dequant::Map::new(sp);
        let f = map.f;
        let sticky = sp.round.needs_sticky();
        let mut looks = vec![];
        // the relation's own drawn elements: `combine(v) = sum alpha_i v_i - z`,
        // which is affine in `v` with the alphas as coefficients
        let rel = |z: SecureField, al: &[SecureField], cols: &[Option<RCol>]| {
            let mut den = Affine { terms: vec![], k: -z };
            for (i, c) in cols.iter().enumerate() {
                // a `None` is the constant-zero column the non-sticky rounding
                // modes pass, and `alpha_i * 0` is not a term at all
                if let Some(c) = c {
                    den.terms.push((*c, al[i]));
                }
            }
            den
        };
        let (nz_ok, al_norm) = rels.norm.parts();
        let (fz, al_frac) = rels.frac.parts();
        let (mz, al_mul) = rels.mul.parts();
        for s in 0..sp.per_row {
            // stream Z: the fold has no normalisation core; its eight range
            // looks stand alone, in the AIR's `add_to_relation` order
            // (`dequant.rs`'s `Fold` arm)
            if sp.variant == crate::gadgets::dequant::Variant::Fold {
                let two = one + one;
                let lim = crate::gadgets::dequant::FOLD_LIMB;
                looks.push(LinkLook { num: None, den: range_den(
                    rels.range(sp.q_bits), Affine::col(RCol::Deq(map.at(s, f.qm)), one)) });
                looks.push(LinkLook { num: None, den: range_den(
                    rels.range(7), Affine::col(RCol::Deq(map.at(s, f.qc)), one)) });
                looks.push(LinkLook { num: None, den: range_den(
                    rels.range(sp.q_bits),
                    Affine { terms: vec![(RCol::Deq(map.at(s, f.qm)), one),
                                         (RCol::Deq(map.at(s, f.qc)), -one)],
                             k: SecureField::zero() }) });
                looks.push(LinkLook { num: None, den: range_den(
                    rels.range(lim), Affine::col(RCol::Deq(map.at(s, f.flo)), one)) });
                looks.push(LinkLook { num: None, den: range_den(
                    rels.range(lim), Affine::col(RCol::Deq(map.at(s, f.fb)), one)) });
                looks.push(LinkLook { num: None, den: range_den(
                    rels.range(16), Affine::col(RCol::Deq(map.at(s, f.fa)), one)) });
                looks.push(LinkLook { num: None, den: range_den(
                    rels.range(sp.rem_bits), Affine::col(RCol::Deq(map.at(s, f.fe)), one)) });
                looks.push(LinkLook { num: None, den: range_den(
                    rels.range(sp.rem_bits),
                    Affine { terms: vec![(RCol::Deq(map.at(s, f.fdp)), two),
                                         (RCol::Deq(map.at(s, f.fe)), -one)],
                             k: -one }) });
                continue;
            }
            let st = sticky.then(|| RCol::Deq(map.at(s, f.st)));
            looks.push(LinkLook {
                num: None,
                den: rel(nz_ok, al_norm, &[
                    Some(RCol::Deq(map.at(s, f.m2))), Some(RCol::Deq(map.at(s, f.ph))), st,
                    Some(RCol::Deq(map.at(s, f.m7))), Some(RCol::Deq(map.at(s, f.ef))),
                    Some(RCol::Deq(map.at(s, f.nz)))]),
            });
            looks.push(LinkLook {
                num: None,
                den: rel(fz, al_frac, &[
                    Some(RCol::Deq(map.at(s, f.ph))), Some(RCol::Deq(map.at(s, f.r))), st]),
            });
            match sp.variant {
                crate::gadgets::dequant::Variant::Pow2 => looks.push(LinkLook {
                    num: None,
                    den: range_den(rels.range(8),
                                   Affine::col(RCol::Deq(map.at(s, f.eo)), one)),
                }),
                crate::gadgets::dequant::Variant::Chain => {
                    for t in 0..2 {
                        let m_in = if t == 0 { RCol::Deq(map.at(s, f.m7)) }
                                   else { RCol::Deq(map.at(s, f.mul[0][0])) };
                        // the activation scale is committed and shared by the
                        // row; the weight scale is offline and preprocessed
                        let ms = if t == 0 { RCol::Deq(map.shared[0]) } else { RCol::DeqPre(s) };
                        looks.push(LinkLook {
                            num: Some(Affine::col(RCol::Deq(map.at(s, f.nz)), one)),
                            den: rel(mz, al_mul, &[
                                Some(m_in), Some(ms),
                                Some(RCol::Deq(map.at(s, f.mul[t][0]))),
                                Some(RCol::Deq(map.at(s, f.mul[t][1])))]),
                        });
                        looks.push(LinkLook {
                            num: None,
                            den: range_den(rels.range(8),
                                           Affine::col(RCol::Deq(map.at(s, f.mul[t][2])), one)),
                        });
                    }
                }
                v => panic!("the dequant GKR backend has no looks for variant {}", v.name()),
            }
        }
        out.push(SiteLooks {
            reg: Comp::Deq(m),
            which: 0,
            log_rows: plan.mat_lays[m].log_rows,
            looks,
        });
    }
    out
}

/// The residual-stream family's lookup sites as affine forms.
///
/// B11 stream V, lever 3. This mirrors `residual_stream::residual_stream_interaction`'s `fr` closure
/// look for look and column for column -- that closure is the ground truth for
/// the *order*, because `plogup::paired` batches in it and the generated kernel
/// lowers `ResidualStreamEval::evaluate` in it, so all three agree or `ZK_GENERATED_LOGUP=2`
/// fails. Every value a residual-stream lookup reads is one of the component's own committed
/// trace masks or the constant zero -- `usize::MAX` is the `+0` operand, which
/// contributes no term at all -- so every denominator is affine in committed
/// columns exactly as memo section 4 needs, and every numerator is the literal
/// one, which puts every residual-stream instance in `Layer::LogUpSingles`.
///
/// Three of the six kinds read a *pair* of masks two apart (`c(a + 2)` is the
/// unary's output beside its input) and one reads an affine combination rather
/// than a bare column (`Pow`'s first slot is `ef - etop + DMAX`, the same shape
/// the per-token component's high limbs take). Neither is new: the fold has
/// carried weighted combinations since stream S.
///
/// `gkrlink::ir_looks` derives the same forms from the fraction IR, and
/// `ZK_GKR_AFFINITY=1` compares the two. This one is what both the prover and
/// the *verifier* run, because it needs the plan and the drawn relations and
/// nothing else -- the verifier has no evaluator to lower.
pub fn residual_stream_looks(plan: &crate::compose::Plan, rels: &crate::compose::Rels)
                  -> Vec<SiteLooks> {
    use crate::gadgets::residual_stream::{Kind, Src, UnKind};
    let Some(gp) = plan.residual_stream.as_ref() else { return vec![] };
    let Some(gr) = rels.residual_stream.as_ref() else { return vec![] };
    let one = SecureField::one();
    let dmax = SecureField::from(M31::from_u32_unchecked(crate::gadgets::residual_stream::DMAX));
    let mut out = vec![];
    for (g, r) in gp.components.iter().enumerate() {
        let prog = &r.prog;
        // `combine(v) = sum alpha_i v_i - z` over the slots, where a slot is
        // itself an affine form over the component's masks. `None` is the constant
        // zero the `+0` operand passes, and `alpha_i * 0` is not a term.
        let rel = |z: SecureField, al: &[SecureField], slots: &[Option<Affine>]| {
            let mut den = Affine { terms: vec![], k: -z };
            for (i, sl) in slots.iter().enumerate() {
                let Some(sl) = sl else { continue };
                for (c, w) in &sl.terms {
                    den.terms.push((*c, *w * al[i]));
                }
                den.k += sl.k * al[i];
            }
            den.terms.retain(|(_, w)| !w.is_zero());
            den
        };
        // one of the component's committed masks, or the constant zero
        let col = |x: usize| -> Option<Affine> {
            (x != usize::MAX).then(|| Affine::col(RCol::ResidualStream(x), one))
        };
        // Stream OO: a slot operand's half `k` (0 the mantissa, 1 the exponent).
        // `Prog::src_cols` answers `None` for `Src::Pre` because the *witness generation*
        // reads a preprocessed operand out of a second vector
        // (`residual_stream::src_cols_any`'s `PRE_TAG`), and the two callers that took the
        // `None` for the constant zero dropped the operand's term from the
        // affine form -- so a tail's slots folded under a form missing one
        // column and `gkrlink`'s own assert fired at the first tail height.
        // A preprocessed operand is a tree-0 column of the same component: the
        // look is the same shape, at a different tree index.
        let src = |s: &Src, k: usize| -> Option<Affine> {
            match s {
                Src::Pre(i) => Some(Affine::col(RCol::ResidualStreamPre(*i, k), one)),
                _ => prog.src_cols(s, plan.round).map(|c| {
                    Affine::col(RCol::ResidualStream(if k == 0 { c.0 } else { c.1 }), one)
                }),
            }
        };
        let unit = |den: Affine| LinkLook { num: None, den };
        let mut looks = vec![];

        // ---- the leaves: one look per committed leaf, `VAL` or the group's
        // unary. A unary group commits its output beside its input, which is
        // the `+2` here and in the builder.
        for (gi, grp) in prog.groups.iter().enumerate() {
            for i in 0..grp.n {
                let b = prog.leaf_at(gi, i);
                if grp.un == UnKind::None {
                    let (z, al) = gr.val.parts();
                    looks.push(unit(rel(z, al, &[col(b), col(b + 1)])));
                } else {
                    let k = gr.uns.iter().position(|x| *x == grp.un)
                        .expect("a residual-stream group's unary kind is not in the plan's union");
                    let (z, al) = gr.un[k].parts();
                    looks.push(unit(rel(z, al,
                        &[col(b), col(b + 1), col(b + 2), col(b + 3)])));
                }
            }
        }

        // ---- the slots: five looks per Add-tree slot, two per Mul slot
        let sticky = plan.round.needs_sticky();
        let k = 8 + usize::from(sticky);
        for (si, (sa, sb)) in prog.slots.iter().enumerate() {
            let b = prog.slot_at(si, plan.round);
            let (o0, o1) = prog.out_cols(si, plan.round);
            match prog.kind {
                Kind::Add => {
                    let ea = src(sa, 1);
                    let eb = src(sb, 1);
                    // `ef - etop + DMAX`: an affine form in two masks with a
                    // constant, not a bare column.  `ef` is a column of either
                    // tree; the constant zero contributes no term.
                    let align = |ef: Option<Affine>| {
                        let mut a = Affine { terms: vec![], k: dmax };
                        if let Some(e) = ef {
                            a.terms.extend(e.terms);
                            a.k += e.k;
                        }
                        a.terms.push((RCol::ResidualStream(b + 2), -one));
                        Some(a)
                    };
                    let (z, al) = gr.pow.parts();
                    looks.push(unit(rel(z, al, &[align(ea), col(b)])));
                    looks.push(unit(rel(z, al, &[align(eb), col(b + 1)])));
                    let st = sticky.then(|| Affine::col(RCol::ResidualStream(b + 8), one));
                    let (z, al) = gr.norm.parts();
                    looks.push(unit(rel(z, al, &[col(b + 5), col(b + 6), st.clone(),
                                                 col(b + k), col(b + k + 1),
                                                 col(b + k + 2)])));
                    let (z, al) = gr.frac.parts();
                    looks.push(unit(rel(z, al, &[col(b + 6), col(b + 7), st])));
                    let (z, al) = gr.val.parts();
                    looks.push(unit(rel(z, al, &[col(o0), col(o1)])));
                }
                Kind::Mul => {
                    let ca = src(sa, 0);
                    let cb = src(sb, 0);
                    let (z, al) = gr.smul.parts();
                    looks.push(unit(rel(z, al, &[ca, cb, col(b),
                                                 col(b + 1), col(b + 2)])));
                    let (z, al) = gr.val.parts();
                    looks.push(unit(rel(z, al, &[col(o0), col(o1)])));
                }
            }
        }
        out.push(SiteLooks {
            reg: Comp::ResidualStream(g),
            which: 0,
            log_rows: r.log_rows,
            looks,
        });
    }
    out
}

/// The Freivalds checks' lookup sites as affine forms.
///
/// B11 stream V, lever 4. One 8-bit range check per query slot, over the
/// component's first `pq` committed columns -- which is exactly what
/// `layer::freivalds_interaction`'s `fr` reads and what `FreivaldsEval::evaluate` adds.
/// Every numerator is the literal one, so every Freivalds instance is
/// `Layer::LogUpSingles`. This is the whole family: the `v`, `u` and `acc`
/// columns beside them are Horner accumulators, they are 38.9% of this family's
/// interaction cells, and none of them is a lookup.
pub fn freivalds_looks(plan: &crate::compose::Plan, rels: &crate::compose::Rels)
                       -> Vec<SiteLooks> {
    let one = SecureField::one();
    plan.freivalds
        .iter()
        .enumerate()
        .map(|(w, wp)| SiteLooks {
            reg: Comp::Freivalds(w),
            which: 0,
            log_rows: wp.lay.g.log_rows,
            looks: (0..wp.lay.pq)
                .map(|i| LinkLook {
                    num: None,
                    // stream Z: the widened attention output's group ranges its
                    // activations at act_bits, not 8
                    den: range_den(rels.range(wp.act_bits), Affine::col(RCol::Freivalds(i), one)),
                })
                .collect(),
        })
        .collect()
}

/// The attention-output component's lookup sites as affine forms.
///
/// B11 stream V, lever 4. Four looks per channel slot in the AIR's order --
/// the `(qm, nz2)` pair check, then the error `d`, then the excluded bound
/// `dp = 2 denp - 1 - d`, then (stream JJ, `witness-slack.md` section 7 item 1)
/// `denp` itself. The third is the one worth naming: `dp` is not a
/// committed column at all, it is an *affine form* over two of them, which is
/// the same shape the per-token component's high limbs take and the reason memo
/// section 4's collapse does not care. Every numerator is the literal one.
pub fn ao_looks(plan: &crate::compose::Plan, rels: &crate::compose::Rels)
                -> Vec<SiteLooks> {
    let one = SecureField::one();
    let two = SecureField::from(M31::from_u32_unchecked(2));
    let lay = plan.ao_lay;
    let (qz, qal) = rels.qnz.parts();
    let rem = rels.range(plan.p.ao_rem_bits);
    // stream JJ (E11): the exponent window and the accumulator window
    let (ez, eal) = rels.epow.parts();
    let (rlo, rhi) = (rels.range(plan.p.ao_acc_lo_bits), rels.range(plan.p.ao_acc_hi_bits));
    let inv_lo = SecureField::from(M31::from_u32_unchecked(1 << plan.p.ao_acc_lo_bits))
        .inverse();
    // stream TT (E11), log row 73: the magnitude form offsets nothing -- the
    // high limb reads the committed `amag` at `c0 + 9` instead
    let acc_off = SecureField::from(M31::from_u32_unchecked(
        if plan.p.ao_acc_mag { 0 }
        else { 1 << (plan.p.ao_acc_lo_bits + plan.p.ao_acc_hi_bits - 1) }));
    let ncs = crate::gadgets::attn_out::trace::slot_cols(plan.p.ao_acc_mag, plan.p.ao_acc_prod);
    let mut looks = vec![];
    for sl in 0..lay.per_row {
        // `ao_interaction`'s own indexing: slot `sl` starts at `2 + 9 * sl`,
        // and the builder reads `+2`..`+8` off it.  `joff` is column 1 and the
        // slot's `jvp` is preprocessed column `sl`.
        let c0 = 2 + ncs * sl;
        // the `EPOW` window: `(jvp + joff, num, pj)`
        looks.push(LinkLook {
            num: None,
            den: Affine {
                terms: vec![(RCol::AoPre(sl), eal[0]), (RCol::Ao(1), eal[0]),
                            (RCol::Ao(c0 + 6), eal[1]), (RCol::Ao(c0 + 7), eal[2])],
                k: -ez,
            },
        });
        looks.push(LinkLook {
            num: None,
            den: Affine {
                terms: vec![(RCol::Ao(c0 + 2), qal[0]), (RCol::Ao(c0 + 3), qal[1])],
                k: -qz,
            },
        });
        looks.push(LinkLook {
            num: None,
            den: range_den(rem, Affine::col(RCol::Ao(c0 + 5), one)),
        });
        looks.push(LinkLook {
            num: None,
            den: range_den(rem, Affine {
                terms: vec![(RCol::Ao(c0 + 4), two), (RCol::Ao(c0 + 5), -one)],
                k: -one,
            }),
        });
        // stream JJ: `denp`'s own range check -- stream TT's `denp2` form
        // looks up `2 denp`, one bit better on the same table
        looks.push(LinkLook {
            num: None,
            den: range_den(rem, Affine::col(RCol::Ao(c0 + 4),
                                            if plan.p.ao_denp2 { two } else { one })),
        });
        // stream JJ: the accumulator window's two limbs
        looks.push(LinkLook {
            num: None,
            den: range_den(rlo, Affine::col(RCol::Ao(c0 + 8), one)),
        });
        looks.push(LinkLook {
            num: None,
            den: range_den(rhi, Affine {
                terms: vec![(RCol::Ao(if plan.p.ao_acc_mag { c0 + 9 } else { c0 }), inv_lo),
                            (RCol::Ao(c0 + 8), -inv_lo)],
                k: acc_off * inv_lo,
            }),
        });
        // the product form: `ahn = ahi num`, committed at `c0 + 10`, in its
        // `30 - lo`-bit table
        if plan.p.ao_acc_prod {
            looks.push(LinkLook {
                num: None,
                den: range_den(rels.range(crate::compose::ao_prod_bits(&plan.p)),
                               Affine::col(RCol::Ao(c0 + 10), one)),
            });
        }
    }
    vec![SiteLooks { reg: Comp::Ao, which: 0, log_rows: lay.log_rows, looks }]
}

/// How many looks a residual-stream component has, from its program alone.
///
/// The verifier's `sites_of` needs the look count and the numerator kinds
/// without an evaluator to lower, and this is the count `residual_stream_looks` produces:
/// one per committed leaf, then five per Add-tree slot or two per Mul slot.
/// Every one is unit.
pub fn residual_stream_look_count(prog: &crate::gadgets::residual_stream::Prog) -> usize {
    let per_slot = match prog.kind {
        crate::gadgets::residual_stream::Kind::Add => 5,
        crate::gadgets::residual_stream::Kind::Mul => 2,
    };
    prog.groups.iter().map(|g| g.n).sum::<usize>() + per_slot * prog.slots.len()
}

// ------------------------------------------------------------------------
// B11 stream V, lever 2: whether a family's lookups can be linked at all.
// ------------------------------------------------------------------------

/// One node's value as an affine form over the columns the IR loads.
///
/// `terms` is keyed by `(tree, column)` -- the same pair `Ir::Op::Load` carries
/// -- so the mapper that turns it into an `RCol` is the only family-specific
/// part of the derivation.
#[derive(Clone, Debug, Default)]
pub struct AffForm {
    pub terms: BTreeMap<(usize, usize), SecureField>,
    pub k: SecureField,
}

impl AffForm {
    fn konst(k: SecureField) -> Self {
        AffForm { terms: BTreeMap::new(), k }
    }
    fn is_const(&self) -> bool {
        self.terms.is_empty()
    }
    fn scaled(&self, w: SecureField) -> Self {
        AffForm {
            terms: self.terms.iter().map(|(c, v)| (*c, *v * w)).collect(),
            k: self.k * w,
        }
    }
    fn add(&self, o: &Self, sign: SecureField) -> Self {
        let mut terms = self.terms.clone();
        for (c, v) in &o.terms {
            let e = terms.entry(*c).or_insert_with(SecureField::zero);
            *e += *v * sign;
            if e.is_zero() {
                terms.remove(c);
            }
        }
        AffForm { terms, k: self.k + o.k * sign }
    }
}

/// Every node's affine form, or the first reason it has none.
///
/// One pass in dependency order, which `Ir` guarantees ("every operand is the
/// index of an earlier instruction"). A product is affine only when one side is
/// a constant -- which is what makes `2 * dp - 1 - lo` and
/// `a_ef - etop + DMAX` affine and a genuine product of two masks not.
fn aff_nodes(ir: &crate::cudagen::Ir) -> Vec<Result<AffForm, String>> {
    use crate::cudagen::Op;
    let basis = |i: usize| SecureField::from_m31_array(std::array::from_fn(|j| {
        if i == j { M31::one() } else { M31::zero() }
    }));
    let mut out: Vec<Result<AffForm, String>> = Vec::with_capacity(ir.ops.len());
    for op in &ir.ops {
        let v = (|| -> Result<AffForm, String> {
            let get = |i: usize| -> Result<AffForm, String> {
                out[i].clone()
            };
            Ok(match op {
                Op::Load { tree, col, off } => {
                    // tree 0 is preprocessed and tree 1 the committed base; an
                    // interaction column is one this proof is still building,
                    // and a nonzero offset is a different polynomial than the
                    // one the univariate IOP evaluates at the row point.
                    if *tree > 1 {
                        return Err(format!("reads interaction tree {tree} column {col}"));
                    }
                    if *off != 0 {
                        return Err(format!(
                            "reads tree {tree} column {col} at offset {off}"));
                    }
                    AffForm {
                        terms: [((*tree, *col), SecureField::one())].into_iter().collect(),
                        k: SecureField::zero(),
                    }
                }
                Op::ConstM31(v) => {
                    AffForm::konst(SecureField::from(M31::from_u32_unchecked(*v)))
                }
                Op::Scalar(k) => AffForm::konst(
                    *ir.scalars.get(*k)
                        .ok_or_else(|| "a scalar slot the blob does not have".to_string())?),
                // the logup shift is a runtime argument, and it enters the
                // constraint half and never a lookup
                Op::CumsumShift => return Err("reads the logup shift".into()),
                Op::AddB(a, b) | Op::AddE(a, b) => {
                    get(*a)?.add(&get(*b)?, SecureField::one())
                }
                Op::SubB(a, b) | Op::SubE(a, b) => {
                    get(*a)?.add(&get(*b)?, -SecureField::one())
                }
                Op::MulB(a, b) | Op::MulE(a, b) => {
                    let (x, y) = (get(*a)?, get(*b)?);
                    if x.is_const() {
                        y.scaled(x.k)
                    } else if y.is_const() {
                        x.scaled(y.k)
                    } else {
                        return Err(format!(
                            "multiplies two column forms ({} by {} terms)",
                            x.terms.len(), y.terms.len()));
                    }
                }
                Op::NegB(a) | Op::NegE(a) => get(*a)?.scaled(-SecureField::one()),
                // four base values into one extension value: a fixed linear
                // combination over the extension basis, so it stays affine
                Op::Pack(a, b, c, d) => {
                    let mut f = AffForm::default();
                    for (j, n) in [*a, *b, *c, *d].into_iter().enumerate() {
                        f = f.add(&get(n)?.scaled(basis(j)), SecureField::one());
                    }
                    f
                }
            })
        })();
        out.push(v);
    }
    out
}

/// Every look's `(numerator, denominator)` as affine forms.
///
/// This is memo section 4's precondition, decided rather than asserted, and it
/// is also the link's own derivation: the same walk answers "can this family be
/// linked" and "what are the weights". `link_looks` states the key tiles' and
/// dequant's forms by hand -- those two are recorded and their digests gate
/// every change -- and the families stream V adds take them from here, so a
/// family's link cannot drift from the builder the kernel was compiled from.
///
/// The backend is sound only because an instance's input-layer claim collapses to
/// one affine combination of committed-column MLEs at the row part of the OOD
/// point. That needs three things of every look, and all three are properties
/// of the fraction rather than of the prover: the denominator affine in what it
/// reads; everything it reads a committed base mask or a preprocessed column,
/// at this row and no other; and the numerator the constant one or itself
/// affine in those same columns (dequant's committed `nz` is the precedent).
pub fn aff_looks(ir: &crate::cudagen::Ir) -> Vec<Result<(AffForm, AffForm), String>> {
    let nodes = aff_nodes(ir);
    ir.fracs
        .iter()
        .map(|(n, d)| {
            let den = nodes[*d].clone().map_err(|e| format!("denominator {e}"))?;
            let num = nodes[*n].clone().map_err(|e| format!("numerator {e}"))?;
            Ok((num, den))
        })
        .collect()
}

/// The family's looks as the link wants them, from the IR.
///
/// `map` names the column a `(tree, index)` pair is; it is the only
/// family-specific part. A look whose numerator is the literal one becomes a
/// unit look, which is what puts its instance in `Layer::LogUpSingles`.
pub fn ir_looks<F: Fn(usize, usize) -> RCol>(
    tag: &str,
    ir: &crate::cudagen::Ir,
    map: F,
) -> Vec<LinkLook> {
    let one = SecureField::one();
    let conv = |f: &AffForm| Affine {
        terms: f.terms.iter().map(|((t, c), w)| (map(*t, *c), *w)).collect(),
        k: f.k,
    };
    aff_looks(ir)
        .into_iter()
        .enumerate()
        .map(|(i, r)| {
            let (num, den) = r.unwrap_or_else(|e| panic!(
                "{tag}: look {i} cannot be linked -- {e}. The affinity check \
                 (`ZK_GKR_AFFINITY=1`) is what this family's backend was built on."));
            LinkLook {
                num: (!(num.is_const() && num.k == one)).then(|| conv(&num)),
                den: conv(&den),
            }
        })
        .collect()
}

/// The numerator kinds, in `add_to_relation` order.
///
/// A family's `look_units` list, derived rather than written down: the
/// collector needs it to split a site into unit and non-unit instances, and
/// `gkr_key_tile::plan` needs the two to agree with what the fractions actually are.
pub fn ir_units(ir: &crate::cudagen::Ir) -> Vec<bool> {
    let one = SecureField::one();
    aff_looks(ir)
        .into_iter()
        .map(|r| matches!(&r, Ok((n, _)) if n.is_const() && n.k == one))
        .collect()
}

/// One evaluator's verdict, for the record.
#[derive(Clone, Debug, serde::Serialize)]
pub struct Affinity {
    pub family: String,
    pub component: String,
    pub n_looks: usize,
    /// the numerator kinds, in `add_to_relation` order
    pub units: Vec<bool>,
    pub ok: bool,
    /// the looks that could not be linked, with the reason
    pub bad: Vec<(usize, String)>,
    /// `(tree, column)` pairs the looks read, deduplicated -- the columns the
    /// link would have to name
    pub reads: Vec<(usize, usize)>,
    /// the widest denominator, in columns
    pub max_den_terms: usize,
}

/// Decide one lowered evaluator's looks, for the record.
pub fn affinity_of(family: &str, component: &str, ir: &crate::cudagen::Ir) -> Affinity {
    let looks = aff_looks(ir);
    let mut bad = vec![];
    let mut reads: Vec<(usize, usize)> = vec![];
    let mut units = vec![];
    let mut max_den = 0usize;
    let one = SecureField::one();
    for (i, r) in looks.iter().enumerate() {
        match r {
            Err(e) => {
                bad.push((i, e.clone()));
                units.push(false);
            }
            Ok((n, d)) => {
                units.push(n.is_const() && n.k == one);
                max_den = max_den.max(d.terms.len());
                for c in d.terms.keys().chain(n.terms.keys()) {
                    if !reads.contains(c) {
                        reads.push(*c);
                    }
                }
            }
        }
    }
    reads.sort_unstable();
    Affinity {
        family: family.to_string(),
        component: component.to_string(),
        n_looks: looks.len(),
        units,
        ok: bad.is_empty(),
        bad,
        reads,
        max_den_terms: max_den,
    }
}

// ------------------------------------------------------------------------
// Per-height parameters: one folded mle_eval instance per distinct component
// height, derived identically by the prover and the verifier.
// ------------------------------------------------------------------------

/// One folded univariate IOP: everything both sides need about one height.
pub struct HeightLink {
    pub h: u32,
    /// the row part of the OOD point: the last `h` coordinates
    pub point: Vec<SecureField>,
    pub mle_point: MleEvalPoint,
    /// the folded claim, and the shift `claim / 2^h` the prefix column carries
    pub claim: SecureField,
    pub shift: SecureField,
    /// folded weights per referenced column, and the folded constant
    pub weights: Vec<(Comp, RCol, SecureField)>,
    pub konst: SecureField,
}

/// eq weights over one instance's looks: position `p`'s bits pair MSB-first
/// with the look part of the instance's OOD point -- the convention the MLE
/// machinery fixes (`eval_at_point` splits halves on `point[0]`, and the CPU
/// `gen_eq_evals` walks `y` reversed). Each pass doubles the table with the
/// processed coordinate as the new index MSB, so the walk runs over `x_look`
/// *reversed*: the last pass, and therefore the MSB, is `x_look[0]`.
fn eq_weights(k: usize, x_look: &[SecureField]) -> Vec<SecureField> {
    assert_eq!(k, x_look.len());
    let mut out = vec![SecureField::one()];
    for x in x_look.iter().rev() {
        let mut next = Vec::with_capacity(out.len() * 2);
        for w in &out {
            next.push(*w * (SecureField::one() - *x));
        }
        for w in &out {
            next.push(*w * *x);
        }
        out = next;
    }
    out
}

/// Derive the per-height fold from the plan, the drawn relations, the GKR
/// artifact and the post-GKR challenge. The verifier calls this to know what
/// the binding constraints assert; the prover calls it to build the very columns
/// those constraints check. `Err` is a rejection, never a panic: the artifact
/// is proof-derived data.
pub fn link_params(
    plan: &crate::compose::Plan,
    rels: &crate::compose::Rels,
    insts: &[Inst],
    artifact: &GkrArtifact,
    alpha: SecureField,
) -> Result<Vec<HeightLink>, String> {
    let looks = link_looks(plan, rels);
    let el = artifact.ood_point.len();
    if artifact.claims_to_verify_by_instance.len() != insts.len() {
        return Err(format!("the GKR artifact carries {} instances, the plan says {}",
                           artifact.claims_to_verify_by_instance.len(), insts.len()));
    }

    struct Acc {
        point: Vec<SecureField>,
        claim: SecureField,
        weights: BTreeMap<(Comp, RCol), SecureField>,
        konst: SecureField,
        n: usize,
    }
    let mut by_h: BTreeMap<u32, Acc> = BTreeMap::new();

    for (i, inst) in insts.iter().enumerate() {
        let n_i = inst.n_variables();
        if el < n_i {
            return Err("the GKR OOD point is shorter than an instance".into());
        }
        let x_i = &artifact.ood_point[el - n_i..];
        let k_look = inst.ents.len().ilog2() as usize;
        let (x_look, x_row) = x_i.split_at(k_look);
        let eqw = eq_weights(k_look, x_look);
        let site = &looks[inst.site];
        let h = inst.log_rows;
        debug_assert_eq!(site.log_rows, h);

        let claims = &artifact.claims_to_verify_by_instance[i];
        if claims.len() != 2 {
            return Err(format!("instance {i} carries {} claims, not 2", claims.len()));
        }
        let (num_claim, den_claim) = (claims[0], claims[1]);

        let acc = by_h.entry(h).or_insert_with(|| Acc {
            point: x_row.to_vec(),
            claim: SecureField::zero(),
            weights: BTreeMap::new(),
            konst: SecureField::zero(),
            n: 0,
        });
        debug_assert_eq!(acc.point.as_slice(), x_row);

        fn push(acc: &mut Acc, alpha: SecureField, reg: Comp, claim: SecureField,
                parts: &mut dyn Iterator<Item = (SecureField, Affine)>) {
            let a = alpha.pow(acc.n as u128);
            acc.n += 1;
            acc.claim += a * claim;
            for (w, aff) in parts {
                for (c, cw) in &aff.terms {
                    *acc.weights.entry((reg, *c)).or_insert(SecureField::zero()) +=
                        a * w * *cw;
                }
                acc.konst += a * w * aff.k;
            }
        }

        if inst.unit {
            // a singles instance's numerator MLE is the constant one, so its
            // claim must be exactly one -- checked here, no column involved
            if num_claim != SecureField::one() {
                return Err(format!("instance {i} claims a unit numerator of {num_claim:?}"));
            }
        } else {
            let mut parts = inst.ents.iter().zip(&eqw).map(|(l, w)| {
                (*w, site.looks[*l].num.clone().unwrap_or(Affine {
                    terms: vec![],
                    k: SecureField::one(),
                }))
            });
            push(acc, alpha, site.reg, num_claim, &mut parts);
        }
        let mut parts = inst.ents.iter().zip(&eqw)
            .map(|(l, w)| (*w, site.looks[*l].den.clone()));
        push(acc, alpha, site.reg, den_claim, &mut parts);
    }

    Ok(by_h
        .into_iter()
        .map(|(h, acc)| {
            let shift = acc.claim / BaseField::from(1u32 << h);
            HeightLink {
                h,
                mle_point: MleEvalPoint::new(&acc.point),
                point: acc.point,
                claim: acc.claim,
                shift,
                weights: acc.weights.into_iter().map(|((r, c), w)| (r, c, w)).collect(),
                konst: acc.konst,
            }
        })
        .collect())
}

/// The distinct link heights of a plan, ascending -- what decides how many
/// `LinkEval` components a statement has before any proof exists (the emit
/// path needs exactly this).
pub fn link_heights(plan: &crate::compose::Plan) -> Vec<u32> {
    let mut hs: Vec<u32> = crate::gkr_key_tile::sites_of(plan).iter().map(|s| s.log_rows).collect();
    hs.sort_unstable();
    hs.dedup();
    hs
}

pub fn first_id(h: u32) -> PreProcessedColumnId {
    PreProcessedColumnId { id: format!("s_link_first_{h}") }
}
pub fn second_id(h: u32) -> PreProcessedColumnId {
    PreProcessedColumnId { id: format!("s_link_second_{h}") }
}

/// The two boundary selectors of one height: `1` at the first and second
/// coset rows. Committed to the preprocessed tree only under the flag.
pub fn boundary_cols(h: u32) -> [Eval; 2] {
    let domain = CanonicCoset::new(h).circle_domain();
    let size = 1usize << h;
    [0usize, 1].map(|coset_i| {
        let mut packed = Col::<SimdBackend, BaseField>::zeros(size);
        packed.set(bit_reverse_index(coset_index_to_circle_domain_index(coset_i, h), h),
                   M31::one());
        CircleEvaluation::new(domain, packed)
    })
}

// ------------------------------------------------------------------------
// The prover's columns: per height, [q, coeff, eq, prefix] -- 16 M31 columns
// -- in the exact order `LinkEval::evaluate` claims its masks.
// ------------------------------------------------------------------------

fn coords_to_evals(h: u32, c: SecureColumnByCoords<SimdBackend>) -> Vec<Eval> {
    let domain = CanonicCoset::new(h).circle_domain();
    c.columns.into_iter().map(|col| CircleEvaluation::new(domain, col)).collect()
}

/// Where one referenced column of the fold lives.
///
/// B11 stream T: under a device witness generation a key tile's committed base columns have no
/// host mirror, so the fold has to reach them where they are. A diagonal block key tile is
/// always host-built and the causal-mask columns are always host-built, so a
/// single height can mix the two -- which costs nothing, because field addition
/// is exact and associative and the two partial folds add up to the same column
/// whichever order they are taken in.
pub enum Src<'a> {
    Host(&'a Eval),
    /// a device-resident base column, as the witness generation left it
    Dev(*const u32),
}

/// The folded coefficient column: `konst + sum_j w_j * col_j`, pointwise.
///
/// The device-resident terms are combined by one kernel launch and brought
/// back; the host-resident ones are added on the host. With every column on the
/// host this is exactly the loop `build_height_cols` used to open with.
pub fn fold_coeff<'a, F: Fn(Comp, &RCol) -> Src<'a>>(hl: &HeightLink, src_of: &F)
                                                      -> SecureColumn {
    let h = hl.h;
    assert!(h >= LOG_N_LANES, "a link height below the SIMD lane count");
    let n_vec = 1usize << (h - LOG_N_LANES);

    let mut host: Vec<(&Eval, SecureField)> = vec![];
    let mut dev: Vec<(*const u32, SecureField)> = vec![];
    for (reg, c, w) in &hl.weights {
        match src_of(*reg, c) {
            Src::Host(e) => {
                assert_eq!(e.len(), 1usize << h, "column height disagrees with the fold");
                host.push((e, *w));
            }
            Src::Dev(p) => dev.push((p, *w)),
        }
    }

    // the device half, with the constant folded in there so it is added once
    let mut coeff = if dev.is_empty() {
        vec![PackedSecureField::broadcast(hl.konst); n_vec]
    } else {
        #[cfg(not(feature = "cuda"))]
        panic!("a link column is on the device on a build without the cuda feature");
        #[cfg(feature = "cuda")]
        fold_dev(h, hl.konst, &dev)
    };
    for (e, w) in host {
        let w = PackedSecureField::broadcast(w);
        for (vr, slot) in coeff.iter_mut().enumerate() {
            *slot += w * e.data[vr];
        }
    }
    SecureColumn { data: coeff, length: 1 << h }
}

/// The device half of the fold: one launch of the link's affine combine over
/// the resident columns, then one download of the folded column.
#[cfg(feature = "cuda")]
fn fold_dev(h: u32, konst: SecureField, terms: &[(*const u32, SecureField)])
            -> Vec<PackedSecureField> {
    use stwo::stwo_cuda::secure_field_vec::SecureFieldVec;
    unsafe extern "C" {
        fn zk_link_coeff(cols: *const *const u32, w_host: *const u32, n_cols: u32,
                          k: [u32; 4], out: *const u32, n_rows: u32);
    }
    let n_rows = 1usize << h;
    let cols: Vec<*const u32> = terms.iter().map(|(p, _)| *p).collect();
    let w: Vec<u32> =
        terms.iter().flat_map(|(_, w)| w.to_m31_array().map(|x| x.0)).collect();
    let out = SecureFieldVec::new_uninitialized(n_rows);
    unsafe {
        zk_link_coeff(cols.as_ptr(), w.as_ptr(), cols.len() as u32,
                       konst.to_m31_array().map(|x| x.0), out.device_ptr, n_rows as u32);
    }
    out.to_vec().into_iter().collect::<SecureColumn>().data
}

/// Build one height's committed columns from its folded coefficient column.
/// The fold resolves referenced columns against this chunk's committed data
/// (the prover's ground truth); the binding constraints are what force it to agree
/// with the verifier's own resolution through the component spans.
pub fn build_height_cols(hl: &HeightLink, coeff_col: SecureColumn) -> Vec<Eval> {
    let h = hl.h;
    assert_eq!(coeff_col.length, 1usize << h, "the folded column is the wrong height");

    // the eq column at the row point, and the shifted prefix sum of
    // coeff * eq (upstream `build_trace`, function for function)
    let t = std::time::Instant::now();
    let eq_mle = SimdBackend::gen_eq_evals(&hl.point, SecureField::one());
    crate::util::add_phase("int/link_eq", t.elapsed().as_secs_f64());
    assert_eq!(eq_mle.len(), 1usize << h);
    let t = std::time::Instant::now();
    // B11 stream T: one packed multiply per lane group, in parallel -- the same
    // products in the same slots, which is all the digest asks of it.
    let mut terms: Vec<PackedSecureField> = coeff_col
        .data
        .par_iter()
        .zip(eq_mle.data.par_iter())
        .map(|(a, b)| *a * *b)
        .collect();
    // the prefix-sum boundary is the claim itself: assert it here, loudly,
    // so a weight bug fails the prover and not just the verifier
    let total: SecureField = terms
        .iter()
        .map(|p| p.to_array().iter().fold(SecureField::zero(), |a, b| a + *b))
        .fold(SecureField::zero(), |a, b| a + b);
    assert_eq!(total, hl.claim,
               "the folded coefficient column disagrees with the GKR claim at height {h}");
    let shift = PackedSecureField::broadcast(hl.shift);
    terms.par_iter_mut().for_each(|t| *t -= shift);
    let shifted = SecureColumn { data: terms, length: 1 << h };
    crate::util::add_phase("int/link_terms", t.elapsed().as_secs_f64());
    let t = std::time::Instant::now();
    let prefix_coords = shifted
        .into_secure_column_by_coords()
        .columns
        .map(inclusive_prefix_sum);
    let prefix = SecureColumnByCoords { columns: prefix_coords };
    crate::util::add_phase("int/link_prefix", t.elapsed().as_secs_f64());

    let t = std::time::Instant::now();
    let q = gen_carry_quotient_col(&hl.mle_point);
    crate::util::add_phase("int/link_q", t.elapsed().as_secs_f64());

    let t = std::time::Instant::now();
    let mut out = coords_to_evals(h, q);
    out.extend(coords_to_evals(h, coeff_col.into_secure_column_by_coords()));
    out.extend(coords_to_evals(h, eq_mle.into_evals().into_secure_column_by_coords()));
    out.extend(coords_to_evals(h, prefix));
    crate::util::add_phase("int/link_split", t.elapsed().as_secs_f64());
    check_height_cols(hl, &out);
    out
}

// ------------------------------------------------------------------------
// The framework component: one per height, carrying the eq and prefix-sum
// constraints over the committed [q, coeff, eq, prefix] columns.
// ------------------------------------------------------------------------

#[derive(Clone)]
pub struct LinkEval {
    pub h: u32,
    pub mle_point: MleEvalPoint,
    pub shift: SecureField,
}

impl FrameworkEval for LinkEval {
    /// milestone 2b: the flat parameter block for the generated
    /// constraint kernel (`gpu/m2b/cudagen.rs`).
    fn cuda_eval_blob(&self) -> Option<Vec<u8>> {
        crate::cudagen::blob_for(self)
    }

    fn cuda_eval_name(&self) -> &'static str { "zk_link_eval" }

    fn log_size(&self) -> u32 {
        self.h
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.h + 1
    }
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        // mask claim order defines the committed column order: q, coeff, eq,
        // prefix -- exactly what `build_height_cols` emits
        let [q_cur] = eval.next_extension_interaction_mask(INTERACTION_TRACE_IDX, [0]);
        let [coeff_cur] = eval.next_extension_interaction_mask(INTERACTION_TRACE_IDX, [0]);
        let is_first = eval.get_preprocessed_column(first_id(self.h));
        let is_second = eval.get_preprocessed_column(second_id(self.h));
        let eq_cur = eval_eq_constraints(INTERACTION_TRACE_IDX, &mut eval, &self.mle_point,
                                         q_cur, is_first, is_second);
        let terms = coeff_cur * eq_cur;
        // stream RR: `shift` is `claim / 2^h`, a value of the statement -- so a
        // height whose folded claim is zero must not lower to a shorter shape
        let shift = eval.claim(self.shift);
        eval_prefix_sum_constraints(INTERACTION_TRACE_IDX, &mut eval, terms, shift);
        eval
    }
}

// ------------------------------------------------------------------------
// The binding component: the two OODS equalities per height that upstream's
// oracle and succinct column performed implicitly. Identically zero for an
// honest prover, so the prover side only consumes its accumulator slots.
// ------------------------------------------------------------------------

pub struct TieHeight {
    pub h: u32,
    pub mle_point: MleEvalPoint,
    pub konst: SecureField,
    /// the affine combination's committed side: `(tree, global column, weight)`
    pub terms: Vec<(usize, usize, SecureField)>,
    /// first global tree-2 column of the committed carry-quotient quadruple
    pub q_at: usize,
    /// first global tree-2 column of the committed coefficient quadruple
    pub coeff_at: usize,
}

pub struct LinkTie {
    pub heights: Vec<TieHeight>,
    /// preprocessed columns the bindings read (the diagonal block masks), globally indexed
    pub pre_idx: Vec<usize>,
}

impl Component for LinkTie {
    fn n_constraints(&self) -> usize {
        2 * self.heights.len()
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.heights.iter().map(|t| t.h + 1).max().unwrap_or(1)
    }

    fn trace_log_degree_bounds(&self) -> TreeVec<ColumnVec<u32>> {
        TreeVec::default()
    }

    fn mask_points(
        &self,
        _point: CirclePoint<SecureField>,
        _max_log_degree_bound: u32,
    ) -> TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>> {
        TreeVec::default()
    }

    fn preprocessed_column_indices(&self) -> ColumnVec<usize> {
        self.pre_idx.clone()
    }

    fn evaluate_constraint_quotients_at_point(
        &self,
        point: CirclePoint<SecureField>,
        mask: &TreeVec<ColumnVec<Vec<SecureField>>>,
        evaluation_accumulator: &mut PointEvaluationAccumulator,
        max_log_degree_bound: u32,
    ) {
        let denom_inverse =
            coset_vanishing(CanonicCoset::new(max_log_degree_bound).coset, point).inverse();
        let ext = |at: usize| {
            SecureField::from_partial_evals([
                mask[INTERACTION_TRACE_IDX][at][0],
                mask[INTERACTION_TRACE_IDX][at + 1][0],
                mask[INTERACTION_TRACE_IDX][at + 2][0],
                mask[INTERACTION_TRACE_IDX][at + 3][0],
            ])
        };
        for t in &self.heights {
            // the committed coefficient column is the `w_c` combination of
            // the base columns -- as polynomials, checked at the OODS point
            let mut binding_coeff = ext(t.coeff_at) - t.konst;
            for (tree, col, w) in &t.terms {
                // `[0]` is the column's value at its *first* sampled offset,
                // and the weight beside it is the one a look that reads the
                // column at offset 0 produced -- so this line is right only
                // where offset 0 is the first offset the owning component
                // sampled. It is: `aff_nodes` refuses to link a look that
                // loads at any other offset, and the one linkable column that
                // is read at two offsets, the attention output's `joff`, is
                // read `[0, -1]` for exactly this reason. `check_mask_offsets`
                // is that sentence as a refusal, run by the prover and by the
                // verifier on the components they both build.
                binding_coeff -= *w * mask[*tree][*col][0];
            }
            evaluation_accumulator.accumulate(denom_inverse * binding_coeff);
            // the committed carry-quotient column is the succinct one. The
            // lifted protocol samples a 2^h column's polynomial at the point
            // doubled down to its own domain, so the analytic side evaluates
            // at the same mapped point -- the coefficient binding above needs no
            // mapping because every column it compares shares one height and
            // therefore one lift.
            let mapped = point.repeated_double(max_log_degree_bound - t.h);
            let binding_q = ext(t.q_at) - eval_carry_quotient_col(&t.mle_point, mapped);
            evaluation_accumulator.accumulate(denom_inverse * binding_q);
        }
    }
}

impl<B: stwo::prover::backend::Backend> ComponentProver<B> for LinkTie {
    fn evaluate_constraint_quotients_on_domain(
        &self,
        _trace: &Trace<'_, B>,
        evaluation_accumulator: &mut DomainEvaluationAccumulator<B>,
    ) {
        // an honest prover's binding constraints are identically zero as
        // polynomials, so their quotient contribution is the zero polynomial:
        // consume the accumulator coefficients and add nothing
        evaluation_accumulator.skip_coeffs(self.n_constraints());
    }
}

// ------------------------------------------------------------------------
// Open question 7.3: the offset a named column is read at
// ------------------------------------------------------------------------

/// Every column a link names is sampled at the offset the link reads.
///
/// `LinkTie` resolves a named column as `mask[tree][col][0]` -- the value at the
/// column's *first* sampled offset -- while the weight it multiplies comes from
/// a look that reads that column at offset 0 and at no other: `aff_nodes`
/// refuses to link a look that loads at any other offset, and refuses an
/// interaction tree outright. The two sides mean the same value only when the
/// component that owns the column put offset 0 first in its own mask, and that
/// is a property of how each linked evaluator happens to be written rather than
/// of a check. It holds today, and one comment is what holds it: the attention
/// output reads `den` at `[-1, 0]` and `joff` at `[0, -1]`, and `joff` is the
/// linked one. A lookup added over a column whose component reads the previous
/// row first would have the link compare the previous row's value against this
/// row's claim -- which agrees on a witness whose column is constant down the
/// rows and on no other, so the backend would accept or reject by the data.
///
/// The check is over the sampled *points*, which is what the offsets become: a
/// column read at offsets `o_0, o_1, ...` is sampled at `point + step o_i` in
/// that order, so `points[0] == point` says exactly "the first sampled offset
/// is 0". The point is arbitrary here -- offset to point is injective at any
/// point, for every offset smaller than the step's order -- so this runs off the
/// component list alone, before a proof and before a verification, and both
/// sides run it because both build the same components.
///
/// The two preprocessed trees are not checked and need none: the verifier
/// samples every preprocessed column a component names at the row point and at
/// nothing else (`Components::mask_points` gives each of them the one point),
/// so `[0]` is that column's value at the row by construction.
pub fn check_mask_offsets(tie: &LinkTie, components: &[&dyn Component]) -> Result<(), String> {
    let max_log = components
        .iter()
        .map(|c| c.max_constraint_log_degree_bound())
        .max()
        .unwrap_or(1);
    let point = CirclePoint::<SecureField>::zero();
    let pts = TreeVec::concat_cols(components.iter().map(|c| c.mask_points(point, max_log)));
    check_mask_points(tie, &pts, point)
}

/// `check_mask_offsets` against a sampling already built, so a control can hand
/// it the two offset orders and see which of them is refused.
pub fn check_mask_points(
    tie: &LinkTie,
    pts: &TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>>,
    point: CirclePoint<SecureField>,
) -> Result<(), String> {
    let at = |what: String, tree: usize, col: usize| -> Result<(), String> {
        if tree < crate::common::BASE_TREE {
            return Ok(());
        }
        let ps = pts.get(tree).and_then(|t| t.get(col)).ok_or_else(|| format!(
            "{what}: tree {tree} column {col} is sampled by no component, so the link \
             would read a value the proof never opened"))?;
        match ps.first() {
            Some(p) if *p == point => Ok(()),
            Some(_) => Err(format!(
                "{what}: tree {tree} column {col} is sampled at {} offsets and offset 0 is \
                 not the first of them, so the link would read a neighbouring row where its \
                 weight is this row's -- the owning evaluator has to read this column at \
                 offset 0 first",
                ps.len())),
            None => Err(format!("{what}: tree {tree} column {col} is sampled at no offset")),
        }
    };
    for t in &tie.heights {
        for (tree, col, _) in &t.terms {
            at(format!("height {}", t.h), *tree, *col)?;
        }
        // the two committed quadruples this height's bindings read, on the same
        // rule and by the same indexing
        for (what, c0) in [("the carry quotient", t.q_at), ("the coefficient", t.coeff_at)] {
            for k in 0..4 {
                at(format!("height {}, {what}", t.h), INTERACTION_TRACE_IDX, c0 + k)?;
            }
        }
    }
    Ok(())
}

/// Placeholder link parameters for the emit path: the heights are the plan's
/// own, the scalars distinct nonzero pseudo-random values. The blob replays
/// the real scalars at prove time; distinctness keeps `lit` from folding
/// slots away, exactly as `fill_binding_claims` does for the residual-stream bindings.
pub fn link_for(plan: &crate::compose::Plan, seed: u64) -> Vec<HeightLink> {
    let mut x = seed | 1;
    let mut felt = || {
        let mut a = [0u32; 4];
        for l in a.iter_mut() {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *l = ((x % 0x7fff_fffe) + 1) as u32;
        }
        SecureField::from_m31_array(a.map(M31::from_u32_unchecked))
    };
    link_heights(plan)
        .into_iter()
        .map(|h| {
            let point: Vec<SecureField> = (0..h).map(|_| felt()).collect();
            let claim = felt();
            HeightLink {
                h,
                mle_point: MleEvalPoint::new(&point),
                point,
                claim,
                shift: claim / BaseField::from(1u32 << h),
                weights: vec![],
                konst: felt(),
            }
        })
        .collect()
}

/// The eq-weighted total of the folded coefficient column -- what the honest
/// claim equals. The `ZK_MUT_LINK` control bends one weight and then makes its
/// own columns self-consistent with this total, so everything the prover can
/// check still holds and only the verifier's side of the link can reject.
pub fn eq_total(hl: &HeightLink, coeff: &SecureColumn) -> SecureField {
    let n_vec = 1usize << (hl.h - LOG_N_LANES);
    let eq_mle = SimdBackend::gen_eq_evals(&hl.point, SecureField::one());
    let mut acc = PackedSecureField::broadcast(SecureField::zero());
    for vr in 0..n_vec {
        acc += eq_mle.data[vr] * coeff.data[vr];
    }
    acc.to_array().iter().fold(SecureField::zero(), |a, b| a + *b)
}

/// `ZK_LINK_CHECK=1`: evaluate the height's constraints pointwise on the CPU
/// and name the first violated relation. A development probe for the composed
/// run -- the constraints here are the same expressions `LinkEval` hands the
/// framework, walked in coset order over the committed values.
pub fn check_height_cols(hl: &HeightLink, cols: &[Eval]) {
    if std::env::var("ZK_LINK_CHECK").map(|v| v != "1").unwrap_or(true) {
        return;
    }
    let h = hl.h;
    let n = 1usize << h;
    let at = |group: usize, coset_i: usize| -> SecureField {
        let s = bit_reverse_index(coset_index_to_circle_domain_index(coset_i, h), h);
        let (vr, lane) = (s / 16, s % 16);
        SecureField::from_m31_array([0, 1, 2, 3].map(|k| {
            cols[group * 4 + k].data[vr].to_array()[lane]
        }))
    };
    let (q, coeff, eqc, prefix) = (0usize, 1usize, 2usize, 3usize);
    // the eq boundaries
    if at(eqc, 0) != hl.mle_point.eq_0_p {
        eprintln!("linkcheck 2^{h}: eq[coset 0] = {:?}, want eq_0_p {:?}",
                  at(eqc, 0), hl.mle_point.eq_0_p);
    }
    if at(eqc, 1) != hl.mle_point.eq_1_p {
        eprintln!("linkcheck 2^{h}: eq[coset 1] = {:?}, want eq_1_p {:?}",
                  at(eqc, 1), hl.mle_point.eq_1_p);
    }
    // the eq steps: curr == next_next * q(curr), wrapping mod the domain
    let mut bad_step = 0usize;
    for i in 0..n {
        let expect = at(eqc, (i + 2) % n) * at(q, i);
        if at(eqc, i) != expect && bad_step < 3 {
            eprintln!("linkcheck 2^{h}: eq step at coset {i}: {:?} != next_next*q {:?}",
                      at(eqc, i), expect);
            bad_step += 1;
        }
    }
    // the prefix recurrence: curr - prev == coeff*eq - shift, wrapping
    let mut bad_pre = 0usize;
    for i in 0..n {
        let prev = at(prefix, (i + n - 1) % n);
        let diff = at(prefix, i) - prev;
        let want = at(coeff, i) * at(eqc, i) - hl.shift;
        if diff != want && bad_pre < 3 {
            eprintln!("linkcheck 2^{h}: prefix at coset {i}: diff {:?} != term {:?}",
                      diff, want);
            bad_pre += 1;
        }
    }
    eprintln!("linkcheck 2^{h}: done ({bad_step} step, {bad_pre} prefix violations shown)");
}

// ------------------------------------------------------------------------
// B11 stream U, lever 1: the link's committed columns on the card.
//
// Stream T measured the five host loops this section replaces -- the carry
// quotient at 0.078 s, the eq column at 0.041 s, the coordinate split at
// 0.032 s, the prefix sums at 0.024 s and the term products at 0.014 s, so
// 0.190 s per C=16384 chunk -- and named them the backend's single largest
// remaining item, host work in a regime where every other interaction column
// is built on the card. Nothing about *what* is computed changes here: the
// same values in the same slots, which is why every one of T's digests is a
// reproduction gate for this lever rather than a number to re-record.
// ------------------------------------------------------------------------

/// Where the link's own columns are built.
///
/// On the cuda backend the whole chain runs on the card and the sixteen columns per
/// height reach the commitment device-resident, the way the key tiles' fraction
/// columns already do. `ZK_DEVICE_LINK=0` puts the build back on the host: that is
/// the control the stream's own Δ is measured against, the same shape as T's
/// `ZK_GKR_FUSE=0`.
#[cfg(feature = "cuda")]
pub fn dev_link() -> bool {
    std::env::var("ZK_DEVICE_LINK").map(|v| v != "0").unwrap_or(true)
}

/// One height's sixteen committed columns, built wherever the prover can build
/// them, ready for the interaction tree.
pub fn height_cols<'a, F: Fn(Comp, &RCol) -> Src<'a>>(
    hl: &HeightLink,
    src_of: &F,
) -> Vec<crate::device::AnyCol<'static>> {
    #[cfg(feature = "cuda")]
    if dev_link() {
        let cols = dev::height_cols(hl, src_of);
        dev::col_check(hl, src_of, &cols);
        return cols;
    }
    let t = std::time::Instant::now();
    let coeff = fold_coeff(hl, src_of);
    crate::util::add_phase("int/link_fold", t.elapsed().as_secs_f64());
    let t = std::time::Instant::now();
    let cols = build_height_cols(hl, coeff);
    crate::util::add_phase("int/link_cols", t.elapsed().as_secs_f64());
    crate::device::host_cols(cols)
}

/// The eq-weighted total of one height's folded coefficient column, taken
/// wherever the fold ran. This is what `ZK_MUT_LINK` re-derives its claim from,
/// so it has to exist on both paths or the control would only be armed on one.
pub fn fold_total<'a, F: Fn(Comp, &RCol) -> Src<'a>>(
    hl: &HeightLink,
    src_of: &F,
) -> SecureField {
    #[cfg(feature = "cuda")]
    if dev_link() {
        return dev::fold_total(hl, src_of);
    }
    eq_total(hl, &fold_coeff(hl, src_of))
}

/// The link's column build on the card.
///
/// Five kernels in `nitroo/.../cuda/gkr.cu` beside `zk_link_coeff`, one per
/// host loop, each reproducing its loop value for value. The intermediate
/// columns -- the folded coefficients, the eq evaluations, the shifted
/// products -- never come home; only the claim's own total does, as the block
/// partials of the products, because the prover asserts that boundary loudly.
#[cfg(feature = "cuda")]
pub mod dev {
    use super::*;
    use crate::device::AnyCol;
    use stwo::prover::backend::cuda::CudaBackend;
    use stwo::stwo_cuda::base_field_vec::BaseFieldVec;
    use stwo::stwo_cuda::secure_field_vec::SecureFieldVec;

    unsafe extern "C" {
        fn zk_link_coeff(cols: *const *const u32, w_host: *const u32, n_cols: u32,
                          k: [u32; 4], out: *const u32, n_rows: u32);
        fn zk_link_split(src: *const u32, o0: *const u32, o1: *const u32,
                          o2: *const u32, o3: *const u32, n_rows: u32);
        fn zk_link_terms(coeff: *const u32, eq: *const u32, shift: [u32; 4],
                          out: *const u32, partials: *mut u32, n_blocks: u32,
                          n_rows: u32);
        fn zk_link_carry(half0: *const u32, half1: *const u32, n_vars: u32,
                          log_size: u32, o0: *const u32, o1: *const u32,
                          o2: *const u32, o3: *const u32);
        fn zk_link_prefix(src: *const u32, o0: *const u32, o1: *const u32,
                           o2: *const u32, o3: *const u32, log_size: u32);
    }

    fn sync() {
        unsafe { stwo::stwo_cuda::bindings::gkr_device_synchronize() };
    }

    /// Four fresh device columns of one height -- one coordinate each.
    fn quad(n: usize) -> [BaseFieldVec; 4] {
        std::array::from_fn(|_| BaseFieldVec::new_uninitialized(n))
    }

    fn flat(v: &[SecureField]) -> Vec<u32> {
        v.iter().flat_map(|x| x.to_m31_array().map(|c| c.0)).collect()
    }

    /// The fold, with the coefficient column left where it is built.
    ///
    /// T's `fold_dev` combined the device-resident terms on the card and then
    /// downloaded the column so the host loops could read it, adding the
    /// host-resident terms on the way. There is nothing left on the host to
    /// read it, so the host-resident terms -- the preprocessed causal masks,
    /// and every base column of a host-built key tile -- are uploaded instead and
    /// the whole affine combine is one launch. The accumulation order is the
    /// weight order either way and field addition is exact, so it is the same
    /// column.
    fn fold<'a, F: Fn(Comp, &RCol) -> Src<'a>>(hl: &HeightLink, src_of: &F)
                                                -> SecureFieldVec {
        let n = 1usize << hl.h;
        let mut keep: Vec<BaseFieldVec> = vec![];
        let mut ptrs: Vec<*const u32> = vec![];
        let mut ws: Vec<u32> = vec![];
        for (reg, c, w) in &hl.weights {
            let p = match src_of(*reg, c) {
                Src::Host(e) => {
                    assert_eq!(e.len(), n, "column height disagrees with the fold");
                    let d = crate::device::upload_col(e);
                    let p = d.device_ptr;
                    keep.push(d);
                    p
                }
                Src::Dev(p) => p,
            };
            ptrs.push(p);
            ws.extend(w.to_m31_array().map(|x| x.0));
        }
        let out = SecureFieldVec::new_uninitialized(n);
        unsafe {
            zk_link_coeff(ptrs.as_ptr(), ws.as_ptr(), ptrs.len() as u32,
                           hl.konst.to_m31_array().map(|x| x.0),
                           out.device_ptr, n as u32);
        }
        drop(keep);
        out
    }

    /// `coeff * eq - shift` into `out`, and the unshifted products' total.
    /// A `None` output asks for the total alone, which is what the
    /// `ZK_MUT_LINK` control needs of a bent column.
    fn terms_of(coeff: &SecureFieldVec, eq: &SecureFieldVec, shift: SecureField,
                out: Option<&SecureFieldVec>, n_rows: usize) -> SecureField {
        let n_blocks = n_rows.div_ceil(256).clamp(1, 1024);
        let mut part = vec![0u32; 4 * n_blocks];
        unsafe {
            zk_link_terms(coeff.device_ptr, eq.device_ptr,
                           shift.to_m31_array().map(|x| x.0),
                           out.map(|o| o.device_ptr).unwrap_or(std::ptr::null()),
                           part.as_mut_ptr(), n_blocks as u32, n_rows as u32);
        }
        part.chunks(4).fold(SecureField::zero(), |a, c| {
            a + SecureField::from_m31_array(
                [0usize, 1, 2, 3].map(|i| M31::from_u32_unchecked(c[i])))
        })
    }

    fn eq_col(hl: &HeightLink) -> SecureFieldVec {
        CudaBackend::gen_eq_evals(&hl.point, SecureField::one()).into_evals()
    }

    pub fn fold_total<'a, F: Fn(Comp, &RCol) -> Src<'a>>(hl: &HeightLink, src_of: &F)
                                                          -> SecureField {
        let n = 1usize << hl.h;
        let coeff = fold(hl, src_of);
        let eq = eq_col(hl);
        terms_of(&coeff, &eq, hl.shift, None, n)
    }

    pub fn height_cols<'a, F: Fn(Comp, &RCol) -> Src<'a>>(hl: &HeightLink, src_of: &F)
                                                           -> Vec<AnyCol<'static>> {
        let h = hl.h;
        let n = 1usize << h;

        let t = std::time::Instant::now();
        let coeff = fold(hl, src_of);
        sync();
        crate::util::add_phase("int/link_fold", t.elapsed().as_secs_f64());

        let cols_t = std::time::Instant::now();
        let t = std::time::Instant::now();
        let eq = eq_col(hl);
        sync();
        crate::util::add_phase("int/link_eq", t.elapsed().as_secs_f64());

        // the products, and the prefix sum's boundary asserted where the host
        // builder asserts it: a weight bug fails the prover, not just the
        // verifier
        let t = std::time::Instant::now();
        let terms = SecureFieldVec::new_uninitialized(n);
        let total = terms_of(&coeff, &eq, hl.shift, Some(&terms), n);
        assert_eq!(total, hl.claim,
                   "the folded coefficient column disagrees with the GKR claim at height {h}");
        crate::util::add_phase("int/link_terms", t.elapsed().as_secs_f64());

        let t = std::time::Instant::now();
        let prefix = quad(n);
        unsafe {
            zk_link_prefix(terms.device_ptr, prefix[0].device_ptr, prefix[1].device_ptr,
                            prefix[2].device_ptr, prefix[3].device_ptr, h);
        }
        sync();
        drop(terms);
        crate::util::add_phase("int/link_prefix", t.elapsed().as_secs_f64());

        let t = std::time::Instant::now();
        let (half0, half1) = gen_half_coset_carry_quotients(&hl.mle_point);
        let (f0, f1) = (flat(&half0), flat(&half1));
        let q = quad(n);
        unsafe {
            zk_link_carry(f0.as_ptr(), f1.as_ptr(), half0.len() as u32, h,
                           q[0].device_ptr, q[1].device_ptr, q[2].device_ptr,
                           q[3].device_ptr);
        }
        sync();
        crate::util::add_phase("int/link_q", t.elapsed().as_secs_f64());

        let t = std::time::Instant::now();
        let coeff4 = quad(n);
        let eq4 = quad(n);
        unsafe {
            zk_link_split(coeff.device_ptr, coeff4[0].device_ptr, coeff4[1].device_ptr,
                           coeff4[2].device_ptr, coeff4[3].device_ptr, n as u32);
            zk_link_split(eq.device_ptr, eq4[0].device_ptr, eq4[1].device_ptr,
                           eq4[2].device_ptr, eq4[3].device_ptr, n as u32);
        }
        sync();
        drop(coeff);
        drop(eq);
        crate::util::add_phase("int/link_split", t.elapsed().as_secs_f64());

        crate::util::add_phase("int/link_cols", cols_t.elapsed().as_secs_f64());
        let mut out: Vec<AnyCol<'static>> = Vec::with_capacity(16);
        for group in [q, coeff4, eq4, prefix] {
            out.extend(group.into_iter().map(|c| AnyCol::OwnedDev(c, h)));
        }
        out
    }

    /// `ZK_LINK_COLUMN_CHECK=1`: the card's sixteen columns against the host
    /// builder's, cell for cell -- stream T's `ZK_GKR_FEED_CHECK` for this lever.
    /// It pays a download and a second, host build of the same height, so it is
    /// a control and never a measurement.
    pub fn col_check<'a, F: Fn(Comp, &RCol) -> Src<'a>>(
        hl: &HeightLink, src_of: &F, cols: &[AnyCol<'static>],
    ) {
        if std::env::var("ZK_LINK_COLUMN_CHECK").map(|v| v != "1").unwrap_or(true) {
            return;
        }
        let want = build_height_cols(hl, fold_coeff(hl, src_of));
        assert_eq!(want.len(), cols.len(), "the two link builds differ in column count");
        let mut bad = 0usize;
        for (i, (w, g)) in want.iter().zip(cols).enumerate() {
            let got = match g {
                AnyCol::OwnedDev(d, _) => d.to_vec(),
                _ => panic!("the link's device build handed back a host column"),
            };
            let host = w.values.as_slice();
            assert_eq!(host.len(), got.len(), "link column {i} differs in length");
            for (r, (a, b)) in host.iter().zip(&got).enumerate() {
                if a != b && bad < 8 {
                    eprintln!("linkcolcheck 2^{}: column {i} row {r}: host {} device {}",
                              hl.h, a.0, b.0);
                    bad += 1;
                }
            }
        }
        eprintln!("linkcolcheck 2^{}: {} columns, {} mismatching cells shown",
                  hl.h, cols.len(), bad);
        assert_eq!(bad, 0, "the link's device columns disagree with the host builder");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One height naming one base column, and the two committed quadruples at
    /// the front of the interaction tree.
    fn tie_naming(col: usize) -> LinkTie {
        LinkTie {
            heights: vec![TieHeight {
                h: 4,
                // any point whose coordinates are neither 0 nor 1: the eq
                // carry quotients divide by `eq(0, p)` and `eq(1, p)`
                mle_point: MleEvalPoint::new(
                    &std::array::from_fn::<_, 4, _>(|i| {
                        SecureField::from(M31::from_u32_unchecked(i as u32 + 2))
                    })),
                konst: SecureField::zero(),
                terms: vec![(crate::common::BASE_TREE, col, SecureField::one())],
                q_at: 0,
                coeff_at: 4,
            }],
            pre_idx: vec![],
        }
    }

    /// What the framework samples for a column read at these offsets, in the
    /// order it reads them -- `FrameworkComponent::mask_points`, by hand.
    fn sampling(max_log: u32, base: &[&[isize]])
                -> (CirclePoint<SecureField>, TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>>) {
        let point = CirclePoint::<SecureField>::zero();
        let step = CanonicCoset::new(max_log).step();
        let of = |offs: &[isize]| -> Vec<CirclePoint<SecureField>> {
            offs.iter().map(|o| point + step.mul_signed(*o).into_ef()).collect()
        };
        let mut trees: Vec<ColumnVec<Vec<CirclePoint<SecureField>>>> = vec![vec![]; 4];
        trees[crate::common::BASE_TREE] = base.iter().map(|o| of(o)).collect();
        // the eight interaction columns the two quadruples take, all at offset 0
        trees[INTERACTION_TRACE_IDX] = (0..8).map(|_| of(&[0])).collect();
        (point, TreeVec::new(trees))
    }

    /// Open question 7.3. The attention output reads its two masked base
    /// columns in opposite orders on purpose: `den` at `[-1, 0]`, because
    /// nothing links it, and `joff` at `[0, -1]`, because the link reads a
    /// named column at its first sampled offset. Naming the first is the
    /// control; naming the second is the shape the prover actually builds.
    #[test]
    fn a_linked_column_read_at_the_previous_row_first_is_refused() {
        let (point, pts) = sampling(6, &[&[-1, 0], &[0, -1], &[0]]);

        // column 1 is `joff`'s order, and column 2 a column read at this row
        // and nowhere else, which is every other linked column there is
        for ok in [1usize, 2] {
            assert_eq!(check_mask_points(&tie_naming(ok), &pts, point), Ok(()),
                       "column {ok} is read at offset 0 first, so the link may name it");
        }

        // column 0 is `den`'s order: the link's weight is this row's and the
        // value it would multiply is the row before
        let e = check_mask_points(&tie_naming(0), &pts, point)
            .expect_err("a column whose first sampled offset is not 0 must be refused");
        assert!(e.contains("offset 0 is not the first"), "the refusal must say why: {e}");
        assert!(e.contains("column 0"), "the refusal must name the column: {e}");

        // and a column no component sampled at all is refused rather than
        // indexed into
        let e = check_mask_points(&tie_naming(3), &pts, point)
            .expect_err("a column nothing sampled must be refused");
        assert!(e.contains("sampled by no component"), "{e}");
    }
}
