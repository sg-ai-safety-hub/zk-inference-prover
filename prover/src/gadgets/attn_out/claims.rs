//! What the attnout component publishes, and its LogUp columns.

use num_traits::{One, Zero};
use rayon::prelude::*;
use stwo::core::fields::FieldExpOps;
use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::SecureField;

use crate::encoding::m31_of;
use crate::gadgets::attn_out::trace::AoComponent;
use crate::gadgets::attn_out::witness::AttnOut;

pub struct AoInteraction {
    pub cols: Vec<crate::device::AnyCol<'static>>,
    pub claimed_out: SecureField,
    pub claimed_den: SecureField,
    /// Stream JJ (E11): the component's own requantized output, folded under
    /// `claimed_out`'s weights -- what the `o` Freivalds check's `qX` equals.
    pub claimed_q: SecureField,
    pub use_sum: SecureField,
}

#[allow(clippy::too_many_arguments)]
pub fn ao_interaction(ao: &AttnOut, r: &AoComponent, rr: SecureField, s_ch: SecureField,
                      rv: SecureField, c_ctx: usize, rels: &crate::compose::Rels,
                      air: &crate::gadgets::attn_out::air::AttnOutEval, gkr: bool) -> AoInteraction {
    use stwo::prover::backend::simd::qm31::PackedQM31;
    use stwo_constraint_framework::Relation;
    let lay = r.lay;
    let rows = 1usize << lay.log_rows;
    let pr = lay.per_row;
    let (rv_pow, rv_step) = crate::relations::powers(rv, pr);
    let (_, rv_hd) = crate::relations::powers(rv, ao.hd);
    let (_, r_c) = crate::relations::powers(rr, c_ctx);
    let (_, s_span) = crate::relations::powers(s_ch, ao.s - 1);
    let mu_out = rv_hd * s_span.inverse();
    let mu_den = r_c * s_span.inverse();

    // B7a milestone 4 (section 3 row 1): the attnout's z, h and dz, on the same
    // three-value mode every witness generation in this series uses.
    let acc_dev = r.dev.as_ref().filter(|_| crate::devacc::mode() >= 1);
    let acc_host = acc_dev.is_none() || crate::devacc::mode() == 2;
    // stream JJ's fourth accumulator is the host's whatever the device witness generation
    // did with the other three (stream M's lesson in `mat_interaction`: keep
    // the real height before `rows` is shadowed to zero)
    let n_rows = rows;
    let rows = if acc_host { rows } else { 0 };
    let inner: Vec<SecureField> = (0..rows)
        .into_par_iter()
        .map(|j| {
            let mut zf = SecureField::zero();
            if let Some((h, t, b)) = lay.at(ao.nh, ao.s, j) {
                for s in 0..pr {
                    let a = ao.acc[(h * ao.s + t) * ao.hd + b * pr + s];
                    if a != 0 {
                        zf += rv_pow[s] * SecureField::from(m31_of(a));
                    }
                }
            }
            zf
        })
        .collect();
    let (mut z, mut h, mut dz) = (vec![SecureField::zero(); rows],
                                  vec![SecureField::zero(); rows],
                                  vec![SecureField::zero(); rows]);
    for j in 0..rows {
        let nofirst = matches!(lay.at(ao.nh, ao.s, j), Some((_, _, b)) if b + 1 < lay.chunks);
        let last = matches!(lay.at(ao.nh, ao.s, j), Some((_, _, 0)));
        let hfirst = matches!(lay.at(ao.nh, ao.s, j), Some((_, t, 0)) if t == ao.s - 1);
        z[j] = if nofirst { z[j - 1] * rv_step } else { SecureField::zero() } + inner[j];
        let (mo, md) = if j == 0 {
            (SecureField::zero(), SecureField::zero())
        } else if last && hfirst {
            (mu_out, mu_den)
        } else if last {
            (s_ch, s_ch)
        } else {
            (SecureField::one(), SecureField::one())
        };
        h[j] = if j > 0 { h[j - 1] * mo } else { SecureField::zero() };
        dz[j] = if j > 0 { dz[j - 1] * md } else { SecureField::zero() };
        if last {
            h[j] += z[j];
            if let Some((hh, t, _)) = lay.at(ao.nh, ao.s, j) {
                dz[j] += SecureField::from(m31_of(ao.den[hh * ao.s + t]));
            }
        }
    }
    let mut cols = crate::device::ColSink::new();
    let (claimed_out, claimed_den) = match acc_dev {
        Some(d) => {
            let got = crate::devacc::ao(d, &lay, ao.nh, ao.s, rv, s_ch, mu_out, mu_den);
            let hh = acc_host.then(|| {
                let mut e = crate::encoding::secure_cols_pub(lay.log_rows, &z);
                e.extend(crate::encoding::secure_cols_pub(lay.log_rows, &h));
                e.extend(crate::encoding::secure_cols_pub(lay.log_rows, &dz));
                (e, vec![h[rows - 1], dz[rows - 1]])
            });
            let c = crate::devacc::place(&format!("attnout 2^{}", lay.log_rows), got, hh,
                                         &mut cols);
            (c[0], c[1])
        }
        None => {
            cols.extend(crate::encoding::secure_cols_pub(lay.log_rows, &z));
            cols.extend(crate::encoding::secure_cols_pub(lay.log_rows, &h));
            cols.extend(crate::encoding::secure_cols_pub(lay.log_rows, &dz));
            (h[rows - 1], dz[rows - 1])
        }
    };

    // ---- stream JJ (E11): the component's own output, published --------------
    //
    // `z`/`h`'s shape exactly, over `q = qm (1 - 2 sg)` instead of over `acc`.
    // `AttnOutEval` *is* the requantization of the attention output, so this
    // claim is what the `o` Freivalds check's committed activations have to equal, row for
    // row -- `q_out` below is the same value this folds.  The `eqs` entry that
    // compares them is stream II's, on the consumer's side.
    let (mut zq, mut hq) = (vec![SecureField::zero(); n_rows],
                            vec![SecureField::zero(); n_rows]);
    for j in 0..n_rows {
        let nofirst = matches!(lay.at(ao.nh, ao.s, j), Some((_, _, b)) if b + 1 < lay.chunks);
        let last = matches!(lay.at(ao.nh, ao.s, j), Some((_, _, 0)));
        let hfirst = matches!(lay.at(ao.nh, ao.s, j), Some((_, t, 0)) if t == ao.s - 1);
        let mut qf = SecureField::zero();
        if let Some((h, t, b)) = lay.at(ao.nh, ao.s, j) {
            for s in 0..pr {
                let q = r.q_out[(h * ao.s + t) * ao.hd + b * pr + s];
                if q != 0 {
                    qf += rv_pow[s] * SecureField::from(m31_of(q));
                }
            }
        }
        zq[j] = if nofirst { zq[j - 1] * rv_step } else { SecureField::zero() } + qf;
        let mq = if j == 0 {
            SecureField::zero()
        } else if last && hfirst {
            mu_out
        } else if last {
            s_ch
        } else {
            SecureField::one()
        };
        hq[j] = if j > 0 { hq[j - 1] * mq } else { SecureField::zero() };
        if last {
            hq[j] += zq[j];
        }
    }
    let claimed_q = hq[n_rows - 1];
    cols.extend_any(crate::encoding::secure_cols_pub(lay.log_rows, &zq)
                        .into_iter().map(crate::device::AnyCol::OwnedHost).collect::<Vec<_>>());
    cols.extend_any(crate::encoding::secure_cols_pub(lay.log_rows, &hq)
                        .into_iter().map(crate::device::AnyCol::OwnedHost).collect::<Vec<_>>());

    // lookups in AIR order, per slot: the `EPOW` window, (qm, nz2), d, dp,
    // denp, the offset accumulator's low limb, then its high limb, and in the
    // product form `ahn = ahi num` last
    let one = PackedQM31::broadcast(SecureField::one());
    let lps = r.looks_per_slot();
    let n_looks = lps * pr;
    let inv_lo = stwo::prover::backend::simd::m31::PackedM31::broadcast(
        M31::from_u32_unchecked(1 << r.acc_lo_bits).inverse());
    let acc_off = stwo::prover::backend::simd::m31::PackedM31::broadcast(
        M31::from_u32_unchecked(r.acc_off() as u32));
    let ncs = r.per_slot();
    let fr = |i: usize, vr: usize| -> (PackedQM31, PackedQM31) {
        let (s, which) = (i / lps, i % lps);
        let c0 = 2 + ncs * s;
        match which {
            // the window: `(jvp + joff, num, pj)`, `jvp` preprocessed and
            // model-only, `joff` the row's committed output exponent
            0 => (one, rels.epow.combine(&[r.pre[s].data[vr] + r.base[1].data[vr],
                                           r.base[c0 + 6].data[vr],
                                           r.base[c0 + 7].data[vr]])),
            1 => (one, rels.qnz.combine(&[r.base[c0 + 2].data[vr], r.base[c0 + 3].data[vr]])),
            2 => (one, rels.range(r.rem_bits).combine(&[r.base[c0 + 5].data[vr]])),
            3 => {
                let two = stwo::prover::backend::simd::m31::PackedM31::broadcast(
                    M31::from_u32_unchecked(2));
                let onem = stwo::prover::backend::simd::m31::PackedM31::broadcast(M31::one());
                let dp = r.base[c0 + 4].data[vr] * two - onem - r.base[c0 + 5].data[vr];
                (one, rels.range(r.rem_bits).combine(&[dp]))
            }
            // section 7 item 1: `denp` itself
            4 => {
                // stream TT: the `denp2` form looks up `2 denp`, which states
                // the width an honest `denp` has
                let v = r.base[c0 + 4].data[vr];
                let v = if r.denp2 {
                    v * stwo::prover::backend::simd::m31::PackedM31::broadcast(
                        M31::from_u32_unchecked(2))
                } else { v };
                (one, rels.range(r.rem_bits).combine(&[v]))
            }
            // the no-wrap window's two limbs on `acc + acc_off` -- and, in
            // stream TT's magnitude form, on the committed `|acc|` at `c0 + 9`
            5 => (one, rels.range(r.acc_lo_bits).combine(&[r.base[c0 + 8].data[vr]])),
            // the product form's per-slot bound
            7 => (one, rels.range(r.acc_prod.expect("an eighth look is the product form's"))
                           .combine(&[r.base[c0 + 10].data[vr]])),
            _ => {
                let v = if r.acc_mag { r.base[c0 + 9].data[vr] }
                        else { r.base[c0].data[vr] + acc_off };
                let hi = (v - r.base[c0 + 8].data[vr]) * inv_lo;
                (one, rels.range(r.acc_hi_bits).combine(&[hi]))
            }
        }
    };
    // B11 stream V, lever 4: as the Freivalds checks, and for the same reason. The `z`,
    // `h` and `dz` accumulators stay.
    if gkr {
        let _ = fr;
        return AoInteraction {
            cols: cols.into_inner(),
            claimed_out,
            claimed_den,
            claimed_q,
            use_sum: SecureField::zero(),
        };
    }
    let (lcols, use_sum) = crate::devfrac::frac_cols(
        air, "attnout", lay.log_rows,
        crate::devfrac::cols_of(&r.pre, r.dev.as_ref().map(|d| &d.pre[..])),
        crate::devfrac::cols_of(&r.base, r.dev.as_ref().map(|d| &d.base[..])),
        || {
            let (n_pairs, fpair) = crate::plogup::paired(n_looks, fr);
            crate::plogup::build(lay.log_rows, n_pairs, fpair)
        });
    cols.extend_any(lcols);
    AoInteraction { cols: cols.into_inner(), claimed_out, claimed_den, claimed_q, use_sum }
}
