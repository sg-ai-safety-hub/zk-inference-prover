//! The attnout component's committed columns.

use num_traits::{One, Zero};
use stwo::core::fields::m31::M31;

use crate::common::{Eval, eval_of, m31_u32};
use crate::device_witness::sub_slice;
use crate::encoding::m31_of;
use crate::gadgets::attn_out::layout::AoLayout;
use crate::gadgets::attn_out::witness::AttnOut;
use crate::layer::{Layer, Tables};

pub struct AoComponent {
    pub lay: AoLayout,
    /// as `FreivaldsComponent`'s: the mirror under `ZK_DEVICE_WITNESS_GEN_ATTENTION_FREIVALDS=1`
    pub pre: Vec<Eval>,
    pub base: Vec<Eval>,
    /// B7a milestone 4 (section 3 row 1): the same columns, born on the device.
    pub dev: Option<crate::gadgets::attn_out::device::DevAo>,
    pub ref_bad: usize,
    /// the committed int8 outputs, for cross-checks
    pub q_out: Vec<i64>,
    pub n: usize,
    pub rem_bits: u32,
    pub q_bits: u32,
    /// stream JJ: the two limb widths of the range check on the accumulator,
    /// and the offset that makes it non-negative
    pub acc_lo_bits: u32,
    pub acc_hi_bits: u32,
    /// stream TT (log row 73): the magnitude form -- the slot commits
    /// `mag = |acc|` as a tenth column and the two limbs range-check *it*, so
    /// `acc_off` is not used and the widths cover one bit less
    pub acc_mag: bool,
    /// the product form: the width of the `ahn = ahi num` range table, and
    /// with it an eleventh column per slot; `None` in the other two forms
    pub acc_prod: Option<u32>,
    /// stream TT (log row 73): the sibling bound's half -- `2 denp` is looked
    /// up in the `rem_bits` table instead of `denp`, which is the width an
    /// honest `denp` has and one bit better than the plain form states
    pub denp2: bool,
}

impl AoComponent {
    /// `2^(acc_lo_bits + acc_hi_bits - 1)`: what a committed `acc` is shifted by
    /// so its two limbs are a range check on `|acc|`.  Zero in the magnitude
    /// form, where the limbs read the committed magnitude directly.
    pub fn acc_off(&self) -> i64 {
        if self.acc_mag { 0 } else { 1i64 << (self.acc_lo_bits + self.acc_hi_bits - 1) }
    }
    /// Columns per channel slot: `acc, sg, qm, nz2, denp, d, num, pj, alo`, in
    /// the magnitude form `mag` after them, and in the product form `ahn` last.
    pub fn per_slot(&self) -> usize {
        slot_cols(self.acc_mag, self.acc_prod.is_some())
    }
    /// Lookups per channel slot, in AIR order: the `EPOW` window, `(qm, nz2)`,
    /// `d`, `dp`, `denp`, the two limbs, and in the product form `ahn`.
    pub fn looks_per_slot(&self) -> usize {
        slot_looks(self.acc_prod.is_some())
    }
}

/// `AoComponent::per_slot`, from the two form flags alone.
pub fn slot_cols(acc_mag: bool, acc_prod: bool) -> usize {
    9 + usize::from(acc_mag) + usize::from(acc_prod)
}

/// `AoComponent::looks_per_slot`, from the form flag alone, for the
/// verifier's site list, which has no component.
pub fn slot_looks(acc_prod: bool) -> usize {
    7 + usize::from(acc_prod)
}

impl AoComponent {
    /// Committed cells, wherever the columns are.
    pub fn base_cells(&self) -> usize {
        let n = match &self.dev { Some(d) => d.base.len(), None => self.base.len() };
        n << self.lay.log_rows
    }
}

/// Column order: the row's `den` and `joff`, then per slot acc, sg, qm, nz2,
/// denp, d, num, pj, alo -- and, in stream TT's magnitude form, `mag` tenth.
///
/// Stream JJ (E11), `witness-slack.md` section 1f: `joff = jo_max - jo` is the
/// per-token output exponent, committed because `jo` is prover data and pinned
/// across the token's rows the way `den` is.  `num = 2^max(e,0)` and
/// `pj = 2^max(-e,0)` with `e = jv - jo` are committed too and pinned by one
/// lookup per slot into the preprocessed `EPOW` window table at the key
/// `jvp + joff`, where `jvp = jv - jv_min` is preprocessed and model-only.
/// `alo` is the low limb of the offset accumulator, the other half of the
/// no-wrap window.
///
/// Stream TT (E11), log row 73: under `acc_mag` the slot carries a tenth
/// column `mag = |acc|` and `alo` is *its* low limb, not the offset
/// accumulator's.  `acc` stays where it was, so the `z` fold, the device
/// accumulator and `gkrlink`'s binding read the same column they always did.
#[allow(clippy::too_many_arguments)]
pub fn ao_component(ao: &AttnOut, per_row_target: usize, q_bits: u32, rem_bits: u32,
                 acc_bits: (u32, u32), acc_mag: bool, acc_prod: Option<u32>, denp2: bool,
                 jv_min: i32, jo_max: i32,
                 tables: &mut Tables, corrupt: Option<usize>,
                 corrupt_q: Option<usize>, corrupt_den: Option<usize>,
                 corrupt_jo: Option<usize>, corrupt_num: Option<usize>,
                 corrupt_acc: Option<usize>, corrupt_jo_tok: Option<usize>,
                 corrupt_mag: Option<usize>, corrupt_ahn: Option<usize>) -> AoComponent {
    let lay = AoLayout::choose(ao.nh, ao.s, ao.hd, per_row_target);
    let rows = 1usize << lay.log_rows;
    let pr = lay.per_row;
    // B7a milestone 4 (section 3 row 1): the same three-value mode.
    //
    // Stream TT: the magnitude form generates its witness on the host on both backends.  The
    // fork's `zk_witness_gen_attnout` writes the nine-column layout and bumps the
    // limbs at `acc + acc_off`; it has no tenth column and no `mag`, so a
    // device witness generation here would write the wrong component rather than a slow one.
    // `compose::note_host_witness_gen` knows, which is what keeps the shared
    // multiplicity columns on the host with it.
    let witness_gen_mode = if corrupt_q.is_some() || corrupt_den.is_some() || corrupt_jo.is_some()
                             || corrupt_num.is_some() || corrupt_acc.is_some()
                             || corrupt_jo_tok.is_some() || corrupt_mag.is_some()
                             || corrupt_ahn.is_some() || acc_mag || denp2 {
        0
    } else {
        crate::util::path_mode("ZK_DEVICE_WITNESS_GEN_ATTENTION_FREIVALDS")
    };
    assert!(witness_gen_mode == 0 || corrupt.is_none(),
            "attnout: the device witness generation has no `corrupt` path");
    // Stream JJ (witness-slack section 7 item 1, and the row-`den` hole beside
    // it): the row whose denominator is bent.  `den` was read once per row and
    // folded into `dz` only on the token's *last* row, so on any other row it
    // -- and with it `denp` -- was free, and the slot's rounding was a rounding
    // by a denominator of the prover's choosing.  This control rounds one
    // non-last row honestly against `den - 1`: every constraint and every lookup
    // of the pre-change AIR holds, so it accepts there, and the `den` pin is
    // what rejects it now.
    // stream JJ: the row whose `jo` is moved by one, for the control that the
    // `nofirst` pin on `pjo`/`njo` exists to catch.  Same resolution as
    // `bend_den`'s, and it must also land off a token's first row -- the pin
    // compares a row with its predecessor.
    let row_of_site = |idx: usize, what: &str| -> usize {
        let (ht, ch) = (idx / ao.hd, idx % ao.hd);
        let (h, t) = (ht / ao.s, ht % ao.s);
        let j = (0..rows)
            .find(|j| matches!(lay.at(ao.nh, ao.s, *j), Some((hh, tt, b))
                               if hh == h && tt == t && b * pr <= ch && ch < (b + 1) * pr))
            .unwrap_or_else(|| panic!("{what}: no row carries channel {ch}"));
        let b = lay.at(ao.nh, ao.s, j).unwrap().2;
        assert!(b + 1 < lay.chunks,
                "{what} needs a row that is not its token's first (b = chunks - 1);                  index {idx} lands on b = {b}");
        j
    };
    let bend_jo: Option<usize> = corrupt_jo.map(|idx| row_of_site(idx, "--corrupt-ao-jo"));
    // Which way the `jo` control moves: `AttnOut::jo_step`, which is also what
    // `compose-probes` asks before it runs the control at all.  A shape whose
    // windows admit one `jo` and no other has no move to make, and that is a
    // statement about the shape and not a defect here; the caller picks a row
    // that has room (`AttnOut::jo_bend_site`) or skips the control by name.
    let bend_jo_step: i32 = match bend_jo {
        None => 0,
        Some(j) => {
            let (h, t, b) = lay.at(ao.nh, ao.s, j).expect("--corrupt-ao-jo: an inert row");
            let es = ao.row_es(&lay, h, t, b);
            AttnOut::jo_step(&es, ao.jo[t], jo_max, tables.e_lo, tables.e_hi)
                .unwrap_or_else(|| panic!(
                    "--corrupt-ao-jo: the jo window (top {jo_max}) and the exponent \
                     window [{}, {}] admit one jo for this row and no other, so there is \
                     no move to make on it; ask `AttnOut::jo_bend_site` for a row that \
                     has room, or widen a window with --ao-jo-min / --ao-jo-max",
                    tables.e_lo, tables.e_hi))
        }
    };
    // The same choice for the whole-token control, over that token's every row.
    let bend_tok_step: i32 = match corrupt_jo_tok {
        None => 0,
        Some(t) => {
            let t = t % ao.s;
            let es: Vec<i32> = (0..ao.nh)
                .flat_map(|h| (0..ao.hd).map(move |c| (h, c)))
                .map(|(h, c)| ao.jv[h * ao.hd + c] - ao.jo[t])
                .collect();
            AttnOut::jo_step(&es, ao.jo[t], jo_max, tables.e_lo, tables.e_hi)
                .unwrap_or_else(|| panic!(
                    "--corrupt-ao-jo-tok: the windows admit one jo for token {t} and no \
                     other, so there is no move to make; ask `AttnOut::jo_tok_site` for a \
                     token that has room, or widen a window with --ao-jo-min / --ao-jo-max"))
        }
    };
    let (acc_lo_bits, acc_hi_bits) = acc_bits;
    // stream TT: zero in the magnitude form -- the limbs read `mag` itself
    let acc_off = if acc_mag { 0 } else { 1i64 << (acc_lo_bits + acc_hi_bits - 1) };
    assert!(acc_prod.is_none() || acc_mag, "the product form needs the magnitude form");
    assert!(corrupt_ahn.is_none() || acc_prod.is_some(),
            "--corrupt-ao-ahn is the product form's control; this statement is not in it");
    let ncs = slot_cols(acc_mag, acc_prod.is_some());
    let bend_den: Option<usize> = corrupt_den.map(|idx| {
        let (ht, ch) = (idx / ao.hd, idx % ao.hd);
        let (h, t) = (ht / ao.s, ht % ao.s);
        let j = (0..rows)
            .find(|j| matches!(lay.at(ao.nh, ao.s, *j), Some((hh, tt, b))
                               if hh == h && tt == t && b * pr <= ch && ch < (b + 1) * pr))
            .expect("--corrupt-ao-den: no row carries that channel");
        let b = lay.at(ao.nh, ao.s, j).unwrap().2;
        assert!(b != 0,
                "--corrupt-ao-den needs a non-last row of a token (b = 0 is the row \
                 whose `den` the `dz` fold publishes); index {idx} lands on b = 0");
        assert!(ao.den[h * ao.s + t] > 1, "--corrupt-ao-den: this token's Z is 1");
        j
    });
    if witness_gen_mode == 1 {
        let dev = crate::gadgets::attn_out::device::ao_component_dev(ao, &lay, q_bits, rem_bits, acc_bits,
                                                 jv_min, jo_max, tables, false);
        let (ref_bad, q_out) = (dev.ref_bad, dev.q_out.clone());
        return AoComponent { lay, pre: vec![], base: vec![], dev: Some(dev), ref_bad, q_out,
                          n: ao.nh * ao.s * ao.hd, rem_bits, q_bits,
                          acc_lo_bits, acc_hi_bits, acc_mag, acc_prod, denp2 };
    }
    // `ZK_DEVICE_WITNESS_GEN_ATTENTION_FREIVALDS=2` builds this component twice -- host here, device below --
    // and the device call bumps the same counters a second time, so every
    // counter this witness generation touches has to be snapshotted before, compared as a
    // delta, and restored after.  Stream JJ (E11) added three: the two
    // accumulator limb tables and `m_epow` (whose bumps are a host loop on
    // *both* sides, `attn_out::device::ao_component_dev`).  A counter left out of this
    // tuple is counted twice and the honest proof is then rejected with
    // `sums_cancel: false` -- which is what it did on the H200's reference
    // chunk, and what no desk shape shows, because the desk never runs mode 2.
    let snap_tables = |t: &Tables| -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>, usize) {
        (t.m_qnz.clone(), t.m_r[t.ridx(rem_bits)].clone(),
         t.m_r[t.ridx(acc_lo_bits)].clone(), t.m_r[t.ridx(acc_hi_bits)].clone(),
         t.m_epow.clone(), t.off_table)
    };
    let snap_before = (witness_gen_mode == 2).then(|| snap_tables(tables));
    let mut want_host = vec![0u32; if witness_gen_mode == 2 { rows * pr } else { 0 }];
    let n_cols = 2 + ncs * pr;
    let mut cols: Vec<Vec<M31>> = vec![vec![M31::zero(); rows]; n_cols];
    let mut q_out = vec![0i64; ao.nh * ao.s * ao.hd];
    let mut ref_bad = 0usize;
    for j in 0..rows {
        match lay.at(ao.nh, ao.s, j) {
            None => {
                // inert lead row: den = 1, everything else 0 -> d = denp = 1,
                // dp = 0, (qm, nz2) = (0, 0); all constraints hold, all lookups
                // land on real rows
                // `joff = jo_max - jo` with `jo = jo_max` and `jvp = 0`, so the
                // `EPOW` key is 0 and the pair is `(2^max(e_lo,0), 2^max(-e_lo,0))`
                cols[0][j] = M31::one();
                let (num0, pj0) = (1i64 << tables.e_lo.max(0),
                                   1i64 << (-tables.e_lo).max(0));
                for s in 0..pr {
                    let c0 = 2 + ncs * s;
                    // `acc` stays the raw accumulator (the `z` fold reads it);
                    // what the two limbs range-check is `acc + acc_off` --
                    // and in the magnitude form the tenth column `mag`, which
                    // on an inert row is `|0| = 0`, so both limbs are 0 too
                    cols[c0 + 4][j] = m31_of(pj0);                 // denp = den pj
                    cols[c0 + 5][j] = m31_of(pj0);                 // d = denp
                    cols[c0 + 6][j] = m31_of(num0);
                    cols[c0 + 7][j] = m31_of(pj0);
                    cols[c0 + 8][j] = m31_of(acc_off & ((1i64 << acc_lo_bits) - 1));
                    tables.bump_ek(tables.e_lo, "ao_epow_lead");
                    tables.bump_qnzk(0, "ao_qnz_lead");
                    tables.bump_rk(rem_bits, pj0, "ao_rem_lead");           // d
                    tables.bump_rk(rem_bits, pj0 - 1, "ao_rem_lead");       // dp
                    tables.bump_rk(rem_bits, if denp2 { 2 * pj0 } else { pj0 },
                                   "ao_rem_lead");                           // denp
                    tables.bump_rk(acc_lo_bits, acc_off & ((1i64 << acc_lo_bits) - 1),
                                   "ao_acc_lead");
                    tables.bump_rk(acc_hi_bits, acc_off >> acc_lo_bits, "ao_acc_lead");
                    // the product form: `mag = 0` on an inert row, so `ahn = 0`
                    if let Some(pb) = acc_prod {
                        tables.bump_rk(pb, 0, "ao_acc_prod_lead");
                    }
                }
            }
            Some((h, t, b)) => {
                // stream JJ's control bends this row's denominator only
                let den = ao.den[h * ao.s + t] - i64::from(bend_den == Some(j));
                assert!(den > 0, "attnout denominator must be positive");
                // stream JJ: `jo` is committed now, as `joff = jo_max - jo`.
                // The control moves this row's `jo` by one and leaves the
                // token's other rows alone, which only the `nofirst` pin sees.
                let jo = ao.jo[t]
                    + if bend_jo == Some(j) { bend_jo_step } else { 0 }
                    + if corrupt_jo_tok.map(|x| x % ao.s) == Some(t) { bend_tok_step }
                      else { 0 };
                let joff = i64::from(jo_max - jo);
                assert!(joff >= 0, "attnout: jo above the window's top");
                cols[0][j] = m31_of(den);
                cols[1][j] = m31_of(joff);
                for s in 0..pr {
                    let ch = b * pr + s;
                    let idx = (h * ao.s + t) * ao.hd + ch;
                    let mut a = ao.acc[idx];
                    if corrupt == Some(idx) {
                        a += 1;
                    }
                    assert!(corrupt_mag.is_none() || acc_mag,
                            "--corrupt-ao-mag is the magnitude form's control; this \
                             statement is in the offset form");
                    let jv = ao.jv[h * ao.hd + ch];
                    let e = jv - jo;
                    // the window pair, and the control that bends it: a `num`
                    // one doubling out is a triple the `EPOW` table does not
                    // carry, so only the lookup can reject it
                    let num = (1i64 << e.max(0)) << u32::from(corrupt_num == Some(idx));
                    let pj = 1i64 << (-e).max(0);
                    let denp = den * pj;
                    let mag = a.abs();
                    let pm = mag * num;
                    // the no-wrap window: every term of the error identity must be
                    // an integer below p, or the field reading is ambiguous
                    assert!(2 * pm < (1i64 << 31) - 1 && 2 * denp < (1i64 << 31) - 1,
                            "attnout fold outgrows the field: pm={pm} denp={denp}");
                    let mut qm = (2 * pm + denp) / (2 * denp);
                    // stream Z's bend control: one quotient pushed past the pair
                    // table's domain (the clamp), everything else derived from it
                    if corrupt_q == Some(idx) {
                        qm = 1i64 << q_bits;
                    }
                    let sg = i64::from(a < 0 && qm != 0);
                    let d = if a < 0 && qm == 0 { denp - 2 * pm }
                            else { 2 * pm - (2 * qm - 1) * denp };
                    let c0 = 2 + ncs * s;
                    // the offset accumulator's two limbs.  The control moves the
                    // committed low limb by one *inside its own table*, so both
                    // range lookups still land on real rows and the only thing
                    // that can see it is that `(acc + acc_off - alo) / 2^lo` is
                    // no longer the integer the high table was bumped at.
                    //
                    // stream TT: in the magnitude form the limbs read `mag`
                    // itself -- `acc_off` is 0 and the value decomposed is the
                    // tenth column's, not the accumulator's.
                    //
                    // stream TT's control commits the *other* root of that pin:
                    // `amag = -|acc|`, which satisfies `(amag - acc)(amag + acc)
                    // = 0` exactly as the honest magnitude does and leaves every
                    // other column of the slot alone.  As a field element it is
                    // `p - |acc|`, about `2^31`, so it is the forgery the bound
                    // exists to stop -- `2 mag num` would run past `p` -- and
                    // the only thing in the statement that can see it is the
                    // high limb's range lookup.
                    let magv = if corrupt_mag == Some(idx) { -mag } else { mag };
                    let ao_v = if acc_mag { magv } else { a + acc_off };
                    assert!(corrupt_mag == Some(idx)
                                || (ao_v >= 0
                                    && ao_v < 1i64 << (acc_lo_bits + acc_hi_bits)),
                            "attnout: |acc| outside its stated window: {a}");
                    // the limbs are of the *field* representative, so a bent
                    // negative magnitude decomposes the way the AIR reads it
                    let fv = if ao_v < 0 { ((1i64 << 31) - 1) + ao_v } else { ao_v };
                    let alo = if corrupt_acc == Some(idx) {
                        ((fv & ((1i64 << acc_lo_bits) - 1)) + 1) & ((1i64 << acc_lo_bits) - 1)
                    } else {
                        fv & ((1i64 << acc_lo_bits) - 1)
                    };
                    cols[c0][j] = m31_of(a);
                    cols[c0 + 1][j] = m31_of(sg);
                    cols[c0 + 2][j] = m31_of(qm);
                    cols[c0 + 3][j] = m31_of(i64::from(qm != 0));
                    cols[c0 + 4][j] = m31_of(denp);
                    cols[c0 + 5][j] = m31_of(d);
                    cols[c0 + 6][j] = m31_of(num);
                    cols[c0 + 7][j] = m31_of(pj);
                    cols[c0 + 8][j] = m31_of(alo);
                    // stream TT (log row 73): the tenth column, `|acc|`.  It is
                    // committed rather than derived so its two limbs stay
                    // degree 1; the AIR pins it with `(amag - acc)(amag + acc)
                    // = 0`, which with `amag` in `[0, 2^(lo+hi))` says `amag`
                    // is the accumulator's magnitude and nothing else.  Note it
                    // is *not* the AIR's `mag = acc (1 - 2 sg)`: on a negative
                    // accumulator that rounds to zero the sign pin forces
                    // `sg = 0` and that expression is the negative value, which
                    // is why the bound needs a column of its own.
                    if acc_mag {
                        cols[c0 + 9][j] = m31_of(magv);
                    }
                    // the product form's eleventh column, `ahn = ahi num`, of
                    // the same field representative the limbs decompose.  An
                    // honest slot has `2 mag num < 2^31 - 1` (asserted above),
                    // so `ahn <= mag num / 2^lo < 2^(30 - lo)` lands in its
                    // table; a bent magnitude does not, which is the point.
                    let ahi = (fv - alo) >> acc_lo_bits;
                    if let Some(pb) = acc_prod {
                        let ahn = ahi * num;
                        assert!(corrupt_mag == Some(idx) || corrupt_acc == Some(idx)
                                    || corrupt_num == Some(idx) || ahn < 1i64 << pb,
                                "attnout: ahi num = {ahn} outside the product form's \
                                 {pb}-bit table (|acc| = {mag}, num = {num})");
                        // the product form's control: `ahn` one off `ahi num`,
                        // a step that stays inside its own table, and the
                        // table bumped at what is committed -- so the lookup
                        // lands and only the pin `ahn = ahi num` can see it
                        let ahn = match corrupt_ahn == Some(idx) {
                            true if ahn + 1 < 1i64 << pb => ahn + 1,
                            true => ahn - 1,
                            false => ahn,
                        };
                        cols[c0 + 10][j] = m31_of(ahn);
                        tables.bump_rk(pb, ahn, "ao_acc_prod");
                    }
                    tables.bump_ek(e, "ao_epow");
                    tables.bump_qnzk(qm, "ao_qnz");
                    tables.bump_rk(rem_bits, d, "ao_rem_d");
                    tables.bump_rk(rem_bits, 2 * denp - 1 - d, "ao_rem_dp");
                    // section 7 item 1: `denp`'s own bound, without which the
                    // pair above leaves the `2 denp = d + dp + 1` halving branch
                    tables.bump_rk(rem_bits, if denp2 { 2 * denp } else { denp },
                                   "ao_rem_denp");
                    // and the other half of the no-wrap window: `2 mag num < p`
                    // is `2 mag num` with `num <= 2^e_hi` from the `EPOW` table
                    // and `mag` bounded here, in two limbs
                    tables.bump_rk(acc_lo_bits, alo, "ao_acc_lo");
                    tables.bump_rk(acc_hi_bits, ahi, "ao_acc_hi");
                    q_out[idx] = if a < 0 { -qm } else { qm };
                    // the reference: one f64 rounding, half away from zero.  It
                    // reads `num / pj` rather than `2^(jv - jo)` -- the same
                    // number on an honest row, and on a bent one the reference
                    // follows the bend instead of counting it twice.
                    let want = ((mag as f64 * (num as f64 / pj as f64)) / den as f64 + 0.5)
                        .floor() as i64;
                    if !want_host.is_empty() {
                        want_host[j * pr + s] = want as u32;
                    }
                    if want != qm {
                        ref_bad += 1;
                    }
                }
            }
        }
    }
    // preprocessed: one per-slot column, then the selectors.  Stream JJ (E11):
    // `jvp = jv - jv_min`, a per-(head, channel) model constant offset to be
    // non-negative.  It replaces the old `(num, pj)` pair, which was a function
    // of `e = jv - jo` and so of this chunk's tokens; tree 0 no longer is.
    let mut pre = vec![];
    for s in 0..pr {
        pre.push(eval_of(lay.log_rows, (0..rows).map(|j| {
            match lay.at(ao.nh, ao.s, j) {
                None => M31::zero(),
                Some((h, _t, b)) => m31_of(i64::from(ao.jv[h * ao.hd + b * pr + s] - jv_min)),
            }
        }).collect()));
    }
    let active = |j: usize| lay.at(ao.nh, ao.s, j).is_some();
    let last = |j: usize| matches!(lay.at(ao.nh, ao.s, j), Some((_, _, 0)));
    let head_first = |j: usize| matches!(lay.at(ao.nh, ao.s, j),
                                          Some((_, t, 0)) if t == ao.s - 1);
    let sel = |fsel: &dyn Fn(usize) -> bool| {
        eval_of(lay.log_rows, (0..rows).map(|j| m31_u32(fsel(j) as u32)).collect())
    };
    pre.push(sel(&|j| matches!(lay.at(ao.nh, ao.s, j), Some((_, _, b)) if b + 1 < lay.chunks)));
    pre.push(sel(&|j| last(j)));
    pre.push(sel(&|j| j > 0 && !last(j)));                       // m1 (incl. inert)
    pre.push(sel(&|j| j > 0 && last(j) && !head_first(j)));      // m2: token step s
    pre.push(sel(&|j| j > 0 && last(j) && head_first(j)));       // m3: head step mu
    pre.push(sel(&|j| j + 1 == rows));                           // end
    let _ = active;
    let base: Vec<Eval> = cols.into_iter().map(|c| eval_of(lay.log_rows, c)).collect();
    if witness_gen_mode == 2 {
        let after = snap_tables(tables);
        let before = snap_before.unwrap();
        let dev = crate::gadgets::attn_out::device::ao_component_dev(ao, &lay, q_bits, rem_bits, acc_bits,
                                                 jv_min, jo_max, tables, true);
        crate::device_witness::check_cols("attnout base", &base, &dev.base);
        crate::device_witness::check_cols("attnout pre", &pre, &dev.pre);
        let ri = tables.ridx(rem_bits);
        let (rlo, rhi) = (tables.ridx(acc_lo_bits), tables.ridx(acc_hi_bits));
        crate::device_witness::check_delta("attnout m_qnz", &before.0, &after.0,
                &sub_slice(&tables.m_qnz, &after.0));
        crate::device_witness::check_delta("attnout m_rem", &before.1, &after.1,
                &sub_slice(&tables.m_r[ri], &after.1));
        // stream JJ (E11): the accumulator window's two limbs, and the `EPOW`
        // window, whose column is a host loop on both sides
        crate::device_witness::check_delta("attnout m_acc_lo", &before.2, &after.2,
                &sub_slice(&tables.m_r[rlo], &after.2));
        crate::device_witness::check_delta("attnout m_acc_hi", &before.3, &after.3,
                &sub_slice(&tables.m_r[rhi], &after.3));
        crate::device_witness::check_delta("attnout m_epow", &before.4, &after.4,
                &sub_slice(&tables.m_epow, &after.4));
        assert_eq!(tables.off_table - after.5, after.5 - before.5,
                   "attnout: off-table bumps differ");
        assert_eq!(dev.ref_bad, ref_bad, "attnout: ref_bad differs");
        assert_eq!(dev.q_out, q_out, "attnout: the committed int8 outputs differ");
        // the one part of this family's witness generation that is not integer arithmetic,
        // compared per (row, slot) rather than as a count -- which on an honest
        // witness is zero on both sides and would agree vacuously
        let w = dev.want.as_ref().expect("mode 2 asks for the reference");
        assert_eq!(w.len(), want_host.len(), "attnout: want length");
        if let Some(k) = (0..w.len()).find(|k| w[*k] != want_host[*k]) {
            panic!("attnout: the f64 reference differs at row {} slot {}: host {}, device {}",
                   k / pr, k % pr, want_host[k], w[k]);
        }
        // restore in table order, so a width shared between two of these (say
        // `rem_bits == acc_lo_bits`) ends on the one snapshot that covers it
        tables.m_qnz.copy_from_slice(&after.0);
        tables.m_r[ri].copy_from_slice(&after.1);
        tables.m_r[rlo].copy_from_slice(&after.2);
        tables.m_r[rhi].copy_from_slice(&after.3);
        tables.m_epow.copy_from_slice(&after.4);
        tables.off_table = after.5;
        eprintln!("witness_gen-dev: attnout -- {} base and {} pre columns, {} cells, \
all five multiplicity tables and the f64 reference all match",
                  base.len(), pre.len(), (base.len() + pre.len()) << lay.log_rows);
    }
    // Stream II (E11): the witness-side reference and the component's own
    // committed output are the same function of `AttnOut`, and a synthetic
    // layer's `o` input is built from the former -- so an untampered run
    // asserts they agree rather than trusting that they do.  Every tamper index
    // here is `None` on an honest run, which is exactly when the assert holds.
    if bend_den.is_none() && bend_jo.is_none() && corrupt.is_none() && corrupt_q.is_none()
        && corrupt_num.is_none() && corrupt_acc.is_none() && corrupt_jo_tok.is_none()
        && corrupt_mag.is_none()
    {
        debug_assert_eq!(q_out, ao.q_ref(), "attnout: q_out and AttnOut::q_ref disagree");
    }
    AoComponent { lay, pre, base, dev: None, ref_bad, q_out, n: ao.nh * ao.s * ao.hd,
               rem_bits, q_bits, acc_lo_bits, acc_hi_bits, acc_mag, acc_prod, denp2 }
}

impl Layer {
    /// Rebuild the attnout witness from the (possibly corrupted) key tiles, keeping
    /// the scales: the probes mutate a key tile and need `ao` to follow it.
    pub fn rebuild_ao(&mut self) {
        let (nh, s, hd) = (self.ao.nh, self.ao.s, self.ao.hd);
        // zero first, then accumulate: a head's totals are the sum over every key tile
        // that covers it, which with B10 step 3's diagonal block is two of them
        self.ao.den.iter_mut().for_each(|v| *v = 0);
        self.ao.acc.iter_mut().for_each(|v| *v = 0);
        for (bi, b) in self.key_tiles.iter().enumerate() {
            for i in 0..self.key_tile_sizes[bi] {
                let h = self.head_offsets[bi] + i;
                for t in 0..s {
                    self.ao.den[h * s + t] += b.zz[i * s + t];
                    for j in 0..hd {
                        self.ao.acc[(h * s + t) * hd + j] += b.out[(i * s + t) * hd + j];
                    }
                }
            }
        }
        let _ = nh;
    }
}
