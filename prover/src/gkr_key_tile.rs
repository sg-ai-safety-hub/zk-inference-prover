//! Stream Q: the attention key tiles' lookups proved by LogUp-GKR.
//!
//! Today a key tile component's lookups are proved by committing running-sum columns:
//! `attn_trace::logup_cols` pairs the component's lookups, builds the pointwise
//! partial sums, and the AIR's `finalize_logup_in_pairs` asserts the recurrence
//! while a boundary constraint bindings the last row to the component's entry in
//! `claims.use_sums`. Those columns are 88% of the key tiles' interaction cells and
//! the key tiles are two thirds of the interaction tree at long context, so they are
//! the largest single thing being committed that does not have to be.
//!
//! This module proves the same fractional sums with a sumcheck instead. It
//! commits nothing. What it produces is one `GkrBatchProof` per chunk and, from
//! that proof's output claims, exactly the same claimed sums the committed
//! columns produce today -- which is what `verify_composed`'s `sums_cancel`
//! consumes, so the global LogUp identity is unchanged.
//!
//! ## The shape of the argument
//!
//! A component's fractions are indexed by `(look, row)`. Laid out that way, a
//! component is a natural GKR instance: its input layer is an MLE over
//! `log2(n_looks) + log_rows` variables. Two details decide the layout.
//!
//! *The lookups reach the circuit unpaired* (B11 stream S, lever 3; it undoes
//! stream R's outside-the-circuit pairing). The input layer holds one entry
//! per look -- numerator `1` or `1 - mask`, denominator `combine(cols) - z`,
//! both *affine* in the committed columns -- which is what lets the `mle_eval`
//! link (memo section 4) binding the GKR's input-layer claims to the committed
//! base columns. R's pairing is not lost: fraction addition is exactly the
//! GKR's gate, so the pairing the committed path performs in `plogup::paired`
//! is now the circuit's own first layer, sumchecked inside the proof instead
//! of precomputed outside it. That first layer is the price of soundness, and
//! it is measured, not assumed.
//!
//! *The entry axis wants a power of two.* Padding a 66-look channel component up
//! to 128 would nearly double its sumcheck. Instead a component is split along the
//! binary decomposition of its look count -- 66 becomes a 64-look instance
//! and a 2-look one -- which pads nothing and costs only instance bookkeeping.
//!
//! *The numerators come in two kinds.* A unit look stores no numerators at all
//! (`Layer::LogUpSingles`); a diagonal block grid's exp looks carry `1 - mask` and
//! live in `Layer::LogUpGeneric` instances. A component therefore contributes up
//! to two groups of instances, and the split is by numerator kind first,
//! binary decomposition second.
//!
//! ## One `prove_batch` per chunk
//!
//! Every instance from every key tile goes into a single `prove_batch` call. This is
//! not an optimisation to add later: a sumcheck's domain halves every round, so
//! a per-component GKR would pay per-round launch overhead across dozens of small
//! trees, which stream C named as the likeliest way the sumcheck's cost model
//! turns out optimistic. `prove_batch` already draws one challenge pair per
//! layer and runs one batched sumcheck per layer across all instances, starting
//! each instance when the layer count reaches it, so the batching costs us
//! nothing but putting the instances in one vector.

use std::time::Instant;

use num_traits::{One, Zero};
use stwo::core::fields::FieldExpOps;
use rayon::prelude::*;
use stwo::core::channel::Channel;
use stwo::core::fields::qm31::SecureField;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::simd::column::SecureColumn;
use stwo::prover::backend::simd::m31::{LOG_N_LANES, N_LANES};
use stwo::prover::backend::simd::qm31::PackedSecureField;
use stwo::prover::lookups::gkr_prover::{Layer, prove_batch};
use stwo::prover::lookups::gkr_verifier::{
    Gate, GkrArtifact, GkrBatchProof, GkrError, partially_verify_batch,
};
use stwo::prover::lookups::mle::Mle;
use stwo_constraint_framework::FrameworkEval;

use crate::gadgets::attention::witness::Variant as AttnVariant;

/// One key tile component's lookup site, as `interaction_phase` reaches it.
///
/// `which` is the component index `devfrac::key_tile_cols` uses -- 0 for the channel
/// component, 1 for the grid, 3 for the per-token -- so a site names the same thing
/// the existing code names.
#[derive(Clone, Debug)]
pub struct Site {
    /// B11 stream U, lever 3: which family and which component within it. Stream S
    /// carried a key tile index alone; the dequant family's components index into the
    /// plan's matmul list, and the two spaces are unrelated.
    pub reg: crate::gkrlink::Comp,
    pub which: usize,
    pub log_rows: u32,
    /// numerator kind per lookup, in `add_to_relation` order: true when the
    /// numerator is one
    pub unit: Vec<bool>,
}

impl Site {
    pub fn tag(&self) -> String {
        match self.reg {
            crate::gkrlink::Comp::KeyTile(b) =>
                format!("key_tile{b}/{}", ["chan", "key", "out", "row"][self.which]),
            crate::gkrlink::Comp::Deq(m) => format!("deq{m}"),
            crate::gkrlink::Comp::ResidualStream(g) => format!("residual_stream{g}"),
            crate::gkrlink::Comp::Freivalds(w) => format!("freivalds{w}"),
            crate::gkrlink::Comp::Ao => "attnout".to_string(),
        }
    }
    pub fn n_looks(&self) -> usize {
        self.unit.len()
    }
    /// B11 stream S, lever 3: the input layer holds the *unpaired* looks, one
    /// entry each, so the GKR's input-layer claims are affine in the committed
    /// columns and the `mle_eval` link (memo section 4) can binding them down.
    /// Stream R's pairing is not lost -- fraction addition is exactly the
    /// GKR's gate, so the pairing is now the circuit's own first layer,
    /// sumchecked instead of precomputed outside the proof.
    pub fn n_ents(&self) -> usize {
        self.n_looks()
    }
    /// Whether entry `c` (one look now) has a unit numerator.
    pub fn ent_unit(&self, c: usize) -> bool {
        self.unit[c]
    }
}

/// One GKR instance: a set of one site's looks whose count is a power of two,
/// over that site's rows.
#[derive(Clone, Debug)]
pub struct Inst {
    /// index into the site list
    pub site: usize,
    /// which of the site's looks this instance covers
    pub ents: Vec<usize>,
    pub log_rows: u32,
    /// every numerator in this instance is one
    pub unit: bool,
}

impl Inst {
    pub fn n_variables(&self) -> usize {
        self.ents.len().ilog2() as usize + self.log_rows as usize
    }
    /// Unpaired fractions: what the input layer holds and the sumcheck runs over.
    pub fn fractions(&self) -> usize {
        self.ents.len() << self.log_rows
    }
}

/// The instances the sites become, in site order.
///
/// Within a site the unit-numerator entries come first, then the rest; each
/// group is cut into power-of-two blocks largest first. The order is a pure
/// function of the site list, so the prover and the verifier derive the same
/// instance list from the plan and nothing about it travels in the proof.
pub fn plan(sites: &[Site]) -> Vec<Inst> {
    let mut out = vec![];
    for (i, s) in sites.iter().enumerate() {
        for unit in [true, false] {
            let group: Vec<usize> = (0..s.n_ents()).filter(|c| s.ent_unit(*c) == unit).collect();
            let mut at = 0usize;
            // largest block first: the highest set bit of what is left
            let mut left = group.len();
            while left > 0 {
                let take = 1usize << (usize::BITS - 1 - left.leading_zeros());
                out.push(Inst {
                    site: i,
                    ents: group[at..at + take].to_vec(),
                    log_rows: s.log_rows,
                    unit,
                });
                at += take;
                left -= take;
            }
        }
    }
    out
}

/// The gates the verifier needs, one per instance. Every key tile instance is a
/// LogUp fractional sum.
pub fn gates(insts: &[Inst]) -> Vec<Gate> {
    vec![Gate::LogUp; insts.len()]
}

/// Build one instance's input layer from a packed fraction source.
///
/// `frac(ent, vec_row)` is `look_frac` directly -- the numerator and
/// denominator of one *look* at one packed row, unpaired
/// -- so the fractions proved here are the same function of the same committed
/// columns that the committed running-sum columns are built from. Nothing is
/// re-derived and nothing is fed unpaired.
fn input_layer<F>(inst: &Inst, frac: &F) -> Layer<SimdBackend>
where
    F: Fn(usize, usize) -> (PackedSecureField, PackedSecureField) + Sync,
{
    let n_vec_row = 1usize << (inst.log_rows - LOG_N_LANES);
    let n_vec = inst.ents.len() * n_vec_row;
    let length = inst.fractions();

    // `(entry, row)` in that order, so the leading `log2(n_ents)` variables are
    // the entry axis. The link in the memo reduces the input-layer claim over
    // exactly this split.
    let build = |pick: fn(&(PackedSecureField, PackedSecureField)) -> PackedSecureField| {
        let data: Vec<PackedSecureField> = (0..n_vec)
            .into_par_iter()
            .map(|k| {
                let (b, vr) = (k / n_vec_row, k % n_vec_row);
                pick(&frac(inst.ents[b], vr))
            })
            .collect();
        SecureColumn { data, length }
    };

    if inst.unit {
        Layer::LogUpSingles { denominators: Mle::new(build(|f| f.1)) }
    } else {
        Layer::LogUpGeneric {
            numerators: Mle::new(build(|f| f.0)),
            denominators: Mle::new(build(|f| f.1)),
        }
    }
}

/// The sites and input layers gathered from a pass over the key tiles.
///
/// `interaction_phase` builds each component's lookup list already, and that list
/// plus the column slice it indexes is only in scope right there. Rather than
/// reconstruct either outside -- two definitions of the same thing is how the
/// two backends drift apart -- the interaction pass hands them to this collector at
/// the same three sites where it calls `devfrac::key_tile_cols`. Collecting is
/// purely additive: it reads the same columns and writes nothing the committed
/// path can see.
#[derive(Default)]
pub struct Collect {
    /// the component whose lookups are arriving; the caller sets it before each
    /// component's interaction pass, because a diagonal block key tile reaches
    /// `interaction_phase` with no `KeyTileAir` to carry its index
    pub reg: crate::gkrlink::Comp,
    pub sites: Vec<Site>,
    /// the host-built input layers, in the order the host sites arrived
    pub layers: Vec<Layer<SimdBackend>>,
    /// B11 stream T: the device feed's input layers, in the order the fed sites
    /// arrived. `on_dev` says, per instance and in instance order, which of the
    /// two lists holds it -- the instance order is the transcript's, so the two
    /// are interleaved back together and never sorted.
    #[cfg(feature = "cuda")]
    pub dev_layers: Vec<Layer<stwo::prover::backend::cuda::CudaBackend>>,
    /// per instance, in instance order: the device feed built this one
    pub on_dev: Vec<bool>,
    pub insts: Vec<Inst>,
    pub layers_secs: f64,
    /// B11 stream T: the feed's launches, apart from the host builds
    pub feed_secs: f64,
    /// the claimed sum of each site, summed directly from its fractions rather
    /// than through the sumcheck -- the self-test's reference
    pub direct: Vec<SecureField>,
    /// fill `direct` as sites arrive; off for a proving run, on for the test
    pub want_direct: bool,
    /// B11 stream S, lever 1: the backend owns the key tiles' fractional sums, so
    /// `interaction_phase` builds no committed running-sum columns and pushes
    /// no use sums at the collected sites -- the GKR proves the same sums and
    /// `verify_composed` feeds them to `sums_cancel` directly. Off, collecting
    /// stays purely additive, which is what the self-test relies on.
    pub replace: bool,
    /// B11 stream V, lever 1: what each fed site would need to build its input
    /// layers a second time -- the dispatch key, the committed column pointers
    /// and the scalar blob, all of which outlive the GKR. `prove_batch` frees a
    /// dropped input layer as soon as its first halving exists and asks for it
    /// back in the last round; this is what answers.
    #[cfg(feature = "cuda")]
    pub feeds: Vec<feedimpl::FeedPlan>,
}

impl Collect {
    pub fn with_direct(want_direct: bool) -> Self {
        Self { want_direct, ..Default::default() }
    }

    /// Take one component's lookups and build its instances' input layers.
    pub fn push(
        &mut self,
        which: usize,
        log_rows: u32,
        looks: &[crate::gadgets::attention::trace::Look],
        cols: &[crate::common::Eval],
        pre: &[crate::common::Eval],
        exp: &crate::gadgets::attention::air::ExpRel,
        ranges: &[crate::relations::RangeRel],
    ) {
        let t0 = Instant::now();
        let site = Site {
            reg: self.reg,
            which,
            log_rows,
            unit: looks.iter().map(crate::gadgets::attention::trace::look_is_unit).collect(),
        };
        let at = self.sites.len();
        // B11 stream S, lever 3: the looks feed the circuit *unpaired* -- one
        // input-layer entry per look, `look_frac` read directly, so the
        // input-layer claims stay affine in the committed columns and the
        // `mle_eval` link can binding them down. The circuit's own first layer now
        // does the pairing stream R precomputed outside the proof.
        let frac = |look: usize, vr: usize| {
            crate::gadgets::attention::trace::look_frac(&looks[look], vr, cols, pre, exp, ranges)
        };
        let mut direct = SecureField::zero();
        for inst in plan(std::slice::from_ref(&site)) {
            let inst = Inst { site: at, ..inst };
            if self.want_direct {
                direct += direct_sum(&inst, &frac);
            }
            self.layers.push(input_layer(&inst, &frac));
            self.on_dev.push(false);
            self.insts.push(inst);
        }
        self.sites.push(site);
        self.direct.push(direct);
        self.layers_secs += t0.elapsed().as_secs_f64();
    }

    /// B11 stream U, lever 3: one component's input layers from a fraction table
    /// the caller has already built.
    ///
    /// `Collect::push` above is the key tiles' path, and it is key-tile-shaped:
    /// `attn_trace::look_frac` names the columns a key tile lookup reads. The
    /// dequant family's lookups come from its fraction IR instead, so the
    /// fractions arrive as a table and everything downstream -- the plan, the
    /// layers, the direct sum -- is the same code.
    pub fn push_flat(
        &mut self,
        which: usize,
        log_rows: u32,
        unit: Vec<bool>,
        flat: &[Vec<(PackedSecureField, PackedSecureField)>],
    ) {
        let t0 = Instant::now();
        assert!(flat.iter().all(|r| r.len() == unit.len()),
                "the fraction table has a row with a different look count");
        let site = Site { reg: self.reg, which, log_rows, unit };
        let at = self.sites.len();
        let frac = |look: usize, vr: usize| flat[vr][look];
        let mut direct = SecureField::zero();
        for inst in plan(std::slice::from_ref(&site)) {
            let inst = Inst { site: at, ..inst };
            if self.want_direct {
                direct += direct_sum(&inst, &frac);
            }
            self.layers.push(input_layer(&inst, &frac));
            self.on_dev.push(false);
            self.insts.push(inst);
        }
        self.sites.push(site);
        self.direct.push(direct);
        self.layers_secs += t0.elapsed().as_secs_f64();
    }
}

/// B11 stream T: one site's fractions, computed where its columns already are.
///
/// This is the call `interaction_phase` makes at each of the three key tile sites,
/// in place of `Collect::push` alone. When the key tile's witness generation left its columns on
/// the device there is nothing on the host to read -- `com.base` is empty and
/// the collector's own `look_frac` would index straight out of it, which is
/// what stream S's empty-env control recorded -- so the fractions are computed
/// by the *feed*: the generated fraction kernel's unpaired variant, one launch
/// per site, writing each look straight into the GKR instance layer that holds
/// it, in MLE layout. Otherwise the host builder runs exactly as before.
///
/// `at` is the committed side's component split; the device's own copy has to
/// agree with it, because the base columns the kernel reads are named by it.
#[allow(clippy::too_many_arguments)]
/// `ZK_GKR_FEED_CHECK=1`: run the feed beside the host builder and compare. Opt-in,
/// spelled the way `ZK_LINK_CHECK` is -- `util::switch_on` reads "on unless set
/// to 0", which is the wrong default for a probe that costs an upload and a
/// full cell-for-cell comparison.
fn feedcheck_on() -> bool {
    std::env::var("ZK_GKR_FEED_CHECK").map(|v| v == "1").unwrap_or(false)
}

/// The key tiles' numerator kinds, in look order.
fn unit_of(looks: &[crate::gadgets::attention::trace::Look]) -> Vec<bool> {
    looks.iter().map(crate::gadgets::attention::trace::look_is_unit).collect()
}

/// Whether `cols` actually holds this component's committed span. A
/// device-built key tile has no host columns at all -- but the per-token site
/// passes a list that always carries its two derived high-limb blocks, so
/// "non-empty" is not the question; "long enough to hold the span" is.
fn host_span(at: &[usize], which: usize, cols: &[crate::common::Eval]) -> bool {
    cols.len() >= at[which + 1]
}

pub fn feed_site(
    c: &mut Collect,
    air: Option<&crate::devfrac::KeyTileAir<'_>>,
    which: usize,
    log_rows: u32,
    looks: &[crate::gadgets::attention::trace::Look],
    at: &[usize],
    cols: &[crate::common::Eval],
    pre: &[crate::common::Eval],
    exp: &crate::gadgets::attention::air::ExpRel,
    ranges: &[crate::relations::RangeRel],
) {
    let dev = air.and_then(|a| a.dev);
    let Some(d) = dev else {
        // `ZK_GKR_FEED_CHECK=1` with the key tile on the host: run the feed anyway over
        // an *uploaded* copy of the same committed columns and compare every
        // cell against the host builder, then throw the fed layers away. The
        // upload is the check's, exactly as `ZK_GENERATED_LOGUP=3` pays one; nothing
        // on the prover's path moves, so this can run in the regime that has
        // host columns to compare against -- which the fed regime, by
        // definition, does not.
        #[cfg(feature = "cuda")]
        if feedcheck_on() && host_span(at, which, cols) && air.is_some() {
            c.feed_check(air.unwrap(), which, log_rows, looks, at, cols, pre, exp, ranges);
        }
        c.push(which, log_rows, looks, cols, pre, exp, ranges);
        return;
    };
    let air = air.expect("a key tile with device columns and no AIR");
    assert_eq!(at, &d.at[..],
               "the committed component split disagrees with the device witness generation's");
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (air, which, log_rows, looks, exp, ranges);
        panic!("a key tile's columns are on the device on a build without the cuda feature");
    }
    #[cfg(feature = "cuda")]
    {
        let (lo, hi) = (at[which], at[which + 1]);
        let base: Vec<*const u32> = d.base[lo..hi].iter().map(|c| c.device_ptr).collect();
        // The fractions the feed computes are the ones the *committed* backend would
        // have built, so the evaluator it lowers is the un-ablated one: stream
        // S's lever 1 makes the flag-on evaluator emit no lookups at all, and
        // its fraction IR would be empty. Flipping the flag back for the
        // lowering reproduces the flag-off fingerprint exactly, which is why the
        // kernel store already carries this shape.
        let first = c.insts.len();
        match which {
            0 => {
                let mut e = air.chan.clone();
                e.gkr = false;
                c.push_dev(which, log_rows, unit_of(looks), &e, &[], &base);
            }
            1 => {
                let mut e = air.key.clone();
                e.gkr = false;
                c.push_dev(which, log_rows, unit_of(looks), &e, &[], &base);
            }
            3 => {
                let mut e = air.row.clone();
                e.gkr = false;
                c.push_dev(which, log_rows, unit_of(looks), &e, &[], &base);
            }
            _ => panic!("key tile component {which} has no eval"),
        }
        // `ZK_GKR_FEED_CHECK=1`: the same check `ZK_GENERATED_LOGUP=2` is for the paired
        // kernel -- every cell of every instance against the host builder, which
        // needs host columns and so only runs in a regime that has them.
        if feedcheck_on() && host_span(at, which, cols) {
            c.check_fed(first, looks, cols, pre, exp, ranges);
        }
    }
}

/// B11 stream U, lever 3: one dequant component's lookups, wherever its columns
/// are.
///
/// This is `feed_site`'s shape for the second family. `mat_interaction` calls it
/// in place of `mat_logup` under `--gkr-dequant`: the component's fractions become
/// GKR input layers instead of committed running-sum columns, and its claimed
/// sum leaves `use_sums` for `sums_cancel`.
///
/// The fractions come from the component's own fraction IR, which is the lowering
/// of the *un-ablated* `DequantEval` -- stream T's precedent, and the reason the
/// kernel store already carries this fingerprint: under the flag the evaluator
/// emits no lookups at all and its IR would be empty. One definition of the
/// fraction, three readers: this, `mat_logup`'s hand-written builder (the one
/// `ZK_GENERATED_LOGUP=2` compares against) and `gkrlink::deq_looks`.
pub fn deq_site(
    c: &mut Collect,
    reg: usize,
    eval: &crate::gadgets::dequant::DequantEval,
    r: &crate::layer::MatComponent,
) {
    fam_site(c, FamSite {
        reg: crate::gkrlink::Comp::Deq(reg),
        tag: format!("deq{reg}"),
        mut_env: "ZK_MUT_DEQUANT",
        mut_key: reg,
        log_rows: r.lay.log_rows,
        unit: crate::gadgets::dequant::look_units(&r.spec),
        pre: &r.pre,
        base: &r.base,
        #[cfg(feature = "cuda")]
        dev: r.dev.as_ref().map(|d| (&d.pre[..], &d.base[..])),
    }, eval);
}

/// B11 stream V, lever 3: one residual-stream component's lookups, wherever its columns are.
///
/// `residual_stream_interaction` stops building running-sum columns under `--gkr-residual-stream` and
/// `compose` calls this in the collection pass instead. Every residual-stream look is unit
/// -- the numerator is the literal one at all six kinds -- so every instance is
/// `Layer::LogUpSingles`, and the unit list is asserted against the fraction
/// IR's own numerators rather than assumed: `gkrlink::residual_stream_look_count` is what
/// the *verifier* derives, `ir_units` is what the fractions are, and the two
/// disagreeing is a stop before a proof rather than after one.
pub fn residual_stream_site(
    c: &mut Collect,
    reg: usize,
    eval: &crate::gadgets::residual_stream::ResidualStreamEval,
    r: &crate::gadgets::residual_stream::ResidualStreamComponent,
) {
    let log_rows = r.log_rows;
    let ir = crate::scalars::cached_ir(eval, log_rows, log_rows, true, || {
        crate::devfrac::ir_of(eval, log_rows)
    });
    let want = crate::gkrlink::residual_stream_look_count(&r.prog);
    let units = crate::gkrlink::ir_units(&ir);
    assert_eq!(units.len(), want,
               "residual_stream/{}: the fraction IR has {} lookups, the plan's program {want}",
               r.tag, units.len());
    assert!(units.iter().all(|u| *u),
            "residual_stream/{}: a lookup's numerator is not the constant one, so the \
             site's instance split is not the one `sites_of` derives", r.tag);
    fam_site(c, FamSite {
        reg: crate::gkrlink::Comp::ResidualStream(reg),
        tag: format!("residual_stream/{}", r.tag),
        mut_env: "ZK_MUT_RESIDUAL_STREAM",
        mut_key: reg,
        log_rows,
        unit: units,
        pre: &r.pre,
        base: &r.base,
        #[cfg(feature = "cuda")]
        dev: r.dev.as_ref().map(|d| (&d.pre[..], &d.base[..])),
    }, eval);
}

/// B11 stream V, lever 4: one Freivalds group's range checks.
pub fn freivalds_site(
    c: &mut Collect,
    reg: usize,
    eval: &crate::gadgets::freivalds::FreivaldsEval,
    r: &crate::layer::FreivaldsComponent,
) {
    fam_site(c, FamSite {
        reg: crate::gkrlink::Comp::Freivalds(reg),
        tag: format!("freivalds/{}", r.name),
        mut_env: "ZK_MUT_FREIVALDS",
        mut_key: reg,
        log_rows: r.lay.g.log_rows,
        unit: vec![true; r.lay.pq],
        pre: &r.pre,
        base: &r.base,
        #[cfg(feature = "cuda")]
        dev: r.dev.as_ref().map(|d| (&d.pre[..], &d.base[..])),
    }, eval);
}

/// B11 stream V, lever 4: the attention-output component's looks -- seven per
/// slot since stream JJ (E11) added the exponent window and the accumulator's
/// two limbs to `denp`'s own range check.
pub fn ao_site(
    c: &mut Collect,
    eval: &crate::gadgets::freivalds::AttnOutEval,
    r: &crate::layer::AoComponent,
) {
    fam_site(c, FamSite {
        reg: crate::gkrlink::Comp::Ao,
        tag: "attnout".to_string(),
        mut_env: "ZK_MUT_ATTNOUT",
        mut_key: 0,
        log_rows: r.lay.log_rows,
        unit: vec![true; r.looks_per_slot() * r.lay.per_row],
        pre: &r.pre,
        base: &r.base,
        #[cfg(feature = "cuda")]
        dev: r.dev.as_ref().map(|d| (&d.pre[..], &d.base[..])),
    }, eval);
}

/// One family component's column sources, as `fam_site` needs them.
///
/// B11 stream V, lever 3: stream U wrote this resolution once, for dequant. The
/// three families this stream adds want it unchanged -- the same prefix rule,
/// the same feed-or-host decision, the same two controls -- and the way to keep
/// them the same is to have one copy rather than four. `deq_site` is now the
/// first caller and its recorded digests are what say the extraction moved
/// nothing.
pub struct FamSite<'a> {
    pub reg: crate::gkrlink::Comp,
    /// what a control's message calls this component
    pub tag: String,
    /// the environment variable that bends one of this family's fractions
    pub mut_env: &'static str,
    /// the component index that variable names
    pub mut_key: usize,
    pub log_rows: u32,
    pub unit: Vec<bool>,
    /// the host mirrors; empty when the witness generation left the columns on the card
    pub pre: &'a [crate::common::Eval],
    pub base: &'a [crate::common::Eval],
    #[cfg(feature = "cuda")]
    pub dev: Option<(&'a [stwo::stwo_cuda::base_field_vec::BaseFieldVec],
                     &'a [stwo::stwo_cuda::base_field_vec::BaseFieldVec])>,
}

/// One family component's lookups, wherever its columns are.
///
/// This is `feed_site`'s shape for a family that is not the key tiles. A family's
/// interaction builder calls it in place of its own LogUp build under that
/// family's flag: the component's fractions become GKR input layers instead of
/// committed running-sum columns, and its claimed sum leaves `use_sums` for
/// `sums_cancel`.
///
/// The fractions come from the component's own fraction IR, which is the lowering
/// of the *un-ablated* evaluator -- stream T's precedent, and the reason the
/// kernel store already carries these fingerprints: under the flag the
/// evaluator emits no lookups at all and its IR would be empty. One definition
/// of the fraction, three readers: this, the family's hand-written builder (the
/// one `ZK_GENERATED_LOGUP=2` compares against) and `gkrlink`'s affine forms.
pub fn fam_site<E: FrameworkEval>(c: &mut Collect, s: FamSite<'_>, eval: &E) {
    let (reg, log_rows) = (s.reg, s.log_rows);
    let tag = &s.tag;
    c.reg = reg;
    let unit = s.unit;
    let ir = crate::scalars::cached_ir(eval, log_rows, log_rows, true, || {
        crate::devfrac::ir_of(eval, log_rows)
    });
    // `mat_logup_ir`'s prefix rule, asserted the same way: an o-side dequant
    // component commits three extra preprocessed selectors that belong to the
    // *argmax* component, its evaluator never reads them, and they sit after
    // the ones the IR loads -- so the prefix is exactly what to supply. A
    // family whose components carry no spare preprocessed columns passes it
    // unchanged.
    let np = ir.cols[0];
    let prefix = |n: usize| {
        assert!(n >= np,
                "{tag}: the component has {n} preprocessed columns, fewer than the \
                 {np} its fraction IR loads");
        np
    };
    // The family's own control bends one of *this component's* fractions, which is
    // a statement about the host builder's values, so it takes the host path
    // even where the witness generation left the columns on the card. Nothing else moves.
    let mutating = std::env::var(s.mut_env).ok()
        .and_then(|v| v.parse::<usize>().ok()) == Some(s.mut_key);
    #[cfg(feature = "cuda")]
    if let Some((dpre, dbase)) = s.dev {
        // The prover's path: the columns are on the card and so are the
        // fractions. The two controls below both need the host builder's values
        // -- one to bend them, one to compare against -- so each takes the
        // component off the feed and neither is on the prover's path.
        if !mutating && !feedcheck_on() {
            let pp: Vec<*const u32> =
                dpre[..prefix(dpre.len())].iter().map(|x| x.device_ptr).collect();
            let bp: Vec<*const u32> = dbase.iter().map(|x| x.device_ptr).collect();
            c.push_dev(0, log_rows, unit, eval, &pp, &bp);
            return;
        }
        eprintln!("{tag}: off the feed for a control, columns downloaded");
    }
    // The host builder's inputs: the component's own columns. Only the controls
    // reach here with a device witness generation, and the download they need is theirs.
    #[cfg(feature = "cuda")]
    let down = s.dev.map(|(dpre, dbase)| {
        // `common::eval_of` takes values in *coset* order and permutes them into
        // the committed one; a device column is already committed order, so the
        // download reinterprets rather than permutes -- the same construction
        // `device::take_all` makes on a build without the cuda feature.
        let domain = stwo::core::poly::circle::CanonicCoset::new(log_rows).circle_domain();
        let ev = |c: &stwo::stwo_cuda::base_field_vec::BaseFieldVec| {
            crate::common::Eval::new(
                domain,
                stwo::prover::backend::simd::column::BaseColumn::from_iter(c.to_vec()))
        };
        (dpre[..prefix(dpre.len())].iter().map(ev).collect::<Vec<_>>(),
         dbase.iter().map(ev).collect::<Vec<_>>())
    });
    #[cfg(not(feature = "cuda"))]
    let down: Option<(Vec<crate::common::Eval>, Vec<crate::common::Eval>)> = None;
    let (hpre, hbase) = match &down {
        Some((pp, bb)) => (&pp[..], &bb[..]),
        None => (&s.pre[..prefix(s.pre.len())], s.base),
    };
    let mut flat = deq_fracs(&ir, log_rows, hpre, hbase);
    // `ZK_GKR_FEED_CHECK=1`: the feed over an uploaded copy of the very columns the
    // host builder just read, cell for cell against it, then thrown away. Stream
    // T's control for the key tiles, on the family that joins them; the upload is
    // the check's, exactly as `ZK_GENERATED_LOGUP=3` pays one.
    #[cfg(feature = "cuda")]
    if feedcheck_on() {
        use stwo::stwo_cuda::base_field_vec::BaseFieldVec;
        use stwo::prover::backend::Column as _;
        let up = |v: &[crate::common::Eval]| -> Vec<BaseFieldVec> {
            v.iter().map(|e| BaseFieldVec::from_vec(e.values.to_cpu())).collect()
        };
        let (upre, ubase) = (up(hpre), up(hbase));
        let pp: Vec<*const u32> = upre.iter().map(|x| x.device_ptr).collect();
        let bp: Vec<*const u32> = ubase.iter().map(|x| x.device_ptr).collect();
        let mut scratch = Collect { reg: c.reg, ..Default::default() };
        scratch.push_dev(0, log_rows, unit.clone(), eval, &pp, &bp);
        scratch.check_fed_flat(0, &flat);
    }
    // B11 stream U, lever 3: the family's own instance-aimed control.
    //
    // `<FAM>MUT=<k>` bends one denominator of this family's component `k`'s input
    // layer -- one fraction of one look of one row -- and changes nothing else.
    // The GKR then proves the bent layer honestly, so its sumcheck is internally
    // consistent and every claim it yields is the claim of the layer it saw; the
    // *link*'s coefficient column is built from the committed columns, which are
    // untouched. The two can only disagree, and the prover's own boundary assert
    // is where they do, naming the height. That is the property the backend's
    // soundness rests on for this family -- the GKR's input layer and the
    // committed columns are one function -- and this is what says it is load
    // bearing rather than assumed.
    if mutating {
        eprintln!("{}: {tag}, look 0 of row 0, denominator bent", s.mut_env);
        flat[0][0].1 += PackedSecureField::broadcast(SecureField::one());
    }
    c.push_flat(0, log_rows, unit, &flat);
}

/// One component's fractions, every look at every packed row, laid out
/// `[look][vec_row]`.
///
/// `cudagen::frac_at` runs the whole straight-line IR once per packed row and
/// hands back every look, so asking it per `(look, row)` -- which is the shape
/// `input_layer` wants -- would run it `n_looks` times over. It runs once here
/// and the layers index the table.
pub fn deq_fracs(ir: &crate::cudagen::Ir, log_rows: u32,
                 pre: &[crate::common::Eval], base: &[crate::common::Eval])
                 -> Vec<Vec<(PackedSecureField, PackedSecureField)>> {
    use stwo::prover::backend::simd::m31::PackedM31;
    let sc = crate::cudagen::FracScalars::new(ir);
    let p: Vec<&[PackedM31]> = pre.iter().map(|e| &e.data[..]).collect();
    let b: Vec<&[PackedM31]> = base.iter().map(|e| &e.data[..]).collect();
    let n_vec_row = 1usize << (log_rows - LOG_N_LANES);
    (0..n_vec_row)
        .into_par_iter()
        .map(|vr| {
            let mut v = Vec::with_capacity(ir.ops.len());
            let mut out = Vec::with_capacity(ir.fracs.len());
            crate::cudagen::frac_at(ir, &sc, &p, &b, vr, &mut v, &mut out);
            out
        })
        .collect()
}

/// B11 stream T: the device feed's host half.
#[cfg(feature = "cuda")]
mod feedimpl {
    use std::ffi::{c_char, CString};

    use stwo::prover::backend::cuda::CudaBackend;
    use stwo::prover::lookups::gkr_prover::Layer;
    use stwo::prover::lookups::mle::Mle;
    use stwo::stwo_cuda::secure_field_vec::SecureFieldVec;
    use stwo_constraint_framework::FrameworkEval;

    use super::{Collect, Inst, Site, input_layer, plan, unit_of};

    /// One fed site's launch, kept so it can be launched a second time.
    ///
    /// B11 stream V, lever 1. Every field is a value or a device pointer into a
    /// *committed* column -- tree 0 and tree 1 -- and those live from their
    /// commit until the proof ends, which is strictly longer than the GKR. The
    /// site is kept whole because `plan` derives the instance split from it, so
    /// the re-feed produces the same instances in the same order with the same
    /// entry-to-slot map; the layers it returns are bit-identical to the ones
    /// the first launch produced, and the transcript cannot tell the two apart.
    pub struct FeedPlan {
        site: Site,
        /// this site's index in `Collect::sites`, which its instances carry.
        /// Not the plan's own index: a host-built site takes a slot in `sites`
        /// and none here.
        site_at: usize,
        /// the global instance indices this launch fills, in `plan` order
        pub insts: Vec<usize>,
        family: String,
        shape: u32,
        fingerprint: u32,
        pre_ptrs: Vec<*const u32>,
        base_ptrs: Vec<*const u32>,
        sc: Vec<u32>,
    }

    impl FeedPlan {
        /// Allocate this site's input layers and fill them with one launch.
        ///
        /// `at` is the site index the instances carry. The layers are allocated
        /// first and their slots handed to the kernel, so the write-back *is*
        /// the MLE layout: look `e` of instance `j` at position `p` is written
        /// at `p << log_rows` of that instance's column, which is exactly where
        /// `input_layer` puts it on the host. Nothing is copied, permuted or
        /// reshaped afterwards -- the permutation the plan's unit/non-unit
        /// split induces lives in the pointer table.
        pub fn witness_gen(&self) -> (Vec<Layer<CudaBackend>>, Vec<Inst>) {
            let at = self.site_at;
            let n_rows = 1usize << self.site.log_rows;
            let n_looks = self.site.n_looks();
            let mut num_at = vec![std::ptr::null_mut::<u32>(); n_looks];
            let mut den_at = vec![std::ptr::null_mut::<u32>(); n_looks];
            let mut layers = vec![];
            let mut insts = vec![];
            for inst in plan(std::slice::from_ref(&self.site)) {
                let inst = Inst { site: at, ..inst };
                let n = inst.fractions();
                let den = SecureFieldVec::new_uninitialized(n);
                let dp = den.device_ptr as *mut u32;
                for (p, e) in inst.ents.iter().enumerate() {
                    den_at[*e] = unsafe { dp.add(4 * p * n_rows) };
                }
                let layer = if inst.unit {
                    Layer::LogUpSingles { denominators: Mle::new(den) }
                } else {
                    let num = SecureFieldVec::new_uninitialized(n);
                    let np = num.device_ptr as *mut u32;
                    for (p, e) in inst.ents.iter().enumerate() {
                        num_at[*e] = unsafe { np.add(4 * p * n_rows) };
                    }
                    Layer::LogUpGeneric {
                        numerators: Mle::new(num),
                        denominators: Mle::new(den),
                    }
                };
                layers.push(layer);
                insts.push(inst);
            }
            assert!(den_at.iter().all(|p| !p.is_null()),
                    "the feed's plan left a look with no denominator slot");
            let cname = CString::new(self.family.clone()).unwrap();
            unsafe {
                gen_zk_feed_dispatch(
                    cname.as_ptr(),
                    self.shape,
                    self.fingerprint,
                    self.pre_ptrs.as_ptr(),
                    self.pre_ptrs.len() as u32,
                    self.base_ptrs.as_ptr(),
                    self.base_ptrs.len() as u32,
                    self.sc.as_ptr(),
                    (self.sc.len() / 4) as u32,
                    num_at.as_ptr(),
                    den_at.as_ptr(),
                    n_rows as u32,
                    n_looks as u32,
                );
            }
            (layers, insts)
        }
    }

    /// The GKR's answer when it asks for a dropped input layer back.
    ///
    /// One launch fills every instance of a site, so the first instance of a
    /// site to be asked for builds all of that site's layers and the rest are
    /// handed the ones already built. The cache is `RefCell` because
    /// `InputRefeed` takes `&self`: `prove_batch` holds the layers it is
    /// folding and can only borrow this.
    pub struct Refeed<'a> {
        feeds: &'a [FeedPlan],
        /// which feed plan owns each instance, and `None` for an instance the
        /// host built and uploaded -- those are not dropped and never asked for
        owner: Vec<Option<usize>>,
        built: std::cell::RefCell<Vec<Option<Layer<CudaBackend>>>>,
        pub calls: std::cell::Cell<usize>,
    }

    impl<'a> Refeed<'a> {
        /// `n` is the instance count; `sites_of` gives each fed instance's site.
        pub fn new(feeds: &'a [FeedPlan], n: usize) -> Self {
            let mut owner = vec![None; n];
            for (f, fp) in feeds.iter().enumerate() {
                for i in &fp.insts {
                    owner[*i] = Some(f);
                }
            }
            Refeed {
                feeds,
                owner,
                built: std::cell::RefCell::new((0..n).map(|_| None).collect()),
                calls: std::cell::Cell::new(0),
            }
        }
    }

    impl stwo::prover::lookups::gkr_prover::InputRefeed<CudaBackend> for Refeed<'_> {
        fn droppable(&self, instance: usize) -> bool {
            self.owner[instance].is_some()
        }

        fn refeed(&self, instance: usize) -> Layer<CudaBackend> {
            let f = self.owner[instance].expect("a host instance was never dropped");
            if self.built.borrow()[instance].is_none() {
                let fp = &self.feeds[f];
                let (layers, _) = fp.witness_gen();
                self.calls.set(self.calls.get() + 1);
                let mut b = self.built.borrow_mut();
                for (k, l) in fp.insts.iter().zip(layers) {
                    b[*k] = Some(l);
                }
            }
            self.built.borrow_mut()[instance].take().expect("the re-feed built no layer here")
        }
    }

    unsafe extern "C" {
        /// The generated fraction kernel's unpaired variant, behind the same
        /// per-family dispatcher key the paired one uses. `num_at` and `den_at`
        /// carry one device pointer per *look*, aimed at that look's slot in the
        /// GKR instance layer; a null numerator slot is a singles instance,
        /// which stores no numerators at all.
        fn gen_zk_feed_dispatch(
            family: *const c_char,
            shape: u32,
            fingerprint: u32,
            pre_cols: *const *const u32,
            n_pre: u32,
            base_cols: *const *const u32,
            n_base: u32,
            sc_host: *const u32,
            n_sc: u32,
            num_at: *const *mut u32,
            den_at: *const *mut u32,
            n_rows: u32,
            n_looks: u32,
        );
    }

    impl Collect {
        /// One site's input layers, allocated on the device and filled by one
        /// feed launch.
        ///
        /// The layers are allocated first and their slots handed to the kernel,
        /// so the write-back *is* the MLE layout: look `e` of instance `j` at
        /// position `p` is written at `p << log_rows` of that instance's column,
        /// which is exactly where `input_layer` puts it on the host. Nothing is
        /// copied, permuted or reshaped afterwards -- the permutation the plan's
        /// unit/non-unit split induces lives in the pointer table.
        #[allow(clippy::too_many_arguments)]
        pub fn push_dev<E: FrameworkEval>(
            &mut self,
            which: usize,
            log_rows: u32,
            unit: Vec<bool>,
            eval: &E,
            pre_ptrs: &[*const u32],
            base_ptrs: &[*const u32],
        ) {
            let t0 = std::time::Instant::now();
            let site = Site {
                reg: self.reg,
                which,
                log_rows,
                unit,
            };
            let at = self.sites.len();
            let n_rows = 1usize << log_rows;
            let n_looks = site.n_looks();

            // The IR, and the kernel compiled for it. Same lowering, same
            // fingerprint and same shape id as the committed backend's paired
            // kernel: the two differ only in the write-back.
            let ir = crate::scalars::cached_ir(eval, log_rows, log_rows, true, || {
                crate::devfrac::ir_of(eval, log_rows)
            });
            let family = eval.cuda_eval_name();
            assert_eq!(ir.fracs.len(), n_looks,
                       "{family}: the fraction IR has {} lookups, the site {n_looks}",
                       ir.fracs.len());
            assert_eq!(ir.cols[1], base_ptrs.len(),
                       "{family}: base column count");
            // The key tiles read the base tree alone (18.7's measurement, asserted
            // per shape here as `devfrac::assert_supplied` asserts it there), so
            // they hand over no preprocessed pointers and a preprocessed load
            // would index a null. B11 stream U: the dequant family *does* read
            // one -- the `chain` variant's offline weight-scale mantissa -- so a
            // caller may supply the tree instead, and the assert holds for
            // whichever case it is.
            if pre_ptrs.is_empty() {
                assert!(
                    !ir.ops.iter().any(|o| matches!(*o, crate::cudagen::Op::Load { tree: 0, .. })),
                    "{family}: the feed's IR loads a preprocessed column and the \
                     caller supplied none"
                );
            } else {
                assert_eq!(ir.cols[0], pre_ptrs.len(),
                           "{family}: the feed's IR loads {} preprocessed columns, the \
                            caller supplied {}", ir.cols[0], pre_ptrs.len());
            }
            let (shape, kernel_looks) = crate::gencudakernels::gen_shape_of(
                family, ir.fingerprint())
                .unwrap_or_else(|| panic!(
                    "no feed kernel for {family} fingerprint {:#010x} ({n_looks} lookups): \
                     regenerate with `zk-prover genemit --families {family}` and rebuild \
                     libstwo_cuda.a", ir.fingerprint()));
            assert_eq!(kernel_looks, n_looks, "{family}: the kernel's look count");

            let sc: Vec<u32> =
                ir.scalars.iter().flat_map(|s| s.to_m31_array().map(|x| x.0)).collect();
            let nulls = vec![std::ptr::null::<u32>(); ir.cols[0]];
            let pre = if pre_ptrs.is_empty() { nulls } else { pre_ptrs.to_vec() };

            // B11 stream V, lever 1: the launch is now `FeedPlan::witness_gen`, and
            // everything it needs is plain data that outlives the GKR -- the
            // dispatch key, the committed columns' device pointers, the scalar
            // blob, and the site, whose instance split `plan` reproduces. So
            // the second call in the last round is the first call again, and
            // "idempotent" is a property of the code rather than a hope.
            let fp = FeedPlan {
                site: site.clone(),
                site_at: at,
                insts: (0..plan(std::slice::from_ref(&site)).len())
                    .map(|k| self.insts.len() + k)
                    .collect(),
                family: family.to_string(),
                shape: shape as u32,
                fingerprint: ir.fingerprint(),
                pre_ptrs: pre,
                base_ptrs: base_ptrs.to_vec(),
                sc,
            };
            let (layers, insts) = fp.witness_gen();
            self.dev_layers.extend(layers);
            self.on_dev.extend(std::iter::repeat_n(true, insts.len()));
            self.insts.extend(insts);
            self.feeds.push(fp);
            self.sites.push(site);
            self.direct.push(stwo::core::fields::qm31::SecureField::default());
            self.feed_secs += t0.elapsed().as_secs_f64();
        }

        /// `ZK_GKR_FEED_CHECK=1` on a host-built key tile: the feed over an uploaded
        /// copy of this site's committed columns, against the host builder,
        /// cell for cell. The fed layers are the check's and are dropped.
        #[allow(clippy::too_many_arguments)]
        pub fn feed_check(
            &self,
            air: &crate::devfrac::KeyTileAir<'_>,
            which: usize,
            log_rows: u32,
            looks: &[crate::gadgets::attention::trace::Look],
            at: &[usize],
            cols: &[crate::common::Eval],
            pre: &[crate::common::Eval],
            exp: &crate::gadgets::attention::air::ExpRel,
            ranges: &[crate::relations::RangeRel],
        ) {
            use stwo::prover::backend::Column as _;
            use stwo::stwo_cuda::base_field_vec::BaseFieldVec;
            let (lo, hi) = (at[which], at[which + 1]);
            let up: Vec<BaseFieldVec> =
                cols[lo..hi].iter().map(|e| BaseFieldVec::from_vec(e.values.to_cpu())).collect();
            let ptrs: Vec<*const u32> = up.iter().map(|c| c.device_ptr).collect();
            let mut scratch = Collect { reg: self.reg, ..Default::default() };
            match which {
                0 => {
                    let mut e = air.chan.clone();
                    e.gkr = false;
                    scratch.push_dev(which, log_rows, unit_of(looks), &e, &[], &ptrs);
                }
                1 => {
                    let mut e = air.key.clone();
                    e.gkr = false;
                    scratch.push_dev(which, log_rows, unit_of(looks), &e, &[], &ptrs);
                }
                3 => {
                    let mut e = air.row.clone();
                    e.gkr = false;
                    scratch.push_dev(which, log_rows, unit_of(looks), &e, &[], &ptrs);
                }
                _ => panic!("key tile component {which} has no eval"),
            }
            scratch.check_fed(0, looks, cols, pre, exp, ranges);
        }

        /// `ZK_GKR_FEED_CHECK=1`: every fed instance from `first` on, cell for cell
        /// against the host builder reading the same committed columns.
        pub fn check_fed(
            &self,
            first: usize,
            looks: &[crate::gadgets::attention::trace::Look],
            cols: &[crate::common::Eval],
            pre: &[crate::common::Eval],
            exp: &crate::gadgets::attention::air::ExpRel,
            ranges: &[crate::relations::RangeRel],
        ) {
            use stwo::prover::backend::Column as _;
            let frac = |look: usize, vr: usize| {
                crate::gadgets::attention::trace::look_frac(&looks[look], vr, cols, pre, exp, ranges)
            };
            let n_dev = self.dev_layers.len();
            let mut cells = 0usize;
            for (k, inst) in self.insts[first..].iter().enumerate() {
                let got = &self.dev_layers[n_dev - (self.insts.len() - first) + k];
                let want = input_layer(inst, &frac);
                let cmp = |tag: &str,
                           g: &Mle<CudaBackend, stwo::core::fields::qm31::SecureField>,
                           w: &Mle<
                    stwo::prover::backend::simd::SimdBackend,
                    stwo::core::fields::qm31::SecureField,
                >| {
                    let (a, b) = (g.to_cpu(), w.to_cpu());
                    assert_eq!(a.len(), b.len(), "feedcheck: {tag} length");
                    if let Some(i) = (0..a.len()).find(|i| a[*i] != b[*i]) {
                        panic!(
                            "feedcheck: instance {k} {tag} differs at {i}: \
                             the feed says {:?}, the host builder {:?}",
                            a[i], b[i]
                        );
                    }
                    a.len()
                };
                match (got, &want) {
                    (
                        Layer::LogUpSingles { denominators: g },
                        Layer::LogUpSingles { denominators: w },
                    ) => cells += cmp("denominators", g, w),
                    (
                        Layer::LogUpGeneric { numerators: gn, denominators: gd },
                        Layer::LogUpGeneric { numerators: wn, denominators: wd },
                    ) => {
                        cells += cmp("numerators", gn, wn);
                        cells += cmp("denominators", gd, wd);
                    }
                    _ => panic!("feedcheck: instance {k} has a different layer kind"),
                }
            }
            eprintln!(
                "feedcheck: {} fed instances, {cells} cells match the host builder",
                self.insts.len() - first
            );
        }

        /// The same check for a family whose fractions arrive as a table
        /// (B11 stream U, lever 3: the dequant components).
        pub fn check_fed_flat(
            &self,
            first: usize,
            flat: &[Vec<(
                stwo::prover::backend::simd::qm31::PackedSecureField,
                stwo::prover::backend::simd::qm31::PackedSecureField,
            )>],
        ) {
            use stwo::prover::backend::Column as _;
            let frac = |look: usize, vr: usize| flat[vr][look];
            let n_dev = self.dev_layers.len();
            let mut cells = 0usize;
            for (k, inst) in self.insts[first..].iter().enumerate() {
                let got = &self.dev_layers[n_dev - (self.insts.len() - first) + k];
                let want = input_layer(inst, &frac);
                let cmp = |tag: &str,
                           g: &Mle<CudaBackend, stwo::core::fields::qm31::SecureField>,
                           w: &Mle<
                    stwo::prover::backend::simd::SimdBackend,
                    stwo::core::fields::qm31::SecureField,
                >| {
                    let (a, b) = (g.to_cpu(), w.to_cpu());
                    assert_eq!(a.len(), b.len(), "feedcheck: {tag} length");
                    if let Some(i) = (0..a.len()).find(|i| a[*i] != b[*i]) {
                        panic!(
                            "feedcheck: instance {k} {tag} differs at {i}: \
                             the feed says {:?}, the host builder {:?}",
                            a[i], b[i]
                        );
                    }
                    a.len()
                };
                match (got, &want) {
                    (
                        Layer::LogUpSingles { denominators: g },
                        Layer::LogUpSingles { denominators: w },
                    ) => cells += cmp("denominators", g, w),
                    (
                        Layer::LogUpGeneric { numerators: gn, denominators: gd },
                        Layer::LogUpGeneric { numerators: wn, denominators: wd },
                    ) => {
                        cells += cmp("numerators", gn, wn);
                        cells += cmp("denominators", gd, wd);
                    }
                    _ => panic!("feedcheck: instance {k} has a different layer kind"),
                }
            }
            eprintln!(
                "feedcheck: {} fed instances, {cells} cells match the host builder",
                self.insts.len() - first
            );
        }
    }
}

/// B11 stream R: the device backend's layer plumbing.
///
/// The collected input layers are host (simd) columns; the sumcheck runs on
/// `CudaBackend`. The upload is timed on its own line because on the composed
/// cuda backend it is a stand-in: `gen_zk_dispatch` computes these very fractions
/// on the device already, so a resident backend pays a device-side *reshape* (one
/// pass moving the paired fractions into MLE layout), not a PCIe transfer. The
/// reshape's cost is measured directly -- one device-to-device copy of every
/// layer -- and reported beside the upload so neither is assumed.
#[cfg(feature = "cuda")]
pub mod dev {
    use stwo::prover::backend::Column as _;
    use stwo::prover::backend::cuda::CudaBackend;
    use stwo::prover::backend::simd::SimdBackend;
    use stwo::prover::lookups::gkr_prover::Layer;
    use stwo::prover::lookups::mle::Mle;
    use stwo::stwo_cuda::base_field_vec::BaseFieldVec;
    use stwo::stwo_cuda::secure_field_vec::SecureFieldVec;

    pub fn sync() {
        unsafe { stwo::stwo_cuda::bindings::gkr_device_synchronize() };
    }

    pub fn upload_layer(l: &Layer<SimdBackend>) -> Layer<CudaBackend> {
        let up = |m: &Mle<SimdBackend, stwo::core::fields::qm31::SecureField>| {
            Mle::new(SecureFieldVec::from_vec(m.to_cpu()))
        };
        match l {
            Layer::GrandProduct(mle) => Layer::GrandProduct(up(mle)),
            Layer::LogUpGeneric { numerators, denominators } => Layer::LogUpGeneric {
                numerators: up(numerators),
                denominators: up(denominators),
            },
            Layer::LogUpMultiplicities { numerators, denominators } => {
                Layer::LogUpMultiplicities {
                    numerators: Mle::new(BaseFieldVec::from_vec(numerators.to_cpu())),
                    denominators: up(denominators),
                }
            }
            Layer::LogUpSingles { denominators } => Layer::LogUpSingles {
                denominators: up(denominators),
            },
        }
    }

    /// Grow the device memory pool once, ahead of the circuit evaluation.
    ///
    /// `gen_layers` materialises every GKR layer of every instance before the
    /// first sumcheck round, and nothing is freed until the rounds consume
    /// them -- so without this, every one of its allocations extends the
    /// pool's high-water mark and pays the driver's synchronous mapping cost,
    /// which profiling showed as ~0.5-1.5 ms per `next_layer` call regardless
    /// of size. One big allocation maps the same memory in one driver call;
    /// freeing it hands the range to the pool's free list.
    pub fn prewarm(layers: &[Layer<CudaBackend>]) -> f64 {
        use stwo::stwo_cuda::bindings;
        let input_qm31: usize = layers
            .iter()
            .map(|l| match l {
                Layer::GrandProduct(m) => m.len(),
                Layer::LogUpGeneric { numerators, denominators } => {
                    numerators.len() + denominators.len()
                }
                Layer::LogUpMultiplicities { denominators, .. } => 2 * denominators.len(),
                Layer::LogUpSingles { denominators } => denominators.len(),
            })
            .sum();
        // every level of every tree sums to under 2x the input layer; the
        // halved trees are generic (num + den) even where the input was
        // singles, so budget 4x, plus the eq tables and slack
        // B11 stream T: the pre-grow is **off** by default now, and the reason
        // is the feed. Stream R budgeted four times the input layer in one
        // allocation so that `gen_layers`' own allocations would not each extend
        // the pool's high-water mark -- worth it when the input layers arrived by
        // upload. The feed allocates every input layer before this runs, so the
        // pool is already at size and the extra budget only raises the mark:
        // measured on the card at C=16384, the backend's peak is 57,185 MiB with the
        // 4x pre-grow and 44,705 MiB without, against 46,337 MiB flag off -- so
        // the pre-grow alone was 12.4 GiB, and it turned a 1.6 GiB saving into a
        // 10.6 GiB regression. The wall is the same either way (2.050 s against
        // 2.068 s) and at C=4096 the peak is unmoved (33,109 MiB both ways).
        // `ZK_GKR_POOL_FACTOR=<n>` puts stream R's budget back at factor n.
        //
        // The *kernel* half of prewarm below is untouched: that is the module
        // load, which is what R's pass 4 actually measured at -59%.
        let factor: usize = std::env::var("ZK_GKR_POOL_FACTOR")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        let budget = factor * input_qm31 + (1 << 22);
        let t = std::time::Instant::now();
        if factor > 0 {
            unsafe {
                let p = bindings::gkr_alloc_qm31(budget as u32);
                bindings::cuda_free_memory(p as *const std::ffi::c_void);
            }
        }
        // Launch every kernel the sumcheck uses once, on two elements. The
        // first launch of a kernel pays its module load -- measured at ~88 ms
        // for the first `next_layer` of a process, against ~13 us for every
        // call after it -- and the chunk loop pays that once per *process*,
        // not per chunk, exactly like `wall/cuda_init`.
        let two = SecureFieldVec::from_vec(vec![
            stwo::core::fields::qm31::SecureField::from_u32_unchecked(1, 0, 0, 0),
            stwo::core::fields::qm31::SecureField::from_u32_unchecked(2, 0, 0, 0),
        ]);
        let two_base = BaseFieldVec::from_vec(vec![
            stwo::core::fields::m31::M31::from_u32_unchecked(1),
            stwo::core::fields::m31::M31::from_u32_unchecked(2),
        ]);
        let out = |n: usize| SecureFieldVec::new_uninitialized(n);
        let (o1, o2, o3) = (out(1), out(1), out(2));
        let lambda = stwo::stwo_cuda::bindings::CudaSecureField::zero;
        unsafe {
            bindings::gkr_next_layer_grand_product(two.device_ptr, o1.device_ptr, 1);
            bindings::gkr_next_layer_logup_generic(
                two.device_ptr, two.device_ptr, o1.device_ptr, o2.device_ptr, 1);
            bindings::gkr_next_layer_logup_multiplicities(
                two_base.device_ptr, two.device_ptr, o1.device_ptr, o2.device_ptr, 1);
            bindings::gkr_next_layer_logup_singles(
                two.device_ptr, o1.device_ptr, o2.device_ptr, 1);
            bindings::gkr_sum_grand_product(two.device_ptr, two.device_ptr, 1, o3.device_ptr, 1);
            bindings::gkr_sum_logup_generic(
                two.device_ptr, two.device_ptr, two.device_ptr, lambda(), 1, o3.device_ptr, 1);
            bindings::gkr_sum_logup_multiplicities(
                two_base.device_ptr, two.device_ptr, two.device_ptr, lambda(), 1,
                o3.device_ptr, 1);
            bindings::gkr_sum_logup_singles(
                two.device_ptr, two.device_ptr, lambda(), 1, o3.device_ptr, 1);
            bindings::fix_first_variable_secure_field(
                two.device_ptr, 2, lambda(), o1.device_ptr);
            bindings::fix_first_variable_base_field(
                two_base.device_ptr, 2, lambda(), o1.device_ptr);
            bindings::gen_eq_evals(
                lambda(),
                two.device_ptr as *const stwo::stwo_cuda::bindings::CudaSecureField,
                0,
                o1.device_ptr as *const stwo::stwo_cuda::bindings::CudaSecureField,
                1,
            );
        }
        sync();
        t.elapsed().as_secs_f64()
    }

    /// One device-to-device pass over every layer column: what putting already
    /// resident fractions into MLE layout costs. The copies are dropped; only
    /// the time is kept.
    pub fn reshape_pass(layers: &[Layer<CudaBackend>]) {
        let copy = |src: &SecureFieldVec| {
            let mut dst = SecureFieldVec::new_uninitialized(src.len());
            dst.copy_from(src);
        };
        for l in layers {
            match l {
                Layer::GrandProduct(m) => copy(m),
                Layer::LogUpGeneric { numerators, denominators } => {
                    copy(numerators);
                    copy(denominators);
                }
                Layer::LogUpMultiplicities { denominators, .. } => copy(denominators),
                Layer::LogUpSingles { denominators } => copy(denominators),
            }
        }
    }
}

/// Prove everything a `Collect` gathered, in one batched GKR.
///
/// On the cuda build the sumcheck runs on the device -- that is the backend this
/// build exists to measure -- unless `ZK_GKR_HOST=1` pins it to the host simd
/// backend for an on-box comparison. The proof bytes are the same either way,
/// which is the port's digest gate.
pub fn prove_collected(channel: &mut impl Channel, c: Collect) -> Proved {
    mix_shape(channel, &c.sites);
    let fractions = c.insts.iter().map(|i| i.fractions()).sum();
    let unpaired = c.sites.iter().map(|s| s.n_looks() << s.log_rows).sum();
    let n_instances = c.insts.len();

    let n_fed = c.on_dev.iter().filter(|d| **d).count();

    #[cfg(feature = "cuda")]
    let (proof, artifact, backend, upload_secs, reshape_secs, prewarm_secs, prove_secs, phases,
         refeeds) = {
        if std::env::var("ZK_GKR_HOST").map(|v| v != "0").unwrap_or(false) {
            assert_eq!(n_fed, 0,
                       "`ZK_GKR_HOST=1` proves on the host, which needs every input                         layer there, and the device feed built {n_fed} of them");
            let t1 = Instant::now();
            let (proof, artifact) = prove_batch(channel, c.layers);
            (proof, artifact, "host simd", 0.0, 0.0, 0.0, t1.elapsed().as_secs_f64(), None, 0)
        } else {
            use stwo::prover::backend::cuda::lookups::gkr::prof;
            let t0 = Instant::now();
            // B11 stream T: a fed instance is already device-resident in MLE
            // layout and crosses nothing; only a host-built one -- a diagonal block
            // key tile, whose witness generation stays on the host -- is uploaded. The two lists
            // interleave back into instance order, which is the transcript's.
            let mut hs = c.layers.iter();
            let mut ds = c.dev_layers.into_iter();
            let layers: Vec<_> = c
                .on_dev
                .iter()
                .map(|d| {
                    if *d {
                        ds.next().expect("an instance marked device-built has no layer")
                    } else {
                        dev::upload_layer(hs.next().expect("a host instance has no layer"))
                    }
                })
                .collect();
            dev::sync();
            let upload_secs = t0.elapsed().as_secs_f64();
            // B11 stream V, lever 1: three boundaries the wall report does not
            // name, because until this stream nothing asked *when* the card was
            // fullest. Every input layer is resident here; the towers are not
            // built yet.
            crate::util::mark_timeline("gkr/inputs_resident");
            // Once per process in production -- the pool keeps its high-water
            // mark across chunks -- so it is timed and reported, not counted
            // among the backend's per-chunk adds. Before the reshape, so no later
            // allocation pays the pool's first-touch mapping.
            let prewarm_secs = dev::prewarm(&layers);
            // The reshape pass was stream R's *measurement* of what a backend whose
            // fractions were already resident would pay to put them into MLE
            // layout. The feed writes that layout directly, so where it ran
            // there is no pass to time and none to pay.
            let reshape_secs = if n_fed == 0 {
                let t0 = Instant::now();
                dev::reshape_pass(&layers);
                dev::sync();
                t0.elapsed().as_secs_f64()
            } else {
                0.0
            };
            crate::util::mark_timeline("gkr/prewarmed");
            prof::reset();
            // B11 stream X, lever 1: the fork's own split of what `prove_batch`
            // spends *between* the backend's four phases -- towers, oracles,
            // the sumcheck call, the masks and the channel. Reported beside
            // them because the question this stream asks is which term grows
            // with the instance count, and the backend cannot see the terms
            // that do not launch a kernel.
            stwo::prover::lookups::gkr_prover::roundprof::reset();
            let t1 = Instant::now();
            // B11 stream V, lever 1: the fed instances' input layers are freed
            // as their first halving is produced and re-fed in the last round,
            // which is the only round that reads them. The uploaded instances
            // -- a host-built diagonal block key tile's -- have no feed behind them, so
            // `Refeed::droppable` says no and they are held as stream Q held
            // them. `ZK_GKR_REFEED=0` puts every input layer back for the
            // whole run: the same binary, the before against the after.
            let refeed = crate::util::switch_on("ZK_GKR_REFEED")
                .then(|| feedimpl::Refeed::new(&c.feeds, n_instances));
            let (proof, artifact) = stwo::prover::lookups::gkr_prover::prove_batch_refeed(
                channel,
                layers,
                refeed.as_ref().map(|r| r as &dyn stwo::prover::lookups::gkr_prover::InputRefeed<_>),
            );
            dev::sync();
            let prove_secs = t1.elapsed().as_secs_f64();
            let refeeds = refeed.as_ref().map(|r| r.calls.get()).unwrap_or(0);
            crate::util::mark_timeline("gkr/proved");
            let phases = prof::on().then(|| {
                let mut v = prof::snapshot();
                v.extend(stwo::prover::lookups::gkr_prover::roundprof::snapshot());
                v
            });
            (proof, artifact, "device", upload_secs, reshape_secs, prewarm_secs, prove_secs,
             phases, refeeds)
        }
    };
    #[cfg(not(feature = "cuda"))]
    let (proof, artifact, backend, upload_secs, reshape_secs, prewarm_secs, prove_secs, phases,
         refeeds) = {
        let t1 = Instant::now();
        let (proof, artifact) = prove_batch(channel, c.layers);
        let phases: Option<Vec<(&'static str, f64, u64)>> = None;
        (proof, artifact, "host simd", 0.0, 0.0, 0.0, t1.elapsed().as_secs_f64(), phases, 0)
    };

    let sums = sums_of(c.sites.len(), &c.insts, &proof.output_claims_by_instance);
    let t2 = Instant::now();
    let proof_bytes = serde_json::to_vec(&proof).map(|b| b.len()).unwrap_or(0);
    let assembly_secs = t2.elapsed().as_secs_f64();
    Proved {
        proof,
        artifact,
        sums,
        cost: Cost {
            layers_secs: c.layers_secs,
            feed_secs: c.feed_secs,
            n_fed,
            prove_secs,
            n_instances,
            fractions,
            unpaired,
            proof_bytes,
            backend,
            upload_secs,
            reshape_secs,
            prewarm_secs,
            assembly_secs,
            phases,
            refeeds,
            // the caller adds what it measured around the whole call
            total_secs: c.layers_secs + c.feed_secs + prove_secs,
        },
        insts: c.insts,
    }
}

/// What proving the key tiles' lookups cost, apart from the proof itself.
#[derive(Default, Clone, Debug)]
pub struct Cost {
    /// building every instance's input layer on the host
    pub layers_secs: f64,
    /// B11 stream T: the device feed's launches -- the same input layers, built
    /// where the columns already are, straight into MLE layout
    pub feed_secs: f64,
    /// how many instances the feed built
    pub n_fed: usize,
    /// `prove_batch`: the circuit evaluation and every layer's sumcheck
    pub prove_secs: f64,
    pub n_instances: usize,
    /// the fractions the input layers hold -- unpaired since B11 stream S
    pub fractions: usize,
    /// one per lookup -- equal to `fractions` since the input went unpaired
    pub unpaired: usize,
    /// the serialised `GkrBatchProof`
    pub proof_bytes: usize,
    /// where the sumcheck ran: "host simd" or "device"
    pub backend: &'static str,
    /// host-to-device transfer of the collected layers (device backend only). On a
    /// composed cuda backend the fractions are already resident, so this line is a
    /// stand-in and `reshape_secs` is the honest input cost.
    pub upload_secs: f64,
    /// one device-to-device pass over every layer: putting already resident
    /// fractions into MLE layout
    pub reshape_secs: f64,
    /// growing the device pool to the circuit's high-water mark, once per
    /// process in production (the pool persists across chunks)
    pub prewarm_secs: f64,
    /// serialising the proof
    pub assembly_secs: f64,
    /// the device sumcheck's phase split, when `ZK_GKR_PROFILE=1`
    pub phases: Option<Vec<(&'static str, f64, u64)>>,
    /// B11 stream V, lever 1: how many feed launches the last round asked for,
    /// one per fed site. Zero under `ZK_GKR_REFEED=0` and on the host backend.
    pub refeeds: usize,
    /// everything the backend costs where the committed path costs its LogUp build:
    /// the input layers plus `prove_batch`
    pub total_secs: f64,
}

impl Cost {
    pub fn json(&self) -> serde_json::Value {
        let mut v = serde_json::json!({
            "instances": self.n_instances,
            "fractions": self.fractions,
            "unpaired": self.unpaired,
            "layers_secs": self.layers_secs,
            "feed_secs": self.feed_secs,
            "fed_instances": self.n_fed,
            "prove_secs": self.prove_secs,
            "total_secs": self.total_secs,
            "proof_bytes": self.proof_bytes,
            "refeeds": self.refeeds,
            "backend": self.backend,
            "upload_secs": self.upload_secs,
            "reshape_secs": self.reshape_secs,
            "prewarm_secs": self.prewarm_secs,
            "assembly_secs": self.assembly_secs,
        });
        if let Some(ph) = &self.phases {
            v["phases"] = ph
                .iter()
                .map(|(n, s, c)| serde_json::json!({"phase": n, "secs": s, "calls": c}))
                .collect();
        }
        v
    }
}

pub struct Proved {
    pub proof: GkrBatchProof,
    /// B11 stream S, lever 3: the OOD point and input-layer claims the link's
    /// columns are built from -- the same artifact the verifier re-derives
    /// from the proof.
    pub artifact: GkrArtifact,
    /// one claimed fractional sum per site, in site order -- what the committed
    /// running-sum columns claim today
    pub sums: Vec<SecureField>,
    pub insts: Vec<Inst>,
    pub cost: Cost,
}

/// Fold the batch's per-instance output claims into one claimed sum per site.
///
/// A LogUp circuit's output layer is the pair `[numerator, denominator]` of the
/// whole instance's fractional sum, so an instance contributes `p / q` and a
/// site is the sum of its instances. This is the only place the GKR's answer
/// crosses into the statement, and it is the same quantity
/// `interaction_phase` pushes onto `use_sums`.
fn sums_of(n_sites: usize, insts: &[Inst], claims: &[Vec<SecureField>]) -> Vec<SecureField> {
    let mut sums = vec![SecureField::zero(); n_sites];
    for (i, c) in insts.iter().zip(claims) {
        // an empty or malformed claim is a rejected proof, not a panic here
        if c.len() == 2 && !c[1].is_zero() {
            sums[i.site] += c[0] / c[1];
        }
    }
    sums
}

/// Verify a serialised batch GKR proof and return the claimed sum per site.
///
/// The instance list is derived from the plan by `plan`, never read from the
/// proof, so a prover cannot change the shape of what it proved. What comes back
/// is unchecked in one specific sense that the caller has to close: the
/// `GkrArtifact` carries claimed evaluations of the input layers at a random
/// point, and nothing here bindings those to the committed base columns.
pub fn verify(
    channel: &mut impl Channel,
    sites: &[Site],
    bytes: &[u8],
) -> Result<(Vec<SecureField>, GkrArtifact, Vec<Inst>), String> {
    let proof: GkrBatchProof =
        serde_json::from_slice(bytes).map_err(|e| format!("unparseable GKR proof: {e}"))?;
    mix_shape(channel, sites);
    let insts = plan(sites);
    if proof.output_claims_by_instance.len() != insts.len() {
        return Err(format!(
            "the GKR proof carries {} instances, the plan says {}",
            proof.output_claims_by_instance.len(),
            insts.len()
        ));
    }
    // every instance's variable count is fixed by the plan
    let artifact = partially_verify_batch(gates(&insts), &proof, channel)
        .map_err(|e: GkrError| format!("{e}"))?;
    for (i, n) in insts.iter().zip(&artifact.n_variables_by_instance) {
        if i.n_variables() != *n {
            return Err(format!(
                "instance over {} variables, the plan says {}",
                n,
                i.n_variables()
            ));
        }
    }
    let sums = sums_of(sites.len(), &insts, &proof.output_claims_by_instance);
    Ok((sums, artifact, insts))
}

/// The claimed sum of one instance, read straight off its input layer.
///
/// This is the reference the GKR is checked against in the self-test: the plain
/// sum of `num / den` over every entry, by inversion, with no sumcheck
/// involved. It is `O(size)` and only for the test.
pub fn direct_sum<F>(inst: &Inst, frac: &F) -> SecureField
where
    F: Fn(usize, usize) -> (PackedSecureField, PackedSecureField) + Sync,
{
    let n_vec_row = 1usize << (inst.log_rows - LOG_N_LANES);
    let packed: PackedSecureField = (0..inst.ents.len() * n_vec_row)
        .into_par_iter()
        .map(|k| {
            let (b, vr) = (k / n_vec_row, k % n_vec_row);
            let (n, d) = frac(inst.ents[b], vr);
            n * d.inverse()
        })
        .reduce(|| PackedSecureField::zero(), |a, b| a + b);
    packed.to_array().iter().fold(SecureField::zero(), |a, b| a + *b)
}

/// A domain separator, so the GKR's transcript position cannot be confused with
/// anything else the channel carries.
pub fn mix_shape(channel: &mut impl Channel, sites: &[Site]) {
    let mut felts = vec![SecureField::from_u32_unchecked(0x51, 0x47, 0x4b, 0x52)];
    for s in sites {
        // B11 stream U, lever 3: the family joins the separator, because two
        // families' component indices are unrelated and a transcript that could
        // not tell `key_tile0/chan` from `deq0` would be mixing the same felt for
        // two different shapes.
        // B11 stream V, lever 3: the residual-stream family takes the next code, 2. Key tiles
        // stay 0 and dequant 1, so a key-tiles-only or key tiles+dequant statement
        // mixes exactly the felts stream U's did -- which is what the four
        // key tiles and two dequant wall digests gate.
        let (fam, reg) = match s.reg {
            crate::gkrlink::Comp::KeyTile(b) => (0u32, b as u32),
            crate::gkrlink::Comp::Deq(m) => (1u32, m as u32),
            crate::gkrlink::Comp::ResidualStream(g) => (2u32, g as u32),
            crate::gkrlink::Comp::Freivalds(w) => (3u32, w as u32),
            crate::gkrlink::Comp::Ao => (4u32, 0u32),
        };
        felts.push(SecureField::from_u32_unchecked(
            reg,
            (fam << 8) | s.which as u32,
            s.log_rows,
            s.n_looks() as u32,
        ));
    }
    channel.mix_felts(&felts);
}

#[allow(dead_code)]
const _: () = assert!(N_LANES == 1usize << LOG_N_LANES);

// ------------------------------------------------------------------ the test ---

/// `zk-prover gkr` -- the GKR backend's own self-test on a synthetic key tile.
///
/// What it establishes, in the order the record reports it:
///
/// 1. **The sums agree.** For every key tile component, the fractional sum the batched
///    GKR yields from its output claims equals, exactly, the claimed sum
///    `interaction_phase` pushes onto `use_sums` from the committed running-sum
///    columns. This is the strongest correctness statement available without the
///    `mle_eval` link: the two backends prove the same number about the same
///    fractions, bit for bit, and `sums_cancel` is what consumes that number.
/// 2. **A direct sum agrees too**, computed by plain inversion over every
///    fraction with no sumcheck in it, so a bug that moved the committed columns
///    and the GKR the same way would still be caught.
/// 3. **The proof verifies from bytes.** The `GkrBatchProof` is serialised, a
///    fresh channel is replayed from the same starting state, and
///    `partially_verify_batch` accepts.
/// 4. **A corrupted proof is rejected**, and the rejection is named.
/// 5. **What it cost**: instances, fractions, the input-layer build, the
///    sumcheck, and the proof bytes.
pub fn cmd(args: &[String]) {
    use crate::gadgets::attention::witness::{Attn, Variant};
    use crate::gadgets::attention::layout::Shape;
    use stwo::core::channel::Blake2sM31Channel;

    // The first CUDA call in the process pays context creation and the fatbin
    // load; on the cuda build the sumcheck below must not be billed for it.
    crate::util::touch_device();

    let src = crate::cli::src_of(args);
    let sh = Shape {
        s: crate::cli::arg_scalar(args, "--s", 64usize),
        c: crate::cli::arg_scalar(args, "--ctx", 256usize),
        hd: crate::cli::arg_scalar(args, "--hd", 64usize),
        insts: crate::cli::arg_scalar(args, "--insts", 4usize),
    };
    let cfg = crate::cli::attn_cfg(args, crate::cli::arg_scalar(args, "--scores-per-row", 16usize));
    let a = Attn::get(&src, sh, Variant::IntExp);
    let l = crate::gadgets::attention::trace::Layouts::choose(&sh, &cfg);
    let widths = crate::gadgets::attention::trace::range_widths(&a, &cfg);
    let com = crate::gadgets::attention::trace::commit_phase(&a, &l, &cfg);

    // The challenges have to be drawn the way the proof draws them, or the
    // relations are different relations and the sums are incomparable. This is
    // the order `attn_run::run` uses, without the commitment scheme: what the
    // fractions depend on is the drawn elements, not the Merkle roots.
    let mut ch = Blake2sM31Channel::default();
    crate::util::pcs_config().mix_into(&mut ch);
    let rel_exp = crate::gadgets::attention::air::ExpRel::draw(&mut ch);
    let rels: Vec<crate::relations::RangeRel> =
        widths.iter().map(|_| crate::relations::RangeRel::draw(&mut ch)).collect();
    let f = ch.draw_secure_felts(3);
    let pow = crate::gadgets::attention::air::Powers::new(f[0], f[1], f[2], &sh, &l.chan, &l.key, &l.out, l.rw);

    let mut collect = Collect::with_direct(true);
    collect.reg = crate::gkrlink::Comp::KeyTile(0);
    let t0 = Instant::now();
    let inter = crate::gadgets::attention::trace::interaction_phase(
        &a, &l, &cfg, &pow, &rel_exp, &rels, &com, &widths, true, !a.diagonal_block,
        None, Some(&mut collect), None);
    let committed_secs = t0.elapsed().as_secs_f64();

    let sites: Vec<Site> = collect.sites.clone();
    let direct = collect.direct.clone();
    assert_eq!(sites.len(), inter.use_sums.len(),
               "the collector saw {} key tile sites, `interaction_phase` pushed {} use sums",
               sites.len(), inter.use_sums.len());

    // The GKR runs on a channel replayed to the point the backend would reach it at,
    // and the verifier gets a channel in the same state.
    let mut pch = ch.clone();
    let proved = prove_collected(&mut pch, collect);
    let bytes = serde_json::to_vec(&proved.proof).expect("the GKR proof serialises");

    let mut vch = ch.clone();
    let verified = verify(&mut vch, &sites, &bytes);
    let vsums = match &verified {
        Ok((s, _, _)) => s.clone(),
        Err(_) => vec![],
    };
    let ood_len = match &verified {
        Ok((_, art, _)) => Some(art.ood_point.len()),
        Err(_) => None,
    };

    // the aimed corruption: one felt of the proof, which the sumcheck must catch
    let mut bad = proved.proof.clone();
    let tampered = tamper(&mut bad);
    let mut bch = ch.clone();
    let bad_bytes = serde_json::to_vec(&bad).unwrap();
    let bad_verdict = verify(&mut bch, &sites, &bad_bytes);
    let bad_rejected = bad_verdict.is_err();
    let bad_rejection = bad_verdict.err();

    let per_site: Vec<serde_json::Value> = sites
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let want = inter.use_sums[i];
            serde_json::json!({
                "site": s.tag(),
                "log_rows": s.log_rows,
                "looks": s.n_looks(),
                "unit_looks": s.unit.iter().filter(|u| **u).count(),
                "committed_sum": format!("{want:?}"),
                "gkr_sum": format!("{:?}", proved.sums[i]),
                "gkr_matches_committed": proved.sums[i] == want,
                "direct_matches_committed": direct[i] == want,
                "verified_sum_matches": vsums.get(i) == Some(&proved.sums[i]),
            })
        })
        .collect();

    let all_match = per_site.iter().all(|v| {
        v["gkr_matches_committed"] == true
            && v["direct_matches_committed"] == true
            && v["verified_sum_matches"] == true
    });
    let insts: Vec<serde_json::Value> = proved
        .insts
        .iter()
        .map(|i| serde_json::json!({
            "site": sites[i.site].tag(), "entries": i.ents.len(),
            "log_rows": i.log_rows, "n_variables": i.n_variables(),
            "unit": i.unit, "fractions": i.fractions(),
        }))
        .collect();

    crate::util::emit("gkr_selftest", serde_json::json!({
        "shape": sh.label(), "diagonal_block": a.diagonal_block, "variant": a.variant.name(),
        "scores_per_row": l.key.q, "keys_per_row": l.chan.pk, "rw": l.rw,
        "sites": per_site,
        "instances": insts,
        "sums_agree": all_match,
        "verifies_from_bytes": verified.is_ok(),
        "verify_error": verified.as_ref().err().cloned(),
        "ood_variables": ood_len,
        "control": {
            "what": tampered,
            "rejected": bad_rejected,
            "rejection": bad_rejection,
        },
        "cost": proved.cost.json(),
        "committed_interaction_secs": committed_secs,
    }));

    assert!(all_match, "the GKR backend's sums do not match the committed columns'");
    assert!(verified.is_ok(), "the GKR proof did not verify from bytes");
    assert!(bad_rejected, "a corrupted GKR proof was accepted");
}

/// Corrupt one felt of a batch GKR proof, and say what was corrupted.
///
/// The first round polynomial of the first layer's sumcheck is the earliest
/// thing the verifier checks, so a rejection there says the sumcheck is
/// load-bearing rather than that some later shape check tripped.
fn tamper(proof: &mut GkrBatchProof) -> String {
    if let Some(sc) = proof.sumcheck_proofs.first_mut() {
        if let Some(poly) = sc.round_polys.first_mut() {
            let mut c = poly.to_vec();
            if !c.is_empty() {
                c[0] += SecureField::one();
                *poly = stwo::prover::lookups::utils::UnivariatePoly::new(c);
                return "one coefficient of the first round polynomial, plus one".into();
            }
        }
    }
    if let Some(c) = proof.output_claims_by_instance.first_mut() {
        if let Some(v) = c.first_mut() {
            *v += SecureField::one();
            return "the first instance's output numerator claim, plus one".into();
        }
    }
    "nothing (the proof was empty)".into()
}

/// The key tile sites, derived from the plan alone.
///
/// This is the verifier's route to the instance list, and it has to agree with
/// what the prover's collector saw at the three `devfrac::key_tile_cols` sites
/// exactly -- the same rows, the same lookup counts, the same numerator kinds,
/// in the same order. `Plan` is the object whose whole purpose is that the two
/// sides cannot disagree about layouts, so the derivation reads only from it.
///
/// The prover asserts this function against the collector's own sites before it
/// proves anything, so a drift between the two is a loud failure on the proving
/// side rather than a rejected proof on the verifying side.
pub fn sites_of(plan: &crate::compose::Plan) -> Vec<Site> {
    let mut out = vec![];
    for (b, diagonal_block) in plan.diagonal_block.iter().enumerate() {
        let l = &plan.lays[b];
        let (lc, lk) = (&l.chan, &l.key);
        let cfg = &plan.cfg;

        // ---- the channel component: the key and query range checks, all unit
        if cfg.range_check {
            out.push(Site {
                reg: crate::gkrlink::Comp::KeyTile(b),
                which: 0,
                log_rows: lc.g.log_rows,
                unit: vec![true; lc.pk + lc.pq],
            });
        }

        // ---- the grid component: one exp lookup per score slot, then the
        // variant's remainder checks, then the V range checks. On a diagonal block key tile
        // the exp lookups carry `1 - mask` and are the only non-unit lookups in
        // the whole set.
        let mut unit = vec![!*diagonal_block; lk.q];
        if let AttnVariant::IntSplit(_) = plan.attn_variant {
            unit.extend(vec![true; lk.q]);
        }
        if cfg.range_check {
            unit.extend(vec![true; lk.pv]);
        }
        out.push(Site { reg: crate::gkrlink::Comp::KeyTile(b), which: 1,
                        log_rows: lk.g.log_rows, unit });

        // ---- the per-token component: four denominator limbs per token. There is
        // one per head key tile *group* -- one row maximum and one denominator per
        // token, not per key block -- so it lives on the group's first key tile and
        // every other key tile of the group rides on it. With `--kv-boundary` off that
        // is `!diagonal_block`, which is what this read used to say; with it on, a
        // group is `c/s` prefix sub-key-tiles plus a diagonal block and only the first
        // sub-key-tile carries the component. `plan.rows_key_tile` is the plan's own
        // answer and `attn_trace::interaction_phase`'s `with_rows` argument is
        // fed from it, so this is the derivation the collector actually saw.
        if plan.rows_key_tile[b] {
            out.push(Site {
                reg: crate::gkrlink::Comp::KeyTile(b),
                which: 3,
                log_rows: l.log_row,
                unit: vec![true; 4 * l.rw],
            });
        }
    }
    // ---- B11 stream U, lever 3: the dequant family, one site per component, in
    // the plan's matmul order -- which is the order `interaction_phase` reaches
    // them in, so the collector's arrival order is this order and `same_sites`
    // is what says so.
    if plan.p.gkr_dequant {
        for (m, sp) in plan.specs.iter().enumerate() {
            out.push(Site {
                reg: crate::gkrlink::Comp::Deq(m),
                which: 0,
                log_rows: plan.mat_lays[m].log_rows,
                unit: crate::gadgets::dequant::look_units(sp),
            });
        }
    }
    // ---- B11 stream V, lever 3: the residual-stream family, after dequant -- memo
    // section 8's order, which is the order the interaction pass collects in.
    // Every residual-stream look is unit, and the count comes from the component's program,
    // which is what the verifier has; `residual_stream_site` asserts the fractions agree.
    if plan.p.gkr_residual_stream {
        if let Some(gp) = plan.residual_stream.as_ref() {
            for (g, r) in gp.components.iter().enumerate() {
                out.push(Site {
                    reg: crate::gkrlink::Comp::ResidualStream(g),
                    which: 0,
                    log_rows: r.log_rows,
                    unit: vec![true; crate::gkrlink::residual_stream_look_count(&r.prog)],
                });
            }
        }
    }
    // ---- B11 stream V, lever 4: the Freivalds checks, then attnout -- memo section 8's
    // order again. Both are all-unit and both counts come from the plan.
    if plan.p.gkr_freivalds {
        for (w, wp) in plan.freivalds.iter().enumerate() {
            out.push(Site {
                reg: crate::gkrlink::Comp::Freivalds(w),
                which: 0,
                log_rows: wp.lay.g.log_rows,
                unit: vec![true; wp.lay.pq],
            });
        }
    }
    if plan.p.gkr_attnout {
        out.push(Site {
            reg: crate::gkrlink::Comp::Ao,
            which: 0,
            log_rows: plan.ao_lay.log_rows,
            // stream JJ (E11): seven per slot -- the exponent window, (qm, nz2),
            // `d`, `dp`, `denp` and the accumulator's two limbs
            unit: vec![true; crate::gadgets::attn_out::trace::slot_looks(plan.p.ao_acc_prod)
                             * plan.ao_lay.per_row],
        });
    }
    out
}

/// The two sides' site lists, compared field by field, with the first
/// disagreement named.
pub fn same_sites(a: &[Site], b: &[Site]) -> Result<(), String> {
    if a.len() != b.len() {
        return Err(format!("{} sites against {}", a.len(), b.len()));
    }
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        if x.reg != y.reg || x.which != y.which || x.log_rows != y.log_rows
            || x.unit != y.unit
        {
            return Err(format!(
                "site {i} differs: {} 2^{} {} looks ({} unit) against {} 2^{} {} looks ({} unit)",
                x.tag(), x.log_rows, x.n_looks(),
                x.unit.iter().filter(|u| **u).count(),
                y.tag(), y.log_rows, y.n_looks(),
                y.unit.iter().filter(|u| **u).count()));
        }
    }
    Ok(())
}
