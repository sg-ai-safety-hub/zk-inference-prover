//! Stream BB (E11) -- the embedding gadget: token ids in, residual stream out.
//!
//! Today the layer chain's first chunk begins at an *unchecked* residual-stream
//! input: `briefs/h-bounds.md` records it as "the first chunk's `bin` is the
//! embedding's output ... unchecked".  This module closes that end.  The
//! statement it adds is
//!
//! ```text
//!   for every position t of the block and every channel c of the stream,
//!       bin[t][c] == E[tok[t]][c]
//! ```
//!
//! with `E` the model's embedding matrix, committed in tree 0 beside the
//! weights the Freivalds checks already pin there, and `tok` committed in tree 1 -- before
//! any relation challenge is drawn, which is what makes the lookup below bind
//! it.
//!
//! # Why a scan of the whole table, and why that is not the expensive part
//!
//! Selecting one row of a committed table by a committed index, with no
//! positional opening available to the AIR, costs one pass over the table.
//! There is no cheaper sound shape here: a digest small enough to look up is
//! either challenge-dependent -- and a challenge-dependent value cannot be
//! committed before the challenge that binds it -- or not injective.  So the
//! table side is `V` rows wide in the vocabulary and the pass is `V x d`.
//!
//! What that pass costs is almost all *preprocessed*: `V x d` cells in tree 0,
//! the same kind of cell the `head` Freivalds group already spends 570 M of on
//! `lm_head`'s weight matrix.  The fresh cost -- what a chunk pays again -- is
//! one multiplicity cell and one LogUp column per table row, which is
//! `5 V d / pe` cells at `pe` channels per row, plus the use side's own copy of
//! the stream.
//!
//! # The two components
//!
//! **The use side** (`EmbedEval`), one row per `(token, channel block)`:
//! commits the token id and the block's `pe` channels as `(sm, ef)` pairs, and
//! carries two Horner claims -- one over the stream values, one over the ids.
//! The stream claim uses the wiring family's own binding challenge and weight
//! convention, so the verifier compares it against `stream_bindings[stream_in]` by
//! equality.  That comparison is the binding: it says the row this component proved is
//! the row the layer consumed.
//!
//! **The table side** (`EmbTableEval`), one row per `(vocabulary entry, channel
//! block)`: `pe` preprocessed value columns and one committed multiplicity.
//!
//! # The padded rows, and why they carry an out-of-range block index
//!
//! The vocabulary is not a power of two (Llama-3.1-8B has 128,256 entries, so
//! `vpad = 131,072`) and the table's row count must be, so `2^log_tab` always
//! holds rows for ids no model can emit.  Stream KK closes what that left open
//! (`notes/witness-slack.md` §1a): before it, a padded row was a legal, distinct
//! table row `(v_pad, cb, 0, .., 0)`, so a prover could commit `tk = v_pad` with
//! an all-zero stream row, put the multiplicity on the padded row, and prove a
//! block whose id names no vocabulary entry.
//!
//! The fix is one preprocessed value: a padded row's block index is `k`, which
//! is **out of the use side's range**.  `cb` on the use side is preprocessed
//! (`j % k`, one of the component's five preprocessed columns), so no use row can
//! ever present `k` in that slot, so a padded row is not the tuple of any use
//! row, so its multiplicity must be zero for the LogUp sums to cancel.  That
//! confines `tk` to `[0, vocab)` at zero marginal cost: no column, no
//! constraint, no arity change and no new kernel fingerprint.  Tree 0's values
//! move, so `pre_root` and the digest of an `--embed` chunk move with them.
//!
//! # What the lookup carries, and why one cell per channel is enough
//!
//! The tuple is `(v, cb, y_0 .. y_{pe-1})` with `y_k = PACK sm_k + ef_k` --
//! `boundary::PACK`'s packing, one preprocessed cell per table entry instead of
//! two.  The soundness of the halving is the packing's injectivity on the valid
//! bf16 domain, and the use side buys the domain itself with one 8-bit range
//! check per channel on `ef`:
//!
//!   * the table's `(sm_E, ef_E)` is valid by construction;
//!   * `ef_x` is range-checked to `[0, 256)`;
//!   * `PACK (sm_x - sm_E) = ef_E - ef_x` lies in `(-256, 256)`, and the only
//!     multiple of `PACK = 512` there is zero, so `sm_x = sm_E`, `ef_x = ef_E`.
//!
//! The alternative -- `sm` and `ef` as two preprocessed columns and a `2pe + 2`
//! tuple -- needs no range check and costs `2 V d` preprocessed cells instead
//! of `V d`.  It is priced in the record and not built.
//!
//! # Where the values are committed, and why the order matters
//!
//! Both components' columns are base columns (tree 1), committed before the
//! relation challenges are drawn.  The two Horner accumulators are interaction
//! columns (tree 2), built after the binding challenge `w` and its companion `lam`
//! are drawn -- exactly where the wiring family builds its own.  Nothing this
//! module commits lands in a boundary tree, and it adds no tree.

use num_traits::{One, Zero};
use serde_json::{Value, json};
use stwo::core::air::Component;
use stwo::core::channel::Channel;
use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::SecureField;
use stwo::prover::ComponentProver;
use stwo::prover::backend::simd::m31::PackedM31;
use stwo::prover::backend::simd::qm31::PackedQM31;
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;
use stwo_constraint_framework::{
    EvalAtRow, FrameworkComponent, FrameworkEval, INTERACTION_TRACE_IDX, ORIGINAL_TRACE_IDX,
    Relation, RelationEFTraitBound, RelationEntry, TraceLocationAllocator,
};

use crate::encoding::{canon, m31_of, secure_cols_pub, split};
use crate::relations::{RangeRel, RangeTableEval, id, table_col, tbl_range};
use crate::boundary::PACK;
use crate::common::{Eval, Proven, Session, Sums, eval_of, log_rows_for, m31_u32, twiddles};

/// The exponent's width: `ef` is a bf16 exponent field, so `[0, 256)`, and the
/// packing argument above needs exactly this check and no other.
pub const EF_BITS: u32 = 8;

/// The use component's preprocessed columns, in the order `pre_ids` writes them.
/// `act`, `in` and `bd` are selectors; `cb` is the channel-block index, which
/// the lookup tuple carries so that two blocks of one token cannot be swapped.
pub const EMB_SEL: [&str; 4] = ["b11_emb_act", "b11_emb_in", "b11_emb_bd", "b11_emb_end"];

pub fn cb_id() -> PreProcessedColumnId {
    id("b11_emb_cb")
}
pub fn tab_v_id() -> PreProcessedColumnId {
    id("b11_embt_v")
}
pub fn tab_cb_id() -> PreProcessedColumnId {
    id("b11_embt_cb")
}
pub fn tab_e_id(k: usize) -> PreProcessedColumnId {
    id(&format!("b11_embt_e{k:04}"))
}

// ------------------------------------------------------------------ layout ----

/// How the stream and the vocabulary are cut into rows.
///
/// `k` blocks of `pe = d / k` channels; `k` is the smallest power of two that
/// brings a block at or under the target, so the use component's row count
/// `tokens * k` stays a power of two and the table's row count is the padded
/// vocabulary times `k`.  The target is a knob because the block width is the
/// lookup tuple's arity and therefore the generated kernel's per-row width --
/// the one number in this gadget that the emit chain cares about.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EmbLayout {
    /// channels per row
    pub pe: usize,
    /// channel blocks per token
    pub k: usize,
    pub log_use: u32,
    pub log_tab: u32,
    /// the padded vocabulary, which is the table's row count divided by `k`
    pub vpad: usize,
}

impl EmbLayout {
    pub fn choose(d: usize, tokens: usize, vocab: usize, target: usize) -> Self {
        assert!(d >= 2 && target >= 1, "the embedding needs a stream and a target");
        let mut k = 1usize;
        while d % k != 0 || d / k > target {
            k *= 2;
            assert!(k <= d, "no power-of-two block count divides d = {d} at target {target}");
        }
        let log_v = log_rows_for(vocab, 1);
        Self {
            pe: d / k,
            k,
            log_use: log_rows_for(tokens * k, 1),
            log_tab: log_v + k.trailing_zeros(),
            vpad: 1usize << log_v,
        }
    }
    /// `(token, block)` at use-component row `j`, or `None` on a padded row.
    pub fn at(&self, tokens: usize, j: usize) -> Option<(usize, usize)> {
        let (t, cb) = (j / self.k, j % self.k);
        (t < tokens).then_some((t, cb))
    }
    /// The lookup tuple's arity: `(v, cb)` and one packed cell per channel.
    pub fn arity(&self) -> usize {
        self.pe + 2
    }
    pub fn json(&self, tokens: usize, vocab: usize) -> Value {
        json!({"per_row": self.pe, "blocks": self.k, "arity": self.arity(),
               "log_use": self.log_use, "log_tab": self.log_tab,
               "use_rows": 1usize << self.log_use, "tab_rows": 1usize << self.log_tab,
               "tokens": tokens, "vocab": vocab, "vocab_padded": self.vpad,
               "vocab_pad_frac": (self.vpad - vocab) as f64 / self.vpad as f64})
    }
}

// ---------------------------------------------------------------- relation ----

/// The embedding relation: `(v, cb, y_0 .. y_{pe-1})`.
///
/// Written out rather than taken from `relation!` because its arity is a
/// runtime number -- the macro's `alpha_powers` is a `[SecureField; N]` with `N`
/// a literal.  `draw` consumes the channel exactly as `LookupElements::draw`
/// does (two secure felts, `z` then `alpha`), and `combine` is the same
/// `sum alpha^i v_i - z`, so `cudagen`'s `RelationProbe` recovers `z` and every
/// `alpha_i` from it the way it does for every other relation here.
#[derive(Clone, Debug, PartialEq)]
pub struct EmbRel {
    pub z: SecureField,
    pub alpha: SecureField,
    pub powers: Vec<SecureField>,
}

impl EmbRel {
    pub fn draw(channel: &mut impl Channel, n: usize) -> Self {
        let f = channel.draw_secure_felts(2);
        let (z, alpha) = (f[0], f[1]);
        let mut cur = SecureField::one();
        let powers = (0..n)
            .map(|_| {
                let r = cur;
                cur *= alpha;
                r
            })
            .collect();
        Self { z, alpha, powers }
    }
}

impl<F: Clone, EF: RelationEFTraitBound<F>> Relation<F, EF> for EmbRel {
    fn combine(&self, values: &[F]) -> EF {
        values
            .iter()
            .zip(&self.powers)
            .fold(EF::zero(), |acc, (v, p)| acc + EF::from(*p) * v.clone())
            - EF::from(self.z)
    }
    fn get_name(&self) -> &str {
        "EmbRel"
    }
    fn get_size(&self) -> usize {
        self.powers.len()
    }
}

// -------------------------------------------------------------------- bindings ----

/// The stream binding's weights, in the wiring family's own convention: the value
/// at canonical stream position `q = c * tokens + t` weighs `w^(N - 1 - q)` with
/// `N = d * tokens`, and a value is `sm + lam ef`.  That is exactly what
/// `ResidualStreamPlan`'s `stream_in` binding accumulates over `gn1`'s group 0, so the two
/// claims are comparable by equality and nothing else has to be arranged.
#[derive(Clone, Debug)]
pub struct ETie {
    /// one weight per channel slot, at the component's last row
    pub w: Vec<SecureField>,
    pub lam: SecureField,
    /// the row step inside a token, and across a token boundary
    pub k_in: SecureField,
    pub k_bd: SecureField,
    /// the id claim's per-token step
    pub k_tok: SecureField,
    pub claimed: SecureField,
    pub claimed_tok: SecureField,
}

fn wpow(w: SecureField, e: i128) -> SecureField {
    use stwo::core::fields::FieldExpOps;
    if e >= 0 { w.pow(e as u128) } else { w.pow((-e) as u128).inverse() }
}

/// The weights both sides derive from the drawn `w` and `lam` and the layout.
pub fn binding_w(w: SecureField, lam: SecureField, d: usize, tokens: usize, lay: &EmbLayout) -> ETie {
    let (n, tok, pe) = ((d * tokens) as i128, tokens as i128, lay.pe as i128);
    let last = (lay.k - 1) as i128 * pe;
    ETie {
        w: (0..lay.pe)
            .map(|k| wpow(w, n - 1 - (last + k as i128) * tok - (tok - 1)))
            .collect(),
        lam,
        k_in: wpow(w, pe * tok),
        k_bd: wpow(w, 1 - (d as i128 - pe) * tok),
        k_tok: w,
        claimed: SecureField::zero(),
        claimed_tok: SecureField::zero(),
    }
}

// --------------------------------------------------------------- the AIR ------

/// The use side: one row per `(token, channel block)`.
#[derive(Clone)]
pub struct EmbedEval {
    pub log_rows: u32,
    /// channels per row
    pub per_row: usize,
    pub rel: EmbRel,
    /// the 8-bit range relation `ef` is checked against
    pub rel_ef: RangeRel,
    pub binding: ETie,
}

impl FrameworkEval for EmbedEval {
    fn cuda_eval_blob(&self) -> Option<Vec<u8>> {
        crate::cudagen::blob_for(self)
    }
    fn cuda_eval_name(&self) -> &'static str {
        "zk_embed_eval"
    }
    fn log_size(&self) -> u32 {
        self.log_rows
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_rows + 1
    }
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let sel: Vec<E::F> = EMB_SEL.iter().map(|n| eval.get_preprocessed_column(id(*n))).collect();
        let (act, s_in, s_bd, s_end) =
            (sel[0].clone(), sel[1].clone(), sel[2].clone(), sel[3].clone());
        let cb = eval.get_preprocessed_column(cb_id());
        let pack = E::F::from(m31_u32(PACK as u32));

        // the token id, constant across the token's blocks
        let [tk_prev, tk] = eval.next_interaction_mask(ORIGINAL_TRACE_IDX, [-1, 0]);
        eval.add_constraint((tk.clone() - tk_prev) * s_in.clone());

        // the block's channels, as the wiring family carries them
        let mut sm = Vec::with_capacity(self.per_row);
        let mut ef = Vec::with_capacity(self.per_row);
        for _ in 0..self.per_row {
            sm.push(eval.next_trace_mask());
            ef.push(eval.next_trace_mask());
        }

        // the lookup: this block of this token is that block of that table row.
        // The multiplicity is the active selector, so a padded row enters
        // nothing -- its tuple is not a table row and would not cancel.
        let mut vals: Vec<E::F> = Vec::with_capacity(self.per_row + 2);
        vals.push(tk.clone());
        vals.push(cb);
        for i in 0..self.per_row {
            vals.push(sm[i].clone() * pack.clone() + ef[i].clone());
        }
        eval.add_to_relation(RelationEntry::new(&self.rel, E::EF::from(act.clone()), &vals));
        // and the one bound the packing's injectivity needs
        for e in &ef {
            eval.add_to_relation(RelationEntry::new(
                &self.rel_ef, E::EF::from(act.clone()), core::slice::from_ref(e)));
        }

        // the stream binding: the counterparty is the wiring family's `stream_in`
        // claim over the very same values, under the same weights
        let mult: E::EF = E::EF::from(s_in.clone()) * self.binding.k_in
            + E::EF::from(s_bd.clone()) * self.binding.k_bd;
        let [h_prev, h] = eval.next_extension_interaction_mask(INTERACTION_TRACE_IDX, [-1, 0]);
        let mut f = E::EF::zero();
        for i in 0..self.per_row {
            f += (E::EF::from(sm[i].clone()) + E::EF::from(ef[i].clone()) * self.binding.lam)
                * self.binding.w[i];
        }
        eval.add_constraint(h.clone() - h_prev * mult - f * act.clone());
        let claim = eval.claim(self.binding.claimed);
        eval.add_constraint((h - claim) * s_end.clone());

        // the id claim: one step per token, so the ids are a named value of the
        // statement and not merely columns inside it
        let kmult: E::EF =
            E::EF::from(s_in.clone()) + E::EF::from(s_bd.clone()) * self.binding.k_tok;
        let [g_prev, g] = eval.next_extension_interaction_mask(INTERACTION_TRACE_IDX, [-1, 0]);
        eval.add_constraint(
            g.clone() - g_prev * kmult - E::EF::from(tk * (E::F::one() - s_in) * act),
        );
        let claim_tok = eval.claim(self.binding.claimed_tok);
        eval.add_constraint((g - claim_tok) * s_end);

        eval.finalize_logup_in_pairs();
        eval
    }
}

/// The table side: one row per `(vocabulary entry, channel block)`.
#[derive(Clone)]
pub struct EmbTableEval {
    pub log_rows: u32,
    pub per_row: usize,
    pub rel: EmbRel,
}

impl FrameworkEval for EmbTableEval {
    fn cuda_eval_blob(&self) -> Option<Vec<u8>> {
        crate::cudagen::blob_for(self)
    }
    fn cuda_eval_name(&self) -> &'static str {
        "zk_emb_table_eval"
    }
    fn log_size(&self) -> u32 {
        self.log_rows
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_rows + 1
    }
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let mut vals: Vec<E::F> = Vec::with_capacity(self.per_row + 2);
        vals.push(eval.get_preprocessed_column(tab_v_id()));
        vals.push(eval.get_preprocessed_column(tab_cb_id()));
        for k in 0..self.per_row {
            vals.push(eval.get_preprocessed_column(tab_e_id(k)));
        }
        let m = eval.next_trace_mask();
        eval.add_to_relation(RelationEntry::new(&self.rel, -E::EF::from(m), &vals));
        eval.finalize_logup();
        eval
    }
}

// ----------------------------------------------------------------- witness ----

/// What a probe corrupts.  Each aims at a different half of the argument.
#[derive(Clone, Copy, Debug)]
pub enum Tamper {
    /// one committed token id moved to a different vocabulary entry, the
    /// committed row left alone: the tuple is no longer a table row, so the
    /// lookup's sums do not cancel
    Tok(usize),
    /// **the aimed corruption.**  The id moved *and* the committed row
    /// recomputed from the table to match it, so the component is internally
    /// consistent and every constraint inside it still holds.  Only the binding to
    /// the layer's `bin` can tell that this is not the row the layer consumed.
    /// This is the shape of `argmax::Tamper::Key`.
    TokRow(usize),
    /// one committed channel of the stream changed: the lookup rejects
    Val(usize),
    /// one cell of the committed *table* changed, with the component rebuilt to
    /// match it -- every internal check passes and the pinned preprocessed root
    /// is what rejects
    Table(usize),
    /// stream KK, `notes/witness-slack.md` §1a's **before** arm: build the table
    /// the way it was built before the sentinel, with every padded row carrying
    /// its own block index and therefore matchable.  Not a corruption of an
    /// honest witness -- it is the open table, and on a witness whose id is a
    /// padded one (`EWitness::padded`) it is what *accepts*.  Pairing it with
    /// the same witness under `None` is the control that shows the fix bites.
    LegacyPad,
}

pub struct EWitness {
    pub tokens: usize,
    pub d: usize,
    pub vocab: usize,
    /// `(vocab, d)` canonical bf16 patterns -- the model's embedding matrix
    pub tab: Vec<u16>,
    /// one vocabulary index per position
    pub tok: Vec<u32>,
    /// `(tokens, d)` canonical bf16 patterns -- the layer's residual-stream
    /// input, i.e. the wiring family's `gn1` group 0
    pub xin: Vec<u16>,
    pub source: String,
}

impl EWitness {
    /// A synthetic block: a random table, random ids, and the stream read off
    /// the table so an honest run accepts.
    pub fn synthetic(tokens: usize, d: usize, vocab: usize, seed: u64) -> Self {
        use rand::rngs::SmallRng;
        use rand::{Rng, SeedableRng};
        let mut rng = SmallRng::seed_from_u64(seed);
        // exponents in [1, 0xfe] and a full significand: exactly the domain
        // `residual_stream::split` maps to a nonzero `(sm, ef)`, plus the zero pattern
        let mut pat = || -> u16 {
            if rng.random::<f64>() < 0.02 {
                0
            } else {
                let sg: u16 = rng.random_range(0..2);
                let e: u16 = rng.random_range(1..0xff);
                let m: u16 = rng.random_range(0..128);
                (sg << 15) | (e << 7) | m
            }
        };
        let tab: Vec<u16> = (0..vocab * d).map(|_| pat()).collect();
        let tok: Vec<u32> = (0..tokens).map(|_| rng.random_range(0..vocab as u32)).collect();
        let mut xin = vec![0u16; tokens * d];
        for t in 0..tokens {
            let v = tok[t] as usize;
            xin[t * d..(t + 1) * d].copy_from_slice(&tab[v * d..(v + 1) * d]);
        }
        Self { tokens, d, vocab, tab, tok, xin, source: format!("synthetic(seed={seed})") }
    }

    /// Stream KK, `notes/witness-slack.md` §1a: the witness the open statement
    /// admitted.  Position `at` commits an id in `[vocab, vpad)` -- outside the
    /// vocabulary, inside the table -- and an all-zero stream row, which is the
    /// row `residual_stream::split(0)` gives and the row the padded table entry held.
    /// Every other position is honest.
    ///
    /// This is not a corruption: every constraint the component carries holds on
    /// it, and before the sentinel the lookup cancelled.  It is a legal witness
    /// for a false statement, which is why it is built here rather than in
    /// `Tamper`.
    pub fn padded(tokens: usize, d: usize, vocab: usize, target: usize, seed: u64, at: usize)
                  -> Self {
        let lay = EmbLayout::choose(d, tokens, vocab, target);
        assert!(lay.vpad > vocab, "this vocabulary needs no padding: {vocab} = 2^k");
        let mut w = Self::synthetic(tokens, d, vocab, seed);
        let at = at % tokens;
        w.tok[at] = vocab as u32;
        w.xin[at * d..(at + 1) * d].fill(0);
        w.source = format!("synthetic(seed={seed}) with the padded id {vocab} at position {at}");
        w
    }

    /// The real columns a `Layer` holds.  The check is the statement's own
    /// premise, so a witness that fails it is a spec mismatch and not something
    /// for the component to prove around -- exactly as `AWitness::from_cols` treats
    /// a dump whose token is not the first maximum.
    pub fn from_cols(tokens: usize, d: usize, vocab: usize, tab: Vec<u16>, tok: Vec<u32>,
                     xin: Vec<u16>) -> Result<Self, String> {
        if tab.len() != vocab * d {
            return Err(format!("embed table: {} values, want {vocab} x {d}", tab.len()));
        }
        if tok.len() != tokens {
            return Err(format!("token ids: {} values, want {tokens}", tok.len()));
        }
        if xin.len() != tokens * d {
            return Err(format!("stream input: {} values, want {tokens} x {d}", xin.len()));
        }
        let tab: Vec<u16> = tab.into_iter().map(canon).collect();
        let xin: Vec<u16> = xin.into_iter().map(canon).collect();
        for t in 0..tokens {
            let v = tok[t] as usize;
            if v >= vocab {
                return Err(format!("position {t}: token {v} is outside the vocabulary"));
            }
            for c in 0..d {
                if xin[t * d + c] != tab[v * d + c] {
                    return Err(format!(
                        "position {t} channel {c}: the stream is {:#06x} and row {v} of the \
                         embedding is {:#06x}",
                        xin[t * d + c], tab[v * d + c]));
                }
            }
        }
        Ok(Self { tokens, d, vocab, tab, tok, xin, source: "the layer's embedding columns".into() })
    }
}

// ------------------------------------------------------------------ component ----

/// The gadget's committed and preprocessed columns, with no transcript of its
/// own -- what the composed proof needs, and what `run` builds its standalone
/// proof out of so the two cannot drift apart.
pub struct EComponent {
    pub lay: EmbLayout,
    pub tokens: usize,
    pub d: usize,
    pub vocab: usize,
    /// the use side's four selectors and its `cb` index, in `pre_ids` order
    pub pre: Vec<Eval>,
    /// the use side's base columns: `tk`, then `(sm, ef)` per channel slot
    pub base: Vec<Eval>,
    /// the table side's `v`, `cb` and `pe` value columns, in `pre_ids` order
    pub tab_pre: Vec<Eval>,
    /// the table side's one multiplicity column
    pub tab_mult: Eval,
    /// the `(sm, ef)` the use side committed, in row-major `(row, slot)` order
    pub vals: Vec<(i64, i64)>,
    /// the ids the use side committed, one per row
    pub ids: Vec<u32>,
    /// values whose `ef` fell outside the 8-bit table -- always zero on an
    /// honest witness, and the number the probes move
    pub off_table: usize,
}

/// Build both components, and bump the 8-bit range table's multiplicities.
pub fn component(w: &EWitness, lay: EmbLayout, m_ef: &mut [u32], bad: Option<Tamper>) -> EComponent {
    let (pe, k) = (lay.pe, lay.k);
    let rows = 1usize << lay.log_use;
    let d = w.d;

    // the table, with the one-cell probe applied before anything reads it
    let mut tab = std::borrow::Cow::Borrowed(&w.tab);
    if let Some(Tamper::Table(i)) = bad {
        let q = i % w.tab.len();
        let owned = tab.to_mut();
        // one significand step, which stays inside the valid bf16 domain
        owned[q] ^= 1;
    }
    let tab: &[u16] = &tab;

    // the ids and the stream the use side commits
    let mut tok = w.tok.clone();
    let mut xin: Vec<u16> = if matches!(bad, Some(Tamper::Table(_))) {
        // the witness is rebuilt to match the corrupted table, so every check
        // inside the proof passes and only the pinned root disagrees
        (0..w.tokens)
            .flat_map(|t| {
                let v = tok[t] as usize;
                tab[v * d..(v + 1) * d].to_vec()
            })
            .collect()
    } else {
        w.xin.clone()
    };
    match bad {
        Some(Tamper::Tok(t)) => {
            let t = t % w.tokens;
            tok[t] = (tok[t] + 1) % w.vocab as u32;
        }
        Some(Tamper::TokRow(t)) => {
            let t = t % w.tokens;
            let v = ((tok[t] + 1) % w.vocab as u32) as usize;
            tok[t] = v as u32;
            xin[t * d..(t + 1) * d].copy_from_slice(&tab[v * d..(v + 1) * d]);
        }
        Some(Tamper::Val(i)) => {
            let q = i % xin.len();
            xin[q] ^= 1;
        }
        Some(Tamper::Table(_)) | Some(Tamper::LegacyPad) | None => {}
    }

    // ---- the use side --------------------------------------------------------
    let n_cols = 1 + 2 * pe;
    let mut cols: Vec<Vec<M31>> = vec![vec![M31::zero(); rows]; n_cols];
    let mut vals: Vec<(i64, i64)> = vec![(0, 0); rows * pe];
    let mut ids = vec![0u32; rows];
    let mut off_table = 0usize;
    for j in 0..rows {
        let Some((t, cb)) = lay.at(w.tokens, j) else { continue };
        cols[0][j] = m31_u32(tok[t]);
        ids[j] = tok[t];
        for sl in 0..pe {
            let (sm, ef) = split(xin[t * d + cb * pe + sl]);
            cols[1 + 2 * sl][j] = m31_of(sm);
            cols[2 + 2 * sl][j] = m31_u32(ef);
            vals[j * pe + sl] = (sm, ef as i64);
            if (ef as usize) < m_ef.len() {
                m_ef[ef as usize] += 1;
            } else {
                off_table += 1;
            }
        }
    }
    let sel = |f: &dyn Fn(usize) -> bool| {
        eval_of(lay.log_use, (0..rows).map(|j| m31_u32(f(j) as u32)).collect())
    };
    let n_use = w.tokens * k;
    let pre = vec![
        sel(&|j| j < n_use),
        sel(&|j| j < n_use && j % k != 0),
        sel(&|j| j < n_use && j % k == 0 && j != 0),
        sel(&|j| j + 1 == n_use),
        eval_of(lay.log_use, (0..rows).map(|j| m31_u32((j % k) as u32)).collect()),
    ];
    let base: Vec<Eval> = cols.into_iter().map(|c| eval_of(lay.log_use, c)).collect();

    // ---- the table side ------------------------------------------------------
    let trows = 1usize << lay.log_tab;
    let legacy_pad = matches!(bad, Some(Tamper::LegacyPad));
    let mut tcols: Vec<Vec<M31>> = vec![vec![M31::zero(); trows]; pe + 2];
    for n in 0..trows {
        let (v, cb) = (n / k, n % k);
        tcols[0][n] = m31_u32(v as u32);
        // §1a: a padded row's block index is `k`, which the use side's
        // preprocessed `cb` (`j % k`) can never present -- so a padded row is
        // the tuple of no use row and its multiplicity must be zero.  The probe
        // arm restores the pre-KK value, under which the row is matchable.
        tcols[1][n] = m31_u32(if v >= w.vocab && !legacy_pad { k as u32 } else { cb as u32 });
        if v >= w.vocab {
            continue;
        }
        for sl in 0..pe {
            let (sm, ef) = split(tab[v * d + cb * pe + sl]);
            tcols[2 + sl][n] = m31_of(PACK * sm + ef as i64);
        }
    }
    let mut mult = vec![0u32; trows];
    for j in 0..rows {
        let Some((t, cb)) = lay.at(w.tokens, j) else { continue };
        mult[tok[t] as usize * k + cb] += 1;
    }
    let tab_pre: Vec<Eval> = tcols.into_iter().map(|c| eval_of(lay.log_tab, c)).collect();
    let tab_mult = eval_of(lay.log_tab, mult.into_iter().map(m31_u32).collect());

    EComponent { lay, tokens: w.tokens, d, vocab: w.vocab, pre, base, tab_pre, tab_mult, vals, ids,
              off_table }
}

impl EComponent {
    /// Stream KK: the committed id of every position, in position order -- the
    /// vector the statement publishes and `boundary-grid` compares against the
    /// previous block's proved argmax.  Read off the committed column, so it is
    /// the component's own ids and not the witness's.
    pub fn token_ids(&self) -> Vec<u32> {
        (0..self.tokens).map(|t| self.ids[t * self.lay.k]).collect()
    }
    /// Committed cells, both sides.
    pub fn base_cells(&self) -> usize {
        self.base.len() * (1usize << self.lay.log_use) + (1usize << self.lay.log_tab)
    }
    pub fn pre_cells(&self) -> usize {
        self.pre.len() * (1usize << self.lay.log_use)
            + self.tab_pre.len() * (1usize << self.lay.log_tab)
    }
    /// How many lookups the use component's AIR adds, in `add_to_relation` order.
    pub fn n_looks(&self) -> usize {
        1 + self.lay.pe
    }
}

/// The use component's LogUp columns and its use sum, in the order the AIR adds
/// the lookups: the embedding entry first, then one `ef` range check per slot.
pub fn logup(r: &EComponent, rel: &EmbRel, rel_ef: &RangeRel) -> (Vec<Eval>, SecureField) {
    let pe = r.lay.pe;
    let pack = PackedM31::broadcast(m31_u32(PACK as u32));
    let one = PackedQM31::broadcast(SecureField::one());
    crate::plogup::build_rows(r.lay.log_use, r.n_looks(), |vr, dst| {
        let act = PackedQM31::from(r.pre[0].data[vr]);
        let mut vals: Vec<PackedM31> = Vec::with_capacity(pe + 2);
        vals.push(r.base[0].data[vr]);
        vals.push(r.pre[4].data[vr]);
        for sl in 0..pe {
            vals.push(r.base[1 + 2 * sl].data[vr] * pack + r.base[2 + 2 * sl].data[vr]);
        }
        dst.push((act, rel.combine(&vals)));
        for sl in 0..pe {
            let ef = r.base[2 + 2 * sl].data[vr];
            dst.push((act, rel_ef.combine(core::slice::from_ref(&ef))));
        }
        let _ = one;
    })
}

/// The table side's one LogUp column and its claimed sum.
pub fn table_side(r: &EComponent, rel: &EmbRel) -> (Vec<Eval>, SecureField) {
    let refs: Vec<&Eval> = r.tab_pre.iter().collect();
    table_col(r.lay.log_tab, rel, &refs, &r.tab_mult)
}

/// The two Horner columns and the two claims they close on: the stream binding and
/// the id claim, in the order the AIR reads them.
pub fn binding_cols(r: &EComponent, t: &ETie) -> (Vec<Eval>, SecureField, SecureField) {
    let rows = 1usize << r.lay.log_use;
    let (pe, k) = (r.lay.pe, r.lay.k);
    let n_use = r.tokens * k;
    let mut h = vec![SecureField::zero(); rows];
    let mut g = vec![SecureField::zero(); rows];
    for j in 0..rows {
        let act = j < n_use;
        let s_in = act && j % k != 0;
        let s_bd = act && j % k == 0 && j != 0;
        let mult = if s_in { t.k_in } else if s_bd { t.k_bd } else { SecureField::zero() };
        let kmult = if s_in {
            SecureField::one()
        } else if s_bd {
            t.k_tok
        } else {
            SecureField::zero()
        };
        let mut f = SecureField::zero();
        if act {
            for sl in 0..pe {
                let (sm, ef) = r.vals[j * pe + sl];
                f += (SecureField::from(m31_of(sm)) + SecureField::from(m31_of(ef)) * t.lam)
                    * t.w[sl];
            }
        }
        let prev_h = if j == 0 { SecureField::zero() } else { h[j - 1] };
        let prev_g = if j == 0 { SecureField::zero() } else { g[j - 1] };
        h[j] = prev_h * mult + f;
        let add = if act && j % k == 0 {
            SecureField::from(m31_u32(r.ids[j]))
        } else {
            SecureField::zero()
        };
        g[j] = prev_g * kmult + add;
    }
    let (hc, gc) = (h[n_use - 1], g[n_use - 1]);
    let mut cols = secure_cols_pub(r.lay.log_use, &h);
    cols.extend(secure_cols_pub(r.lay.log_use, &g));
    (cols, hc, gc)
}

// ------------------------------------------------------------- the standalone --

pub struct Outcome {
    pub proven: Proven,
    pub sums: Sums,
    pub cells: [usize; 3],
    pub n_cols: [usize; 3],
    pub off_table: usize,
    pub binding_ok: bool,
}

impl Outcome {
    pub fn accepts(&self) -> bool {
        self.proven.stark_ok && self.sums.cancels() && self.off_table == 0 && self.binding_ok
    }
    pub fn json(&self, w: &EWitness, lay: &EmbLayout) -> Value {
        let fresh = self.cells[1] + self.cells[2];
        json!({
            "source": w.source, "d": w.d, "tokens": w.tokens, "vocab": w.vocab,
            "layout": lay.json(w.tokens, w.vocab),
            "cells": {"preprocessed": self.cells[0], "base": self.cells[1],
                      "interaction": self.cells[2], "fresh": fresh},
            "columns": {"preprocessed": self.n_cols[0], "base": self.n_cols[1],
                        "interaction": self.n_cols[2]},
            "cells_per_position": fresh as f64 / w.tokens as f64,
            "cells_per_position_channel": fresh as f64 / (w.tokens * w.d) as f64,
            "proof_bytes": self.proven.proof_bytes, "prove_secs": self.proven.prove_secs,
            "verify_secs": self.proven.verify_secs,
            // the byte-exact reference the gate compares across backends: this
            // harness builds no serialized proof, so its three tree roots are
            // what stands in for `proof_sha256`
            "roots": self.proven.roots,
            "off_table": self.off_table,
            "checks": {"stark": self.proven.stark_ok, "sums_cancel": self.sums.cancels(),
                       "off_table": self.off_table, "binding": self.binding_ok},
            "accepts": self.accepts(),
            "prove_error": self.proven.prove_error,
        })
    }
}

/// The standalone proof: the two components, the 8-bit range table, one transcript.
///
/// The binding has no counterparty here -- there is no wiring family in a standalone
/// run -- so the claim is recomputed from the witness's own `xin` and compared,
/// which is the same comparison `verify_composed` makes against
/// `stream_bindings[stream_in]` and is what makes `TokRow` bite in this harness too.
pub fn run(w: &EWitness, target: usize, bad: Option<Tamper>) -> (Outcome, EmbLayout) {
    let lay = EmbLayout::choose(w.d, w.tokens, w.vocab, target);
    let mut m_ef = vec![0u32; 1 << EF_BITS];
    let r = component(w, lay, &mut m_ef, bad);

    let mult_ef = eval_of(EF_BITS, m_ef.iter().map(|v| m31_u32(*v)).collect());
    let tbl_ef = eval_of(EF_BITS, (0..1u32 << EF_BITS).map(m31_u32).collect());
    let widest = lay.log_tab.max(lay.log_use).max(EF_BITS);
    let tw = twiddles(widest);
    let mut sess = Session::new(&tw);

    let mut pre_all: Vec<Eval> = r.pre.clone();
    pre_all.extend(r.tab_pre.iter().cloned());
    pre_all.push(tbl_ef.clone());
    sess.commit(pre_all.clone());
    // A gadget on its own has no model column, but the tree layout is the
    // same for every proof this crate makes: the empty model tree keeps the
    // base trace at the index every component's mask names.
    sess.commit_model_empty();
    let mut base_all: Vec<Eval> = r.base.clone();
    base_all.push(r.tab_mult.clone());
    base_all.push(mult_ef.clone());
    sess.commit(base_all.clone());

    let rel_ef = sess.draw_relation(RangeRel::draw);
    let rel = sess.draw_relation(|ch| EmbRel::draw(ch, lay.arity()));
    let ch = sess.draw_felts(2);
    let binding = {
        let mut t = binding_w(ch[0], ch[1], w.d, w.tokens, &lay);
        let (cols, hc, gc) = binding_cols(&r, &t);
        t.claimed = hc;
        t.claimed_tok = gc;
        (t, cols)
    };
    let (binding, tcols) = binding;
    let (lcols, use_sum) = logup(&r, &rel, &rel_ef);
    let (tbcols, tab_sum) = table_side(&r, &rel);
    let (efcols, ef_sum) = table_col(EF_BITS, &rel_ef, &[&tbl_ef], &mult_ef);
    let mut icols = tcols;
    icols.extend(lcols);
    icols.extend(tbcols);
    icols.extend(efcols);
    sess.commit(icols.clone());

    // the binding's counterparty, recomputed from the witness rather than the trace
    let want = {
        let (n, tok) = ((w.d * w.tokens) as i128, w.tokens as i128);
        let mut acc = SecureField::zero();
        for t in 0..w.tokens {
            for c in 0..w.d {
                let (sm, ef) = split(w.xin[t * w.d + c]);
                acc += (SecureField::from(m31_of(sm))
                        + SecureField::from(m31_u32(ef)) * binding.lam)
                    * wpow(ch[0], n - 1 - (c as i128 * tok + t as i128));
            }
        }
        acc
    };
    let binding_ok = binding.claimed == want;

    let mut ids: Vec<PreProcessedColumnId> = EMB_SEL.iter().map(|n| id(*n)).collect();
    ids.push(cb_id());
    ids.push(tab_v_id());
    ids.push(tab_cb_id());
    ids.extend((0..lay.pe).map(tab_e_id));
    ids.push(tbl_range(EF_BITS));
    let alloc = &mut TraceLocationAllocator::new_with_preprocessed_columns(&ids);
    let em = FrameworkComponent::<EmbedEval>::new(
        alloc,
        EmbedEval { log_rows: lay.log_use, per_row: lay.pe, rel: rel.clone(),
                    rel_ef: rel_ef.clone(), binding },
        use_sum,
    );
    let et = FrameworkComponent::<EmbTableEval>::new(
        alloc,
        EmbTableEval { log_rows: lay.log_tab, per_row: lay.pe, rel },
        tab_sum,
    );
    let tb = FrameworkComponent::<RangeTableEval>::new(
        alloc, RangeTableEval { bits: EF_BITS, rel: rel_ef }, ef_sum);
    let provers: Vec<&dyn ComponentProver<_>> = vec![&em, &et, &tb];
    let components: Vec<&dyn Component> = vec![&em, &et, &tb];
    let proven = sess.prove(&provers, &components);

    let cells = [
        pre_all.iter().map(|e| e.domain.size()).sum::<usize>(),
        base_all.iter().map(|e| e.domain.size()).sum::<usize>(),
        icols.iter().map(|e| e.domain.size()).sum::<usize>(),
    ];
    let n_cols = [pre_all.len(), base_all.len(), icols.len()];
    (Outcome { proven, sums: Sums { uses: use_sum, tables: tab_sum + ef_sum }, cells, n_cols,
               off_table: r.off_table, binding_ok },
     lay)
}

/// The cells the gadget costs at a shape, with no witness and no proof.
///
/// A forecast, and labelled one wherever it is reported: it is the arithmetic
/// of the layout below, not a measurement.  `run` on a real witness is what
/// turns it into one.
pub fn price(d: usize, tokens: usize, vocab: usize, target: usize) -> Value {
    let lay = EmbLayout::choose(d, tokens, vocab, target);
    let (ur, tr) = (1usize << lay.log_use, 1usize << lay.log_tab);
    // the use side: `tk` and two cells per channel slot
    let use_base = ur * (1 + 2 * lay.pe);
    // one LogUp column per pair of lookups, four cells each, plus the two bindings
    let use_inter = ur * (4 * (1 + lay.pe).div_ceil(2) + 8);
    let tab_base = tr;
    let tab_inter = tr * 4;
    let fresh = use_base + use_inter + tab_base + tab_inter;
    json!({
        "layout": lay.json(tokens, vocab),
        "forecast": true,
        "preprocessed": {"use": ur * 5, "table": tr * (lay.pe + 2),
                         "total": ur * 5 + tr * (lay.pe + 2)},
        "base": {"use": use_base, "table": tab_base, "total": use_base + tab_base},
        "interaction": {"use": use_inter, "table": tab_inter,
                        "total": use_inter + tab_inter},
        "fresh": fresh,
        "fresh_per_position": fresh as f64 / tokens as f64,
        "fresh_per_block": fresh,
    })
}

// ------------------------------------------------------------------- tests ----


/// The embedding row: that the layer-0 residual stream is row `tok[t]` of the
/// pinned embedding matrix.
pub struct Embed;

/// Everything about an embedding run that is not the witness.
pub struct Cfg {
    pub per_row: usize,
}

impl crate::gadgets::Gadget for Embed {
    type Witness = EWitness;
    type Cfg = Cfg;
    type Tamper = Tamper;
    type Outcome = Outcome;

    fn run(w: &EWitness, cfg: &Cfg, bad: Option<Tamper>) -> Outcome {
        run(w, cfg.per_row, bad).0
    }

    fn accepts(o: &Outcome) -> bool {
        o.accepts()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The one thing a box run would otherwise be the first to check: that the
    /// claim this module publishes is the *same function of the same values*
    /// the wiring family's `stream_in` binding is.
    ///
    /// The residual-stream side is recomputed here from `ResidualStreamPlan`'s own definition --
    /// weights `w^(a tokens (nreal - 1 - j))` per leaf, a row step of `w`, over
    /// `Tiling`'s natural row order (row `r*tokens + t`, leaf `j` is channel
    /// `r + j a`) -- and compared against `binding_cols`.  Equality is what makes
    /// `verify_composed`'s one-line comparison meaningful; if the conventions
    /// ever drift, this fails here rather than on a card.
    fn residual_stream_claim(xin: &[u16], d: usize, tokens: usize, a: usize,
                             w: SecureField, lam: SecureField) -> SecureField {
        let nreal = d / a;
        let mut h = SecureField::zero();
        for m in 0..tokens * a {
            let (r, t) = (m / tokens, m % tokens);
            let mut f = SecureField::zero();
            for j in 0..nreal {
                let (sm, ef) = split(xin[t * d + r + j * a]);
                f += (SecureField::from(m31_of(sm)) + SecureField::from(m31_u32(ef)) * lam)
                    * wpow(w, (a * tokens * (nreal - 1 - j)) as i128);
            }
            h = h * w + f;
        }
        h
    }

    #[test]
    fn binding_matches_the_wiring_family() {
        for (d, tokens, vocab, a, target) in
            [(128usize, 16usize, 97usize, 4usize, 32usize),
             (128, 16, 97, 32, 16),
             (96, 8, 40, 8, 24),
             (64, 16, 33, 2, 64)]
        {
            let w = EWitness::synthetic(tokens, d, vocab, 3);
            let lay = EmbLayout::choose(d, tokens, vocab, target);
            let mut m_ef = vec![0u32; 1 << EF_BITS];
            let r = component(&w, lay, &mut m_ef, None);
            let ch = SecureField::from_m31_array(
                [m31_u32(7), m31_u32(11), m31_u32(13), m31_u32(17)]);
            let lam = SecureField::from_m31_array(
                [m31_u32(3), m31_u32(5), m31_u32(19), m31_u32(23)]);
            let t = binding_w(ch, lam, d, tokens, &lay);
            let (_, got, _) = binding_cols(&r, &t);
            let want = residual_stream_claim(&w.xin, d, tokens, a, ch, lam);
            assert_eq!(got, want, "d={d} tokens={tokens} a={a} target={target}");
        }
    }

    /// The id claim is `sum_t w^(tokens - 1 - t) tok[t]`, and nothing else.
    #[test]
    fn id_claim_is_the_ids() {
        let (d, tokens, vocab) = (96usize, 8usize, 40usize);
        let w = EWitness::synthetic(tokens, d, vocab, 5);
        let lay = EmbLayout::choose(d, tokens, vocab, 24);
        let mut m_ef = vec![0u32; 1 << EF_BITS];
        let r = component(&w, lay, &mut m_ef, None);
        let ch = SecureField::from_m31_array(
            [m31_u32(29), m31_u32(31), m31_u32(37), m31_u32(41)]);
        let t = binding_w(ch, SecureField::one(), d, tokens, &lay);
        let (_, _, got) = binding_cols(&r, &t);
        let mut want = SecureField::zero();
        for k in 0..tokens {
            want = want * ch + SecureField::from(m31_u32(w.tok[k]));
        }
        assert_eq!(got, want);
    }


    /// Stream KK, `notes/witness-slack.md` §1a, both arms, on tuples and not on
    /// field sums: with the pre-KK table the padded id's use row **is** a table
    /// row, so the lookup has something to cancel against; with the sentinel it
    /// is not the tuple of any row of the table at all, so the multiplicity has
    /// nowhere to go.  The proving half of the same pair is
    /// `embed-probes`' `pad` / `pad_legacy` rows.
    #[test]
    fn a_padded_id_is_a_table_row_only_before_the_sentinel() {
        let (d, tokens, vocab, target) = (96usize, 8usize, 40usize, 24usize);
        let lay = EmbLayout::choose(d, tokens, vocab, target);
        assert!(lay.vpad > vocab);
        let w = EWitness::padded(tokens, d, vocab, target, 11, 3);
        // the committed columns are in the domain's own order, so the rows are
        // read as sets: every use tuple whose id is the padded one, against
        // every row of the table.
        let cells = |e: &Eval| -> Vec<M31> {
            e.data.iter().flat_map(|p| p.to_array()).collect()
        };
        for legacy in [true, false] {
            let mut m_ef = vec![0u32; 1 << EF_BITS];
            let r = component(&w, lay, &mut m_ef, legacy.then_some(Tamper::LegacyPad));
            let (ur, tr) = (1usize << lay.log_use, 1usize << lay.log_tab);
            let (tk, cb) = (cells(&r.base[0]), cells(&r.pre[4]));
            let act = cells(&r.pre[0]);
            let vals: Vec<Vec<M31>> = (0..lay.pe)
                .map(|sl| {
                    let (sm, ef) = (cells(&r.base[1 + 2 * sl]), cells(&r.base[2 + 2 * sl]));
                    (0..ur).map(|j| sm[j] * m31_u32(PACK as u32) + ef[j]).collect()
                })
                .collect();
            let tcols: Vec<Vec<M31>> = r.tab_pre.iter().map(&cells).collect();
            let tab: Vec<Vec<M31>> = (0..tr)
                .map(|n| tcols.iter().map(|c| c[n]).collect())
                .collect();
            let mut seen = 0usize;
            for j in 0..ur {
                if act[j] == M31::zero() || tk[j] != m31_u32(vocab as u32) {
                    continue;
                }
                seen += 1;
                let mut want = vec![tk[j], cb[j]];
                want.extend((0..lay.pe).map(|sl| vals[sl][j]));
                let found = tab.iter().filter(|row| **row == want).count();
                assert_eq!(found, usize::from(legacy),
                           "legacy={legacy}: the padded id's tuple matched {found} table row(s)");
            }
            assert_eq!(seen, lay.k, "the padded position takes one row per channel block");
        }
    }

    /// Stream KK, `notes/witness-slack.md` section 1b: the sum
    /// `verify_composed` recomputes from the statement's published ids is the
    /// claim the component publishes, at four shapes.  One line in the verifier
    /// rests on this; if the two conventions ever drift, it fails here rather
    /// than on a box, and a chunk that is honest reads as one that is not.
    #[test]
    fn the_published_ids_recompute_the_claim() {
        for (d, tokens, vocab, target) in [(96usize, 8usize, 40usize, 24usize),
                                           (128, 16, 97, 32),
                                           (64, 16, 33, 64),
                                           (128, 4, 300, 16)] {
            let w = EWitness::synthetic(tokens, d, vocab, 17);
            let lay = EmbLayout::choose(d, tokens, vocab, target);
            let mut m_ef = vec![0u32; 1 << EF_BITS];
            let r = component(&w, lay, &mut m_ef, None);
            let ch = SecureField::from_m31_array(
                [m31_u32(43), m31_u32(47), m31_u32(53), m31_u32(59)]);
            let t = binding_w(ch, SecureField::one(), d, tokens, &lay);
            let (_, _, claimed) = binding_cols(&r, &t);
            // exactly `verify_composed`'s `horner(.., true)`
            let ids = r.token_ids();
            assert_eq!(ids.len(), tokens);
            assert_eq!(ids, w.tok, "the published ids are the witness's");
            let mut acc = SecureField::zero();
            for v in &ids {
                acc = acc * ch + SecureField::from(m31_u32(*v));
            }
            assert_eq!(acc, claimed, "d={d} tokens={tokens} vocab={vocab}");
        }
    }

    /// A `Params` with the wiring family and nothing else optional: the smallest
    /// statement the gadget can ride on.
    fn params(embed: bool) -> crate::compose::Params {
        let mats: Vec<crate::compose::MatParams> = ["q", "o"]
            .iter()
            .map(|n| crate::compose::MatParams {
                name: (*n).into(), variant: "pow2".into(), n_in: 64, n_out: 64, tokens: 32,
            })
            .collect();
        crate::compose::Params {
            s: 32, c: 128, hd: 32, nh: 3, head_groups: vec![2, 1],
            variant: "intsplit6".into(), log_table: 16,
            q_bits: 8, k_bits: 8, nmax: 1, chan_target: 128, grid_target: 64, rw: 16,
            z_lo_bits: 16, z_hi_bits: 6, z_min: 1, mats, deq_target: 16, kmax: 12,
            round: "away".into(), ao_target: 16, ao_q_bits: 7, ao_rem_bits: 8,
            // stream JJ (E11): the attention output's two windows.  This fixture
            // carries no attnout data, so the values only have to satisfy
            // `Plan::new`'s asserts.
            ao_jv_min: -8, ao_jv_max: -6, ao_jo_min: -10, ao_jo_max: -10,
            ao_acc_lo_bits: 16, ao_acc_hi_bits: 4,
            // stream TT (E11), log row 73: the offset form and the plain
            // quotient bound, which is what every statement that composed
            // before these two fields existed carries.
            ao_acc_mag: false, ao_acc_prod: false, ao_denp2: false,
            fold_rem_bits: 0, fold_alpha: 0, freivalds_p: 136,
            residual_stream: Some(crate::compose::ResidualStreamParams {
                // stream II: the two additive components are unconditional, so this
                // fixture carries a real `dff` and the model's two norm
                // constants.  A mechanical update, not a change of what the
                // test asserts -- both plans it compares gain the same columns.
                d: 64, a: 4, dff: 64, head_norm: false,
                eps: crate::bf16::from_f64(1e-5),
                inv_d: crate::bf16::from_f64(1.0 / 64.0),
            }),
            diagonal_block: false, gkr_key_tiles: false, gkr_dequant: false, gkr_residual_stream: false,
            gkr_freivalds: false, gkr_attnout: false, layer_boundary: false, kv_boundary: false,
            logits_claim: false, prove_argmax: false, prove_argmax_lax: false,
            embed_vocab: if embed { 300 } else { 0 },
            embed_per_row: if embed { 32 } else { 0 },
            // stream NN: this fixture's q/k/v widths do not line up with its
            // key tiles, so it carries no rotation -- exactly as a shape whose
            // matmuls are not the attention's does in the field
            rope: None,
        }
    }

    /// The three append-only invariants, checked against the flag-off plan
    /// rather than asserted in prose: the preprocessed id list gains exactly the
    /// gadget's ids **at the end**, and both claimed-sum cursors gain exactly
    /// one. A mismatch in either cursor is caught today only by `sums_cancel`
    /// after a full prove, which on a box is an hour.
    #[test]
    fn registration_is_append_only() {
        let (off, on) = (crate::compose::Plan::new(&params(false)),
                         crate::compose::Plan::new(&params(true)));
        let l = on.emb.expect("the plan builds the gadget");
        let (a, b) = (off.pre_ids(), on.pre_ids());
        assert_eq!(b.len(), a.len() + 5 + 2 + l.pe, "the id list gained the wrong count");
        assert_eq!(a, b[..a.len()], "an existing preprocessed id moved");
        let want: Vec<String> = EMB_SEL.iter().map(|n| id(*n).id)
            .chain([cb_id().id, tab_v_id().id, tab_cb_id().id])
            .chain((0..l.pe).map(|k| tab_e_id(k).id))
            .collect();
        let got: Vec<String> = b[a.len()..].iter().map(|i| i.id.clone()).collect();
        assert_eq!(got, want);
        assert_eq!(on.n_use(), off.n_use() + 1);
        assert_eq!(on.n_table_sums(), off.n_table_sums() + 1);
    }

    /// The two components exist, are last, and are the last two the prewarm list
    /// covers -- the trap stream X recorded, checked here rather than on a card.
    #[test]
    fn components_are_last_and_prewarmed() {
        for embed in [false, true] {
            let plan = crate::compose::Plan::new(&params(embed));
            let mut ch = stwo::core::channel::Blake2sM31Channel::default();
            Channel::mix_u64(&mut ch, 0xBB_0001);
            let rels = crate::compose::Rels::draw(
                &mut ch, &plan.widths, plan.residual_stream.as_ref().map(|g| g.uns.as_slice()),
                plan.emb.map(|l| l.arity()), plan.needs_rq_rels(),
                                          plan.q_tabs.len());
            let f = Channel::draw_secure_felts(&mut ch, 6);
            let claims = crate::exprprobe::claims_for(&plan, 0x0b65_bb01);
            let b = crate::compose::build_components(&plan, &rels, f[0], f[1], f[2], f[3],
                                                     f[4], f[5], &claims);
            assert_eq!(b.embed.is_some(), embed);
            assert_eq!(b.emb_tab.is_some(), embed);
            let n = b.components().len();
            assert_eq!(b.provers().len(), n);
            // stream X's trap, as a test: `prewarm_blobs` must cover *every*
            // dispatched component. `prewarm` bumps `blob_report().lowerings`
            // once per call whatever the kernel store holds, so the count is
            // the list's length and the components' is what it has to equal.
            // Off the GKR backend there is no `link_binding`, so the two coincide.
            let before = crate::cudagen::blob_report()["lowerings"].as_u64().unwrap();
            b.prewarm_blobs();
            let after = crate::cudagen::blob_report()["lowerings"].as_u64().unwrap();
            assert_eq!(after - before, n as u64,
                       "prewarm_blobs covers {} of {n} dispatched components (embed={embed})",
                       after - before);
            // and the fingerprint report, which keeps `cudaemit`'s order but
            // must keep this list: `n_evals` is `walk`'s entries less the link
            // binding, which carries no evaluator and no kernel.
            assert_eq!(b.fp_rows().len(), b.n_evals());
            if embed {
                // the two are the last two components, so their interaction and
                // base spans are after every family that already exists
                let spans = b.components();
                let last = spans[n - 1].trace_log_degree_bounds();
                let prev = spans[n - 2].trace_log_degree_bounds();
                // stream PP: the base trace is tree 2 -- the shape tree and the
                // model tree are 0 and 1.
                let base = crate::common::BASE_TREE;
                assert_eq!(last[base].len(), 1, "the table component commits one column");
                assert_eq!(prev[base].len(), 1 + 2 * plan.emb.unwrap().pe,
                           "the use component commits `tk` and two cells per channel");
            }
        }
    }
    /// The block count is a power of two and the block width divides `d`, at
    /// both real shapes and at a non-power-of-two width.
    #[test]
    fn layout_divides() {
        for (d, vocab, target, pe, k) in [
            (4096usize, 128_256usize, 256usize, 256usize, 16usize),
            (896, 151_936, 256, 224, 4),
            (3584, 151_936, 256, 224, 16),
            (4096, 128_256, 4096, 4096, 1),
        ] {
            let l = EmbLayout::choose(d, 512, vocab, target);
            assert_eq!((l.pe, l.k), (pe, k), "d={d} target={target}");
            assert_eq!(l.pe * l.k, d);
            assert!(l.k.is_power_of_two());
            assert_eq!(1usize << l.log_tab, l.vpad * l.k);
        }
    }
}
