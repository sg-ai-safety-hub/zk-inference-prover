//! Issue #18: the shape tree's root, cached under exactly what it is a
//! function of, and the calibration it is a function of, published.
//!
//! Deriving a segment's shape root is `expect-root --shape-only`: the whole
//! preprocessed shape tree is built and committed, and the root is read off it.
//! That is the verifier's dominant cost, and a grid's segments re-derive
//! columns that are identical from segment to segment.
//!
//! The shape tree is a function of two things and of nothing else.
//!
//! * `Fixed.params` -- `Plan::new` takes the statement and produces every
//!   layout, every selector's height and every table's width, and
//!   `Plan::pre_split` classifies a column `PreNeeds::Params` exactly when the
//!   statement determines every one of its cells.
//! * the per-layer calibration, which is what `PreNeeds::Manifest` names: the
//!   key tiles' `dinv` (the per-instance table-key scale `2^-(F - frac_i)`) and
//!   the master exponent table's value column, `table_of(1 << log_table, nmax,
//!   frac)`. Both come off the dump's `frac_per_head` and neither is a weight.
//!   In the loaded segment that calibration is the key tiles' own `frac` (the
//!   master exponent `F`), `frac_per_inst` (`frac_per_head`, cycled into the
//!   key tile's instances) and `delta` (`2^(F - frac_i)`, the value `dinv`
//!   inverts).
//!
//! So the key is the SHA-256 of the serialized `Params` and of that
//! calibration, and two segments whose key agrees have the same shape tree cell
//! for cell. Everything else a segment differs in -- its activations, its
//! weights, its model root -- is outside both.
//!
//! The cache is **off by default** and turned on by `--shape-cache DIR`, which
//! is both the switch and the directory. On disk rather than in process alone,
//! because `zk-prover model-roots` runs each plan line through `expect-root` in
//! a process of its own and an in-process cache cannot cross that; and off
//! unless asked for, so that no run which has ever recorded a digest takes a
//! different path than it did. A stored record carries the key's whole input
//! beside the root, and a record whose input is not the current one is a
//! refusal and not a root: a cache that answered on the digest alone would be a
//! second place a shape root could come from, and this one is meant to have
//! none.
//!
//! What the root is a function of is published whether or not the cache is on:
//! `result.shape` names the key and the calibration on every `compose`,
//! `expect-root` and `expect-root --shape-only` run.
//!
//! `--shape-cache-verify` is the control: the tree is committed anyway and the
//! root it gives is asserted equal to the stored one, which is a cell-for-cell
//! comparison by construction (rule 5's shape, as `precache` does it). A tamper
//! is the case the assertion would misread -- `--corrupt-exp` and
//! `--corrupt-norm` bend a shape column and move the root under a statement and
//! a calibration that did not move -- so `--shape-cache` refuses to be set
//! beside one rather than storing a bent root or rejecting an honest run.
//!
//! **Issue #95 asks whether a verifier could afford to rebuild the shape tree
//! inside `verify_composed`, and this is the half of that it can serve.** The
//! key is taken over `Fixed.params`, which a verifier holds, and over a
//! `Calibration`, whose fields are public and need no `Layer`: the model
//! manifest `zk-prover model-roots` writes already publishes the calibration
//! those fields are built from, per layer, beside the model root. So
//! `verify_composed` could compute this key from what it holds, and the first
//! proof at a shape would pay the derivation while every later proof at that
//! shape paid a lookup.
//!
//! What the cache does **not** do is make the comparison: a stored root is only
//! as good as the derivation that wrote it, so it amortises the honest cost and
//! is not itself a check.
//!
//! **Issue #95 makes the comparison.** `verify_composed` derives the shape root
//! from `Fixed.params` and the calibration the model manifest publishes and
//! compares it with the statement's `pre_root`, and the second half of this
//! file is what it derives with: the calibration read off a published document
//! (`--verify-calibration FILE`) and the memo that makes one shape's derivation
//! cost one derivation in a process. That memo is **not** the cache above: a
//! `--shape-cache` record is a prover-side artifact, and a verifier that read a
//! root out of a directory would have taken the root off something other than
//! its own derivation, which is the whole of what #95 refuses.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};

use serde_json::{json, Value};

use crate::compose::Params;

/// The per-key-tile calibration the shape tree reads and the statement does not
/// carry. Floats are their IEEE-754 bit patterns, so the key is exact and a
/// record round-trips through JSON without a parser's rounding in the way.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct KeyTileCalibration {
    /// the master exponent `F` this key tile's table is pinned to
    pub frac: u64,
    /// `frac_per_head`, cycled into this key tile's instances
    pub frac_per_inst: Vec<u64>,
    /// the per-instance table-key scale `2^(F - frac_i)`, which `dinv` inverts
    pub delta: Vec<i64>,
}

/// One segment's calibration, in `Layer::key_tiles` order.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Calibration(pub Vec<KeyTileCalibration>);

impl Calibration {
    /// The calibration of a loaded segment, off the key tiles the loader built.
    ///
    /// It is read here and not off `manifest.json` on purpose: `Layer::load`
    /// and `Layer::load_weights` are the one place `frac_per_head` becomes
    /// `frac`, `frac_per_inst` and `delta`, and a second reader of the manifest
    /// would be a second derivation to drift from them. A synthetic segment has
    /// no manifest at all and this still names its calibration exactly.
    pub fn of(l: &crate::layer::Layer) -> Self {
        Self(l.key_tiles.iter().map(|a| KeyTileCalibration {
            frac: a.frac.to_bits(),
            frac_per_inst: a.frac_per_inst.iter().map(|f| f.to_bits()).collect(),
            delta: a.delta.clone(),
        }).collect())
    }

    /// The published form, which is also the form the key is taken over: the
    /// bit patterns, exact, and the decimal values a reader wants, side by
    /// side.
    pub fn json(&self) -> Value {
        Value::Array(self.0.iter().map(|k| json!({
            "frac": f64::from_bits(k.frac),
            "frac_bits": k.frac,
            "frac_per_inst": k.frac_per_inst.iter().map(|b| f64::from_bits(*b))
                .collect::<Vec<f64>>(),
            "frac_per_inst_bits": k.frac_per_inst,
            "delta": k.delta,
        })).collect())
    }
}

/// What the key is taken over: the statement's own parameters and the
/// calibration. Serialized so a stored record can be compared with the segment
/// in hand field for field, rather than on the digest alone.
pub fn inputs(p: &Params, calib: &Calibration) -> Value {
    json!({
        "version": "zk-shape-root-v1",
        "params": serde_json::to_value(p).expect("params serialize"),
        "calibration": calib.json(),
    })
}

/// The key: SHA-256 of the serialized inputs, as hex.
pub fn key(inputs: &Value) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(serde_json::to_vec(inputs).expect("shape key serialize"));
    format!("{:x}", h.finalize())
}

struct Entry {
    inputs: Value,
    pre_root: String,
}

static MEMO: OnceLock<Mutex<HashMap<String, Entry>>> = OnceLock::new();
static DIR: OnceLock<Option<PathBuf>> = OnceLock::new();
static VERIFY: OnceLock<bool> = OnceLock::new();
/// this segment's own key and inputs, noted once per `generate`
static NOTED: OnceLock<Mutex<Option<(String, Value)>>> = OnceLock::new();
static HITS: AtomicU64 = AtomicU64::new(0);
static MISSES: AtomicU64 = AtomicU64::new(0);
static WRITES: AtomicU64 = AtomicU64::new(0);
static CHECKS: AtomicU64 = AtomicU64::new(0);

/// Every refusal here is a designed one, and a `Mutex` a designed panic
/// crossed is poisoned rather than corrupt, so the guard is taken through this
/// and the second caller gets the map rather than a poison error.
fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn memo() -> &'static Mutex<HashMap<String, Entry>> {
    MEMO.get_or_init(|| Mutex::new(HashMap::new()))
}

fn noted() -> &'static Mutex<Option<(String, Value)>> {
    NOTED.get_or_init(|| Mutex::new(None))
}

/// `--shape-cache DIR` and `--shape-cache-verify`, read once from the command
/// line. Without the directory the cache is off and every run derives its own
/// root, exactly as it did before this existed.
pub fn configure(args: &[String]) {
    let dir = crate::cli::arg_str(args, "--shape-cache").map(PathBuf::from);
    if let Some(d) = dir.as_ref() {
        // A tamper bends a committed shape column after the statement and the
        // calibration are fixed, so the root moves under a key that does not.
        // Storing that root, or comparing an honest one against it, is the one
        // way this cache could answer with a root that is not the shape's, and
        // the refusal is here rather than in the store.
        let bent: Vec<&String> = args.iter().filter(|a| a.starts_with("--corrupt-")).collect();
        assert!(bent.is_empty(),
                "--shape-cache is keyed on the statement and the calibration, and {bent:?}                  moves a committed shape column without moving either -- run the control                  without the cache");
        std::fs::create_dir_all(d)
            .unwrap_or_else(|e| panic!("--shape-cache {}: {e}", d.display()));
    }
    let _ = DIR.set(dir);
    let _ = VERIFY.set(args.iter().any(|a| a == "--shape-cache-verify"));
    // Issue #95: and the verifier's own input, read here because it is read
    // once per process and because every command that verifies takes it.
    let _ = VERIFY_CALIB.set(crate::cli::arg_str(args, "--verify-calibration").map(|f| {
        let bytes = std::fs::read(&f)
            .unwrap_or_else(|e| panic!("--verify-calibration {f}: {e}"));
        let v: Value = serde_json::from_slice(&bytes)
            .unwrap_or_else(|e| panic!("--verify-calibration {f}: not JSON: {e}"));
        (f, v)
    }));
}

/// Whether the cache is on at all: `--shape-cache DIR` is the switch.
pub fn enabled() -> bool {
    dir().is_some()
}

/// The same switch, for the tests below: a directory of this process's own, so
/// a run leaves nothing a later run would read back as a hit.
#[cfg(test)]
fn enable_for_tests() {
    let d = std::env::temp_dir().join(format!("zk-shape-root-test-{}", std::process::id()));
    std::fs::create_dir_all(&d).expect("test cache directory");
    let _ = DIR.set(Some(d));
    let _ = VERIFY.set(false);
}

fn dir() -> Option<&'static PathBuf> {
    DIR.get().and_then(|d| d.as_ref())
}

/// Whether the derivation runs anyway and compares its root with the stored
/// one, instead of taking the stored one and stopping.
pub fn verifying() -> bool {
    *VERIFY.get().unwrap_or(&false)
}

fn path_of(k: &str) -> Option<PathBuf> {
    dir().map(|d| d.join(format!("{k}.json")))
}

/// Note this segment's key, so the commit path can store the root it derives
/// without recomputing either.
pub fn note(k: &str, inputs: &Value) {
    *lock(noted()) = Some((k.to_string(), inputs.clone()));
}

/// The stored shape root for this key, when there is one whose inputs are the
/// ones in hand.
///
/// A record whose digest matches and whose inputs do not is refused by name:
/// either two different shapes collided under SHA-256, or the file was written
/// by something that is not this derivation, and neither is a root to serve.
pub fn take(k: &str, inputs: &Value) -> Option<String> {
    let root = stored(k, inputs)?;
    HITS.fetch_add(1, Relaxed);
    Some(root)
}

/// The root under this key, from the process's own map or from the directory,
/// with the record's inputs compared against the ones in hand.
///
/// A record whose digest matches and whose inputs do not is refused by name:
/// either two shapes collided under SHA-256, or the file was written by
/// something that is not this derivation, and neither is a root to serve.
fn stored(k: &str, inputs: &Value) -> Option<String> {
    if !enabled() {
        return None;
    }
    // read out from under the lock before comparing: the comparison is a
    // designed refusal, and a refusal taken while the guard is alive would
    // poison the map for every later caller instead of naming the one record
    let held = lock(memo()).get(k).map(|e| (e.inputs.clone(), e.pre_root.clone()));
    if let Some((was, root)) = held {
        assert_eq!(&was, inputs,
                   "shape cache: the in-process record under {k} was derived from different \
                    parameters -- the key does not determine the shape tree");
        return Some(root);
    }
    let p = path_of(k)?;
    let bytes = std::fs::read(&p).ok()?;
    let doc: Value = serde_json::from_slice(&bytes)
        .unwrap_or_else(|e| panic!("shape cache: {} is not JSON: {e}", p.display()));
    assert_eq!(&doc["inputs"], inputs,
               "shape cache: {} was derived from different parameters -- the key does not \
                determine the shape tree", p.display());
    let root = doc["pre_root"].as_str()
        .unwrap_or_else(|| panic!("shape cache: {} names no pre_root", p.display()))
        .to_string();
    lock(memo()).insert(k.to_string(),
                        Entry { inputs: inputs.clone(), pre_root: root.clone() });
    Some(root)
}

/// Store the root this run derived under the noted key, or -- when the key
/// already carries one, in this process or in the directory -- assert the fresh
/// root against it.
///
/// The comparison is what `--shape-cache-verify` exists for, and it reads the
/// directory and not only this process's map: `zk-prover model-roots` runs each
/// plan line in a process of its own, so a control that compared in-process
/// alone would have nothing to compare against and would quietly rewrite the
/// record instead.
pub fn fill(pre_root: &str) {
    // Issue #95: a verifier's derivation runs the same builder, and the root it
    // gets is a root of stand-ins for everything but the shape. It has its own
    // memo below; the run's cache and the run's noted key are left as the run
    // itself wrote them.
    if !enabled() || DERIVING.load(Relaxed) {
        return;
    }
    let Some((k, inputs)) = lock(noted()).clone() else { return };
    if let Some(was) = stored(&k, &inputs) {
        CHECKS.fetch_add(1, Relaxed);
        assert_eq!(was, pre_root,
                   "shape cache: the shape tree committed under {k} has a different root from \
                    the stored one -- the key does not determine the tree");
        return;
    }
    MISSES.fetch_add(1, Relaxed);
    lock(memo()).insert(k.clone(),
                        Entry { inputs: inputs.clone(), pre_root: pre_root.to_string() });
    let Some(p) = path_of(&k) else { return };
    let doc = json!({"key": k, "inputs": inputs, "pre_root": pre_root});
    // written beside the target and renamed, so a reader never sees half a
    // record and two processes racing on one key cannot interleave
    let tmp = p.with_extension(format!("json.tmp.{}", std::process::id()));
    std::fs::write(&tmp, serde_json::to_vec_pretty(&doc).expect("shape record"))
        .unwrap_or_else(|e| panic!("shape cache: {}: {e}", tmp.display()));
    std::fs::rename(&tmp, &p)
        .unwrap_or_else(|e| panic!("shape cache: {}: {e}", p.display()));
    WRITES.fetch_add(1, Relaxed);
}

/// The block a run publishes: what the shape root is a function of, and what
/// the cache did with it.
pub fn shape_json() -> Value {
    let noted = lock(noted()).clone();
    match noted {
        None => Value::Null,
        Some((k, inputs)) => json!({
            "key": k,
            "calibration": inputs["calibration"],
            "cache": json!({
                "dir": dir().map(|d| d.display().to_string()),
                "verify": verifying(),
                "hits": HITS.load(Relaxed),
                "misses": MISSES.load(Relaxed),
                "writes": WRITES.load(Relaxed),
                "checked": CHECKS.load(Relaxed),
            }),
        }),
    }
}

// ------------------------------------------------ issue #95: the verifier ----

/// `--verify-calibration FILE`, read once: the published document a verifier
/// derives shape roots with.
static VERIFY_CALIB: OnceLock<Option<(String, Value)>> = OnceLock::new();
/// The roots this process derived, by key, with the inputs each was derived
/// from beside it. Filled by `derived_root` and by nothing else.
static DERIVED: OnceLock<Mutex<HashMap<String, Entry>>> = OnceLock::new();
/// set while a verifier's derivation is running, so the derivation cannot
/// write to the run's own cache or rewrite the key the run published
static DERIVING: AtomicBool = AtomicBool::new(false);
static DERIVATIONS: AtomicU64 = AtomicU64::new(0);
static REUSES: AtomicU64 = AtomicU64::new(0);

fn derived() -> &'static Mutex<HashMap<String, Entry>> {
    DERIVED.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The document `--verify-calibration` named, when the run was given one.
/// Without it a verifier has no calibration and derives nothing.
pub fn verify_calibration() -> Option<&'static (String, Value)> {
    VERIFY_CALIB.get().and_then(|d| d.as_ref())
}

/// The calibration a published document names for this statement.
///
/// Four forms, and each of them is a document this repository already writes:
///
/// * a model manifest -- `{"rows": [...]}` or the bare list of rows `zk-prover
///   model-roots --out` writes -- whose row for this statement's `(layer, c)`
///   carries the calibration. This is the verifier's own case: the manifest is
///   what publishes the model root it compares, and it publishes the
///   calibration beside it;
/// * a run's `result.shape` block, `{"calibration": [...]}`;
/// * that block's per-key-tile list on its own, which is the exact form and the
///   only one a segment whose key tiles do not share one exponent per head has;
/// * `{"frac_per_head": [...]}` -- a dump's manifest, or one manifest row --
///   which `Layer::calibration_of` cycles into the key tiles.
///
/// `layer` is the statement's own coordinate, which is how a manifest row is
/// found; a document that is not a manifest names one segment and is read as
/// it stands.
pub fn calibration_for(f: &str, doc: &Value, p: &Params, layer: Option<usize>)
                       -> Result<Calibration, String> {
    let Some(rows) = manifest_rows(doc) else {
        return calibration_of_doc(f, doc, p);
    };
    let row = rows.iter().find(|r| {
        r["layer"].as_i64() == layer.map(|l| l as i64)
            && r["c"].as_u64() == Some(p.c as u64)
    });
    // A manifest that has no row for a cell is a manifest that says nothing
    // about it, and saying nothing is not a calibration to derive with.
    let row = row.ok_or_else(|| format!(
        "{f}: the manifest has no row for (layer {}, c {}), so it publishes no calibration \
         for this statement",
        layer.map(|l| l.to_string()).unwrap_or_else(|| "unstated".into()), p.c))?;
    calibration_of_doc(f, row, p)
}

/// Whether this document is a model manifest, and its rows. A manifest row
/// names a cell of a grid (`layer`, `c`, `pre_root`, `model_root`); a key
/// tile's calibration names an exponent and a scale, and neither is the other.
fn manifest_rows(doc: &Value) -> Option<&Vec<Value>> {
    let rows = if doc.is_array() { doc.as_array()? } else { doc.get("rows")?.as_array()? };
    let first = rows.first()?;
    ["layer", "c", "pre_root", "model_root"].iter().any(|k| !first[k].is_null())
        .then_some(rows)
}

/// One segment's calibration, off a document that names exactly one: the
/// per-key-tile block a run publishes, or the per-head list a manifest row
/// carries.
///
/// The bit patterns are what the key is taken over, so they are what is read;
/// the decimal fields beside them are a reader's copy and are used only where a
/// document carries no bit pattern at all.
pub fn calibration_of_doc(f: &str, v: &Value, p: &Params) -> Result<Calibration, String> {
    // the per-key-tile block first: it is the exact form, and a document that
    // carries both is a run's own report, where the block is what the root was
    // taken over
    let per_key_tile = if v.is_array() { v.clone() } else { v["calibration"].clone() };
    if let Some(rows) = per_key_tile.as_array() {
        if rows.is_empty() {
            return Err(format!("{f}: the calibration names no key tile"));
        }
        let mut out = vec![];
        for k in rows {
            let bits = |x: &Value, name: &str| -> Result<u64, String> {
                x.as_u64().ok_or_else(|| format!("{f}: {name} is not a bit pattern"))
            };
            let frac = match (&k["frac_bits"], &k["frac"]) {
                (Value::Null, x) => x.as_f64()
                    .ok_or_else(|| format!("{f}: a key tile names no frac"))?
                    .to_bits(),
                (x, _) => bits(x, "frac_bits")?,
            };
            let frac_per_inst: Vec<u64> = match (&k["frac_per_inst_bits"], &k["frac_per_inst"]) {
                (Value::Null, x) => x.as_array()
                    .ok_or_else(|| format!("{f}: a key tile names no frac_per_inst"))?
                    .iter().map(|v| v.as_f64().unwrap_or(0.0).to_bits()).collect(),
                (x, _) => {
                    let a = x.as_array()
                        .ok_or_else(|| format!("{f}: frac_per_inst_bits is not a list"))?;
                    a.iter().map(|v| bits(v, "frac_per_inst_bits"))
                        .collect::<Result<Vec<u64>, String>>()?
                }
            };
            let delta: Vec<i64> = k["delta"].as_array()
                .ok_or_else(|| format!("{f}: a key tile names no delta"))?
                .iter().map(|v| v.as_i64().unwrap_or(0)).collect();
            out.push(KeyTileCalibration { frac, frac_per_inst, delta });
        }
        return Ok(Calibration(out));
    }
    let v = v["frac_per_head"].as_array()
        .ok_or_else(|| format!("{f}: carries neither a `calibration` block nor \
                                `frac_per_head`"))?;
    let mut fracs: Vec<f64> = vec![];
    for x in v {
        fracs.push(x.as_f64()
                   .ok_or_else(|| format!("{f}: frac_per_head is not a list of numbers"))?);
    }
    if fracs.len() < p.nh {
        return Err(format!("{f}: the calibration names {} heads and this statement has {}",
                           fracs.len(), p.nh));
    }
    Ok(crate::layer::Layer::calibration_of(p, &fracs))
}

/// Issue #95: the shape root this statement and this calibration give, derived
/// once per key in a process.
///
/// The key is the one the whole file is about -- the digest of the serialized
/// parameters and of the calibration -- so two segments of a grid at one
/// `(layer, shape)` pay one derivation between them, which is what makes the
/// comparison affordable at all. A record whose key matches and whose inputs do
/// not is refused by name here as it is in the store: the key is what the tree
/// is a function of, and a memo that answered on the digest alone would be a
/// second place a shape root could come from.
///
/// Returns the root and whether this call paid for it.
pub fn derived_root(p: &Params, calib: &Calibration,
                    derive: impl FnOnce() -> String) -> (String, bool) {
    let inputs = inputs(p, calib);
    let k = key(&inputs);
    let held = lock(derived()).get(&k).map(|e| (e.inputs.clone(), e.pre_root.clone()));
    if let Some((was, root)) = held {
        assert_eq!(&was, &inputs,
                   "the verifier's derivation under {k} was made from different parameters -- \
                    the key does not determine the shape tree");
        REUSES.fetch_add(1, Relaxed);
        return (root, false);
    }
    let root = derive();
    DERIVATIONS.fetch_add(1, Relaxed);
    lock(derived()).insert(k, Entry { inputs, pre_root: root.clone() });
    (root, true)
}

/// Held while a verifier's derivation runs: the run's own noted key is put back
/// when it ends, so a verification never rewrites what the run it verifies
/// published as `result.shape`.
pub struct Deriving(Option<(String, Value)>);

impl Drop for Deriving {
    fn drop(&mut self) {
        *lock(noted()) = self.0.take();
        DERIVING.store(false, Relaxed);
    }
}

/// Take that guard.
pub fn deriving() -> Deriving {
    DERIVING.store(true, Relaxed);
    Deriving(lock(noted()).clone())
}

/// What the verifier's derivations cost this process: how many were paid and
/// how many proofs took a root one of them had already given.
pub fn derivations_json() -> Value {
    json!({"derived": DERIVATIONS.load(Relaxed), "reused": REUSES.load(Relaxed)})
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `note` and `fill` are per process, so the two tests that use them take
    /// this in turn rather than racing over one noted key.
    static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

    /// A `Params` with the two fields the tests move, over the smallest shape
    /// `Plan` is not asked about here -- nothing in this module reads more of
    /// it than its serialization.
    fn params() -> Params {
        Params { s: 32, c: 128, hd: 32, nh: 3, head_groups: vec![2, 1],
                 variant: "intsplit6".into(), log_table: 16, q_bits: 8, k_bits: 8,
                 nmax: 16, ao_q_bits: 7, ao_rem_bits: 20,
                 ao_jv_min: -8, ao_jv_max: -6, ao_jo_min: -10, ao_jo_max: -10,
                 ao_acc_lo_bits: 16, ao_acc_hi_bits: 4,
                 ..Default::default() }
    }

    fn calibration() -> Calibration {
        Calibration(vec![KeyTileCalibration {
            frac: 6.5f64.to_bits(),
            frac_per_inst: vec![6.5f64.to_bits(), 5.5f64.to_bits()],
            delta: vec![1, 2],
        }])
    }

    /// The key is a function of the statement and of the calibration, and the
    /// cache answers on it: a root stored under one shape comes back for that
    /// shape.
    #[test]
    fn a_stored_root_comes_back() {
        let _g = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        enable_for_tests();
        let (p, c) = (params(), calibration());
        let inp = inputs(&p, &c);
        let k = key(&inp);
        note(&k, &inp);
        assert_eq!(take(&k, &inp), None, "nothing was stored yet");
        fill("0123456789abcdef");
        assert_eq!(take(&k, &inp).as_deref(), Some("0123456789abcdef"));
        // and the same inputs built a second time give the same key, so the
        // hit is not an artefact of holding one `Value` alive
        assert_eq!(key(&inputs(&params(), &calibration())), k);
    }

    /// One parameter moved is a different shape, so it is a different key and
    /// the stored root is not served for it. Every field below is one the
    /// shape tree reads: the context, the block size, the head width, the head
    /// key tile list, the exponent table's width, the score widths, the
    /// accumulator's two limbs and the exponent window.
    #[test]
    fn one_parameter_moved_is_a_different_key() {
        let base = key(&inputs(&params(), &calibration()));
        let moves: Vec<(&str, Params)> = vec![
            ("c", Params { c: 256, ..params() }),
            ("s", Params { s: 64, ..params() }),
            ("hd", Params { hd: 64, ..params() }),
            ("nh", Params { nh: 4, ..params() }),
            ("head_groups", Params { head_groups: vec![2, 2], ..params() }),
            ("log_table", Params { log_table: 17, ..params() }),
            ("variant", Params { variant: "intexp".into(), ..params() }),
            ("q_bits", Params { q_bits: 9, ..params() }),
            ("k_bits", Params { k_bits: 9, ..params() }),
            ("nmax", Params { nmax: 32, ..params() }),
            ("ao_q_bits", Params { ao_q_bits: 8, ..params() }),
            ("ao_rem_bits", Params { ao_rem_bits: 21, ..params() }),
            ("ao_acc_lo_bits", Params { ao_acc_lo_bits: 15, ..params() }),
            ("ao_acc_hi_bits", Params { ao_acc_hi_bits: 5, ..params() }),
            ("ao_jv_max", Params { ao_jv_max: -5, ..params() }),
            ("ao_jo_min", Params { ao_jo_min: -11, ..params() }),
            ("diagonal_block", Params { diagonal_block: true, ..params() }),
            ("kv_boundary", Params { kv_boundary: true, ..params() }),
            ("ao_acc_mag", Params { ao_acc_mag: true, ..params() }),
            ("ao_acc_prod", Params { ao_acc_mag: true, ao_acc_prod: true, ..params() }),
        ];
        for (name, p) in moves {
            let k = key(&inputs(&p, &calibration()));
            assert_ne!(k, base, "{name} moved and the shape key did not");
        }
    }

    /// And the calibration is in the key too: the same statement over a layer
    /// calibrated differently is a different shape tree, because `dinv` and the
    /// exponent table's value column are what it moves.
    #[test]
    fn a_moved_calibration_is_a_different_key() {
        let base = key(&inputs(&params(), &calibration()));
        let mut c = calibration();
        c.0[0].delta[1] = 4;
        assert_ne!(key(&inputs(&params(), &c)), base, "delta moved and the key did not");
        let mut c = calibration();
        c.0[0].frac = 7.5f64.to_bits();
        assert_ne!(key(&inputs(&params(), &c)), base, "the master exponent moved and the key \
                                                       did not");
        let mut c = calibration();
        c.0[0].frac_per_inst[1] = 4.5f64.to_bits();
        assert_ne!(key(&inputs(&params(), &c)), base,
                   "a per-head exponent moved and the key did not");
    }

    /// A record under the right digest and the wrong parameters is a refusal,
    /// not a root: the cache is not allowed to be a second place a shape root
    /// can come from.
    #[test]
    #[should_panic(expected = "the key does not determine the shape tree")]
    fn a_record_whose_inputs_disagree_is_refused() {
        let _g = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        enable_for_tests();
        let inp = inputs(&Params { c: 512, ..params() }, &calibration());
        let k = key(&inp);
        note(&k, &inp);
        fill("fedcba9876543210");
        // the same key, asked for with parameters that are not the ones it was
        // stored under
        let other = inputs(&Params { c: 1024, ..params() }, &calibration());
        let _ = take(&k, &other);
    }
}
