//! The attention output's product form, proved and attacked end to end.
//!
//! The product form (`compose::Params::ao_acc_prod`) commits `ahn = ahi num`
//! per slot and range-checks it, which bounds `mag num` slot by slot instead of
//! by the chunk-wide pair of the largest `|acc|` and the largest exponent.
//! `params_of` takes it only when the magnitude form's window refuses, which no
//! synthetic shape does, so `--ao-acc-prod` forces it here.
//!
//! This pins three things on the two sound shapes `prover/README.md` records
//! digests for (with and without the five GKR reductions): the honest chunk
//! proves and verifies in the product form, the suite reaches its last control
//! with no forgery accepted, and the magnitude form's own control (the
//! magnitude committed as the negative root, about `2^31`) is still rejected.
//!
//! SIMD build only, for the reason `attnout_jo_control.rs` gives: a CUDA
//! build has no generation kernel for the shapes the suite restates.

use std::process::Command;

const SYN: &[&str] = &[
    "compose-probes", "--round", "away", "--variants", "intsplit6", "--table-bits", "16",
    "--s", "32", "--ctx", "128", "--hd", "32", "--head-groups", "2,1",
    "--nout", "64", "--residual-stream-a", "32", "--kmax", "16", "--synth-residual-stream",
    "--diagonal-block", "--ao-acc-prod",
];
const GKR5: &[&str] =
    &["--gkr-key-tiles", "--gkr-dequant", "--gkr-residual-stream", "--gkr-freivalds", "--gkr-attnout"];
const MAG_ROOT: &str = "attnout: the magnitude committed as the negative root";
const LIMB: &str = "attnout: one accumulator limb off its own decomposition";
const NO_CONSUMER: &str = "nothing -- this chunk has no consumer for it";

fn suite(name: &str, extra: &[&str]) -> Vec<serde_json::Value> {
    let args: Vec<&str> = SYN.iter().chain(extra).copied().collect();
    let out = Command::new(env!("CARGO_BIN_EXE_zk-prover"))
        .args(&args)
        .env("RAYON_NUM_THREADS", "4")
        .output()
        .expect("run the prover binary");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{name}: zk-prover {} failed:\n{err}", args.join(" "));
    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("one JSON document on stdout");
    assert!(v["result"]["crashed"].is_null(), "{name}: the suite was cut short");
    v["result"]["probes"].as_array().expect("result.probes").clone()
}

fn check(name: &str, ps: &[serde_json::Value]) {
    let honest = &ps[0]["result"];
    assert_eq!(honest["accepts"], serde_json::Value::Bool(true),
               "{name}: the honest chunk did not verify");
    assert_eq!(honest["ao_wrap"]["form"], serde_json::json!("product"),
               "{name}: the chunk is not in the product form");
    assert_eq!(ps.last().expect("some probes")["probe"],
               serde_json::json!("tampered claim in the statement"),
               "{name}: the suite did not reach its last control");
    for p in ps {
        if p["accepted"] == serde_json::Value::Bool(true) {
            assert_eq!(p["expect_reject_by"], serde_json::json!(NO_CONSUMER),
                       "{name}: {} was accepted", p["probe"]);
        }
    }
    for control in [MAG_ROOT, LIMB] {
        let p = ps.iter().find(|p| p["probe"].as_str() == Some(control))
            .unwrap_or_else(|| panic!("{name}: no record for {control:?}"));
        assert_eq!(p["accepted"], serde_json::Value::Bool(false),
                   "{name}: {control} was not rejected");
    }
}

#[test]
#[cfg_attr(feature = "cuda", ignore = "the control suite runs on the SIMD build")]
fn the_product_form_proves_and_rejects_every_control() {
    check("sound", &suite("sound", &[]));
}

#[test]
#[cfg_attr(feature = "cuda", ignore = "the control suite runs on the SIMD build")]
fn the_product_form_proves_and_rejects_every_control_under_gkr() {
    check("sound_gkrfive", &suite("sound_gkrfive", GKR5));
}
