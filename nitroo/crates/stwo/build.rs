use std::path::PathBuf;
fn main() {
    #[cfg(target_os = "macos")]
    std::process::exit(0);
    // cargo's job count (`-j`, `CARGO_BUILD_JOBS`), exported to build scripts as NUM_JOBS; 16 when unset
    let jobs: usize = std::env::var("NUM_JOBS").ok().and_then(|s| s.parse().ok()).unwrap_or(16);
    let dst = cmake::Config::new("src/stwo_cuda/cuda")
        .profile("Release")
        .build_arg(format!("--jobs={jobs}"))
        .build();
    println!("cargo:rustc-link-search=native={}", dst.display());
    println!("cargo:rustc-link-lib=static=stwo_cuda");
    let cuda_lib_path: PathBuf = PathBuf::from("/usr/local/cuda/lib64");
    println!("cargo:rustc-link-search=native={}", cuda_lib_path.display());
    println!("cargo:rustc-link-lib=cudart");
    #[cfg(target_os = "linux")]
    println!("cargo:rustc-link-lib=stdc++");
}