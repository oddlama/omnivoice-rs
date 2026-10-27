{
  lib,
  fetchFromGitHub,
  rustPlatform,
  pkg-config,
  openssl,
  makeWrapper,
  symlinkJoin,
  cudaPackages,
  addDriverRunpath,
  config,
  cudaSupport ? config.cudaSupport or false,
  cudaCapability ? "86",
}:

let
  # Match the CUTLASS revision requested by candle-flash-attn's build script.
  cutlass = fetchFromGitHub {
    owner = "NVIDIA";
    repo = "cutlass";
    rev = "7d49e6c7e2f8896c47f586706e67e1fb215529dc";
    hash = "sha256-D/s7eYsa5l/mfx73tE4mnFcTQdYqGmXa9d9TCryw4e4=";
  };

  # Merge split CUDA packages into a single tree so build scripts
  # (candle-kernels) can find headers, nvcc, and libraries together.
  cudaToolkit = symlinkJoin {
    name = "cuda-toolkit-joined";
    paths = with cudaPackages; [
      cuda_nvcc
      cccl
      cuda_cudart
      libcublas
      libcublas.dev
      cuda_nvrtc
      cuda_nvrtc.dev
      libcurand
      libcurand.dev
    ];
  };

  cudaLibraryPath = lib.makeLibraryPath (
    with cudaPackages;
    [
      (cuda_cudart.lib or cuda_cudart)
      (libcublas.lib or libcublas)
      (cuda_nvrtc.lib or cuda_nvrtc)
      (libcurand.lib or libcurand)
    ]
  );
in
rustPlatform.buildRustPackage {
  pname = "omnivoice";
  version = "0.1.0";

  src = lib.cleanSource ./.;

  cargoLock = {
    lockFile = ./Cargo.lock;
    # All Candle crates share this Git source; one hash covers the workspace.
    outputHashes."candle-core-0.1.0" = "sha256-S4ZX4Uv0oksBSafBWo/M8oXrmM7ZlXpP7ct/YA+Bxdk=";
  };

  nativeBuildInputs = [
    pkg-config
  ]
  ++ lib.optionals cudaSupport [
    makeWrapper
    cudaPackages.cuda_nvcc
  ];

  buildInputs = [
    openssl
  ]
  ++ lib.optionals cudaSupport (
    with cudaPackages;
    [
      cuda_cudart
      libcublas
      cuda_nvrtc
      libcurand
    ]
  );

  buildFeatures = lib.optionals cudaSupport [ "cuda" ];

  # Integration tests require network access (Hugging Face Hub) and model
  # weights that are not available inside the Nix build sandbox.
  doCheck = false;

  env = {
    # Override upstream's target-cpu=native: Nix outputs must not depend on
    # the build host's CPU.
    RUSTFLAGS = "-C force-frame-pointers=yes";
  }
  // lib.optionalAttrs cudaSupport {
    CUDA_ROOT = "${cudaToolkit}";
    CUDA_COMPUTE_CAP = cudaCapability;
  };

  # cudaforge otherwise tries to clone CUTLASS during the sandboxed build.
  # cargoSetupHook provides a writable copy of the vendored dependencies.
  postPatch = lib.optionalString cudaSupport ''
    substituteInPlace "$cargoDepsCopy/candle-flash-attn-0.11.0/build.rs" \
      --replace-fail 'const CUTLASS_COMMIT: &str = "${cutlass.rev}";' "" \
      --replace-fail '.with_cutlass(Some(CUTLASS_COMMIT))' \
        '.include_path("${cutlass}/include").include_path("${cutlass}/tools/util/include").arg("-DUSE_CUTLASS")'
  '';

  preBuild = lib.optionalString cudaSupport ''
    export CUDAFORGE_THREADS="$NIX_BUILD_CORES"
  '';

  # cudarc loads CUDA libraries via dlopen at runtime; wrap binaries so
  # they can find both the toolkit libs and the NixOS driver stub.
  postFixup = lib.optionalString cudaSupport ''
    for bin in $out/bin/*; do
      wrapProgram "$bin" \
        --prefix LD_LIBRARY_PATH : "${cudaLibraryPath}" \
        --prefix LD_LIBRARY_PATH : "${addDriverRunpath.driverLink}/lib"
    done
  '';

  # Consumers such as the NixOS module select GPU specific runtime settings
  # based on this flag.
  passthru = {
    inherit cudaSupport;
  };

  meta = {
    description = "GPU-first Rust port of OmniVoice TTS inference";
    homepage = "https://github.com/FerrisMind/omnivoice-rs";
    license = lib.licenses.asl20;
    mainProgram = "omnivoice-infer";
  };
}
