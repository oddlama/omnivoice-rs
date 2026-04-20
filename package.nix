{
  lib,
  rustPlatform,
  pkg-config,
  openssl,
  makeWrapper,
  symlinkJoin,
  cudaPackages,
  addDriverRunpath,
  config,
  cudaCapability ? "86",
}:

let
  cudaSupport = config.cudaSupport or false;

  # Merge split CUDA packages into a single tree so build scripts
  # (candle-kernels) can find headers, nvcc, and libraries together.
  cudaToolkit = symlinkJoin {
    name = "cuda-toolkit-joined";
    paths = with cudaPackages; [
      cuda_nvcc
      cuda_cccl
      cuda_cudart
      libcublas
      libcublas.dev
      cuda_nvrtc
      cuda_nvrtc.dev
      libcurand
      libcurand.dev
    ];
  };

  cudaLibraryPath = lib.makeLibraryPath (with cudaPackages; [
    (cuda_cudart.lib or cuda_cudart)
    (libcublas.lib or libcublas)
    (cuda_nvrtc.lib or cuda_nvrtc)
    (libcurand.lib or libcurand)
  ]);
in
rustPlatform.buildRustPackage {
  pname = "omnivoice";
  version = "0.1.0";

  src = lib.cleanSource ./.;

  cargoLock.lockFile = ./Cargo.lock;

  nativeBuildInputs =
    [ pkg-config ]
    ++ lib.optionals cudaSupport [
      makeWrapper
      cudaPackages.cuda_nvcc
    ];

  buildInputs =
    [ openssl ]
    ++ lib.optionals cudaSupport (with cudaPackages; [
      cuda_cudart
      libcublas
      cuda_nvrtc
      libcurand
    ]);

  buildFeatures = lib.optionals cudaSupport [ "cuda" ];

  # Integration tests require network access (Hugging Face Hub) and model
  # weights that are not available inside the Nix build sandbox.
  doCheck = false;

  env = lib.optionalAttrs cudaSupport {
    CUDA_ROOT = "${cudaToolkit}";
    CUDA_COMPUTE_CAP = cudaCapability;
  };

  # cudarc loads CUDA libraries via dlopen at runtime; wrap binaries so
  # they can find both the toolkit libs and the NixOS driver stub.
  postFixup = lib.optionalString cudaSupport ''
    for bin in $out/bin/*; do
      wrapProgram "$bin" \
        --prefix LD_LIBRARY_PATH : "${cudaLibraryPath}" \
        --prefix LD_LIBRARY_PATH : "${addDriverRunpath.driverLink}/lib"
    done
  '';

  meta = {
    description = "GPU-first Rust port of OmniVoice TTS inference";
    homepage = "https://github.com/FerrisMind/omnivoice-rs";
    license = lib.licenses.asl20;
    mainProgram = "omnivoice-infer";
  };
}
