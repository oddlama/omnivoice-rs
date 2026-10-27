{
  inputs = {
    devshell = {
      url = "github:numtide/devshell";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    flake-parts.url = "github:hercules-ci/flake-parts";
    nci = {
      url = "github:90-008/nix-cargo-integration";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    git-hooks = {
      url = "github:cachix/git-hooks.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    treefmt-nix = {
      url = "github:numtide/treefmt-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    inputs:
    let
      # Use the same pinned upstream toolchain as nci's checks. The nixpkgs
      # Rust/LLVM combination fails to compile Candle's AVX-512 intrinsics.
      mkPackage =
        pkgs: args:
        let
          toolchain = (inputs.nci.inputs.rust-overlay.lib.mkRustBin { } pkgs).stable.latest.default;
        in
        pkgs.callPackage ./package.nix (
          {
            rustPlatform = pkgs.makeRustPlatform {
              cargo = toolchain;
              rustc = toolchain;
            };
          }
          // args
        );
    in
    inputs.flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [
        inputs.devshell.flakeModule
        inputs.nci.flakeModule
        inputs.git-hooks.flakeModule
        inputs.treefmt-nix.flakeModule
      ];

      systems = [
        "x86_64-linux"
        "aarch64-linux"
      ];

      flake = {
        overlays.default = final: _prev: {
          omnivoice = mkPackage final { };
          omnivoice-cuda = mkPackage final { cudaSupport = true; };
        };

        nixosModules = {
          omnivoice = ./module.nix;
          # Convenience module: adds `pkgs.omnivoice` so the package option's
          # default resolves without wiring up the overlay by hand.
          default = {
            imports = [ ./module.nix ];
            nixpkgs.overlays = [ inputs.self.overlays.default ];
          };
        };
      };

      perSystem =
        {
          config,
          lib,
          pkgs,
          system,
          ...
        }:
        let
          cudaPkgs = import inputs.nixpkgs {
            inherit system;
            config = {
              allowUnfree = true;
              cudaSupport = true;
            };
          };
        in
        {
          devshells.default = {
            packages = [
              config.treefmt.build.wrapper
              pkgs.rust-analyzer
              pkgs.pkg-config
              pkgs.openssl
              pkgs.openssl.dev
            ];
            devshell.startup.pre-commit.text = config.pre-commit.installationScript;
            env = [
              {
                name = "RUST_SRC_PATH";
                value = "${pkgs.rustPlatform.rustLibSrc}";
              }
            ];
          };

          pre-commit.settings.hooks.treefmt.enable = true;
          treefmt = {
            projectRootFile = "flake.nix";
            programs = {
              deadnix.enable = true;
              statix.enable = true;
              nixfmt.enable = true;
              rustfmt = {
                enable = true;
                edition = "2021";
              };
            };
          };

          nci.projects.omnivoice = {
            path = inputs.self.outPath;
            export = false;
            numtideDevshell = "default";
            drvConfig = {
              # Reuse the package's hash-pinned vendor tree instead of
              # dream2nix's shallow Git fetches during evaluation.
              imports = [
                {
                  rust-cargo-vendor.vendoredSources = lib.mkForce config.packages.default.cargoDeps;
                }
              ];
              env.RUSTFLAGS = config.packages.default.RUSTFLAGS;
              mkDerivation = {
                nativeBuildInputs = [ pkgs.pkg-config ];
                buildInputs = [ pkgs.openssl ];
              };
            };
            depsDrvConfig = {
              env.RUSTFLAGS = config.packages.default.RUSTFLAGS;
              mkDerivation = {
                nativeBuildInputs = [ pkgs.pkg-config ];
                buildInputs = [ pkgs.openssl ];
              };
            };
          };

          packages = {
            default = mkPackage pkgs { };
            cuda = mkPackage cudaPkgs { };
            with-cuda = config.packages.cuda;
          };

          checks = {
            package = config.packages.default;
            nixos-module =
              let
                nixos = inputs.nixpkgs.lib.nixosSystem {
                  modules = [
                    ./module.nix
                    {
                      nixpkgs.pkgs = pkgs;
                      boot.loader.grub.enable = false;
                      fileSystems."/" = {
                        device = "none";
                        fsType = "tmpfs";
                      };
                      system.stateVersion = "25.05";

                      services.omnivoice = {
                        enable = true;
                        package = config.packages.default;
                        apiKeyFile = "/run/secrets/omnivoice-api-key";
                        host = "0.0.0.0";
                        openFirewall = true;
                        wyoming = {
                          enable = true;
                          openFirewall = true;
                        };
                        defaultVoice = "alice";
                        languages = [ "en" ];
                        voices.alice = {
                          description = "Alice";
                          language = "en";
                          speed = 1;
                          num_step = 32;
                          guidance_scale = 2.0;
                        };
                      };
                    }
                  ];
                };
                failed = builtins.filter (entry: !entry.assertion) nixos.config.assertions;
              in
              if failed != [ ] then
                throw "nixosModule assertions failed: ${builtins.concatStringsSep "\n" (map (e: e.message) failed)}"
              else
                nixos.config.systemd.units."omnivoice-server.service".unit;
            omnivoice-cli-tests = config.nci.outputs.omnivoice-cli.check;
            omnivoice-infer-tests = config.nci.outputs.omnivoice-infer.check;
            omnivoice-server-tests = config.nci.outputs.omnivoice-server.check;
          };
        };
    };
}
