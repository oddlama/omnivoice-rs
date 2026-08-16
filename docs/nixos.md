# NixOS Module

The flake exports a NixOS module that runs `omnivoice-server` as a system service, optionally with the Wyoming endpoint and a declarative voice preset catalog.

Exported outputs:

- `nixosModules.default`: the module plus `overlays.default`, so `pkgs.omnivoice` resolves out of the box
- `nixosModules.omnivoice`: the bare module, for setups that provide the package themselves
- `overlays.default`: adds `pkgs.omnivoice` and `pkgs.omnivoice-cuda`
- `packages.default` / `packages.cuda`: the CPU and CUDA builds

## Minimal Configuration

```nix
{
  inputs.omnivoice.url = "github:FerrisMind/omnivoice-rs";

  outputs = { nixpkgs, omnivoice, ... }: {
    nixosConfigurations.tts = nixpkgs.lib.nixosSystem {
      system = "x86_64-linux";
      modules = [
        omnivoice.nixosModules.default
        {
          services.omnivoice = {
            enable = true;
            apiKeyFile = "/run/secrets/omnivoice-api-key";
          };
        }
      ];
    };
  };
}
```

The HTTP endpoint requires a bearer token, so the service refuses to start without `apiKeyFile` (read through systemd credentials, never copied into the Nix store) or an `environmentFile` that defines `OMNIVOICE_API_KEY`.

## Models

`services.omnivoice.model` is unset by default. The server then resolves the default Hugging Face repository and downloads the weights into `$HOME/.cache/huggingface`, where `$HOME` is `services.omnivoice.dataDir` (`/var/lib/omnivoice-rs`). Because the weights are large, the service runs as a persistent system user rather than a `DynamicUser`.

Model resolution happens before the listeners are bound, so the first start stays silent until the download finishes. `systemctl status omnivoice-server` shows the unit as running during that time; the endpoints only answer afterwards, and `GET /health` returns `503` until the runtime is ready.

Set `services.omnivoice.model` to a repository id or a directory on disk to pin a specific bundle:

```nix
services.omnivoice.model = "k2-fsa/OmniVoice";
```

## CUDA

Point the package option at a CUDA build:

```nix
services.omnivoice = {
  package = pkgs.omnivoice-cuda;
  device = "cuda";
  cuda.visibleDevices = [ "0" ];
};
nixpkgs.config.allowUnfree = true;
hardware.nvidia.open = true; # or whatever the host needs
```

`services.omnivoice.cuda.enable` follows the package's `cudaSupport` flag, so it does not normally need to be set by hand. When it is on, the unit gets `CUDA_DEVICE_ORDER=PCI_BUS_ID` — which makes `cuda.visibleDevices` indices refer to PCI bus order instead of the driver's speed ranking — and the sandbox options that block GPU access (`PrivateDevices`, `MemoryDenyWriteExecute`, `ProcSubset`, `ProtectKernelModules`) are relaxed.

## Wyoming and Voice Presets

`services.omnivoice.voices` mirrors the [voice preset TOML](./cli-and-server.md#voice-preset-toml): every attribute becomes one `[[voice.<name>]]` table, and the rendered catalog is passed as `--voices-config`. Unknown attributes are written to the file verbatim, so fields added upstream can be used before the module knows about them; the server rejects genuinely unknown keys at startup.

Presets are only used by the Wyoming endpoint, which is what `--voices-config` requires.

```nix
services.omnivoice = {
  enable = true;
  apiKeyFile = "/run/secrets/omnivoice-api-key";
  host = "0.0.0.0";
  openFirewall = true;

  wyoming = {
    enable = true;
    port = 10200;
    openFirewall = true;
  };

  defaultVoice = "alice";
  languages = [ "en" ];

  voices = {
    alice = {
      description = "Alice";
      reference = ./references/alice.wav;
      reference_text = "This is what was said by the reference voice.";
      language = "en";
      speed = 1.0;
      num_step = 32;
      guidance_scale = 2.0;
    };

    alice-fast = {
      description = "Alice (fast)";
      reference = ./references/alice.wav;
      reference_text = "This is what was said by the reference voice.";
      language = "en";
      speed = 1.2;
      num_step = 16;
    };
  };
};
```

Reference audio given as a Nix path is copied into the Nix store and is therefore world readable. Use `services.omnivoice.voicesConfigFile` with a catalog outside the store when that matters; relative reference paths inside it resolve against its directory.

Wyoming has no authentication, so `wyoming.openFirewall` exposes the model to everyone who can reach the port. Keep it on trusted networks and let Home Assistant connect from there.

Reference audio is decoded and clone prompts are prepared once during startup, so changing voices makes the next start slower but keeps request latency low. After changing presets, restart the service and reload the Home Assistant Wyoming integration.

## Service Details

- unit name: `omnivoice-server.service`, `Type=exec`, restarted on failure
- user and group: `omnivoice`, created unless the options are pointed at existing accounts
- state directory: `/var/lib/omnivoice-rs` (`StateDirectory=`, mode `0750`); a `dataDir` outside `/var/lib` is created with tmpfiles and added to `ReadWritePaths=`
- network: `network-online.target` is waited for, since the first start may download the model
- hardening: `ProtectSystem=strict`, `ProtectHome`, `ProtectProc=invisible`, `RestrictNamespaces`, `SystemCallFilter=@system-service`, `NoNewPrivileges`, `UMask=0077`, with `CAP_NET_BIND_SERVICE` only when a port below 1024 is configured
- extras: `services.omnivoice.extraArgs`, `services.omnivoice.environment`, and `services.omnivoice.logLevel` (`RUST_LOG`) cover anything the typed options do not

The module and the generated unit are evaluated by the `nixos-module` flake check:

```bash
nix build .#checks.x86_64-linux.nixos-module
```
