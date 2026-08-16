{
  config,
  lib,
  pkgs,
  ...
}:

let
  cfg = config.services.omnivoice;

  inherit (lib)
    literalExpression
    mapAttrs
    mkIf
    mkOption
    mkPackageOption
    optionalAttrs
    optionals
    optionalString
    types
    ;

  defaultUser = "omnivoice";
  defaultGroup = "omnivoice";

  tomlFormat = pkgs.formats.toml { };

  # TOML distinguishes integers from floats and the server deserializes these
  # fields as `f32`, so `speed = 1` has to reach the file as `1.0`.
  float = types.coercedTo types.int (value: value * 1.0) types.float;

  mkNullOption =
    type: description:
    mkOption {
      type = types.nullOr type;
      default = null;
      inherit description;
    };

  voiceModule =
    { name, ... }:
    {
      freeformType = tomlFormat.type;

      options = {
        description = mkOption {
          type = types.str;
          default = name;
          defaultText = literalExpression "<name>";
          description = "Human readable description advertised to Wyoming clients.";
        };

        languages = mkNullOption (types.listOf types.str) ''
          Language tags advertised for this voice, for example Home Assistant's
          voice picker. Falls back to {option}`services.omnivoice.languages`, then
          to {option}`language`, then to `[ "en" ]`.
        '';

        language = mkNullOption types.str ''
          OmniVoice generation language used when synthesizing with this voice.
        '';

        reference = mkNullOption types.path ''
          Reference audio file used for voice cloning. Mutually exclusive with
          {option}`ref_audio`.

          The generated voice catalog lives in the Nix store, so relative paths
          are resolved against the store. Use an absolute path or a Nix path
          (which copies the audio into the store and makes it world readable).
        '';

        ref_audio = mkNullOption types.str ''
          Base64 `data:` URI holding the reference audio, as an alternative to
          {option}`reference`.
        '';

        reference_text = mkNullOption types.str ''
          Transcript of the reference audio. Requires reference audio; when it is
          omitted the server transcribes the reference with ASR instead.
        '';

        instruct = mkNullOption types.str ''
          Voice design instruction describing the speaker.
        '';

        asr_model = mkNullOption types.str ''
          ASR model used to transcribe reference audio when {option}`reference_text`
          is not given.
        '';

        seed = mkNullOption types.ints.unsigned "Sampling seed for this voice.";
        duration = mkNullOption float "Target duration in seconds.";
        speed = mkNullOption float "Speaking rate multiplier.";
        num_step = mkNullOption types.ints.positive "Number of stage1 diffusion steps.";
        guidance_scale = mkNullOption float "Classifier-free guidance scale.";
        t_shift = mkNullOption float "Timestep shift applied during stage1 decoding.";
        layer_penalty_factor = mkNullOption float "Stage0 layer penalty factor.";
        position_temperature = mkNullOption float "Stage0 position sampling temperature.";
        class_temperature = mkNullOption float "Stage0 class sampling temperature.";
        preprocess_prompt = mkNullOption types.bool "Preprocess the reference prompt.";
        postprocess_output = mkNullOption types.bool "Postprocess the generated audio.";
        denoise = mkNullOption types.bool "Denoise the generated audio.";
        audio_chunk_duration = mkNullOption float "Audio chunk duration in seconds.";
        audio_chunk_threshold = mkNullOption float "Audio chunk threshold in seconds.";

        response_format =
          mkNullOption
            (types.enum [
              "wav"
              "pcm"
              "mp3"
            ])
            "Response format, validated for parity with the OpenAI endpoint. Wyoming always transports PCM16.";

        stream_format = mkNullOption (types.enum [
          "audio"
          "sse"
        ]) "Stream format, validated for parity with the OpenAI endpoint. Wyoming always transports PCM16.";
      };
    };

  # The server rejects unknown TOML keys, so unset options must not be emitted.
  voiceToAttrs = lib.filterAttrs (_: value: value != null);

  voicesToml = {
    voice = mapAttrs (_: voice: [ (voiceToAttrs voice) ]) cfg.voices;
  }
  // optionalAttrs (cfg.defaultVoice != null) { default_voice = cfg.defaultVoice; }
  // optionalAttrs (cfg.languages != null) { inherit (cfg) languages; };

  voicesFile =
    if cfg.voicesConfigFile != null then
      cfg.voicesConfigFile
    else if cfg.voices != { } then
      tomlFormat.generate "omnivoice-voices.toml" voicesToml
    else
      null;

  cudaEnabled = cfg.cuda.enable;

  # `StateDirectory=` only manages directories directly below /var/lib.
  stateDirectory =
    let
      relative = lib.removePrefix "/var/lib/" cfg.dataDir;
    in
    if lib.hasPrefix "/var/lib/" cfg.dataDir && !lib.hasInfix "/" relative then relative else null;

  needsNetBindCapability = cfg.port < 1024 || (cfg.wyoming.enable && cfg.wyoming.port < 1024);

  serverArgs = [
    "--host"
    cfg.host
    "--port"
    (toString cfg.port)
    "--device"
    cfg.device
    "--dtype"
    cfg.dtype
    "--served-model-id"
    cfg.servedModelId
    "--max-body-mb"
    (toString cfg.maxBodyMb)
    "--max-concurrent-requests"
    (toString cfg.maxConcurrentRequests)
    "--mp3-bitrate-kbps"
    (toString cfg.mp3BitrateKbps)
    "--request-timeout-secs"
    (toString cfg.requestTimeout)
  ]
  # Without --model the server resolves the default Hugging Face repo and
  # downloads it into $HOME, which is cfg.dataDir.
  ++ optionals (cfg.model != null) [
    "--model"
    (toString cfg.model)
  ]
  ++ optionals (cfg.basePath != "") [
    "--base-path"
    cfg.basePath
  ]
  ++ optionals cfg.wyoming.enable (
    [
      "--wyoming-port"
      (toString cfg.wyoming.port)
    ]
    ++ optionals (cfg.wyoming.host != null) [
      "--wyoming-host"
      cfg.wyoming.host
    ]
    ++ optionals (voicesFile != null) [
      "--voices-config"
      "${voicesFile}"
    ]
  )
  ++ cfg.extraArgs;

  serverCommand = "${lib.getExe' cfg.package "omnivoice-server"} ${lib.escapeShellArgs serverArgs}";

  startScript = pkgs.writeShellScript "omnivoice-server-start" ''
    set -euo pipefail
    ${optionalString (cfg.apiKeyFile != null) ''
      OMNIVOICE_API_KEY="$(< "$CREDENTIALS_DIRECTORY/api-key")"
      export OMNIVOICE_API_KEY
    ''}
    exec ${serverCommand}
  '';
in
{
  options.services.omnivoice = {
    enable = lib.mkEnableOption "the OmniVoice text-to-speech server";

    package = mkPackageOption pkgs "omnivoice" {
      extraDescription = ''
        Use `pkgs.omnivoice-cuda` (or a package built with `cudaSupport = true`)
        for GPU inference; {option}`services.omnivoice.cuda.enable` follows the
        package's `cudaSupport` flag automatically.
      '';
    };

    user = mkOption {
      type = types.str;
      default = defaultUser;
      description = ''
        User the server runs as. A system user is created when this is left at
        its default. Model weights are large, so a persistent user is used
        instead of {option}`DynamicUser`.
      '';
    };

    group = mkOption {
      type = types.str;
      default = defaultGroup;
      description = "Group the server runs as. Created when left at its default.";
    };

    dataDir = mkOption {
      type = types.path;
      default = "/var/lib/omnivoice";
      description = ''
        State directory of the service. It is used as `$HOME`, which is where
        model weights are downloaded to (`$HOME/.cache/huggingface`) unless
        {option}`model` points somewhere else.
      '';
    };

    host = mkOption {
      type = types.str;
      default = "127.0.0.1";
      example = "0.0.0.0";
      description = "Address the OpenAI-compatible HTTP endpoint binds to.";
    };

    port = mkOption {
      type = types.port;
      default = 8000;
      description = "Port of the OpenAI-compatible HTTP endpoint.";
    };

    openFirewall = mkOption {
      type = types.bool;
      default = false;
      description = "Whether to open {option}`port` in the firewall.";
    };

    apiKeyFile = mkOption {
      type = types.nullOr types.path;
      default = null;
      example = "/run/secrets/omnivoice-api-key";
      description = ''
        File containing the bearer token required by the HTTP endpoint. It is
        passed to the service via systemd credentials and exported as
        `OMNIVOICE_API_KEY`, so it never enters the Nix store.

        The server refuses to start without an API key; alternatively provide
        `OMNIVOICE_API_KEY` through {option}`environmentFile`.
      '';
    };

    environmentFile = mkOption {
      type = types.nullOr types.path;
      default = null;
      example = "/run/secrets/omnivoice.env";
      description = ''
        `EnvironmentFile` passed to the unit, for secrets such as
        `OMNIVOICE_API_KEY` or `HF_TOKEN`.
      '';
    };

    model = mkOption {
      type = types.nullOr (types.either types.str types.path);
      default = null;
      example = "k2-fsa/OmniVoice";
      description = ''
        Model to serve, either a Hugging Face repository id or a local directory.

        When `null` (the default) the server resolves the default repository and
        downloads the weights into {option}`dataDir` on first start, which
        requires network access at startup.
      '';
    };

    device = mkOption {
      type = types.str;
      default = "auto";
      example = "cuda";
      description = "Device passed to `--device` (`auto`, `cpu`, `cuda`, `cuda:1`, `metal`).";
    };

    dtype = mkOption {
      type = types.str;
      default = "auto";
      example = "bf16";
      description = "Data type passed to `--dtype`.";
    };

    servedModelId = mkOption {
      type = types.str;
      default = "omnivoice-tts";
      description = "Model id reported by `GET /v1/models` and expected in requests.";
    };

    basePath = mkOption {
      type = types.str;
      default = "";
      example = "/tts";
      description = "Path prefix all HTTP routes are mounted under.";
    };

    maxBodyMb = mkOption {
      type = types.ints.positive;
      default = 50;
      description = "Maximum accepted request body size in mebibytes.";
    };

    maxConcurrentRequests = mkOption {
      type = types.ints.positive;
      default = 1;
      description = ''
        Number of synthesis requests served concurrently. Inference is GPU or CPU
        bound, so values above 1 mostly increase memory pressure.
      '';
    };

    mp3BitrateKbps = mkOption {
      type = types.ints.positive;
      default = 128;
      description = "Bitrate used when `response_format` is `mp3`.";
    };

    requestTimeout = mkOption {
      type = types.ints.positive;
      default = 300;
      description = "Per-request synthesis timeout in seconds.";
    };

    logLevel = mkOption {
      type = types.nullOr types.str;
      default = null;
      example = "omnivoice_server=debug,tower_http=info";
      description = "Value of `RUST_LOG`. Leave `null` to keep the server's default filter.";
    };

    environment = mkOption {
      type = types.attrsOf types.str;
      default = { };
      example = literalExpression ''{ HF_ENDPOINT = "https://hf-mirror.com"; }'';
      description = "Extra environment variables for the service.";
    };

    extraArgs = mkOption {
      type = types.listOf types.str;
      default = [ ];
      description = "Extra command line arguments appended to `omnivoice-server`.";
    };

    cuda = {
      enable = mkOption {
        type = types.bool;
        default = cfg.package.cudaSupport or false;
        defaultText = literalExpression "config.services.omnivoice.package.cudaSupport or false";
        description = ''
          Whether the service runs CUDA inference. Detected from the package by
          default. Enabling it sets `CUDA_DEVICE_ORDER=PCI_BUS_ID` and relaxes the
          sandbox options that block GPU access (device access, W^X memory).
        '';
      };

      visibleDevices = mkOption {
        type = types.nullOr (types.coercedTo types.str lib.singleton (types.listOf types.str));
        default = null;
        example = [
          "0"
          "1"
        ];
        description = ''
          Value of `CUDA_VISIBLE_DEVICES`, given as a list of device indices or
          UUIDs. Because `CUDA_DEVICE_ORDER=PCI_BUS_ID` is set alongside it,
          indices refer to PCI bus order rather than the driver's speed ranking.
        '';
      };
    };

    wyoming = {
      enable = lib.mkEnableOption ''
        the native Wyoming protocol endpoint next to the HTTP endpoint, for use
        with Home Assistant. Both endpoints share the loaded model, the request
        limit and the inference lock'';

      host = mkOption {
        type = types.nullOr types.str;
        default = null;
        example = "0.0.0.0";
        description = "Address the Wyoming endpoint binds to. Defaults to {option}`services.omnivoice.host`.";
      };

      port = mkOption {
        type = types.port;
        default = 10200;
        description = "Port of the Wyoming endpoint.";
      };

      openFirewall = mkOption {
        type = types.bool;
        default = false;
        description = ''
          Whether to open {option}`wyoming.port` in the firewall. The Wyoming
          protocol has no authentication, so only expose it to trusted networks.
        '';
      };
    };

    defaultVoice = mkOption {
      type = types.nullOr types.str;
      default = null;
      example = "alice";
      description = ''
        Voice used when a client does not request one. Defaults to the first
        voice in {option}`voices` (alphabetically).
      '';
    };

    languages = mkOption {
      type = types.nullOr (types.listOf types.str);
      default = null;
      example = [ "en" ];
      description = "Default list of advertised language tags for voices that do not set their own.";
    };

    voices = mkOption {
      type = types.attrsOf (types.submodule voiceModule);
      default = { };
      example = literalExpression ''
        {
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
        }
      '';
      description = ''
        Voice presets advertised over Wyoming, rendered into the `--voices-config`
        TOML catalog. Each attribute becomes one `[[voice.<name>]]` table, and
        unknown attributes are passed through to the file verbatim.

        Home Assistant transports only voice, language and speaker selections, so
        define one preset per parameter combination that should be selectable.
        Reference audio is decoded at startup and clone prompts are prepared once,
        which makes startup slower but every request faster.

        Requires {option}`services.omnivoice.wyoming.enable`; with no voices
        configured the server advertises a single automatic `default` voice.
      '';
    };

    voicesConfigFile = mkOption {
      type = types.nullOr types.path;
      default = null;
      example = "/var/lib/omnivoice/voices.toml";
      description = ''
        Use this voice catalog instead of generating one from {option}`voices`,
        for example to keep reference audio out of the world-readable Nix store.
        Relative reference paths inside it are resolved against its directory.
      '';
    };
  };

  config = mkIf cfg.enable {
    assertions = [
      {
        assertion = cfg.apiKeyFile != null || cfg.environmentFile != null;
        message = ''
          services.omnivoice: the server refuses to start without an API key.
          Set services.omnivoice.apiKeyFile, or provide OMNIVOICE_API_KEY through
          services.omnivoice.environmentFile.
        '';
      }
      {
        assertion = (cfg.voices == { } && cfg.voicesConfigFile == null) || cfg.wyoming.enable;
        message = ''
          services.omnivoice: voice presets are only used by the Wyoming endpoint
          (`--voices-config` requires `--wyoming-port`). Set
          services.omnivoice.wyoming.enable = true.
        '';
      }
      {
        assertion = cfg.voicesConfigFile == null || cfg.voices == { };
        message = ''
          services.omnivoice: voicesConfigFile and voices are mutually exclusive.
        '';
      }
      {
        assertion = cfg.defaultVoice == null || cfg.voices ? ${cfg.defaultVoice};
        message = ''
          services.omnivoice.defaultVoice: `${toString cfg.defaultVoice}` is not defined in
          services.omnivoice.voices.
        '';
      }
      {
        assertion =
          !cfg.wyoming.enable
          || cfg.port != cfg.wyoming.port
          || (cfg.wyoming.host != null && cfg.wyoming.host != cfg.host);
        message = ''
          services.omnivoice: the HTTP and Wyoming endpoints cannot share
          ${cfg.host}:${toString cfg.port}.
        '';
      }
    ];

    warnings =
      optionals (cfg.wyoming.enable && cfg.wyoming.openFirewall) [
        ''
          services.omnivoice.wyoming.openFirewall is enabled. The Wyoming protocol
          has no authentication, so anyone who can reach port
          ${toString cfg.wyoming.port} can use the model.
        ''
      ]
      ++
        optionals
          (cfg.openFirewall && (cfg.host == "127.0.0.1" || cfg.host == "localhost" || cfg.host == "::1"))
          [
            ''
              services.omnivoice.openFirewall is enabled but the server only listens on
              ${cfg.host}. Set services.omnivoice.host to a reachable address.
            ''
          ];

    users.users = optionalAttrs (cfg.user == defaultUser) {
      ${defaultUser} = {
        description = "OmniVoice text-to-speech server";
        isSystemUser = true;
        inherit (cfg) group;
        home = cfg.dataDir;
      };
    };

    users.groups = optionalAttrs (cfg.group == defaultGroup) {
      ${defaultGroup} = { };
    };

    networking.firewall.allowedTCPPorts =
      optionals cfg.openFirewall [ cfg.port ]
      ++ optionals (cfg.wyoming.enable && cfg.wyoming.openFirewall) [ cfg.wyoming.port ];

    systemd.tmpfiles.settings = mkIf (stateDirectory == null) {
      "10-omnivoice".${cfg.dataDir}.d = {
        inherit (cfg) user group;
        mode = "0750";
      };
    };

    systemd.services.omnivoice-server = {
      description = "OmniVoice text-to-speech server";
      documentation = [ "https://github.com/FerrisMind/omnivoice-rs" ];
      wantedBy = [ "multi-user.target" ];
      # The model is downloaded on first start when `model` is unset.
      wants = [ "network-online.target" ];
      after = [ "network-online.target" ];

      environment = {
        HOME = cfg.dataDir;
      }
      // optionalAttrs cudaEnabled { CUDA_DEVICE_ORDER = "PCI_BUS_ID"; }
      // optionalAttrs (cfg.cuda.visibleDevices != null) {
        CUDA_VISIBLE_DEVICES = lib.concatStringsSep "," cfg.cuda.visibleDevices;
      }
      // optionalAttrs (cfg.logLevel != null) { RUST_LOG = cfg.logLevel; }
      // cfg.environment;

      serviceConfig = {
        Type = "exec";
        ExecStart = if cfg.apiKeyFile != null then "${startScript}" else serverCommand;
        User = cfg.user;
        Group = cfg.group;
        WorkingDirectory = cfg.dataDir;
        Restart = "on-failure";
        RestartSec = 5;

        LoadCredential = optionals (cfg.apiKeyFile != null) [ "api-key:${cfg.apiKeyFile}" ];
        EnvironmentFile = optionals (cfg.environmentFile != null) [ cfg.environmentFile ];

        StateDirectory = mkIf (stateDirectory != null) stateDirectory;
        StateDirectoryMode = mkIf (stateDirectory != null) "0750";
        ReadWritePaths = optionals (stateDirectory == null) [ cfg.dataDir ];

        AmbientCapabilities = optionals needsNetBindCapability [ "CAP_NET_BIND_SERVICE" ];
        CapabilityBoundingSet = if needsNetBindCapability then [ "CAP_NET_BIND_SERVICE" ] else [ "" ];

        # Hardening. CUDA needs the GPU device nodes, /proc/driver/nvidia and
        # writable-executable memory for its JIT, so those knobs are relaxed when
        # the package was built with CUDA support.
        DevicePolicy = if cudaEnabled then "auto" else "closed";
        LockPersonality = true;
        MemoryDenyWriteExecute = !cudaEnabled;
        NoNewPrivileges = true;
        PrivateDevices = !cudaEnabled;
        PrivateMounts = true;
        PrivateTmp = true;
        ProcSubset = if cudaEnabled then "all" else "pid";
        ProtectClock = true;
        ProtectControlGroups = true;
        ProtectHome = true;
        ProtectHostname = true;
        ProtectKernelLogs = true;
        ProtectKernelModules = !cudaEnabled;
        ProtectKernelTunables = true;
        ProtectProc = "invisible";
        ProtectSystem = "strict";
        RemoveIPC = true;
        RestrictAddressFamilies = [
          "AF_INET"
          "AF_INET6"
          "AF_UNIX"
        ];
        RestrictNamespaces = true;
        RestrictRealtime = true;
        RestrictSUIDSGID = true;
        SystemCallArchitectures = "native";
        SystemCallFilter = [ "@system-service" ];
        UMask = "0077";
      };
    };
  };
}
