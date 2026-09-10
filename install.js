module.exports = {
  requires: {
    bundle: "ai"
  },
  run: [
    // Intel macOS is unsupported: OmniVoice requires torch>=2.4 and PyTorch
    // ships no x86-64 macOS wheels after 2.2.2. Stop here rather than build an
    // environment that installs cleanly and then fails at TTS.
    {
      when: "{{platform === 'darwin' && arch !== 'arm64'}}",
      method: "notify",
      params: {
        html: "VidLingo does not support Intel macOS. OmniVoice TTS requires PyTorch 2.4 or newer, and PyTorch publishes no Intel-Mac builds past 2.2.2. Supported platforms: Apple Silicon macOS, Windows, and Linux. Installation stopped."
      },
      next: null
    },
    {
      method: "notify",
      params: {
        html: "Installing VidLingo — YouTube → transcribe → translate..."
      }
    },
    // Platform Torch first. `accelerate` requires torch>=2.0, so installing
    // requirements first pulls a generic Torch that torch.js then force-
    // reinstalls — a redundant multi-gigabyte download and a transiently
    // wrong environment. Installed first, it already satisfies that
    // constraint and the requirements install leaves it alone.
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          path: ".",
          xformers: false,
          flashattention: false,
          triton: false
        }
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: ".",
        // Install the resolved lock, not the loose spec, so two machines
        // installing on different days get the same packages. Regenerate with
        // `python tools/relock.py` after editing app/requirements.txt.
        // The torch constraint is a backstop: torch.js has already installed
        // the platform build, which satisfies accelerate's torch>=2.0, so
        // nothing replaces it. Without the constraint, an environment missing
        // Torch would silently resolve whatever generic build is newest.
        message: [
          "uv pip install -r app/requirements.lock.txt -c app/torch-constraint.txt"
        ]
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: ".",
        message: "uv pip install omnivoice==0.2.1 --no-deps"
      }
    },
    {
      method: "notify",
      params: {
        html: "Installed. Accept HF licenses for Cohere Transcribe and TranslateGemma; models download on first use."
      }
    }
  ]
}
