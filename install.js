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
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: ".",
        message: [
          "uv pip install -r app/requirements.txt"
        ]
      }
    },
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
        message: "uv pip install omnivoice --no-deps"
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
