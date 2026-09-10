module.exports = {
  requires: {
    bundle: "ai"
  },
  run: [
    // Intel macOS is unsupported — see install.js.
    {
      when: "{{platform === 'darwin' && arch !== 'arm64'}}",
      method: "notify",
      params: {
        html: "VidLingo does not support Intel macOS. OmniVoice TTS requires PyTorch 2.4 or newer, and PyTorch publishes no Intel-Mac builds past 2.2.2. Supported platforms: Apple Silicon macOS, Windows, and Linux. Update stopped."
      },
      next: null
    },
    {
      method: "shell.run",
      params: {
        message: "git pull"
      }
    },
    // Platform Torch first — see install.js.
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
        // Resolved lock, not the loose spec — see install.js.
        message: "uv pip install -r app/requirements.lock.txt -c app/torch-constraint.txt"
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
        html: "Update complete! Launcher pulled and Python dependencies refreshed."
      }
    }
  ]
}
