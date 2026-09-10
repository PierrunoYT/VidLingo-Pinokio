module.exports = {
  daemon: true,
  run: [
    // Intel macOS is unsupported — see install.js. Refuse here too, so an
    // environment installed before that gate existed fails clearly at launch
    // instead of part-way through the pipeline.
    {
      when: "{{platform === 'darwin' && arch !== 'arm64'}}",
      method: "notify",
      params: {
        html: "VidLingo does not support Intel macOS. OmniVoice TTS requires PyTorch 2.4 or newer, and PyTorch publishes no Intel-Mac builds past 2.2.2. Supported platforms: Apple Silicon macOS, Windows, and Linux."
      },
      next: null
    },
    {
      method: "notify",
      params: {
        html: "Starting VidLingo..."
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        env: {
          GRADIO_SERVER_NAME: "127.0.0.1",
          GRADIO_SERVER_PORT: "{{port}}",
          HF_HUB_ENABLE_HF_TRANSFER: "1",
          HF_HUB_DOWNLOAD_TIMEOUT: "300",
          PYTHONUTF8: "1"
        },
        path: ".",
        message: [
          "python app/app.py"
        ],
        on: [{
          event: "/(http:\\/\\/[0-9.:]+)/",
          done: true
        }]
      }
    },
    {
      method: "local.set",
      params: {
        url: "{{input.event[1]}}"
      }
    },
    {
      method: "notify",
      params: {
        html: "VidLingo is running."
      }
    }
  ]
}
