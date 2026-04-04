module.exports = {
  run: [
    {
      method: "notify",
      params: {
        html: "Installing VidLingo — YouTube → transcribe → translate..."
      }
    },
    {
      method: "shell.run",
      params: {
        path: ".",
        message: "git lfs install"
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install -r requirements.txt"
        ]
      }
    },
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          path: "app"
        }
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
