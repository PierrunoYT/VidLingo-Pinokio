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
          path: "."
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
