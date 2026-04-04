module.exports = {
  daemon: true,
  run: [
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
        path: "app",
        message: [
          "python app.py"
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
