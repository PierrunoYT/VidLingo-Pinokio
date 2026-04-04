module.exports = {
  run: [
    {
      method: "shell.run",
      params: {
        message: "git pull"
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: ".",
        message: "uv pip install -r app/requirements.txt"
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
        html: "Update complete! Launcher pulled and Python dependencies refreshed."
      }
    }
  ]
}
