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
      method: "notify",
      params: {
        html: "Update complete! Launcher pulled and Python dependencies refreshed."
      }
    }
  ]
}
