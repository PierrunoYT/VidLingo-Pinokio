module.exports = {
  run: [
    {
      method: "fs.link",
      params: {
        venv: "env",
        path: "."
      }
    },
    {
      method: "notify",
      params: {
        html: "Deduplication complete! Redundant library files were linked to save disk space."
      }
    }
  ]
}
