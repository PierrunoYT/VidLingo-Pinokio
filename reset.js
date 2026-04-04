module.exports = {
  run: [
    {
      method: "fs.rm",
      params: {
        path: "env"
      }
    },
    {
      method: "notify",
      params: {
        html: "Reset complete! The <code>env</code> folder was removed. Click Install to reinstall."
      }
    }
  ]
}
