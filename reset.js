module.exports = {
  run: [
    {
      method: "fs.rm",
      params: {
        path: "env"
      }
    },
    {
      method: "fs.rm",
      params: {
        path: "app/downloads"
      }
    },
    {
      method: "notify",
      params: {
        html: "Reset complete! The <code>env</code> and <code>app/downloads</code> folders were removed. Click Install to reinstall."
      }
    }
  ]
}
