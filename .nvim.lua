-- ============================================================================
-- tiny-llm 项目本地配置
-- ============================================================================
-- 职责: 把 .nvim 加进 runtimepath, 让 overseer 扫描到项目模板
--   .nvim/lua/overseer/template/*.lua → overseer 默认搜索 "overseer/template"
-- 项目: Python (PDM) + MLX, Apple Silicon 本地运行, 无容器
-- ============================================================================

vim.opt.runtimepath:append(vim.fn.getcwd() .. "/.nvim")

-- 注释里的调用链是相对 site-packages 写的 (transformers/..., mlx_lm/..., jinja2/...),
-- 让 gf / gF 能沿 path 找到它们. 用本文件位置推仓根, 不依赖 cwd.
local _root = vim.fn.fnamemodify(debug.getinfo(1, "S").source:sub(2), ":p:h")
vim.opt.path:append(_root .. "/.venv/lib/python3.10/site-packages")
vim.opt.isfname:append(".")
-- "#" 默认在 isfname 里, 注释行首的 # 会先被 gF 当成文件名; 去掉它,
-- 光标停在行首也能直接跳到路径.
vim.opt.isfname:remove("#")

-- gd: 光标落在 "path:line" 上一律先当跳转目标, 找不到再退回 LSP 定义.
vim.keymap.set("n", "gd", function()
  local cf = vim.fn.expand("<cfile>")
  local name = cf:match("^(.-):%d+$") or cf
  if name ~= "" and vim.fn.findfile(name, vim.o.path) ~= "" then
    vim.cmd("normal! gF")
  else
    vim.lsp.buf.definition()
  end
end, { desc = "gd: path:line 或 LSP definition" })

vim.notify("tiny-llm 项目配置已加载 (overseer: tinyllm 测试 / 构建扩展 / 运行推理)", vim.log.levels.INFO)
