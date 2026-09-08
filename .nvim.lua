-- ============================================================================
-- tiny-llm 项目本地配置
-- ============================================================================
-- 职责: 把 .nvim 加进 runtimepath, 让 overseer 扫描到项目模板
--   .nvim/lua/overseer/template/*.lua → overseer 默认搜索 "overseer/template"
-- 项目: Python (PDM) + MLX, Apple Silicon 本地运行, 无容器
-- ============================================================================

vim.opt.runtimepath:append(vim.fn.getcwd() .. "/.nvim")

vim.notify("tiny-llm 项目配置已加载 (overseer: tinyllm 测试 / 构建扩展 / 运行推理)", vim.log.levels.INFO)
