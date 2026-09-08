-- ============================================================================
-- tinyllm 构建扩展 (nanobind C++/Metal 扩展)
-- ============================================================================
-- 对应 AGENTS.md 的 Extension Rebuild Rule:
--   改了 src/extensions/src/* 后, 测试前先 build-ext
-- 三个变体都是 pyproject.toml [tool.pdm.scripts] 里的真实脚本:
--   build-ext      学生实现 (src/extensions)
--   build-ext-ref  参考实现 (src/extensions_ref)
--   build-ext-test 构建并运行扩展自测 (src/extensions/test.py)
-- ============================================================================

---@type overseer.TemplateFileDefinition
return {
    name = "tinyllm 构建扩展",
    desc = "pdm run build-ext (nanobind C++/Metal 扩展)",
    condition = {
        dir = vim.fn.getcwd(),
    },
    params = {
        variant = {
            type = "enum",
            choices = { "build-ext", "build-ext-ref", "build-ext-test" },
            default = "build-ext",
            desc = "学生扩展 / 参考实现 / 构建并自测",
        },
    },
    builder = function(params)
        return {
            name = "tinyllm 构建扩展: " .. params.variant,
            cmd = "pdm",
            args = { "run", params.variant },
            components = { "default" },
        }
    end,
}
