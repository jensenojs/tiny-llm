-- ============================================================================
-- tinyllm 运行推理 (pdm run main → python main.py)
-- ============================================================================
-- 对应 main.py 的常用参数:
--   --loader week1|week2|week3   各周实现的加载器 (默认 week1)
--   --prompt "..."               提示词 (main.py 内置默认)
--   --model qwen3-0.6b           模型名 (model_names.py)
-- ============================================================================

---@type overseer.TemplateFileDefinition
return {
    name = "tinyllm 运行推理",
    desc = "pdm run main (本地 MLX 推理)",
    condition = {
        dir = vim.fn.getcwd(),
    },
    params = {
        loader = {
            type = "enum",
            choices = { "week1", "week2", "week3" },
            default = "week1",
            desc = "各周实现的加载器",
        },
        prompt = {
            type = "string",
            desc = "提示词 (留空用 main.py 内置默认)",
            optional = true,
        },
        model = {
            type = "string",
            desc = "模型名 (默认 qwen3-0.6b, 见 model_names.py)",
            optional = true,
        },
    },
    builder = function(params)
        local args = { "run", "main", "--loader", params.loader }
        local name = "tinyllm 推理 (" .. params.loader
        if params.model then
            table.insert(args, "--model")
            table.insert(args, params.model)
            name = name .. ", " .. params.model
        end
        if params.prompt then
            table.insert(args, "--prompt")
            table.insert(args, params.prompt)
            name = name .. ")"
        else
            name = name .. ")"
        end

        return {
            name = name,
            cmd = "pdm",
            args = args,
            components = { "default" },
        }
    end,
}
