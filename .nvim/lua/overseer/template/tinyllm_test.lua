-- ============================================================================
-- tinyllm 测试 (pdm run test, 经 scripts/dev-tools.py 包装的 pytest)
-- ============================================================================
-- 用法对应 AGENTS.md 的 Canonical Commands:
--   pdm run test                              全量
--   pdm run test --week 1 --day 3             定位章节 (自动 copy-test)
--   pdm run test --week 1 --day 3 -- -k task_2  pytest 过滤
-- 约束: dev-tools.py 要求 week/day 必须成对提供
-- ============================================================================

---@type overseer.TemplateFileDefinition
return {
    name = "tinyllm 测试",
    desc = "pdm run test (week/day 定位章节, -k 过滤用例)",
    condition = {
        dir = vim.fn.getcwd(),
    },
    params = {
        week = {
            type = "integer",
            desc = "周数 (1-4)",
            choices = { "1", "2", "3", "4" },
        },
        day = {
            type = "integer",
            desc = "天数 (必须与 week 成对)",
            choices = { "1", "2", "3", "4", "5", "6", "7", "8", "9" },
        },
        filter = {
            type = "string",
            desc = "pytest -k 过滤表达式 (如 task_2, gpu)",
            optional = true,
            choices = { "task_1", "task_2", "task_3", "gpu" },
        },
    },
    builder = function(params)
        -- dev-tools.py 的 validate_week_day: week 与 day 必须同时提供
        if (params.week == nil) ~= (params.day == nil) then
            return {
                name = "tinyllm 测试 (参数错误)",
                cmd = { "echo", "week 和 day 必须同时提供" },
                components = {},
            }
        end

        local args = { "run", "test" }
        local name = "tinyllm 测试"
        if params.week then
            table.insert(args, "--week")
            table.insert(args, tostring(params.week))
            table.insert(args, "--day")
            table.insert(args, tostring(params.day))
            name = name .. (" W%sD%s"):format(params.week, params.day)
        end
        if params.filter then
            -- dev-tools.py 的 remainder 参数: pdm run test ... -- -k <filter>
            table.insert(args, "--")
            table.insert(args, "-k")
            table.insert(args, params.filter)
            name = name .. " -k " .. params.filter
        end

        return {
            name = name,
            cmd = "pdm",
            args = args,
            -- pytest 要 TTY 才有颜色。PTY 宽度由 jobstart_window_size 收成输出窗宽，
            -- 避免进度被终端二次折行。
            strategy = { "jobstart", use_terminal = true },
            components = {
                "jobstart_window_size",
                "default",
                { "open_output", on_complete = "failure" },
            },
        }
    end,
}
