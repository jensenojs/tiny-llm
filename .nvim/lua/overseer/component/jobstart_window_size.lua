-- jobstart 默认用 vim.o.columns 当 PTY 宽。输出窗更窄时，pytest 把进度写在
-- 行尾，终端再按窗宽折行。启动前把 PTY 收成 OverseerOutput 的实际尺寸。

---@type overseer.ComponentFileDefinition
return {
    desc = "把 jobstart PTY 设成 OverseerOutput 窗口大小",
    editable = false,
    serializable = false,
    constructor = function()
        return {
            on_pre_start = function(_, task)
                local strat = task.strategy
                if type(strat) ~= "table" or type(strat.opts) ~= "table" then
                    return
                end
                if strat.opts.use_terminal == false then
                    return
                end

                require("overseer").open({ enter = false, direction = "bottom" })

                local width, height
                for _, win in ipairs(vim.api.nvim_tabpage_list_wins(0)) do
                    local buf = vim.api.nvim_win_get_buf(win)
                    if vim.bo[buf].filetype == "OverseerOutput" or vim.b[buf].overseer_task then
                        width = vim.api.nvim_win_get_width(win)
                        height = vim.api.nvim_win_get_height(win)
                        break
                    end
                end
                if not width then
                    return
                end

                strat.opts.wrap_opts = vim.tbl_extend("force", strat.opts.wrap_opts or {}, {
                    width = width,
                    height = height,
                })
            end,
        }
    end,
}
