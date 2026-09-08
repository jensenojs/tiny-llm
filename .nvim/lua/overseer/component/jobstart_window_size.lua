-- pytest 把 [ 80%] 写在当前终端宽度的行尾。PTY 若按整屏宽来，输出窗更窄
-- 时进度会被终端折到下一行。启动前把 PTY / COLUMNS 收成输出窗宽。

local function list_width()
    for _, win in ipairs(vim.api.nvim_tabpage_list_wins(0)) do
        local buf = vim.api.nvim_win_get_buf(win)
        if vim.bo[buf].filetype == "OverseerList" then
            return vim.api.nvim_win_get_width(win)
        end
    end
    return 0
end

local function output_win_size(bufnr)
    if bufnr and vim.api.nvim_buf_is_valid(bufnr) then
        for _, win in ipairs(vim.api.nvim_tabpage_list_wins(0)) do
            if vim.api.nvim_win_get_buf(win) == bufnr then
                return vim.api.nvim_win_get_width(win), vim.api.nvim_win_get_height(win)
            end
        end
    end
    for _, win in ipairs(vim.api.nvim_tabpage_list_wins(0)) do
        local buf = vim.api.nvim_win_get_buf(win)
        if vim.bo[buf].filetype == "OverseerOutput" or vim.b[buf].overseer_task then
            return vim.api.nvim_win_get_width(win), vim.api.nvim_win_get_height(win)
        end
    end
end

local function guess_size(bufnr)
    local width, height = output_win_size(bufnr)
    if width then
        return width, height
    end
    local cols = vim.o.columns
    local side = list_width()
    if side > 0 and side < cols * 0.5 then
        width = cols - side
    else
        -- list 还没并排（占满底栏或没开），输出会留出大约 1/5 给 list
        width = math.max(40, cols - math.max(side, math.floor(cols * 0.2)))
    end
    return width, math.max(8, math.floor(vim.o.lines / 5))
end

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

                local width, height = guess_size(task:get_bufnr())
                strat.opts.wrap_opts = vim.tbl_extend("force", strat.opts.wrap_opts or {}, {
                    width = width,
                    height = height,
                })
                -- pytest 启动时缓存 fullwidth；COLUMNS 比 ioctl 先被读到
                task.env = vim.tbl_extend("force", task.env or {}, {
                    COLUMNS = tostring(width),
                    LINES = tostring(height),
                })
            end,
            on_start = function(_, task)
                vim.schedule(function()
                    local strat = task.strategy
                    if type(strat) ~= "table" or not strat.job_id or strat.job_id <= 0 then
                        return
                    end
                    local width, height = output_win_size(task:get_bufnr())
                    if not width or width <= 0 or height <= 0 then
                        return
                    end
                    pcall(vim.fn.jobresize, strat.job_id, width, height)
                end)
            end,
        }
    end,
}
