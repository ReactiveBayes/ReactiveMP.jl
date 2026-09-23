using TestItemRunner

# Same selection syntax as the root test/runtests.jl.

const TEST_ALL = get(ENV, "TEST_ALL", "false") == "true"

const SELECTED_PATHS = String[]
const SELECTED_NAMES = String[]
const SELECTED_TAGS = Symbol[]

for arg in ARGS
    if startswith(arg, "tag:")
        push!(SELECTED_TAGS, Symbol(arg[(length("tag:") + 1):end]))
    elseif startswith(arg, "name:")
        push!(SELECTED_NAMES, arg[(length("name:") + 1):end])
    else
        push!(SELECTED_PATHS, join(split(arg, ":"), "/"))
    end
end

function test_item_filter(ti)
    isempty(SELECTED_PATHS) ||
        any(p -> occursin(p, ti.filename), SELECTED_PATHS) ||
        return false
    isempty(SELECTED_NAMES) ||
        any(n -> occursin(n, ti.name), SELECTED_NAMES) ||
        return false
    isempty(SELECTED_TAGS) ||
        any(t -> t in ti.tags, SELECTED_TAGS) ||
        return false
    if !TEST_ALL && (:slow in ti.tags) && !(:slow in SELECTED_TAGS)
        return false
    end
    return true
end

@run_package_tests(filter = test_item_filter, verbose = true)
