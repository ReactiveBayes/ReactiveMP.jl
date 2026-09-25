using Aqua, TestItemRunner, ReactiveMP

if get(ENV, "RUN_AQUA", "true") == "true"
    Aqua.test_all(
        ReactiveMP;
        # `ambiguities` is on: 322 pairs on `main`, most in the v6 algebra helpers and nodes that
        # left for their packages; each lib suite checks its own.
        deps_compat = (; check_extras = true, check_weakdeps = true),
    )
end

# Test selection.
#
# `make test test_args="..."` passes each entry through as an element of `ARGS`. Three
# prefixes, and they compose:
#
#   engine:fixtures      path   -- maps to `test/engine/fixtures`, matched with `occursin`,
#                                  so prefixes work. This is the original syntax
#   tag:engine           tag    -- only items carrying that tag
#   name:MessageMapping  name   -- only items whose name contains that text
#
# Entries of the same kind are OR'ed together; different kinds are AND'ed. So
# `test_args="tag:engine name:MessageMapping"` runs the engine tests whose names mention MessageMapping.
#
# TestItemRunner's filter already receives `(filename, name, tags)` -- this file previously
# chose to look at `filename` alone, which is why name and tag selection appeared to need a
# different runner. It did not.
#
# `:slow` items are skipped by default, so `make test` is the fast local subset. CI sets
# TEST_ALL=true and therefore runs everything; `make test-all` does the same locally. A
# filter that names `tag:slow` explicitly also runs them.

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

# `lib/` holds packages with suites of their own; `compat/` runs against v6. TestItemRunner
# scans the whole directory, so each is excluded here.
const EXCLUDED_DIRS = ("lib", "compat")

is_excluded(filename) = any(dir -> occursin(joinpath(pkgdir(ReactiveMP), dir) * "/", filename), EXCLUDED_DIRS)

function test_item_filter(ti)
    is_excluded(ti.filename) && return false
    isempty(SELECTED_PATHS) ||
        any(p -> occursin(p, ti.filename), SELECTED_PATHS) ||
        return false
    isempty(SELECTED_NAMES) ||
        any(n -> occursin(n, ti.name), SELECTED_NAMES) ||
        return false
    isempty(SELECTED_TAGS) ||
        any(t -> t in ti.tags, SELECTED_TAGS) ||
        return false
    # Skipping `:slow` must never be how CI loses coverage, so it is opt-out, not opt-in.
    if !TEST_ALL && (:slow in ti.tags) && !(:slow in SELECTED_TAGS)
        return false
    end
    return true
end

@run_package_tests(filter = test_item_filter, verbose = true)
