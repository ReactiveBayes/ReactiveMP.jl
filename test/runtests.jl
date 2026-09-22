using Aqua, TestItemRunner, ReactiveMP

if get(ENV, "RUN_AQUA", "true") == "true"
    Aqua.test_all(
        ReactiveMP;
        # `ambiguities` stays off for now: 322 pairs on `main`, and 253 of them come from
        # `src/helpers/algebra/*`, which the split moves out with Flow/AR anyway. Enabling it
        # is tracked separately in PHASES.md § Phase 2 -- it is a budgeted cleanup, not a flag.
        ambiguities = false,
        # Two deliberate pirate methods, both documented where they are defined:
        # `default_prod_rule`/`prod` for `Uniform`x`Beta` (nodes/predefined/uniform.jl:6,9), a
        # mathematical special case that arguably belongs in ExponentialFamily, and a `dot`
        # overload for `ForwardDiff.Dual` (fixes.jl:12), an upstream hotfix that leaves when
        # `src/fixes.jl` does. Measured: with these two owners declared, zero pirates remain.
        piracies = (;
            treat_as_own = [
                ReactiveMP.Distributions.Uniform, ReactiveMP.ForwardDiff.Dual
            ]
        ),
        deps_compat = (; check_extras = true, check_weakdeps = true),
    )
end

# Test selection.
#
# `make test test_args="..."` passes each entry through as an element of `ARGS`. Three
# prefixes, and they compose:
#
#   rules:beta:out       path   -- maps to `test/rules/beta/out`, matched with `occursin`,
#                                  so prefixes work. This is the original syntax
#   tag:rules            tag    -- only items carrying that tag
#   name:NormalMixture   name   -- only items whose name contains that text
#
# Entries of the same kind are OR'ed together; different kinds are AND'ed. So
# `test_args="tag:rules name:Beta"` runs the rule tests whose names mention Beta.
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
const SELECTED_TAGS  = Symbol[]

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
    # Skipping `:slow` must never be how CI loses coverage, so it is opt-out, not opt-in.
    if !TEST_ALL && (:slow in ti.tags) && !(:slow in SELECTED_TAGS)
        return false
    end
    return true
end

@run_package_tests(filter = test_item_filter, verbose = true)
