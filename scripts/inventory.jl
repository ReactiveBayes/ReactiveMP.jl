# Disposition inventory generator and checker.
#
#   julia --project=compat/v6-comparison scripts/inventory.jl --generate
#   julia --project=compat/v6-comparison scripts/inventory.jl --check
#
# `--generate` enumerates every node, exported symbol, engine hook, extension and
# rule-level exception in ReactiveMP and writes INVENTORY.md. It PRESERVES the
# `destination` and `note` cells already filled in, so it is re-runnable: adding a node
# upstream adds a row with destination `undecided` and leaves every decided row alone.
#
# `--check` fails when a live entity is missing from INVENTORY.md, when a row still says
# `undecided`, when a row refers to something that no longer exists, or when an exported
# deletion has no migration note. Phase P is complete exactly when `--check` exits 0.
#
# Run from the repository root, in the v6 comparison environment: the inventory is the
# record of where everything in ReactiveMP 6.5.0 went, and only v6.5.0 itself still has all
# of it.

using ReactiveMP

const ROOT = normpath(joinpath(@__DIR__, ".."))
const INVENTORY = joinpath(ROOT, "INVENTORY.md")
const SOURCE = pkgdir(ReactiveMP)

const DESTINATIONS = [
    "base",             # MessagePassingRulesBase
    "standard",         # StandardMessagePassingRules: distributions, arithmetic, logic, mixtures
    "approximations",   # MessagePassingRulesApproximations
    "testutils",        # MessagePassingRulesTestUtils
    "engine",           # ReactiveMP itself
    "delete",           # deliberate deletion
    "undecided",        # not yet assigned -- `--check` fails on these
]

# `node:<Name>` is also valid: a node spun out into its own package.
isvaliddestination(d) = d in DESTINATIONS || startswith(d, "node:")

# ---------------------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------------------

relpath_of(p) =
    replace(string(p), SOURCE * "/" => "", ROOT * "/" => "", r"^.*ReactiveMP\.jl/" => "")

srcfiles() = sort!(
    [
        joinpath(r, f) for (r, _, fs) in walkdir(joinpath(SOURCE, "src")) for
            f in fs if endswith(f, ".jl")
    ]
)

"""
Map each exported name to the file whose `export` statement introduced it.
Handles multi-line `export` blocks, which `src/nodes/nodes.jl` and others use.
"""
function export_sites()
    sites = Dict{Symbol, String}()
    for file in srcfiles()
        lines = readlines(file)
        i = 1
        while i <= length(lines)
            line = strip(lines[i])
            if startswith(line, "export ")
                buf = line[(length("export ") + 1):end]
                # Consume continuation lines while the statement is unterminated.
                while endswith(rstrip(buf), ",") && i < length(lines)
                    i += 1
                    buf *= " " * strip(lines[i])
                end
                for tok in split(buf, ",")
                    name = strip(tok)
                    isempty(name) && continue
                    sym = Symbol(name)
                    # First export wins; later ones are the redundant re-exports.
                    get!(sites, sym, relpath_of(file))
                end
            end
            i += 1
        end
    end
    return sites
end

function exported_symbols()
    sites = export_sites()
    rows = NamedTuple[]
    for n in names(ReactiveMP)
        n === :ReactiveMP && continue
        isdefined(ReactiveMP, n) || continue
        v = getfield(ReactiveMP, n)
        kind = if startswith(string(n), "@")
            "macro"
        elseif v isa Type
            "type"
        elseif v isa Function
            "function"
        else
            "const"
        end
        push!(
            rows,
            (
                symbol = string(n),
                kind = kind,
                file = get(sites, n, "(re-export)"),
            ),
        )
    end
    return sort(rows; by = r -> r.symbol)
end

"""
Authoritative node list, taken from the registry rather than from source, so that nodes
declared by hand (`Mixture`, `NormalMixture`, `GammaMixture`, `DiscreteTransition`) are
included alongside the `@node`-generated ones.
"""
function nodes()
    rows = NamedTuple[]
    for m in methods(ReactiveMP.is_predefined_node)
        sig = Base.unwrap_unionall(m.sig)
        length(sig.parameters) == 2 || continue
        T = sig.parameters[2]
        T === Any && continue   # the generic `is_predefined_node(some)` fallback
        name = nodename(T)
        isempty(name) && continue
        push!(rows, (symbol = name, kind = "node", file = locate_node(name)))
    end
    return sort(unique(rows); by = r -> r.symbol)
end

"""
Recover a readable node name from the second parameter of an `is_predefined_node`
signature. Three shapes occur: `Type{X}`, `Type{T} where T <: X` (which is a `UnionAll`,
so the whole signature cannot simply be unwrapped), and `typeof(f)` for the arithmetic
nodes.
"""
function nodename(T)
    if T isa UnionAll
        body = T.body
        if body isa DataType && body.name === Type.body.name
            return nodename(body)
        end
        return string(T)
    elseif T isa DataType && T <: Type && !isempty(T.parameters)
        p = T.parameters[1]
        p isa TypeVar && (p = p.ub)
        return string(p)
    else
        # `typeof(+)`, `typeof(dot)`, ...
        return replace(string(T), "typeof(" => "", ")" => "")
    end
end

"""
Find the file that declares a node. `@node`-generated methods report the macro's own
source location (`nodes.jl`), so the registry cannot answer this and the source must be
searched.
"""
# Nodes declared under a name that differs from the runtime type they register.
# `ExponentialFamily.GammaInverse` is a `const` alias for `Distributions.InverseGamma`,
# so no amount of searching for "InverseGamma" will find `@node GammaInverse`.
const NODE_ALIASES = Dict("InverseGamma" => "GammaInverse")

function locate_node(name)
    # Plain substring search, not a regex: node names include `+`, `*` and `dot`, whose
    # regex escaping would be more error-prone than the search itself.
    bare = last(split(name, "."))               # strip the defining module
    bare = String(first(split(bare, ['{', ' ']))) # strip type parameters and `where`
    bare = get(NODE_ALIASES, bare, bare)
    needles = [
        "@node $bare ",
        "@node typeof($bare) ",
        "is_predefined_node(::Type{$bare}",
        "is_predefined_node(::Type{<:$bare}",
    ]
    predefined = filter(f -> occursin("nodes/predefined", f), srcfiles())
    for file in predefined
        txt = read(file, String)
        any(n -> occursin(n, txt), needles) && return relpath_of(file)
    end
    # Some nodes are declared under an alias whose name differs from the runtime type:
    # `@node GammaInverse` registers `Distributions.InverseGamma`, `@node softdot`
    # registers `SoftDot`. Fall back to any `@node` file that mentions the type at all.
    for file in predefined
        txt = read(file, String)
        (occursin("@node", txt) && occursin(bare, txt)) &&
            return relpath_of(file)
    end
    return "(unlocated)"
end

# Engine hooks cannot be derived from `names()`: five of the eight export nothing at all,
# yet every one of them is documented public API that downstream packages plug into.
const ENGINE_HOOKS = [
    (symbol = "form constraints", file = "src/constraints/form.jl"),
    (symbol = "rule fallbacks", file = "src/rules/fallbacks.jl"),
    (symbol = "callbacks", file = "src/callbacks.jl"),
    (symbol = "stream postprocessors", file = "src/postprocessors.jl"),
    (symbol = "scoring", file = "src/score/"),
    (symbol = "node traits (@node-generated)", file = "src/nodes/nodes.jl"),
    (symbol = "delta rule layouts", file = "src/nodes/predefined/delta/"),
    (symbol = "CVI optimiser hooks", file = "src/approximations/cvi.jl"),
]

const EXTENSIONS = [
    (symbol = "ReactiveMPOptimisersExt", file = "ext/ReactiveMPOptimisersExt/"),
    (symbol = "ReactiveMPProjectionExt", file = "ext/ReactiveMPProjectionExt/"),
]

# Rules inherit their node's destination. These are the ones that cannot: they touch the
# engine, raw tuples, or graph objects, and each needs an individual decision.
const RULE_EXCEPTIONS = [
    (symbol = "mixture/switch.jl", file = "src/rules/mixture/switch.jl"),
    (
        symbol = "delta layout: default",
        file = "src/nodes/predefined/delta/layouts/default.jl",
    ),
    (
        symbol = "delta layout: cvi",
        file = "src/nodes/predefined/delta/layouts/cvi.jl",
    ),
    (
        symbol = "delta layout: cvi-projection",
        file = "ext/ReactiveMPProjectionExt/layout/cvi_projection.jl",
    ),
    (symbol = "mixture rules indexing raw inputs", file = "src/rules/mixture/"),
    (
        symbol = "discrete_transition rules indexing raw inputs",
        file = "src/rules/discrete_transition/",
    ),
    (symbol = "MessageMapping construction sites", file = "src/message.jl"),
]

const SECTIONS = [
    ("Nodes", "node", nodes),
    ("Exported symbols", "export", exported_symbols),
    (
        "Engine hooks",
        "hook",
        () -> [(; r..., kind = "hook") for r in ENGINE_HOOKS],
    ),
    ("Extensions", "ext", () -> [(; r..., kind = "ext") for r in EXTENSIONS]),
    (
        "Rule-level exceptions",
        "rule",
        () -> [(; r..., kind = "rule") for r in RULE_EXCEPTIONS],
    ),
]

# ---------------------------------------------------------------------------------------
# INVENTORY.md read / write
# ---------------------------------------------------------------------------------------

unbacktick(s) = strip(replace(strip(s), "`" => ""))

"Read existing decisions, keyed by (kind, symbol), so regeneration never loses work."
function read_decisions()
    decisions = Dict{Tuple{String, String}, NamedTuple}()
    isfile(INVENTORY) || return decisions
    for line in readlines(INVENTORY)
        startswith(strip(line), "|") || continue
        cells = strip.(split(strip(line), "|"; keepempty = false))
        length(cells) == 5 || continue
        # The `-` node is a legitimate symbol, so a row cannot be dismissed as the
        # markdown separator just because its first cell is dashes -- every cell must be.
        isseparator = all(c -> length(c) >= 3 && all(==('-'), c), cells)
        (isseparator || strip(cells[1]) == "symbol") && continue
        sym, kind = unbacktick(cells[1]), unbacktick(cells[2])
        isempty(sym) && continue
        decisions[(kind, sym)] = (
            destination = unbacktick(cells[4]), note = strip(cells[5]),
        )
    end
    return decisions
end

function generate()
    decisions = read_decisions()
    io = IOBuffer()
    println(
        io,
        """
        # Disposition inventory

        **Generated by `scripts/inventory.jl`. Edit the `destination` and `note` columns only —
        every other column is overwritten on regeneration, and the `destination`/`note` cells
        you fill in are preserved.**

        Phase P deliverable, and open item #14 in `PLAN.md`: every node, exported symbol,
        engine hook, extension and rule-level exception carries either a destination package or
        a deliberate deletion, so that the package split is a lookup rather than a judgement
        call made 400 times under time pressure.

        Regenerate with `julia --project=compat/v6-comparison scripts/inventory.jl --generate`.
        Validate with `julia --project=compat/v6-comparison scripts/inventory.jl --check`.

        ## Destinations

        | value | meaning |
        |---|---|
        | `base` | `MessagePassingRulesBase` |
        | `standard` | `StandardMessagePassingRules` — distributions, arithmetic, logic, mixtures |
        | `approximations` | `MessagePassingRulesApproximations` |
        | `testutils` | `MessagePassingRulesTestUtils` |
        | `engine` | stays in `ReactiveMP` |
        | `node:<Name>` | spun out into its own node package |
        | `delete` | deliberately removed |
        | `undecided` | not yet assigned — `--check` fails while any of these remain |

        **Rules inherit their node's destination.** Only the rules that cannot are listed, under
        *Rule-level exceptions*.

        **`DeltaFn` is deliberately absent from *Nodes*.** It is not registered through
        `is_predefined_node` — it is a special node carrying its own layout system rather than a
        predefined one — so the registry cannot report it. It is covered instead by its exported
        types (`DeltaFn`, `DeltaFnNode`, `DeltaMeta`) and by the three delta layout rows under
        *Rule-level exceptions*.

        A row whose `destination` is `delete` and which is exported **must** carry a `note`: it
        becomes an entry in the migration guide, including when the honest answer is "no
        replacement".
        """,
    )

    for (title, kind, enumerate_fn) in SECTIONS
        rows = enumerate_fn()
        println(io, "\n## $title\n")
        println(io, "$(length(rows)) entries.\n")
        println(io, "| symbol | kind | file | destination | note |")
        println(io, "|---|---|---|---|---|")
        for r in rows
            d = get(decisions, (get(r, :kind, kind), r.symbol), nothing)
            dest = d === nothing ? "undecided" : d.destination
            note = d === nothing ? "" : d.note
            println(
                io,
                "| `$(r.symbol)` | `$(get(r, :kind, kind))` | `$(r.file)` | `$dest` | $note |",
            )
        end
    end

    write(INVENTORY, String(take!(io)))
    return nothing
end

function check()
    isfile(INVENTORY) ||
        error("INVENTORY.md does not exist; run --generate first")
    decisions = read_decisions()
    problems = String[]
    live = Set{Tuple{String, String}}()

    for (title, kind, enumerate_fn) in SECTIONS
        for r in enumerate_fn()
            k = (get(r, :kind, kind), r.symbol)
            push!(live, k)
            d = get(decisions, k, nothing)
            if d === nothing
                push!(
                    problems, "missing from INVENTORY.md: [$title] $(r.symbol)"
                )
                continue
            end
            if d.destination == "undecided" || isempty(d.destination)
                push!(problems, "no destination assigned: [$title] $(r.symbol)")
            elseif !isvaliddestination(d.destination)
                push!(
                    problems,
                    "invalid destination `$(d.destination)`: [$title] $(r.symbol)",
                )
            end
            if d.destination == "delete" && kind == "export" && isempty(d.note)
                push!(
                    problems,
                    "exported deletion without a migration note: $(r.symbol)",
                )
            end
        end
    end

    for k in keys(decisions)
        k in live ||
            push!(problems, "stale row, no longer exists: [$(k[1])] $(k[2])")
    end

    if isempty(problems)
        println("INVENTORY.md is complete: every entity has a destination.")
        return 0
    end
    println(stderr, "INVENTORY.md has $(length(problems)) problem(s):")
    for p in sort(problems)
        println(stderr, "  - ", p)
    end
    return 1
end

function main(args)
    if "--generate" in args
        generate()
        println("wrote ", relpath_of(INVENTORY))
        return 0
    elseif "--check" in args
        return check()
    else
        println(stderr, "usage: inventory.jl --generate | --check")
        return 2
    end
end

exit(main(ARGS))
