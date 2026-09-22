# Phase 0 -- the routing machinery, hand-written.
#
# This is what the macros would lower to. Nothing here is a macro, and nothing here is
# meant to survive: the point is to find out whether the shape in PLAN.md infers and
# allocates the way the design assumes, using the real representation rather than a toy.

module Machinery

using BayesBase, ExponentialFamily, LinearAlgebra, Random

export Target, IndexedTarget, edge, index
export Algorithm, BP, VMP, Unscented, Linearization
export RuleContext, default_context, service
export Annotations, NoAnnotations, annotate!, getannotation
export RuleArgs, RuleSpec, find_rule, message_passing_rule, message_passing_rule!
export RuleNotFound

# ---------------------------------------------------------------------------
# Targets. `towards = :out` and `towards = (:m, k)`.
#
# The edge name is a type parameter, the group index is a field -- mirroring v6, where `on`
# is `Val{:out}` or `Tuple{Val{:in}, Int}`. The index must NOT be a type parameter: it is a
# runtime value, and promoting it would specialise the whole pipeline per group position.
# ---------------------------------------------------------------------------

struct Target{E} end
Target(e::Symbol) = Target{e}()

struct IndexedTarget{E}
    index::Int
end
IndexedTarget(e::Symbol, k::Int) = IndexedTarget{e}(k)

edge(::Target{E}) where {E}        = E
edge(::IndexedTarget{E}) where {E} = E
index(t::IndexedTarget)            = t.index

# ---------------------------------------------------------------------------
# Algorithms. Replaces `meta`: a dispatch selector *and* the parameter bag, which is why
# `Linearization` carries fields while `BP` does not.
# ---------------------------------------------------------------------------

abstract type Algorithm end

struct BP <: Algorithm end
struct VMP <: Algorithm end

struct Unscented{T} <: Algorithm
    alpha::T
    beta::T
    kappa::T
end
Unscented() = Unscented(1.0e-3, 2.0, 0.0)

struct Linearization{I} <: Algorithm
    inverse::I
end
Linearization() = Linearization(nothing)

# ---------------------------------------------------------------------------
# Context: read-only infrastructure, never dispatched on, and explicitly NOT the
# annotations sink (PLAN.md § Dispatch axes -- merging the two is a category error).
# ---------------------------------------------------------------------------

struct RuleContext{L, R, S}
    linalg::L
    rng::R
    services::S
end

const default_context = RuleContext(cholesky, Random.default_rng(), NamedTuple())

service(ctx::RuleContext, name::Symbol) = getfield(ctx.services, name)

# ---------------------------------------------------------------------------
# Annotations: the mutable sink a rule *writes* to. `NoAnnotations` is the zero-field
# default, so a rule that does not annotate pays nothing -- PLAN.md § In-place rules.
# ---------------------------------------------------------------------------

struct NoAnnotations end
annotate!(::NoAnnotations, ::Symbol, _) = nothing
getannotation(::NoAnnotations, ::Symbol, default = nothing) = default

mutable struct Annotations
    entries::Vector{Pair{Symbol, Any}}
end
Annotations() = Annotations(Pair{Symbol, Any}[])
annotate!(a::Annotations, k::Symbol, v) = (push!(a.entries, k => v); nothing)
function getannotation(a::Annotations, k::Symbol, default = nothing)
    for (key, value) in a.entries
        key === k && return value
    end
    return default
end

# ---------------------------------------------------------------------------
# The keyed argument object. `args.m[:μ]` and `args.q[:p][k]`.
#
# NamedTuple-backed, because `getindex(::NamedTuple, ::Symbol)` lowers to `getfield` and is
# constant-folded when the symbol is a literal -- which it always is, since the body is a
# real lambda and a bare `μ` would be an UndefVarError. Declaration and body use the same
# key, so no name is ever derived and nothing is ever split on `_`.
#
# A variadic group `m[:inputs...]` is stored as an ordinary Tuple under the group's name.
# ---------------------------------------------------------------------------

struct RuleArgs{M <: NamedTuple, Q <: NamedTuple}
    m::M
    q::Q
end
RuleArgs(; m = NamedTuple(), q = NamedTuple()) = RuleArgs(m, q)

# ---------------------------------------------------------------------------
# RuleSpec -- NO TYPE PARAMETERS. See DISCUSSION.md §3.14.
#
# The deciding argument is `find_rule`: parameterising on the body would make every rule a
# distinct type, so a lookup that cannot statically pin down which rule fires would return a
# union rather than one type, and that degrades silently. This struct is type-stable by
# construction; there is exactly one `RuleSpec` type.
#
# The cost is an indirect call wherever the compiler cannot see which body sits in the
# field. Accepted by decision; 03_devirt.jl measures it rather than assuming it either way.
# ---------------------------------------------------------------------------

struct RuleSpec
    body::Function
    prealloc::Union{Function, Nothing}
    inplace::Bool
    pure::Bool
    source::String
    file::Symbol
    line::Int
end

function RuleSpec(body; prealloc = nothing, inplace = false, pure = true, source = "", file = :none, line = 0)
    return RuleSpec(body, prealloc, inplace, pure, source, file, line)
end

Base.show(io::IO, spec::RuleSpec) = print(
    io, "RuleSpec(", spec.inplace ? "in-place" : "allocating", spec.pure ? "" : ", impure", " @ ", spec.file, ":", spec.line, ")"
)

# ---------------------------------------------------------------------------
# Resolution and invocation.
#
# `find_rule` is an ordinary generic function: per-rule methods, dispatch on
# (node, target, algorithm, args). NOT a container lookup -- a spec fetched from a Dict
# keyed on runtime values infers as `Any` whatever the spec's own type, which is the one
# constraint that holds regardless of representation.
# ---------------------------------------------------------------------------

struct RuleNotFound
    node::Any
    target::Any
    algorithm::Any
    args::Any
end

# Least-specific fallback. Returns rather than throws, mirroring v6's `RuleMethodError`
# sentinel -- see 04_fallback.jl, which is where that choice is actually decided.
find_rule(node, target, algorithm, args) = RuleNotFound(node, target, algorithm, args)

# Canonical body slot order: (output, algo, ctx, args, ann, node) -- plus `target`, which is
# threaded through but is NOT a user-facing slot.
#
# FINDING (Phase 0). PLAN.md lists six body slots and no target. But an indexed target,
# `towards = (:m, k)`, binds `k` in the body -- v6 does this by injecting `k = on[2]` at
# macro expansion. `k` is a runtime value, so the lowered body cannot close over it and the
# call must carry the target. Resolution: `target` is threaded to every body, and the macro
# emits `k = index(target)` as an ordinary binding when the declaration names an index. It
# stays out of the user-facing slot list -- writing `k` is how you ask for it.
#
# Bodies here take the full list. A macro would read the slot names the user wrote and wrap
# their lambda in a full-arity adapter at definition time, so slot selection costs nothing
# at run time and is not modelled -- see spike/README.md.

@inline function message_passing_rule(node, target, algorithm, args, ctx = default_context, ann = NoAnnotations())
    spec = find_rule(node, target, algorithm, args)
    spec isa RuleNotFound && throw(RuleNotFoundError(spec))
    if spec.inplace
        output = (spec.prealloc)(args)
        return spec.body(output, algorithm, ctx, args, ann, node, target)
    else
        return spec.body(nothing, algorithm, ctx, args, ann, node, target)
    end
end

@inline function message_passing_rule!(output, node, target, algorithm, args, ctx = default_context, ann = NoAnnotations())
    spec = find_rule(node, target, algorithm, args)
    spec isa RuleNotFound && throw(RuleNotFoundError(spec))
    return spec.body(output, algorithm, ctx, args, ann, node, target)
end

struct RuleNotFoundError <: Exception
    spec::RuleNotFound
end

function Base.showerror(io::IO, err::RuleNotFoundError)
    s = err.spec
    print(io, "no rule found: ", s.node, " towards ", s.target, " under ", s.algorithm)
    print(io, "\n  messages : ", keys(s.args.m))
    print(io, "\n  marginals: ", keys(s.args.q))
    return
end

end # module
