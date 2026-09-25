# Opt-in audits of the rules an activated graph runs, all off by default. Each objects to a
# specific rule, by its node, target and the place it is defined, rather than failing vaguely.

"""
    ReactiveMP.EngineDiagnostics(; check_everything_pure = false, check_everything_inplace = false, checked_buffers = false)

The audits a node's rules run under, given to [`ReactiveMP.FactorNodeActivationOptions`](@ref) as
`diagnostics`. All are off by default, and each is checked when a rule is resolved:

- `check_everything_pure` — a rule that declares itself impure (`pure = false`, or an impure
  algorithm it does not override) is an [`ReactiveMP.ImpureRuleError`](@ref). For finding side
  effects, when differentiating through inference or before running rules on several threads.
  Purity is declared, not proved: the audit reads the declaration.
- `check_everything_inplace` — a rule with no in-place form is reported, once per rule, as a
  warning. A coverage audit for latency-sensitive use, not a correctness one.
- `checked_buffers` — the memory the engine recycles between calls is poisoned before each reuse:
  floating-point arrays are filled with `NaN`, so a rule that reads its scratch before writing
  it, against the scratch contract, returns `NaN` rather than a stale value. In-place outputs
  are allocated for every call, so a rule's scratch is the only recycled memory for now.
"""
Base.@kwdef struct EngineDiagnostics
    check_everything_pure::Bool = false
    check_everything_inplace::Bool = false
    checked_buffers::Bool = false
end

"""
    ReactiveMP.ImpureRuleError

The error of `check_everything_pure` (see [`ReactiveMP.EngineDiagnostics`](@ref)): the graph ran a
rule that declares itself impure.
"""
struct ImpureRuleError <: Exception
    spec::MessagePassingRulesBase.RuleSpec
end

rule_description(spec) = string(
    "the ", spec.kind === :average_energy ? "average energy" : "$(spec.kind) rule", " of `", display_name(spec.node), "`",
    spec.target === nothing ? "" : string(spec.kind === :marginal ? " over " : " towards ", target_text(spec.target)),
    " under `", display_name(spec.algorithm), "`, defined at ", spec.file, ":", spec.line,
)

display_name(x::Type) = x isa DataType ? string(nameof(x)) : string(x)
display_name(x) = string(x)

target_text(::Type{MessagePassingRulesBase.Target{E}}) where {E} = ":$E"
target_text(::Type{<:MessagePassingRulesBase.IndexedTarget{E}}) where {E} = "(:$E, k)"
target_text(::Type{MessagePassingRulesBase.ClusterTarget{K}}) where {K} = repr(K)
target_text(target) = string(target)

Base.showerror(io::IO, err::ImpureRuleError) = print(
    io, "ImpureRuleError: `check_everything_pure` is on, and ", rule_description(err.spec),
    " declares itself impure (`pure = false`, or an impure algorithm it does not override)",
)

# The audits a resolved rule passes before it runs. The in-place report names each rule once.
const REPORTED_NOT_INPLACE = Set{Tuple{Symbol, Int, Any}}()
const REPORTED_LOCK = ReentrantLock()

function audit_rule(diagnostics::EngineDiagnostics, spec)
    diagnostics.check_everything_pure && !spec.pure && throw(ImpureRuleError(spec))
    if diagnostics.check_everything_inplace && !spec.inplace
        first_time = lock(REPORTED_LOCK) do
            key = (spec.file, spec.line, spec.target)
            key in REPORTED_NOT_INPLACE ? false : (push!(REPORTED_NOT_INPLACE, key); true)
        end
        first_time && @warn "`check_everything_inplace`: $(rule_description(spec)) has no in-place form"
    end
    return spec
end

"""
    ReactiveMP.poison!(memory)

Fill every floating-point array in `memory`, looking inside tuples and named tuples, with `NaN`,
as `checked_buffers` does to a rule's scratch before each reuse. Anything else is left as it is.
"""
poison!(memory::AbstractArray{<:AbstractFloat}) = fill!(memory, NaN)
poison!(memory::AbstractArray) = (foreach(poison!, memory); memory)
poison!(memory::Union{Tuple, NamedTuple}) = (foreach(poison!, memory); memory)
poison!(memory) = memory
