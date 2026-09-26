# A rule's working memory, one per outbound stream: each message and marginal mapping keeps a
# slot, built at the stream's first call and reused while the same rule runs on it. A rule
# writes its scratch before reading it, so rebuilding it is always safe; it never leaves the
# rule and is never shared with another.

"""
    ReactiveMP.ScratchSlot()

Where a [`ReactiveMP.MessageMapping`](@ref) or a [`ReactiveMP.MarginalMapping`](@ref) keeps the
scratch of the rule it runs: the rule it was built for, and the scratch itself, both `nothing`
until the first call. See [`ReactiveMP.scratch_for!`](@ref).
"""
mutable struct ScratchSlot
    spec::Any
    scratch::Any
    ScratchSlot() = new(nothing, nothing)
end

"""
    ReactiveMP.scratch_for!(slot::ScratchSlot, spec, algorithm, ctx, args, target, checked::Bool = false)

The scratch to run the rule `spec` with: the one `slot` holds, if it was built for this rule, or a
new one from [`rule_scratch`](@extref MessagePassingRulesBase.rule_scratch), which `slot` then
keeps. `nothing` for a rule that declares no scratch.

With `checked`, the `checked_buffers` diagnostic, a reused scratch is poisoned first (see
[`ReactiveMP.poison!`](@ref)), so a rule that reads its scratch before writing it computes `NaN`.
"""
function scratch_for!(slot::ScratchSlot, spec, algorithm, ctx, args, target, checked::Bool = false)
    spec.scratch === nothing && return nothing
    slot.spec === spec && return checked ? poison!(slot.scratch) : slot.scratch
    slot.scratch = MessagePassingRulesBase.rule_scratch(spec, algorithm, ctx, args, target)
    slot.spec = spec
    return slot.scratch
end
