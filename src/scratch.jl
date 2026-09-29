# A rule's working memory, one per outbound stream: each message and marginal mapping whose rule
# declares scratch keeps a slot, built at that rule's first call and reused while it runs there. A rule
# writes its scratch before reading it, so rebuilding it is always safe; it never leaves the
# rule and is never shared with another.

"""
    ReactiveMP.ScratchSlot()

Where a [`ReactiveMP.MessageMapping`](@ref) or a [`ReactiveMP.MarginalMapping`](@ref) keeps the
scratch of the rule it runs: the rule it was built for, and the scratch itself, both `nothing`
until the first call. A mapping creates its slot only when its rule declares scratch. See
[`ReactiveMP.scratch_for!`](@ref).
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
    # The type the scratch has for these inputs, inferred from their types (`Any` where inference
    # cannot pin it down): the slot's value is asserted to it, so the rule runs on a concretely
    # typed scratch; one of another type, built for other input types, is rebuilt.
    T = MessagePassingRulesBase.rule_scratch_type(spec, algorithm, ctx, args, target)
    current = slot.scratch
    if slot.spec === spec && current isa T
        return (checked ? poison!(current) : current)::T
    end
    fresh = MessagePassingRulesBase.rule_scratch(spec, algorithm, ctx, args, target)
    slot.scratch = fresh
    slot.spec = spec
    return fresh::T
end

# The scratch a mapping runs `spec` with, its slot created at the first call of a rule that declares
# scratch: `nothing`, and no slot, for one that does not.
@inline mapping_scratch(mapping, spec, algorithm, ctx, args) =
    spec.scratch === nothing ? nothing :
    scratch_for!(scratch_slot!(mapping), spec, algorithm, ctx, args, mapping.target, mapping.diagnostics.checked_buffers)

function scratch_slot!(mapping)
    slot = mapping.scratch
    slot === nothing || return slot
    fresh = ScratchSlot()
    mapping.scratch = fresh
    return fresh
end
