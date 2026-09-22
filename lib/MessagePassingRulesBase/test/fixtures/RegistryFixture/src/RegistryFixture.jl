module RegistryFixture

using MessagePassingRulesBase: MessagePassingRulesBase, RuleSpec, RuleArgs, Target, BP, register!, @define_registry
import MessagePassingRulesBase: find_message_rule

struct Node end

@define_registry
const SPEC = RuleSpec(kind = :message, node = Node, target = Target{:out}, algorithm = BP, signature = RuleArgs, body = (o, a, c, r, n, t) -> 1.0)
find_message_rule(::Type{Node}, ::Target{:out}, ::BP, ::RuleArgs) = SPEC
register!(__message_passing_registry__, SPEC)

module Nested
    using MessagePassingRulesBase: RuleSpec, RuleArgs, Target, BP, register!, @define_registry
    import MessagePassingRulesBase: find_message_rule

    struct Node end

    @define_registry
    const SPEC = RuleSpec(kind = :message, node = Node, target = Target{:out}, algorithm = BP, signature = RuleArgs, body = (o, a, c, r, n, t) -> 2.0)
    find_message_rule(::Type{Node}, ::Target{:out}, ::BP, ::RuleArgs) = SPEC
    register!(__message_passing_registry__, SPEC)
end

end
