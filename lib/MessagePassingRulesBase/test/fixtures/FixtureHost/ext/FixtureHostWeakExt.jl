module FixtureHostWeakExt

using FixtureHost, FixtureWeak
using MessagePassingRulesBase: RuleSpec, RuleArgs, Target, BP, register!, @define_registry
import MessagePassingRulesBase: find_message_rule

@define_registry
const SPEC = RuleSpec(kind = :message, node = FixtureHost.Node, target = Target{:out}, algorithm = BP, signature = RuleArgs, body = (o, a, c, r, n, t) -> 3.0)
find_message_rule(::Type{FixtureHost.Node}, ::Target{:out}, ::BP, ::RuleArgs) = SPEC
register!(__message_passing_registry__, SPEC)

end
