module FixtureHostWeakExt

using FixtureHost, FixtureWeak
using MessagePassingRulesBase: RuleSpec, RuleArgs, Target, DefaultAlgorithm, register!, @define_registry
import MessagePassingRulesBase: find_message_rule

@define_registry
const SPEC = RuleSpec(kind = :message, node = FixtureHost.Node, target = Target{:out}, algorithm = DefaultAlgorithm, signature = RuleArgs, body = (o, s, a, c, r, n, t) -> 3.0)
find_message_rule(::Type{FixtureHost.Node}, ::Target{:out}, ::DefaultAlgorithm, ::RuleArgs) = SPEC
register!(__message_passing_registry__, SPEC)

end
