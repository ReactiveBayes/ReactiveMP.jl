using ReactiveMP, BayesBase, ExponentialFamily, Profile, MessagePassingRulesBase, StandardMessagePassingRules
import MessagePassingRulesBase: Target, DefaultAlgorithm
mapping = ReactiveMP.MessageMapping(NormalMeanVariance, Target{:out}(), Val((:μ, :v)), nothing, DefaultAlgorithm(), nothing, nothing, nothing)
ms = (Message(NormalMeanVariance(1.0, 2.0), false, false), Message(PointMass(0.5), false, false))
f(n) = for _ in 1:n
    mapping(ms, nothing)
end
f(10); @time f(1_000_000)
Profile.clear(); @profile f(2_000_000)
data, lidict = Profile.retrieve()
# self time per (file:line, func) for Julia frames of ReactiveMP / MessagePassingRulesBase, and top C frames
Profile.print(IOContext(stdout, :displaysize => (300, 400)); format = :flat, sortedby = :overhead, mincount = 150, C = true)
