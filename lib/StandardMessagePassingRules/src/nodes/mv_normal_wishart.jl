# The prior MvNormalWishart(μ, W, λ, ν), ExponentialFamily's (μ, Ψ, κ, ν). Its `W` keeps v6's
# alias `scale`, a model-spec name, although ExponentialFamily's `scale` of an MvNormalWishart
# is κ, the `λ` interface here. It has no average energy, as in v6.
@define_factor_node(node = MvNormalWishart, type = Stochastic, interfaces = [:out, (:μ, aliases = [:mean]), (:W, aliases = [:scale]), :λ, :ν])
