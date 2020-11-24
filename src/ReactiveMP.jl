module ReactiveMP

using Distributions

include("macrotools.jl")
include("helpers.jl")
include("portal.jl")

include("algebra/helpers.jl")
include("algebra/cholinv.jl")

include("approximations.jl")
include("approximations/gausshermite.jl")
include("approximations/sphericalradial.jl")
include("approximations/laplace.jl")

include("distributions.jl")
include("distributions/dirac.jl")
include("distributions/gamma.jl")
include("distributions/beta.jl")
include("distributions/wishart.jl")
include("distributions/normal_mean_precision.jl")
include("distributions/normal_mean_variance.jl")
include("distributions/mv_normal_mean_covariance.jl")
include("distributions/mv_normal_mean_precision.jl")
include("distributions/exp_linear_quadratic.jl")

include("message.jl")
include("marginal.jl")

as_marginal(message::Message)  = Marginal(getdata(message))
as_message(marginal::Marginal) = Message(getdata(marginal))

include("variable.jl")
include("variables/random.jl")
include("variables/constant.jl")
include("variables/data.jl")

include("rule.jl")
include("score.jl")

include("node.jl")

# Stochastic nodes
include("nodes/normal_mean_variance.jl")
include("nodes/normal_mean_precision.jl")
include("nodes/mv_normal_mean_covariance.jl")
include("nodes/mv_normal_mean_precision.jl")
include("nodes/gamma.jl")
include("nodes/beta.jl")
include("nodes/bernoulli.jl")
include("nodes/gcv.jl")
include("nodes/kernel_gcv.jl")
include("nodes/wishart.jl")

# Deterministic nodes
include("nodes/addition.jl")
include("nodes/multiplication.jl")

include("rules/prototypes.jl")

include("actors/prior.jl")

include("model.jl")

end
