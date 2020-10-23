module ReactiveMP

using Distributions

include("macrotools.jl")
include("helpers.jl")
include("stream.jl")

include("algebra/helpers.jl")
include("algebra/cholinv.jl")

include("approximations.jl")
include("approximations/gausshermite.jl")
include("approximations/sphericalradial.jl")
include("approximations/laplace.jl")

include("distributions.jl")
include("distributions/dirac.jl")
include("distributions/gamma.jl")
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

include("node.jl")
include("nodes/normal.jl")
include("nodes/normal_mean_variance.jl")
include("nodes/normal_mean_precision.jl")
include("nodes/mv_normal_mean_covariance.jl")
include("nodes/mv_normal_mean_precision.jl")
include("nodes/gamma.jl")
include("nodes/addition.jl")
include("nodes/gcv.jl")
include("nodes/kernel_gcv.jl")
include("nodes/wishart.jl")

include("rules/prototypes.jl")

include("actors/prior.jl")

include("model.jl")

include("score.jl")

include("score/average_energy/common.jl")
include("score/average_energy/gamma.jl")
include("score/average_energy/normal_mean_precision.jl")
include("score/average_energy/mv_normal_mean_precision.jl")
include("score/average_energy/normal_mean_variance.jl")
include("score/average_energy/gcv.jl")
include("score/average_energy/wishart.jl")

include("score/differential_entropy/common.jl")
include("score/differential_entropy/dirac.jl")

end
