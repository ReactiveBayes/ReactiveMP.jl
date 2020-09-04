module ReactiveMP

using Distributions

include("helpers.jl")

include("distributions.jl")
include("distributions/gamma_ab.jl")
include("distributions/normal_mean_precision.jl")
include("distributions/normal_mean_variance.jl")
include("distributions/mv_normal_mean_covariance.jl")
include("distributions/exp_linear_quadratic.jl")

include("message.jl")
include("marginal.jl")

as_marginal(message::Message)  = Marginal(getdata(message))
as_message(marginal::Marginal) = Message(getdata(marginal))

include("approximations/gausshermite.jl")

include("messages/dummy.jl")
include("messages/delta.jl")
include("messages/normal.jl")
include("messages/normal_mean_precision.jl")
include("messages/normal_mean_variance.jl")
include("messages/mv_normal_mean_covariance.jl")
include("messages/gamma_ab.jl")
include("messages/exp_linear_quadratic.jl")

include("variable.jl")
include("variables/random.jl")
include("variables/constant.jl")
include("variables/data.jl")

include("node.jl")
include("nodes/normal.jl")
include("nodes/normal_mean_variance.jl")
include("nodes/mv_normal_mean_covariance.jl")
include("nodes/normal_mean_precision.jl")
include("nodes/gamma_ab.jl")
include("nodes/addition.jl")
include("nodes/gcv.jl")
include("nodes/kernel_gcv.jl")

include("actors/prior.jl")

include("model.jl")

include("score.jl")

include("score/average_energy/gamma_ab.jl")
include("score/average_energy/normal_mean_precision.jl")
include("score/average_energy/normal_mean_variance.jl")
include("score/average_energy/gcv.jl")

include("score/differential_entropy/common.jl")
include("score/differential_entropy/delta.jl")
include("score/differential_entropy/gamma_ab.jl")
include("score/differential_entropy/normal_mean_precision.jl")
include("score/differential_entropy/normal_mean_variance.jl")
include("score/differential_entropy/mv_normal_mean_covariance.jl")

end
