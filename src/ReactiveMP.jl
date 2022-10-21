
module ReactiveMP

include("helpers/macrohelpers.jl")
include("helpers/helpers.jl")

include("helpers/algebra/cholesky.jl")
include("helpers/algebra/companion_matrix.jl")
include("helpers/algebra/correction.jl")
include("helpers/algebra/common.jl")
include("helpers/algebra/permutation_matrix.jl")
include("helpers/algebra/standard_basis_vector.jl")

include("constraints/prod/prod.jl")
include("constraints/prod/prod_analytical.jl")
include("constraints/prod/prod_generic.jl")
include("constraints/prod/prod_preserve_type.jl")
include("constraints/prod/prod_final.jl")
include("constraints/prod/prod_resolve.jl")

include("constraints/form.jl")

include("message.jl")
include("marginal.jl")
include("distributions.jl")

"""
    to_marginal(any)

Transforms an input to a proper marginal distribution.
Called inside `as_marginal`. Some nodes do not use `Distributions.jl`, but instead implement their own equivalents for messages for better efficiency.
Effectively `to_marginal` is needed to convert internal effective implementation to a user-friendly equivalent (e.g. from `Distributions.jl`).
By default does nothing and returns its input, but some nodes may override this behaviour (see for example `Wishart` and `InverseWishart`).

Note: This function is a part of the private API and is not intended to be used outside of the ReactiveMP package.
"""
to_marginal(any) = any

as_marginal(message::Message)  = Marginal(to_marginal(getdata(message)), is_clamped(message), is_initial(message))
as_message(marginal::Marginal) = Message(getdata(marginal), is_clamped(marginal), is_initial(marginal))

getdata(collection::Tuple)         = map(getdata, collection)
getdata(collection::AbstractArray) = map(getdata, collection)

include("approximations/approximations.jl")
include("approximations/shared.jl")
include("approximations/gausshermite.jl")
include("approximations/gausslaguerre.jl")
include("approximations/sphericalradial.jl")
include("approximations/laplace.jl")
include("approximations/importance.jl")
include("approximations/optimizers.jl")
include("approximations/rts.jl")
include("approximations/linearization.jl")
include("approximations/unscented.jl")

include("distributions/pointmass.jl")
include("distributions/uniform.jl")
include("distributions/gamma_shape_rate.jl")
include("distributions/gamma.jl")
include("distributions/gamma_inverse.jl")
include("distributions/gamma_shape_likelihood.jl")
include("distributions/categorical.jl")
include("distributions/matrix_dirichlet.jl")
include("distributions/dirichlet.jl")
include("distributions/beta.jl")
include("distributions/bernoulli.jl")
include("distributions/normal_mean_variance.jl")
include("distributions/normal_mean_precision.jl")
include("distributions/normal_weighted_mean_precision.jl")
include("distributions/mv_normal_mean_covariance.jl")
include("distributions/mv_normal_mean_precision.jl")
include("distributions/mv_normal_weighted_mean_precision.jl")
include("distributions/normal.jl")
include("distributions/exp_linear_quadratic.jl")
include("distributions/wishart.jl")
include("distributions/wishart_inverse.jl")
include("distributions/contingency.jl")
include("distributions/function.jl")
include("distributions/sample_list.jl")

# Equality node is a special case and needs to be included before random variable implementation
include("nodes/equality.jl")

include("variables/variable.jl")
include("variables/random.jl")
include("variables/constant.jl")
include("variables/data.jl")
include("variables/collection.jl")

include("pipeline/pipeline.jl")
include("pipeline/async.jl")
include("pipeline/discontinue.jl")
include("pipeline/logger.jl")

include("rule.jl")
include("node.jl")

include("score/score.jl")
include("score/variable.jl")
include("score/node.jl")

# Stochastic nodes
include("nodes/uninformative.jl")
include("nodes/uniform.jl")
include("nodes/normal_mean_variance.jl")
include("nodes/normal_mean_precision.jl")
include("nodes/mv_normal_mean_covariance.jl")
include("nodes/mv_normal_mean_precision.jl")
include("nodes/mv_normal_mean_scale_precision.jl")
include("nodes/mv_normal_weighted_mean_precision.jl")
include("nodes/gamma.jl")
include("nodes/gamma_inverse.jl")
include("nodes/gamma_shape_rate.jl")
include("nodes/beta.jl")
include("nodes/categorical.jl")
include("nodes/matrix_dirichlet.jl")
include("nodes/dirichlet.jl")
include("nodes/bernoulli.jl")
include("nodes/gcv.jl")
include("nodes/kernel_gcv.jl")
include("nodes/wishart.jl")
include("nodes/wishart_inverse.jl")
include("nodes/normal_mixture.jl")
include("nodes/gamma_mixture.jl")
include("nodes/dot_product.jl")
include("nodes/transition.jl")
include("nodes/autoregressive.jl")
include("nodes/bifm.jl")
include("nodes/bifm_helper.jl")
include("nodes/probit.jl")
include("nodes/flow/flow.jl")
include("nodes/poisson.jl")

# Deterministic nodes
include("nodes/addition.jl")
include("nodes/subtraction.jl")
include("nodes/multiplication.jl")
include("nodes/and.jl")
include("nodes/or.jl")
include("nodes/not.jl")
include("nodes/implication.jl")

include("rules/prototypes.jl")

include("constraints/specifications/constraints.jl")
include("constraints/specifications/form.jl")
include("constraints/specifications/factorisation.jl")
include("constraints/specifications/meta.jl")

# Delta node depends on model.jl (use AutoVar)
include("nodes/delta/delta.jl")

include("rules/prototypes.jl")

end
