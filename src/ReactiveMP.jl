module ReactiveMP

include("rocket.jl")
include("macrohelpers.jl")
include("helpers.jl")
include("math.jl")

include("constraints/meta/meta.jl")

include("constraints/prod/prod.jl")
include("constraints/form/form.jl")

include("message.jl")
include("marginal.jl")
include("distributions.jl")

include("constraints/prod/prod_analytical.jl")
include("constraints/prod/prod_generic.jl")
include("constraints/prod/prod_preserve_type.jl")
include("constraints/prod/prod_final.jl")
include("constraints/prod/prod_resolve.jl")

as_marginal(message::Message)  = Marginal(getdata(message), is_clamped(message), is_initial(message))
as_message(marginal::Marginal) = Message(getdata(marginal), is_clamped(marginal), is_initial(marginal))

include("variable.jl")
include("pipeline.jl")

include("actors/score.jl")

include("algebra/cholesky.jl")
include("algebra/companion_matrix.jl")
include("algebra/correction.jl")
include("algebra/helpers.jl")
include("algebra/permutation_matrix.jl")
include("algebra/standard_basis_vector.jl")

include("approximations.jl")
include("approximations/gausshermite.jl")
include("approximations/gausslaguerre.jl")
include("approximations/sphericalradial.jl")
include("approximations/laplace.jl")
include("approximations/importance.jl")
include("approximations/optimizers.jl")

include("distributions/pointmass.jl")
include("distributions/uniform.jl")
include("distributions/gamma_shape_rate.jl")
include("distributions/gamma.jl")
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
include("distributions/contingency.jl")
include("distributions/function.jl")
include("distributions/sample_list.jl")
include("distributions/inv_wishart.jl")

# Equality node is a special case and needs to be included before random variable implementation
include("nodes/equality.jl")

include("variables/random.jl")
include("variables/constant.jl")
include("variables/data.jl")

include("pipeline/async.jl")
include("pipeline/discontinue.jl")
include("pipeline/logger.jl")
include("pipeline/vague.jl")

include("rule.jl")

include("node.jl")
include("score.jl")

include("score/variable.jl")
include("score/node.jl")

# Stochastic nodes
include("nodes/uninformative.jl")
include("nodes/uniform.jl")
include("nodes/normal_mean_variance.jl")
include("nodes/normal_mean_precision.jl")
include("nodes/mv_normal_mean_covariance.jl")
include("nodes/mv_normal_mean_precision.jl")
include("nodes/mv_normal_weighted_mean_precision.jl")
include("nodes/gamma.jl")
include("nodes/gamma_shape_rate.jl")
include("nodes/beta.jl")
include("nodes/categorical.jl")
include("nodes/matrix_dirichlet.jl")
include("nodes/dirichlet.jl")
include("nodes/bernoulli.jl")
include("nodes/gcv.jl")
include("nodes/kernel_gcv.jl")
include("nodes/wishart.jl")
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
include("nodes/inv_wishart.jl")

# Deterministic nodes
include("nodes/addition.jl")
include("nodes/subtraction.jl")
include("nodes/multiplication.jl")

include("rules/prototypes.jl")

include("constraints/form/form_unspecified.jl")
include("constraints/form/form_point_mass.jl")
include("constraints/form/form_fixed_marginal.jl")
include("constraints/form/form_sample_list.jl")

include("constraints/spec/spec.jl")
include("constraints/spec/factorisation_spec.jl")
include("constraints/spec/form_spec.jl")

include("model.jl")
include("fixes.jl")
include("inference.jl")

end
