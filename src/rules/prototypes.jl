
include("uninformative/out.jl")

include("uniform/out.jl")

include("addition/marginals.jl")
include("addition/out.jl")
include("addition/in1.jl")
include("addition/in2.jl")

include("subtraction/marginals.jl")
include("subtraction/out.jl")
include("subtraction/in1.jl")
include("subtraction/in2.jl")

include("multiplication/marginals.jl")
include("multiplication/out.jl")
include("multiplication/in.jl")
include("multiplication/A.jl")

include("gamma/out.jl")
include("gamma/marginals.jl")

include("gamma_shape_rate/out.jl")
include("gamma_shape_rate/marginals.jl")
include("gamma_shape_rate/a.jl")

include("beta/out.jl")
include("beta/marginals.jl")

include("dirichlet/marginals.jl")
include("dirichlet/out.jl")

include("matrix_dirichlet/out.jl")

include("categorical/out.jl")
include("categorical/p.jl")
include("categorical/marginals.jl")

include("bernoulli/p.jl")
include("bernoulli/out.jl")
include("bernoulli/marginals.jl")

include("gcv/x.jl")
include("gcv/y.jl")
include("gcv/z.jl")
include("gcv/k.jl")
include("gcv/w.jl")
include("gcv/marginals.jl")
include("gcv/gaussian_extension.jl")

include("kernel_gcv/x.jl")
include("kernel_gcv/y.jl")
include("kernel_gcv/z.jl")
include("kernel_gcv/marginals.jl")

include("mv_normal_mean_covariance/out.jl")
include("mv_normal_mean_covariance/mean.jl")
include("mv_normal_mean_covariance/covariance.jl")
include("mv_normal_mean_covariance/marginals.jl")

include("normal_mean_variance/out.jl")
include("normal_mean_variance/mean.jl")
include("normal_mean_variance/var.jl")
include("normal_mean_variance/marginals.jl")

include("mv_normal_mean_precision/out.jl")
include("mv_normal_mean_precision/mean.jl")
include("mv_normal_mean_precision/precision.jl")
include("mv_normal_mean_precision/marginals.jl")

include("mv_normal_mean_scale_precision/out.jl")
include("mv_normal_mean_scale_precision/mean.jl")
include("mv_normal_mean_scale_precision/precision.jl")
include("mv_normal_mean_scale_precision/marginals.jl")

include("mv_normal_weightedmean_precision/marginals.jl")

include("normal_mean_precision/out.jl")
include("normal_mean_precision/mean.jl")
include("normal_mean_precision/precision.jl")
include("normal_mean_precision/marginals.jl")

include("wishart/out.jl")
include("wishart/marginals.jl")

include("wishart_inverse/out.jl")
include("wishart_inverse/marginals.jl")

include("normal_mixture/switch.jl")
include("normal_mixture/m.jl")
include("normal_mixture/p.jl")
include("normal_mixture/out.jl")

include("gamma_mixture/switch.jl")
include("gamma_mixture/a.jl")
include("gamma_mixture/b.jl")
include("gamma_mixture/out.jl")

include("dot_product/marginals.jl")
include("dot_product/out.jl")
include("dot_product/in1.jl")
include("dot_product/in2.jl")

include("transition/marginals.jl")
include("transition/out.jl")
include("transition/in.jl")
include("transition/a.jl")

include("autoregressive/y.jl")
include("autoregressive/x.jl")
include("autoregressive/theta.jl")
include("autoregressive/gamma.jl")
include("autoregressive/marginals.jl")

include("probit/marginals.jl")
include("probit/in.jl")
include("probit/out.jl")

include("flow/marginals.jl")
include("flow/in.jl")
include("flow/out.jl")

include("bifm/marginals.jl")
include("bifm/in.jl")
include("bifm/out.jl")
include("bifm/znext.jl")
include("bifm/zprev.jl")

include("bifm_helper/in.jl")
include("bifm_helper/out.jl")

include("delta/in.jl")
include("delta/marginals.jl")
include("delta/out.jl")

include("poisson/l.jl")
include("poisson/marginals.jl")
include("poisson/out.jl")

include("or/in1.jl")
include("or/in2.jl")
include("or/out.jl")
include("or/marginals.jl")

include("not/in.jl")
include("not/out.jl")
include("not/marginals.jl")

include("and/in1.jl")
include("and/in2.jl")
include("and/out.jl")
include("and/marginals.jl")

include("implication/in1.jl")
include("implication/in2.jl")
include("implication/out.jl")
include("implication/marginals.jl")
