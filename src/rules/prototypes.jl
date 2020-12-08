
include("uninformative/out.jl")

include("addition/out.jl")
include("addition/in1.jl")
include("addition/in2.jl")

include("multiplication/out.jl")
include("multiplication/in.jl")

include("gamma/out.jl")
include("gamma/marginals.jl")

include("beta/out.jl")
include("beta/marginals.jl")

include("bernoulli/p.jl")
include("bernoulli/marginals.jl")

include("gcv/x.jl")
include("gcv/y.jl")
include("gcv/z.jl")
include("gcv/k.jl")
include("gcv/w.jl")
include("gcv/marginals.jl")

include("kernel_gcv/x.jl")
include("kernel_gcv/y.jl")
include("kernel_gcv/z.jl")
include("kernel_gcv/marginals.jl")

include("mv_normal_mean_covariance/out.jl")
include("mv_normal_mean_covariance/mean.jl")
include("mv_normal_mean_covariance/marginals.jl")

include("mv_normal_mean_precision/out.jl")
include("mv_normal_mean_precision/mean.jl")
include("mv_normal_mean_precision/precision.jl")
include("mv_normal_mean_precision/marginals.jl")

include("normal/out.jl")

include("normal_mean_precision/out.jl")
include("normal_mean_precision/mean.jl")
include("normal_mean_precision/precision.jl")
include("normal_mean_precision/marginals.jl")

include("normal_mean_variance/out.jl")
include("normal_mean_variance/mean.jl")
include("normal_mean_variance/marginals.jl")

include("wishart/out.jl")
include("wishart/marginals.jl")

include("normal_mixture/out.jl")
include("normal_mixture/switch.jl")