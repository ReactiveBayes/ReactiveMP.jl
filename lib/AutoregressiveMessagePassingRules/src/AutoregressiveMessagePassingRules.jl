"""
    AutoregressiveMessagePassingRules

The autoregressive nodes: [`AR`](@ref), a Bayesian autoregressive process of order `p` with
coefficients `θ` and precision `γ` on edges of their own, and [`ConjugateAR`](@ref), the same
likelihood with `(θ, γ)` joint on one `MvNormalGamma` edge. Their rules run under
[`ARVMP`](@ref), which the model must give: the nodes declare no algorithm, and under the
default one no rule exists.
"""
module AutoregressiveMessagePassingRules

using MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions
using MessagePassingRulesBase: AbstractAlgorithm
using BayesBase: huge, promote_paramfloattype, promote_variate_type
using Distributions: VariateForm
using FastCholesky: cholinv
using StatsFuns: log2π
import LinearAlgebra
using LinearAlgebra: dot, pinv
import StandardMessagePassingRules

export AR, Autoregressive, ConjugateAR, ARVMP, ARsafe, ARunsafe

include("algebra/companion_matrix.jl")
include("algebra/standard_basis_vector.jl")
include("algebra/ar_matrices.jl")
include("autoregressive.jl")
include("conjugate_autoregressive.jl")

end
