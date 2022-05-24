export LaplaceApproximation, laplace

laplace() = LaplaceApproximation()

struct LaplaceApproximation <: AbstractApproximationMethod end

approximation_name(::LaplaceApproximation)       = "LaplaceApproximation"
approximation_short_name(::LaplaceApproximation) = "LP"

using ForwardDiff
using Optim

function getweights(::LaplaceApproximation, mean::AbstractVector{T}, covariance::AbstractMatrix{T}) where {T <: Real}
    return getweights(srcubature(), mean, covariance)
end

function getpoints(::LaplaceApproximation, mean::AbstractVector{T}, covariance::AbstractMatrix{T}) where {T <: Real}
    return getpoints(srcubature(), mean, covariance)
end

function approximate_meancov(::LaplaceApproximation, g::Function, distribution)
    logg = (z) -> log(g(z))
    logd = (z) -> logpdf(distribution, z)

    logf   = (z) -> logg(z) + logd(z)
    d_logf = (z) -> ForwardDiff.gradient(logf, z)

    result = optimize((d) -> -(logf(d)), mean(distribution), LBFGS())
    if !Optim.converged(result)
        error("LaplaceApproximation: convergence failed")
    end

    m = Optim.minimizer(result)
    c = -cholinv(ForwardDiff.hessian(logf, m))

    return m, c
end

function approximate_kernel_expectation(::LaplaceApproximation, g::Function, distribution)
    return approximate_kernel_expectation(srcubature(), g, distribution)
end
