export LaplaceApproximation, laplace

laplace() = LaplaceApproximation()

struct LaplaceApproximation <: AbstractApproximationMethod end

approximation_name(::LaplaceApproximation)       = "LaplaceApproximation"
approximation_short_name(::LaplaceApproximation) = "LP"

using ForwardDiff
using Optim

function getweights(
    ::LaplaceApproximation,
    mean::AbstractVector{T},
    covariance::AbstractMatrix{T},
) where {T <: Real}
    return getweights(srcubature(), mean, covariance)
end

function getpoints(
    ::LaplaceApproximation,
    mean::AbstractVector{T},
    covariance::AbstractMatrix{T},
) where {T <: Real}
    return getpoints(srcubature(), mean, covariance)
end

function approximate_meancov(::LaplaceApproximation, g::Function, distribution)
    logg = (z) -> log(g(z))
    logd = (z) -> logpdf(distribution, z)

    logf = (z) -> logg(z) + logd(z)

    result = optimize((d) -> -(logf(d)), mean(distribution), LBFGS())
    if !Optim.converged(result)
        error("LaplaceApproximation: convergence failed")
    end

    m = Optim.minimizer(result)

    # At a maximum of `logf` the Hessian `H` is negative-definite, and the Laplace covariance
    # is `(-H)⁻¹`. Although `-(H⁻¹)` and `(-H)⁻¹` are equal in exact arithmetic, they are *not*
    # equal here: `cholinv` factorizes via Cholesky, which is only defined for
    # positive-definite input. Handing it the negative-definite `H` does not raise -- it
    # silently returns a matrix that is not `H⁻¹` -- so the previous `-cholinv(H)` produced a
    # negated, negative-definite covariance for every input, not just pathological ones.
    # Negating first keeps `cholinv` on the positive-definite matrix it requires.
    #
    # `Symmetric` because `ForwardDiff.hessian` is only symmetric up to rounding, and both
    # `isposdef` and the Cholesky factorization want an exactly symmetric argument.
    neg_hessian = Symmetric(-ForwardDiff.hessian(logf, m))

    # `Optim.converged` above only tells us the optimizer stopped moving; it does not
    # distinguish a strict local maximum from a saddle point or a flat region. For a
    # non-log-concave `g` (an `ExponentialLinearQuadratic` kernel, say) the stationary point
    # may not be a maximum at all, in which case `-H` is not positive-definite and the
    # "covariance" would come out indefinite or negative. Reject that explicitly instead of
    # returning an invalid Gaussian.
    if !isposdef(neg_hessian)
        error(
            """
            LaplaceApproximation: the log-density is not locally concave at the mode found by the optimizer, so no Gaussian approximation exists there.

            The negated Hessian must be positive-definite (equivalently, the stationary point must be a strict local maximum), but it is not — the optimizer likely converged to a saddle point or a flat region. This happens when the integrand is not log-concave.

            Negated Hessian at the stationary point:
            $(neg_hessian)

            Consider a different approximation method (e.g. `GaussHermiteCubature`, `SphericalRadialCubature`, or `ImportanceSamplingApproximation`), which do not assume local concavity.""",
        )
    end

    c = cholinv(neg_hessian)

    return m, c
end

function approximate_kernel_expectation(
    ::LaplaceApproximation, g::Function, distribution
)
    return approximate_kernel_expectation(srcubature(), g, distribution)
end
