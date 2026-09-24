# `out = A * in`: the function `*` is the node, as in v6. A known factor may be a scalar, a
# vector (then the other factor is a scalar) or a matrix; for a matrix, only `A * in` is
# computed, never `in * A`.
@define_factor_node(node = *, type = Deterministic, interfaces = [:out, :A, :in])

# v6's default: a zero on a precision's diagonal is replaced, so it stays invertible.
multiplication_default_correction() = ReplaceZeroDiagonalEntries(tiny)
corrected(ctx, W) = correction!(matrix_correction(ctx, multiplication_default_correction()), W)

# v6's number of draws for the sampled messages (DISCUSSION §3.32).
const MULTIPLICATION_SAMPLES = 3000

# Forward: the distribution of `c * x` for a known c and a Gaussian or Gamma x.
scaled(c::Real, m::UnivariateNormalDistributionsFamily) = NormalMeanVariance(c * mean(m), c^2 * var(m))
scaled(c::Real, m::MvNormalMeanCovariance) = MvNormalMeanCovariance(c * mean(m), c^2 * cov(m))
scaled(c::Real, m::MvNormalMeanPrecision) = MvNormalMeanPrecision(c * mean(m), precision(m) / c^2)
scaled(c::Real, m::MvNormalWeightedMeanPrecision) = ((ξ, W) = weightedmean_precision(m); MvNormalWeightedMeanPrecision(ξ / c, W / c^2))
scaled(c::Real, m::GammaDistributionsFamily) = GammaShapeRate(shape(m), rate(m) / c)
scaled(C::AbstractMatrix, m::NormalDistributionsFamily) =
    ((μ, Σ) = mean_cov(m); promote_variate_type(variate_form(typeof(m)), NormalMeanVariance)(C * μ, C * Σ * C'))
# A vector times a scalar: a covariance of rank one, as in v6, whose TODO noted it is singular.
scaled(c::AbstractVector, m::UnivariateNormalDistributionsFamily) = MvNormalMeanCovariance(mean(m) * c, var(m) * c * c')

# Backward: the likelihood of x from `out = c * x` for a known c, a Gaussian or Gamma in x.
unscaled(ctx, m_out::PointMass, c) = PointMass(c \ mean(m_out))
unscaled(ctx, m_out::GammaDistributionsFamily, c::Real) = GammaShapeRate(shape(m_out), rate(m_out) * c)
unscaled(ctx, m_out::NormalDistributionsFamily, c::Real) =
    ((ξ, W) = weightedmean_precision(m_out); promote_variate_type(variate_form(typeof(m_out)), NormalWeightedMeanPrecision)(c * ξ, corrected(ctx, c^2 * W)))
unscaled(ctx, m_out::MultivariateNormalDistributionsFamily, C::AbstractMatrix) =
    ((ξ, W) = weightedmean_precision(m_out); MvNormalWeightedMeanPrecision(C' * ξ, corrected(ctx, C' * W * C)))
unscaled(ctx, m_out::MultivariateNormalDistributionsFamily, c::AbstractVector) =
    ((ξ, W) = weightedmean_precision(m_out); NormalWeightedMeanPrecision(dot(c, ξ), corrected(ctx, dot(c, W, c))))
# From a covariance, without inverting it first.
function unscaled(ctx, m_out::MvNormalMeanCovariance, C::AbstractMatrix)
    μ, Σ = mean_cov(m_out)
    CᵀΣ⁻¹ = C' / fastcholesky(Σ)
    return MvNormalWeightedMeanPrecision(CᵀΣ⁻¹ * μ, corrected(ctx, CᵀΣ⁻¹ * C))
end
function unscaled(ctx, m_out::MvNormalMeanCovariance, c::AbstractVector)
    μ, Σ = mean_cov(m_out)
    cᵀΣ⁻¹ = c' / fastcholesky(Σ)
    return NormalWeightedMeanPrecision(dot(cᵀΣ⁻¹, μ), corrected(ctx, dot(cᵀΣ⁻¹, c)))
end

# m(x) = N_out(c x) integrates to |c|^(-d) over x ∈ Rᵈ. v6 took -logdet(c), which fails for
# c < 0 and misses d (ReactiveMP.jl#680).
unscaled_logscale(m_out, c::Real) = -length(mean(m_out)) * log(abs(c))

# Between two univariate Gaussians, towards one factor x from `out = x * y`:
# ∫ N_out(x y) N_y(y) dy = N(μ_out / x; μ_y, v_y + v_out / x²) / |x|.
function gaussian_ratio_logpdf(m_out, m_y)
    μ_y, v_y = mean_var(m_y)
    μ_out, v_out = mean_var(m_out)
    return ContinuousUnivariateLogPdf(x -> -log(abs(x)) - (log2π + log(v_y + v_out / x^2)) / 2 - (μ_out - x * μ_y)^2 / (v_y * x^2 + v_out) / 2)
end

# Towards a factor x from any two univariate distributions: ∫ p_out(x y) p_y(y) dy, by draws of
# y. v6 weighted each draw by |y|, the density of out/y instead (ReactiveMP.jl#679). The sum
# is v6's, unnormalised.
function sampled_ratio_logpdf(rng, m_out, m_y)
    ys = rand(rng, m_y, MULTIPLICATION_SAMPLES)
    return ContinuousUnivariateLogPdf(x -> log(sum(y -> pdf(m_out, x * y), ys)))
end

# Towards `out` from any two univariate distributions: ∫ p_A(a) p_in(z / a) / |a| da, by draws of
# a, unnormalised as in v6.
function sampled_product_logpdf(rng, m_A, m_in)
    as = rand(rng, m_A, MULTIPLICATION_SAMPLES)
    return ContinuousUnivariateLogPdf(z -> log(sum(a -> pdf(m_in, z / a) / abs(a), as)))
end

# The log-density of the product of two Gaussians with means `mx`, `my`, variances `vx`, `vy`
# and correlation `rho`, as a series of modified Bessel functions of the second kind truncated
# at `truncation`, evaluated at `x + jitter`.
function besselmod(mx, vx, my, vy, rho; truncation = 10, jitter = 1.0e-8)
    T = float(promote_type(typeof(mx), typeof(vx), typeof(my), typeof(vy), typeof(rho)))
    return function (x)
        x = x + T(jitter)
        s = sqrt(vx * vy)
        term1 = -(mx^2 / vx + my^2 / vy - 2 * rho * (x + mx * my) / s) / (2 * (1 - rho^2))
        term2 = zero(promote_type(T, typeof(x)))
        for n in 0:truncation, m in 0:(2n)
            term2 += x^(2n - m) * abs(x)^(m - n) * sqrt(vx)^(m - n - 1) /
                (T(π) * gamma(T(2n + 1)) * (1 - rho^2)^(2n + T(1) / 2) * sqrt(vy)^(m - n + 1)) *
                (mx / vx - rho * my / s)^m * binomial(2n, m) * (my / vy - rho * mx / s)^(2n - m) *
                besselk(m - n, abs(x) / ((1 - rho^2) * s))
        end
        return term1 + log(term2)
    end
end
