export ExponentialLinearQuadratic

import Distributions: mean, var, cov, pdf, logpdf

struct ExponentialLinearQuadratic{T}
    a :: T
    b :: T
    c :: T
    d :: T
end

function Distributions.pdf(distribution::ExponentialLinearQuadratic, x)
    return exp(logpdf(distribution, x))
end

function Distributions.logpdf(distribution::ExponentialLinearQuadratic, x)
    a = distribution.a
    b = distribution.b
    c = distribution.c
    d = distribution.d
    return -0.5 * (a * x + b * exp(c * x + d * x ^ 2 / 2.0))
end
