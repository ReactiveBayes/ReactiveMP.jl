export DistProduct

"""
    DistProduct

If inference backend cannot return an analytical solution for a product of two distributions it may fallback to the `DistProduct` structure
`DistProduct` is useful to propagate the exact forms of two messages until it hits some approximation method for form-constraint.
However `DistProduct` cannot be used to compute statistics such as mean or variance. 
It has to be approximated before using in actual inference procedure.

Backend exploits form constraints specification which usually help to deal with intractable distributions products. 
User may use EM form constraint with a specific optimisation algorithm or it may approximate intractable product with Gaussian Distribution
using for example Laplace approximation 

See also: [`prod`](@ref)
"""
struct DistProduct{ L, R }
    left  :: L
    right :: R
end

Distributions.mean(product::DistProduct)      = error("mean() is not defined for $(product). DistProduct structure has to be approximated and cannot be used in inference procedure.")
Distributions.median(product::DistProduct)    = error("median() is not defined for $(product). DistProduct structure has to be approximated and cannot be used in inference procedure.")
Distributions.mode(product::DistProduct)      = error("mode() is not defined for $(product). DistProduct structure has to be approximated and cannot be used in inference procedure.")
Distributions.shape(product::DistProduct)     = error("shape() is not defined for $(product). DistProduct structure has to be approximated and cannot be used in inference procedure.")
Distributions.scale(product::DistProduct)     = error("scale() is not defined for $(product). DistProduct structure has to be approximated and cannot be used in inference procedure.")
Distributions.rate(product::DistProduct)      = error("rate() is not defined for $(product). DistProduct structure has to be approximated and cannot be used in inference procedure.")
Distributions.var(product::DistProduct)       = error("var() is not defined for $(product). DistProduct structure has to be approximated and cannot be used in inference procedure.")
Distributions.std(product::DistProduct)       = error("std() is not defined for $(product). DistProduct structure has to be approximated and cannot be used in inference procedure.")
Distributions.cov(product::DistProduct)       = error("cov() is not defined for $(product). DistProduct structure has to be approximated and cannot be used in inference procedure.")
Distributions.invcov(product::DistProduct)    = error("invcov() is not defined for $(product). DistProduct structure has to be approximated and cannot be used in inference procedure.")
Distributions.logdetcov(product::DistProduct) = error("logdetcov() is not defined for $(product). DistProduct structure has to be approximated and cannot be used in inference procedure.")
Distributions.entropy(product::DistProduct)   = error("entropy() is not defined for $(product). DistProduct structure has to be approximated and cannot be used in inference procedure.")
Distributions.params(product::DistProduct)    = error("params() is not defined for $(product). DistProduct structure has to be approximated and cannot be used in inference procedure.")

Distributions.pdf(product::DistProduct, x)    = Distributions.pdf(product.left, x) * Distributions.pdf(product.right, x)
Distributions.logpdf(product::DistProduct, x) = Distributions.logpdf(product.left, x) + Distributions.logpdf(product.right, x)

Base.precision(product::DistProduct) = error("precision() is not defined for $(product). DistProduct structure has to be approximated and cannot be used in inference procedure.")
Base.length(product::DistProduct)    = error("length() is not defined for $(product). DistProduct structure has to be approximated and cannot be used in inference procedure.")
Base.ndims(product::DistProduct)     = error("ndims() is not defined for $(product). DistProduct structure has to be approximated and cannot be used in inference procedure.")
Base.size(product::DistProduct)      = error("size() is not defined for $(product). DistProduct structure has to be approximated and cannot be used in inference procedure.")

probvec(product::DistProduct)         = error("probvec() is not defined for $(product). DistProduct structure has to be approximated and cannot be used in inference procedure.")
weightedmean(product::DistProduct)    = error("weightedmean() is not defined for $(product). DistProduct structure has to be approximated and cannot be used in inference procedure.")
inversemean(product::DistProduct)     = error("inversemean() is not defined for $(product). DistProduct structure has to be approximated and cannot be used in inference procedure.")
logmean(product::DistProduct)         = error("logmean() is not defined for $(product). DistProduct structure has to be approximated and cannot be used in inference procedure.")
meanlogmean(product::DistProduct)     = error("meanlogmean() is not defined for $(product). DistProduct structure has to be approximated and cannot be used in inference procedure.")
mirroredlogmean(product::DistProduct) = error("mirroredlogmean() is not defined for $(product). DistProduct structure has to be approximated and cannot be used in inference procedure.")
loggammamean(product::DistProduct)    = error("loggammamean() is not defined for $(product). DistProduct structure has to be approximated and cannot be used in inference procedure.")