import StatsFuns: log2π
import SpecialFunctions: digamma, gamma

module GammaMixtureHelpers
    function Λ(x::GammaDistributionsFamily)
        α, β = shape(x), rate(x)
        return 0.5*log2π - 0.5*(digamma(α) - log(β)) + mean(x)*(1+digamma(α+1)-log(β))
    end

    function Λ(x::PointMass)
        return log(gamma(mean(x)))
    end
end  # module GammaMixtureHelpres
