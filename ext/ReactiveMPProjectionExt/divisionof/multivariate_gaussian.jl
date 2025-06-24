function BayesBase.prod(
    ::ClosedProd, something::C, division::DivisionOf{A, B}
) where {A <: MultivariateGaussianDistributionsFamily, B <: MultivariateGaussianDistributionsFamily, C <: MultivariateGaussianDistributionsFamily}

    d_numerator = convert(MvNormalMeanPrecision, division.numerator)
    d_denumerator = convert(MvNormalMeanPrecision, division.denumerator)
    d_something = convert(MvNormalMeanPrecision, something)

    ef_a = convert(ExponentialFamilyDistribution, d_numerator)
    ef_b = convert(ExponentialFamilyDistribution, d_denumerator)
    ef_c = convert(ExponentialFamilyDistribution, d_something)

    ef_a_typetag = ExponentialFamily.exponential_family_typetag(ef_a)

    resulting_nat_params = ExponentialFamily.getnaturalparameters(ef_a) - ExponentialFamily.getnaturalparameters(ef_b) + ExponentialFamily.getnaturalparameters(ef_c)
    ef_resulting = ExponentialFamily.ExponentialFamilyDistribution(ef_a_typetag, resulting_nat_params)

    return convert(Distribution, ef_resulting)
end

function BayesBase.prod(
    prodtype::ClosedProd, division::DivisionOf{A, B}, something::C
) where {A <: MultivariateGaussianDistributionsFamily, B <: MultivariateGaussianDistributionsFamily, C <: MultivariateGaussianDistributionsFamily}
    return prod(prodtype, something, division)
end