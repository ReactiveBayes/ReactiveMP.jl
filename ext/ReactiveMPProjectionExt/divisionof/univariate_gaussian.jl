function BayesBase.prod(
    ::ClosedProd, something::C, division::DivisionOf{A, B}
) where {A <: UnivariateGaussianDistributionsFamily, B <: UnivariateGaussianDistributionsFamily, C <: UnivariateGaussianDistributionsFamily}
    ef_a = convert(ExponentialFamilyDistribution, division.numerator)
    ef_b = convert(ExponentialFamilyDistribution, division.denumerator)
    ef_c = convert(ExponentialFamilyDistribution, something)

    ef_a_typetag = ExponentialFamily.exponential_family_typetag(ef_a)

    resulting_nat_params = ExponentialFamily.getnaturalparameters(ef_a) - ExponentialFamily.getnaturalparameters(ef_b) + ExponentialFamily.getnaturalparameters(ef_c)
    ef_resulting = ExponentialFamily.ExponentialFamilyDistribution(ef_a_typetag, resulting_nat_params)

    return convert(Distribution, ef_resulting)
end

function BayesBase.prod(
    prodtype::ClosedProd, division::DivisionOf{A, B}, something::C
) where {A <: UnivariateGaussianDistributionsFamily, B <: UnivariateGaussianDistributionsFamily, C <: UnivariateGaussianDistributionsFamily}
    return prod(prodtype, something, division)
end