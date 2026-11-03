module ReactiveMPExponentialFamilyExt

using ReactiveMP, ExponentialFamily
using ExponentialFamily: ExponentialFamilyDistribution, exponential_family_typetag, isbasemeasureconstant,
    getnaturalparameters, getconditioner
using Distributions: Distribution

# Damping in natural parameters, for two members of one exponential family with one conditioner.
# A family registers its base measure for its type tag, which tells its members from other
# distributions.
function ReactiveMP.damp(β, previous::Distribution, new::Distribution)
    family = exponential_family_typetag(new)
    (exponential_family_typetag(previous) === family && applicable(isbasemeasureconstant, family)) || return new
    left, right = convert(ExponentialFamilyDistribution, previous), convert(ExponentialFamilyDistribution, new)
    getconditioner(left) == getconditioner(right) || return new
    η = β .* getnaturalparameters(left) .+ (1 - β) .* getnaturalparameters(right)
    return convert(Distribution, ExponentialFamilyDistribution(family, η, getconditioner(right)))
end

end
