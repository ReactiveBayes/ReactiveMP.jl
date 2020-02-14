export GaussianMeanVarianceNode

using Rocket

Rocket.@GenerateCombineLatest(2, "gaussianValueForward", AbstractMessage, true, t -> calculate_gaussian_value_output(t[1], t[2]))

struct GaussianMeanVarianceNode <: AbstractFactorNode
    name     :: String
    mean     :: InterfaceIn
    variance :: InterfaceIn
    value    :: InterfaceOut

    GaussianMeanVarianceNode(name::String) = begin
        mean     = InterfaceIn("[$name]: meanInterfaceIn")
        variance = InterfaceIn("[$name]: varianceInterfaceIn")
        value    = InterfaceOut("[$name]: valueInterfaceOut")

        # Forward message over the value
        define_sum_product!(value, gaussianValueForward(joint(mean), joint(variance)))

        # Backward message over the mean
        define_sum_product!(mean, throwError("[$name]: messageOverTheMean is not implemented for all types of incoming messages", AbstractMessage))

        # Backward message over the variance
        define_sum_product!(variance, throwError("[$name]: messageOverTheVariance is not implemented for all types of incoming messages", AbstractMessage))

        return new(name, mean, variance, value)
    end
end

function calculate_gaussian_value_output(mean::DeterministicMessage, variance::DeterministicMessage)::AbstractMessage
    return StochasticMessage(Normal(mean.value, sqrt(variance.value)))::AbstractMessage
end
