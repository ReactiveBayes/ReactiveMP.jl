export GaussianMeanVarianceFactorNode

struct GaussianMeanVarianceFactorNode <: AbstractFactorNode
    name :: String

    mean     :: InterfaceIn
    variance :: InterfaceIn
    value    :: InterfaceOut

    GaussianMeanVarianceFactorNode(name::String) = begin
        mean     = InterfaceIn("[$name]: meanInterfaceIn")
        variance = InterfaceIn("[$name]: varianceInterfaceIn")
        value    = InterfaceOut("[$name]: valueInterfaceOut")

        # Forward message over the value
        define_sum_product!(value,
            combineLatest(mean.joint_message, variance.joint_message) |> map(AbstractMessage, (d) -> calculate_gaussian_value_output(d[1], d[2]))
        )

        # Backward message over the mean
        define_sum_product!(mean, throwError("[$name]: messageOverTheMean is not implemented for all types of incoming messages", AbstractMessage))

        # Backward message over the variance
        define_sum_product!(variance, throwError("[$name]: messageOverTheVariance is not implemented for all types of incoming messages", AbstractMessage))

        return new(name, mean, variance, value)
    end
end

function calculate_gaussian_value_output(mean::DeterministicMessage, variance::DeterministicMessage)::AbstractMessage
    return StochasticMessage(Normal(mean.value, variance.value))::AbstractMessage
end
