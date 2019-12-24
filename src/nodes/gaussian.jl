using Rx

export GaussianMeanVarianceFactorNode, gaussian_mean_variance

@CreateMapOperator(GaussianValueForward, Tuple{DeterministicMessage, DeterministicMessage}, StochasticMessage{Normal{Float64}}, (d::Tuple{DeterministicMessage, DeterministicMessage}) -> calculate_gaussian_value_output(d[1], d[2]))

struct GaussianMeanVarianceFactorNode <: AbstractFactorNode
    name :: String

    mean     :: InterfaceIn{AbstractMessage, DeterministicMessage}
    variance :: InterfaceIn{AbstractMessage, DeterministicMessage}
    value    :: InterfaceOut{StochasticMessage{Normal{Float64}}, StochasticMessage{Normal{Float64}}}

    GaussianMeanVarianceFactorNode(name::String) = begin
        mean     = InterfaceIn{AbstractMessage, DeterministicMessage}("[$name]: meanInterfaceIn")
        variance = InterfaceIn{AbstractMessage, DeterministicMessage}("[$name]: varianceInterfaceIn")
        value    = InterfaceOut{StochasticMessage{Normal{Float64}}, StochasticMessage{Normal{Float64}}}("[$name]: valueInterfaceOut")

        # Forward message over the value
        define_sum_product!(value, combineLatest(mean.joint_message, variance.joint_message) |> GaussianValueForwardMapOperator())

        # Backward message over the mean
        define_sum_product!(mean, throwError("[$name]: messageOverTheMean is not implemented for all types of incoming messages", AbstractMessage))

        # Backward message over the variance
        define_sum_product!(variance, throwError("[$name]: messageOverTheVariance is not implemented for all types of incoming messages", AbstractMessage))

        return new(name, mean, variance, value)
    end
end

gaussian_mean_variance(name::String) = GaussianMeanVarianceFactorNode(name)

function calculate_gaussian_value_output(mean::DeterministicMessage, variance::DeterministicMessage)::AbstractMessage
    return StochasticMessage(Normal(mean.value, variance.value))::AbstractMessage
end
