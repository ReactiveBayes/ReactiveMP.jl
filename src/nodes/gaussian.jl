export GaussianMeanVarianceNode

using Rocket

gaussianValueForward(args...) = combineLatest(args..., isbatch = true, transformType = AbstractMessage, transformFn = calculate_gaussian_value_output)

struct GaussianMeanVarianceNode <: AbstractStochasticNode
    name     :: String
    mean     :: Interface
    variance :: Interface
    value    :: Interface

    GaussianMeanVarianceNode(name::String) = begin
        mean     = Interface("[$name]: mean")
        variance = Interface("[$name]: variance")
        value    = Interface("[$name]: value")

        # Forward message over the value
        define_sum_product_message!(value, gaussianValueForward(partner_message(mean), partner_message(variance)))

        # Backward message over the mean
        define_sum_product_message!(mean, throwError("[$name]: messageOverTheMean is not implemented for all types of incoming messages", AbstractMessage))

        # Backward message over the variance
        define_sum_product_message!(variance, throwError("[$name]: messageOverTheVariance is not implemented for all types of incoming messages", AbstractMessage))

        return new(name, mean, variance, value)
    end
end

function calculate_gaussian_value_output(t::Tuple)
    return calculate_gaussian_value_output(t[1], t[2])
end

function calculate_gaussian_value_output(mean::DeterministicMessage, variance::DeterministicMessage)
    return StochasticMessage(Normal(mean.value, sqrt(variance.value)))
end
