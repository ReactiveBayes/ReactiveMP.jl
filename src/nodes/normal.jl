import Distributions: Normal, MvNormal

function ReactiveMP.make_node(::Type{<:Normal}, options::FactorNodeCreationOptions, args...)
    error("Creating `Normal` (`Gaussian`) node is not allowed, please use a more specific version (e.g. `NormalMeanVariance`.")
end

function ReactiveMP.make_node(::Type{<:MvNormal}, options::FactorNodeCreationOptions, args...)
    error("Creating `MvNormal` node is not allowed, please use a more specific version (e.g. `MvNormalMeanCovariance`.")
end
