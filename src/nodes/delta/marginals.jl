export DeltaMarginal

# from ForneyLab.jl
"""
DeltaMarginal is an auxilary marginal struct for Delta node.
DeltaMarginal stores a vector with the original dimensionalities (ds), so statistics can later be re-separated.
"""
struct DeltaMarginal
    dist::NormalDistributionsFamily
    # ds is a vector with the original dimensionalities of interfaces
    ds::Vector{Any}
end

entropy(d::DeltaMarginal) = entropy(d.dist)

# comparing 2 DeltaMarginals - similar to src/distributions/pointmass.jl
Base.isapprox(left::DeltaMarginal, right::DeltaMarginal; kwargs...) = isapprox(left.dist, right.dist; kwargs...) && left.ds == right.ds
