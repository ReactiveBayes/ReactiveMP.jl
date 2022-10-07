export DeltaMarginal

struct DeltaMarginal
    dist :: NormalDistributionsFamily
    ds   :: Vector{Any}
end

entropy(d::DeltaMarginal) = entropy(d.dist)

# comparing 2 DeltaMarginals - similar to src/distributions/pointmass.jl
Base.isapprox(left::DeltaMarginal, right::DeltaMarginal; kwargs...) = isapprox(left.dist, right.dist; kwargs...) && left.ds == right.ds
