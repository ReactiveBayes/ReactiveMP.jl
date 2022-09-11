export DeltaMarginal

struct DeltaMarginal
    dist :: MultivariateNormalDistributionsFamily
    ds   :: Vector{Any}
end

entropy(d::DeltaMarginal) = entropy(d.dist)
