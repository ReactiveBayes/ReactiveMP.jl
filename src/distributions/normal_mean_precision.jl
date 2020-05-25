export NormalMeanPrecision

import StatsBase: mean

struct NormalMeanPrecision{T}
    mean      :: T
    precision :: T
end

mean(nmp::NormalMeanPrecision) = npm.mean
precision(nmp::NormalMeanPrecision) = npm.precision
