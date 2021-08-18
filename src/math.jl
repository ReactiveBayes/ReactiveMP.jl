import Base: maximum

const tiny = 1e-12
const huge = 1e+12

function softmax(v)
    ret = exp.(clamp.(v .- maximum(v), -100.0, 0.0))
    ret ./ sum(ret)
end