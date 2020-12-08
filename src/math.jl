import Base: maximum

function softmax(v)
    max = maximum(v)
    ret = exp.(v .- max)
    ret ./ sum(ret)
end