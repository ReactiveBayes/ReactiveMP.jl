export AdditionNode

function AdditionNode()
    return Node(typeof(+), SA[ :in1, :in2, :out ], SA[ SA[ 1, 2, 3 ] ])
end

### Out ###

function rule(::Type{ typeof(+) }, ::Val{:out}, messages::Tuple{Message{T}, Message{T}}, beliefs::Nothing, meta::Nothing) where T
    return Message(getdata(messages[1]) + getdata(messages[2]))
end

function rule(::Type{ typeof(+) }, ::Val{:out}, messages::Tuple{Message{Normal{T}}, Message{Normal{T}}}, beliefs::Nothing, meta::Nothing) where T
    in1d = getdata(messages[1])
    in2d = getdata(messages[2])
    return Message(Normal{T}(mean(in1d) + mean(in2d), sqrt(var(in1d) + var(in2d))))
end

function rule(::Type{ typeof(+) }, ::Val{:out}, messages::Tuple{Message{Normal{T}}, Message{T}}, beliefs::Nothing, meta::Nothing) where T
    in1d = getdata(messages[1])
    in2v = getdata(messages[2])
    return Message(Normal{T}(mean(in1d) + in2v, std(in1d)))
end

function rule(::Type{ typeof(+) }, ::Val{:out}, messages::Tuple{Message{T}, Message{Normal{T}}}, beliefs::Nothing, meta::Nothing) where T
    in1v = getdata(messages[1])
    in2d = getdata(messages[2])
    return Message(Normal{T}(mean(in2d) + in1v, std(in2d)))
end

### In 1 ###

function rule(::Type{ typeof(+) }, ::Val{:in1}, messages::Tuple{Message{T}, Message{T}}, beliefs::Nothing, meta::Nothing) where T
    in2 = getdata(messages[1])
    out = getdata(messages[2])
    return Message(out - in2)
end

function rule(::Type{ typeof(+) }, ::Val{:in1}, messages::Tuple{Message{Normal{T}}, Message{Normal{T}}}, beliefs::Nothing, meta::Nothing) where T
    in2d = getdata(messages[1])
    outd = getdata(messages[2])
    return Message(Normal{T}(mean(outd) - mean(in2d), sqrt(var(outd) + var(in2d))))
end

function rule(::Type{ typeof(+) }, ::Val{:in1}, messages::Tuple{Message{Normal{T}}, Message{T}}, beliefs::Nothing, meta::Nothing) where T
    in2d = getdata(messages[1])
    outv = getdata(messages[2])
    return Message(Normal{T}(outv - mean(in2d), std(in2d)))
end

function rule(::Type{ typeof(+) }, ::Val{:in1}, messages::Tuple{Message{T}, Message{Normal{T}}}, beliefs::Nothing, meta::Nothing) where T
    in2v = getdata(messages[1])
    outd = getdata(messages[2])
    return Message(Normal{T}(mean(outd) - in2v, std(outd)))
end

### In 2 ###

function rule(::Type{ typeof(+) }, ::Val{:in2}, messages::Tuple{Message{T}, Message{T}}, beliefs::Nothing, meta::Nothing) where T
    in1 = getdata(messages[1])
    out = getdata(messages[2])
    return Message(out - in1)
end

function rule(::Type{ typeof(+) }, ::Val{:in2}, messages::Tuple{Message{Normal{T}}, Message{Normal{T}}}, beliefs::Nothing, meta::Nothing) where T
    in1d = getdata(messages[1])
    outd = getdata(messages[2])
    return Message(Normal{T}(mean(outd) - mean(in1d), sqrt(var(outd) + var(in1d))))
end

function rule(::Type{ typeof(+) }, ::Val{:in2}, messages::Tuple{Message{Normal{T}}, Message{T}}, beliefs::Nothing, meta::Nothing) where T
    in1d = getdata(messages[1])
    outv = getdata(messages[2])
    return Message(Normal{T}(outv - mean(in1d), std(in1d)))
end

function rule(::Type{ typeof(+) }, ::Val{:in2}, messages::Tuple{Message{T}, Message{Normal{T}}}, beliefs::Nothing, meta::Nothing) where T
    in1v = getdata(messages[1])
    outd = getdata(messages[2])
    return Message(Normal{T}(mean(outd) - in1v, std(outd)))
end

##
