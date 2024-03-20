
using Rocket

import ReactiveMP: AbstractMessage

function check_stream_updated_once(f, stream)
    stream_updated = false
    value = Ref{Any}(missing)
    subscription = subscribe!(stream, (new_value) -> begin 
        if stream_updated
            error("Stream was updated more than once")
        end
        value[] = new_value
        stream_updated = true
    end)
    f()
    @test stream_updated
    unsubscribe!(subscription)
    return value[]
end

function check_stream_updated_once(stream) 
    return check_stream_updated_once(() -> nothing, stream)
end

function check_stream_not_updated(f, stream)
    subscription = subscribe!(stream, (new_value) -> begin 
        error("Stream was updated. It should not be updated. The value was `$(new_value)`")
    end)
    f()
    unsubscribe!(subscription)
    return true
end

function check_stream_not_updated(stream)
    return check_stream_not_updated(() -> nothing, stream)
end

msg(value) = Message(value, false, false, nothing)
mgl(value) = Marginal(value, false, false, nothing)