
using Rocket

import ReactiveMP: AbstractMessage

function fetch_stream_updated(f, stream)
    stream_updated = false
    value = Ref{Any}(missing)
    subscribe!(stream |> take(1), (new_value) -> begin 
        value[] = new_value
        stream_updated = true
    end)
    f()
    @test stream_updated
    return value[]
end

msg(value) = Message(value, false, false, nothing)
mgl(value) = Marginal(value, false, false, nothing)