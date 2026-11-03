"""
    ReactiveMP.DampingPostprocessor(β)

A stream postprocessor that damps messages: each message of an outbound message stream becomes
a mix of the new message and the stream's previous one, as [`ReactiveMP.damp`](@ref)`(β,
previous, new)` gives it, which for two members of one exponential family is
`β η_previous + (1 - β) η_new` in natural parameters (with ExponentialFamily.jl loaded).
Damping keeps the fixed points of message passing and slows the steps towards them, which can
make a loopy or a non-conjugate schedule converge where it oscillates undamped.

- The first message of a stream, a clamped message and a pair `damp` cannot mix pass as they
  are; a damped message keeps the new one's flags and annotations, and a tracked log scale
  becomes undefined.
- The state is the subscription's: every subscriber sees its own damped sequence.
- An equality chain's streams carry triggers rather than messages, and pass unchanged, as do
  marginals and scores.

# Fields

- `β`: the weight of the previous message, in `[0, 1)`; `0` does not damp.

# Throws

- `ArgumentError` when `β` is outside `[0, 1)`.
"""
struct DampingPostprocessor{T <: Real} <: AbstractStreamPostprocessor
    β::T
    function DampingPostprocessor(β::T) where {T <: Real}
        0 <= β < 1 || throw(ArgumentError("the damping factor must be in [0, 1), got $β"))
        return new{T}(β)
    end
end

postprocess_stream_of_outbound_messages(p::DampingPostprocessor, stream) =
    stream |> scan((new, previous) -> damp_message(p.β, previous, new))
postprocess_stream_of_marginals(::DampingPostprocessor, stream) = stream
postprocess_stream_of_scores(::DampingPostprocessor, stream) = stream

# The log scale of a damped message: a mix of two messages has none of its own.
const DAMPING_LOGSCALE = UndefinedLogScale(:damping)

# Messages only: anything else passes.
damp_message(β, previous, new) = new
function damp_message(β, previous::AbstractMessage, new::AbstractMessage)
    message = as_message(new)
    is_clamped(message) && return message
    data = damp(β, getdata(as_message(previous)), getdata(message))
    data === getdata(message) && return message
    return Message(data, false, is_initial(message), getannotations(message), damped_logscale(message.logscale))
end

damped_logscale(::Nothing) = nothing
damped_logscale(_) = DAMPING_LOGSCALE

"""
    ReactiveMP.damp(β, previous, new)

The mix of the message data `previous` and `new` that [`ReactiveMP.DampingPostprocessor`](@ref)
emits: `new` itself by default. With ExponentialFamily.jl loaded, two members of one exponential
family with one conditioner mix in natural parameters, `β η_previous + (1 - β) η_new`, as the
family's default parameterisation. A method for other data adds damping for it.
"""
damp(β, previous, new) = new
