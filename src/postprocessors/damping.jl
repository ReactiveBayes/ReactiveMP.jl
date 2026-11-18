"""
    ReactiveMP.DampingPostprocessor(β)

A stream postprocessor that damps outbound messages. Each message of an outbound message stream
is replaced by [`ReactiveMP.damp`](@ref)`(β, previous, new)`, a mix of the new message and the
previous message of the stream. With ExponentialFamily.jl loaded, two members of one exponential
family are mixed in natural parameters, `β η_previous + (1 - β) η_new`. Damping does not move the
fixed points of message passing but takes smaller steps towards them, so it can make a loopy or
non-conjugate schedule converge where it oscillates without damping.

- The first message of a stream, a clamped message and a pair of messages that `damp` cannot mix
  pass unchanged. A damped message keeps the flags and annotations of the new message, and a
  tracked log scale becomes undefined.
- Each subscription keeps its own state, so every subscriber sees its own damped sequence.
- The streams of an equality chain carry triggers and pass unchanged, and so do marginals and
  scores.

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

Return the mix of the message data `previous` and `new` that
[`ReactiveMP.DampingPostprocessor`](@ref) emits. The default returns `new`. With
ExponentialFamily.jl loaded, two members of one exponential family with the same conditioner are
mixed in natural parameters, `β η_previous + (1 - β) η_new`, and the result is returned in the
family's default parameterisation. To damp other data, add a method for its type.
"""
damp(β, previous, new) = new
