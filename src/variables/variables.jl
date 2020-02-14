export AbstractVariable, forward_message, backward_message
export inference

using Rocket

abstract type AbstractVariable end

Rocket.@GenerateCombineLatest(2, "inferenceMessage", AbstractMessage, true, t -> multiply(t[1], t[2]))

inference(variable) = inferenceMessage(forward_message(variable), backward_message(variable))

forward_message(variable::V)  where V = error("You probably forgot to implement forward_message(variable::$V)")
backward_message(variable::V) where V = error("You probably forgot to implement backward_message(variable::$V)")

