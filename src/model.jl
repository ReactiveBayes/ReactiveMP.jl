export Model, add!

struct Model{T}
    # Here will be nodes and edge etc
    message_gate :: T
end

message_gate(model::Model) = model.message_gate

# placeholder for future
add!(model, some) = some
