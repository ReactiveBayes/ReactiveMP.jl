@testitem "NodeInterface" begin
    using Rocket

    import ReactiveMP:
        AbstractVariable,
        NodeInterface,
        get_stream_of_outbound_messages,
        get_stream_of_inbound_messages,
        tag,
        getvariable,
        MessageObservable,
        connect!,
        name

    struct AbstractVariableImplemention <: AbstractVariable
        stream_of_outbound_messages::MessageObservable
    end

    ReactiveMP.create_new_stream_of_inbound_messages!(
        variable::AbstractVariableImplemention
    ) = (variable.stream_of_outbound_messages, 1)
    ReactiveMP.get_stream_of_outbound_messages(
        variable::AbstractVariableImplemention, ::Int
    ) = variable.stream_of_outbound_messages

    stream_of_outbound_messages = MessageObservable()
    stream = Subject(AbstractMessage)
    connect!(stream_of_outbound_messages, stream)
    variable = AbstractVariableImplemention(stream_of_outbound_messages)
    interface = NodeInterface(:name, variable)

    @test name(interface) === :name
    @test occursin("name", repr(interface))
    @test tag(interface) === Val{:name}()
    @test getvariable(interface) === variable
    @test get_stream_of_inbound_messages(interface) ===
        stream_of_outbound_messages

    actor = keep(AbstractMessage)
    subscription = subscribe!(get_stream_of_outbound_messages(interface), actor)

    next!(stream, Message(1, false, false))

    @test getvalues(actor) == [Message(1, false, false)]

    next!(stream, Message(2, false, false))
    next!(stream, Message(3, false, false))

    @test getvalues(actor) == [
        Message(1, false, false),
        Message(2, false, false),
        Message(3, false, false),
    ]

    unsubscribe!(subscription)

    next!(stream, Message(4, false, false))
    next!(stream, Message(5, false, false))

    @test getvalues(actor) == [
        Message(1, false, false),
        Message(2, false, false),
        Message(3, false, false),
    ]
end

@testitem "NodeInterface israndom/isdata/isconst" begin
    import ReactiveMP:
        AbstractVariable, NodeInterface, israndom, isdata, isconst

    let randomvar = randomvar()
        @test israndom(NodeInterface(:random, randomvar))
        @test !isdata(NodeInterface(:random, randomvar))
        @test !isconst(NodeInterface(:random, randomvar))
    end

    let datavar = datavar()
        @test !israndom(NodeInterface(:random, datavar))
        @test isdata(NodeInterface(:random, datavar))
        @test !isconst(NodeInterface(:random, datavar))
    end

    let constvar = constvar(1)
        @test !israndom(NodeInterface(:random, constvar))
        @test !isdata(NodeInterface(:random, constvar))
        @test isconst(NodeInterface(:random, constvar))
    end
end
