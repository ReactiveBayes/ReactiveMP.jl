@testitem "NodeInterface" begin
    using Rocket

    import ReactiveMP: AbstractVariable, NodeInterface, messageout, messagein, tag, getvariable, MessageObservable, connect!, name

    struct AbstractVariableImplemention <: AbstractVariable
        messageout::MessageObservable
    end

    ReactiveMP.create_messagein!(variable::AbstractVariableImplemention) = (variable.messageout, 1)
    ReactiveMP.messageout(variable::AbstractVariableImplemention, ::Int) = variable.messageout

    varmessageout = MessageObservable()
    stream = Subject(AbstractMessage)
    connect!(varmessageout, stream)
    variable = AbstractVariableImplemention(varmessageout)
    interface = NodeInterface(:name, variable)

    @test name(interface) === :name
    @test occursin("name", repr(interface))
    @test tag(interface) === Val{:name}()
    @test getvariable(interface) === variable
    @test messagein(interface) === varmessageout

    actor = keep(AbstractMessage)
    subscription = subscribe!(messageout(interface), actor)

    next!(stream, Message(1, false, false, nothing))

    @test getvalues(actor) == [Message(1, false, false, nothing)]

    next!(stream, Message(2, false, false, nothing))
    next!(stream, Message(3, false, false, nothing))

    @test getvalues(actor) == [Message(1, false, false, nothing), Message(2, false, false, nothing), Message(3, false, false, nothing)]

    unsubscribe!(subscription)

    next!(stream, Message(4, false, false, nothing))
    next!(stream, Message(5, false, false, nothing))

    @test getvalues(actor) == [Message(1, false, false, nothing), Message(2, false, false, nothing), Message(3, false, false, nothing)]
end

@testitem "NodeInterface israndom/isdata/isconst" begin
    import ReactiveMP: AbstractVariable, NodeInterface, israndom, isdata, isconst

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