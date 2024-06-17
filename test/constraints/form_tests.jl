@testitem "`UnspecifiedFormConstraint` should not error on `Distribution` objects" begin
    using Distributions
    import ReactiveMP: constrain_form

    @test constrain_form(UnspecifiedFormConstraint(), Beta(1, 1)) == Beta(1, 1)
    @test constrain_form(UnspecifiedFormConstraint(), Normal(0, 1)) == Normal(0, 1)
    @test constrain_form(UnspecifiedFormConstraint(), MvNormal([0.0, 0.0])) == MvNormal([0.0, 0.0])
end

@testitem "`UnspecifiedFormConstraint` should error on `ProductOf` and `LinearizedProductOf` objects" begin
    using Distributions, BayesBase
    import ReactiveMP: constrain_form

    @test_throws "object cannot be used as a functional form in inference backend" constrain_form(UnspecifiedFormConstraint(), ProductOf(Beta(1, 1), Normal(0, 1)))
    @test_throws "object cannot be used as a functional form in inference backend" constrain_form(UnspecifiedFormConstraint(), LinearizedProductOf([Beta(1, 1), Beta(1, 1)], 2))
end

@testitem "`CompositeFormConstraint` should call the constraints in the specified order" begin
    import ReactiveMP: constrain_form

    struct FormConstraint1 end
    struct FormConstraint2 end

    constrain_form(::FormConstraint1, x) = x + 1
    constrain_form(::FormConstraint2, x) = x * 2

    composite = CompositeFormConstraint((FormConstraint1(), FormConstraint2()))
    @test constrain_form(composite, 1) == 4

    composite = CompositeFormConstraint((FormConstraint2(), FormConstraint1()))
    @test constrain_form(composite, 1) == 3
end

@testitem "`preprocess_form_constraints` should create `CompositeFormConstraint` from a tuple of constraints" begin
    import ReactiveMP: preprocess_form_constraints, AbstractFormConstraint

    struct FormConstraint1 <: AbstractFormConstraint end
    struct FormConstraint2 <: AbstractFormConstraint end

    constraints = (FormConstraint1(), FormConstraint2())
    @test preprocess_form_constraints(constraints) == CompositeFormConstraint(constraints)
    @test preprocess_form_constraints(FormConstraint1()) == FormConstraint1()
    @test preprocess_form_constraints(FormConstraint2()) == FormConstraint2()
end

@testitem "`preprocess_form_constraints` should wrap unknown form constraints into a `WrappedFormConstraint`" begin
    import ReactiveMP: preprocess_form_constraints, AbstractFormConstraint, WrappedFormConstraint, WrappedFormConstraintNoContext

    struct FormConstraint1 <: AbstractFormConstraint end
    struct FormConstraint2 end
    struct FormConstraint3WithContext end
    struct FormConstraint3Context end

    ReactiveMP.prepare_context(::FormConstraint3WithContext) = FormConstraint3Context()

    @test preprocess_form_constraints(FormConstraint1()) == FormConstraint1()
    @test preprocess_form_constraints(FormConstraint2()) == WrappedFormConstraint(FormConstraint2(), WrappedFormConstraintNoContext())
    @test preprocess_form_constraints(FormConstraint3WithContext()) == WrappedFormConstraint(FormConstraint3WithContext(), FormConstraint3Context())
    @test preprocess_form_constraints((FormConstraint1(), FormConstraint2())) ==
        CompositeFormConstraint((FormConstraint1(), WrappedFormConstraint(FormConstraint2(), WrappedFormConstraintNoContext())))
    @test preprocess_form_constraints((FormConstraint1(), FormConstraint3WithContext())) ==
        CompositeFormConstraint((FormConstraint1(), WrappedFormConstraint(FormConstraint3WithContext(), FormConstraint3Context())))
    @test preprocess_form_constraints((FormConstraint2(), FormConstraint3WithContext())) == CompositeFormConstraint((
        WrappedFormConstraint(FormConstraint2(), WrappedFormConstraintNoContext()), WrappedFormConstraint(FormConstraint3WithContext(), FormConstraint3Context())
    ))
    @test preprocess_form_constraints((FormConstraint2(), FormConstraint3WithContext(), FormConstraint1())) == CompositeFormConstraint((
        WrappedFormConstraint(FormConstraint2(), WrappedFormConstraintNoContext()), WrappedFormConstraint(FormConstraint3WithContext(), FormConstraint3Context()), FormConstraint1()
    ))

    @test preprocess_form_constraints(preprocess_form_constraints(FormConstraint2())) == WrappedFormConstraint(FormConstraint2(), WrappedFormConstraintNoContext())
    @test preprocess_form_constraints(preprocess_form_constraints(FormConstraint3WithContext())) == WrappedFormConstraint(FormConstraint3WithContext(), FormConstraint3Context())
end

@testitem "`WrappedFormConstraint` should not pass empty context to the `constrain_form` call" begin
    import ReactiveMP: constrain_form, preprocess_form_constraints

    struct FormConstraintWithoutContext end

    function constrain_form(::FormConstraintWithoutContext, x)
        return x + 1
    end

    function constrain_form(::FormConstraintWithoutContext, context, x)
        error("This function should not be called")
    end

    constraint = preprocess_form_constraints(FormConstraintWithoutContext())

    @test constrain_form(constraint, 1) == 2
    @test constrain_form(constraint, 2) == 3
    @test constrain_form(constraint, 7) == 8
end

@testitem "`WrappedFormConstraint` should be able to reuse the context between multiple `constrain_form` calls" begin
    import ReactiveMP: constrain_form, preprocess_form_constraints

    struct FormConstraintWithContext end
    mutable struct FormConstraintContext
        value::Int
    end

    ReactiveMP.prepare_context(::FormConstraintWithContext) = FormConstraintContext(0)

    function constrain_form(::FormConstraintWithContext, x)
        error("This function should not be called")
    end

    function constrain_form(::FormConstraintWithContext, context::FormConstraintContext, x)
        context.value += 1
        return x + context.value
    end

    constraint = preprocess_form_constraints(FormConstraintWithContext())

    @test constrain_form(constraint, 1) == 2
    @test constraint.context.value === 1

    @test constrain_form(constraint, 2) == 4
    @test constraint.context.value === 2

    @test constrain_form(constraint, 6) == 9
    @test constraint.context.value === 3
end
