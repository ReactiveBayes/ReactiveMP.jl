@testitem "DivisionOf" begin
    using ExponentialFamily, ExponentialFamilyProjection, BayesBase

    # `DivisionOf` is internal to the extension
    ext = Base.get_extension(ReactiveMP, :ReactiveMPProjectionExt)
    @test !isnothing(ext)
    using .ext

    d1 = NormalMeanVariance(0, 1)
    d2 = NormalMeanVariance(1, 2)

    @test d1 ≈ prod(GenericProd(), ext.DivisionOf(d1, d2), d2)
    @test d1 ≈ prod(GenericProd(), d2, ext.DivisionOf(d1, d2))
    @test ext.DivisionOf(d1, d2) == prod(GenericProd(), ext.DivisionOf(d1, d2), missing)
    @test ext.DivisionOf(d1, d2) == prod(GenericProd(), missing, ext.DivisionOf(d1, d2))
end

@testitem "create_project_to_ins type stability" begin
    using ExponentialFamily, ExponentialFamilyProjection, BayesBase, Test
    using ReactiveMP: CVIProjection
    using JET

    # `create_project_to_ins` is internal to the extension
    ext = Base.get_extension(ReactiveMP, :ReactiveMPProjectionExt)
    @test !isnothing(ext)
    using .ext

    @testset "Complete type stability tests for create_project_to_ins" begin
        # Test Case 1: Default form (nothing)
        let
            method = CVIProjection(in_prjparams = nothing)
            m_in = NormalMeanVariance(0.0, 1.0)
            k = 1

            @test_opt ext.create_project_to_ins(method, m_in, k)
            result = ext.create_project_to_ins(method, m_in, k)
            @test result isa ProjectedTo{<:NormalMeanVariance}
        end

        # Test Case 2: Custom form specified
        let
            form = ProjectedTo(MvNormalMeanCovariance, 2)
            method = CVIProjection(in_prjparams = (in_1 = form,))
            m_in = NormalMeanVariance(0.0, 1.0)  # Input type different from target
            k = 1

            @test_opt ext.create_project_to_ins(method, m_in, k)
            result = ext.create_project_to_ins(method, m_in, k)
            @test result isa ProjectedTo{<:MvNormalMeanCovariance}
        end

        # Test Case 3: Empty NamedTuple for forms
        let
            method = CVIProjection()
            m_in = Gamma(2.0, 2.0)
            k = 1

            @test_opt ext.create_project_to_ins(method, m_in, k)
            result = ext.create_project_to_ins(method, m_in, k)
            @test result isa ProjectedTo{<:Gamma}
        end

        # Test Case 4: Multiple forms specified
        let
            forms = (in_1 = ProjectedTo(NormalMeanVariance), in_2 = ProjectedTo(MvNormalMeanCovariance))
            method = CVIProjection(in_prjparams = forms)
            m_in = Gamma(2.0, 2.0)

            for k in 1:2
                @test_opt ext.create_project_to_ins(method, m_in, k)
                result = ext.create_project_to_ins(method, m_in, k)

                if k == 1
                    @test result isa ProjectedTo{<:NormalMeanVariance}
                else
                    @test result isa ProjectedTo{<:MvNormalMeanCovariance}
                end
            end
        end

        # Test Case 5: not form but just a gradient descent parameters
        let
            params = DefaultProjectionParameters()
            method = CVIProjection(in_prjparams = (in_1 = params,))
            m_in = Gamma(0.0, 1.0)
            k = 1

            @test_opt ext.create_project_to_ins(method, m_in, k)
            result = ext.create_project_to_ins(method, m_in, k)
            @test result isa ProjectedTo{<:Gamma}
        end
    end
end
