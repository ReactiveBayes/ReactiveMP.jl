
@testset "marginalrules:ContinuousTransition" begin
    using Test, ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions, LinearAlgebra
    import ReactiveMP: @test_marginalrules

    rng = MersenneTwister(42)
    @testset "Linear transformation" begin
        # the following rule is used for testing purposes only
        # It is derived separately by Thijs van de Laar
        function benchmark_rule(m_x, m_y, q_W, q_A)
            xix, Wx = weightedmean_precision(m_x)
            xiy, Wy = weightedmean_precision(m_y)

            mW = mean(q_W)
            mA, ΣA, UA = q_A.M, q_A.U, q_A.V

            U = Wx + tr(mW * ΣA) * UA

            Wq = [Wy+mW -mW*mA; -mA'*mW U+mA' * mW * mA]
            return MvNormalWeightedMeanPrecision([xiy; xix], Wq)
        end
        @testset "y_x: (m_y::NormalDistributionsFamily, m_x::NormalDistributionsFamily, q_a::NormalDistributionsFamily, q_W::Any)" begin
            for (dy, dx) in [(1, 3), (2, 3), (3, 2), (2, 2)]
                transformation = (a) -> reshape(a, dy, dx)

                mA, ΣA, UA = rand(rng, dy, dx), diageye(dy), diageye(dx)
                qA = MatrixNormal(mA, ΣA, UA)

                Lx, Ly = rand(rng, dx, dx), rand(rng, dy, dy)
                μx, Σx = rand(rng, dx), Lx * Lx' + dx * I
                μy, Σy = rand(rng, dy), Ly * Ly' + dy * I

                my = MvNormalMeanCovariance(μy, Σy)
                mx = MvNormalMeanCovariance(μx, Σx)
                qa = MvNormalMeanCovariance(vec(mA), kron(UA, ΣA))
                qW = Wishart(dy + 1, diageye(dy))

                metal = CTMeta(transformation)

                @test_marginalrules [check_type_promotion = true, atol = 1e-5] ContinuousTransition(:y_x) [(
                    input = (m_y = my, m_x = mx, q_a = qa, q_W = qW, meta = metal), output = benchmark_rule(mx, my, qW, qA)
                )]
            end
        end
    end

    @testset "Nonlinear transformation" begin
        @testset "y_x: (m_y::NormalDistributionsFamily, m_x::NormalDistributionsFamily, q_a::NormalDistributionsFamily, q_W::Any)" begin
            dy, dx = 2, 2
            transformation = (a) -> [cos(a[1]) -sin(a[1]); sin(a[1]) cos(a[1])]

            μx, Σx = zeros(dx), diageye(dx)
            μy, Σy = zeros(dy), diageye(dy)

            my = MvNormalMeanCovariance(μy, Σy)
            mx = MvNormalMeanCovariance(μx, Σx)
            qa = MvNormalMeanCovariance(zeros(1), tiny * diageye(1))
            qW = Wishart(dy, diageye(dy))

            metanl = CTMeta(transformation)

            @test_marginalrules [check_type_promotion = true, atol = 1e-5] ContinuousTransition(:y_x) [(
                input = (m_y = my, m_x = mx, q_a = qa, q_W = qW, meta = metanl),
                output = MvNormalWeightedMeanPrecision(zeros(4), [(dy + qW.df - 1)*diageye(dy) -(qW.df)diageye(dx); -(qW.df)diageye(dx) (dy + qW.df - 1)diageye(dy)])
            )]
        end
    end
end
