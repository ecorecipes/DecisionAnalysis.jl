# The built-in exact simplex against brute-force vertex enumeration.
using LinearAlgebra: rank

# All basic feasible solutions of {Ax = b, x >= 0}, by solving every square subsystem.
function oracle_lp_minimum(c, A, b)
    m, n = size(A)
    best = nothing
    for mask in 0:(2 ^ n - 1)
        cols = [j for j in 1:n if (mask >> (j - 1)) & 1 == 1]
        length(cols) == m || continue
        B = A[:, cols]
        rank(Float64.(B)) == m || continue
        xB = B \ b
        all(>=(0), xB) || continue
        value = sum(c[cols] .* xB)
        best = best === nothing ? value : min(best, value)
    end
    return best
end

@testset "linear programmes" begin
    solve = DecisionAnalysis._solve_lp
    R = BigRat

    @testset "textbook cases" begin
        # maximise 3x + 2y s.t. x + y <= 4, x + 3y <= 6, x <= 3: optimum 11 at (3, 1).
        sol = solve([-3, -2], zeros(Int, 0, 2), Int[], [1 1; 1 3; 1 0], [4, 6, 3])
        @test sol.status === :optimal
        @test sol.objective == -11
        @test sol.x == [3, 1]
        # Infeasible: x <= -1 with x >= 0.
        @test solve([1], zeros(Int, 0, 1), Int[], reshape([1], 1, 1), [-1]).status ===
              :infeasible
        # Unbounded: minimise -x with no upper bound.
        @test solve([-1], zeros(Int, 0, 1), Int[], reshape([-1], 1, 1), [0]).status ===
              :unbounded
        # A free variable: minimise t s.t. t >= -5.
        sol = solve([1], zeros(Int, 0, 1), Int[], reshape([-1], 1, 1), [5]; free=[1])
        @test sol.status === :optimal && sol.x == [-5]
        # Redundant equality rows are dropped, not reported infeasible.
        sol = solve([1, 2], [1 1; 2 2], [1, 2], zeros(Int, 0, 2), Int[])
        @test sol.status === :optimal && sol.objective == 1
        # Floating-point data are solved as the exact rationals they are.
        sol = solve([-1.0, -1.0], zeros(0, 2), Float64[], [1.0 2.0; 3.0 1.0], [4.0, 6.0])
        @test sol.objective == -R(14, 5)
    end

    @testset "random programmes against vertex enumeration" begin
        rng = Random.Xoshiro(20261123)
        done = 0
        while done < 60
            m, n = rand(rng, 1:3), rand(rng, 2:5)
            A = R.(rand(rng, -4:4, m, n))
            # A known feasible point makes the programme feasible; the extra row
            # Σx + slack = bound makes it bounded.
            x0 = R.(rand(rng, 0:3, n))
            b = A * x0
            bound = sum(x0) + rand(rng, 1:5)
            Astd = [A zeros(R, m); ones(R, 1, n) one(R)]
            bstd = vcat(b, bound)
            # The oracle enumerates square bases, so it needs independent rows; the
            # redundant-row case is covered above.
            rank(Float64.(Astd)) == m + 1 || continue
            done += 1
            c = R.(rand(rng, -5:5, n))
            sol = solve(c, A, b, ones(R, 1, n), [bound])
            @test sol.status === :optimal
            @test sol.objective == oracle_lp_minimum(vcat(c, 0), Astd, bstd)
            @test all(>=(0), sol.x)
            @test A * sol.x == b
            @test sum(sol.x) <= bound
            @test sum(c .* sol.x) == sol.objective
        end
    end
end
