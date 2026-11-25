# The per-outcome profile, the estimation value and the choice among candidate studies,
# against explicit loops from the definitions (posterior form, as in helpers.jl).

# Posterior of the states after signal z, or nothing when the signal has probability 0.
function oracle_posterior(p, L, z)
    pz = sum(p[t] * L[z, t] for t in axes(L, 2))
    iszero(pz) && return pz, nothing
    return pz, [p[t] * L[z, t] / pz for t in axes(L, 2)]
end

function oracle_variance(p, g)
    m = sum(p[t] * g[t] for t in eachindex(g))
    return sum(p[t] * (g[t] - m)^2 for t in eachindex(g))
end

@testset "profiles, estimation value and optimal design" begin
    @testset "value of information by outcome" begin
        rng = Xoshiro(71)
        for _ in 1:40
            m, n, k = rand(rng, 2:4), rand(rng, 2:5), rand(rng, 2:4)
            problem = exact_problem(rng, m, n)
            Q, p = problem.payoff.Q, problem.belief.probabilities
            e = Experiment(random_stochastic(rng, BigRat, k, n))
            L = e.likelihood
            profile = voi_profile(problem, e)
            prior_scores = oracle_expected_utility(Q, p)
            a0 = first(argmax_set(prior_scores))
            @test profile.act_now_action == action_labels(problem.payoff)[a0]
            total, second, zero_mass = big(0 // 1), big(0 // 1), big(0 // 1)
            for (z, row) in enumerate(profile.outcomes)
                pz, posterior = oracle_posterior(p, L, z)
                @test row.probability == pz
                if posterior === nothing
                    @test iszero(row.value) && row.posterior_value === nothing
                    continue
                end
                scores = oracle_expected_utility(Q, posterior)
                value = maximum(scores) - scores[a0]
                @test row.value == value >= 0
                @test row.posterior_value == maximum(scores)
                @test row.best == action_labels(problem.payoff)[argmax_set(scores)]
                @test row.changes_decision == (value > 0)
                total += pz * value
                second += pz * value^2
                iszero(value) && (zero_mass += pz)
            end
            # The profile averages to the EVSI, exactly.
            @test profile.evsi == total == evsi(problem, e) == oracle_evsi(Q, p, L)
            @test profile.variance == second - total^2
            @test profile.standard_deviation ≈ sqrt(Float64(second - total^2))
            @test profile.probability_zero == zero_mass
            @test (profile.probability_zero == 1) ==
                  !information_can_change_decision(problem, e)
        end
        credal = DecisionProblem(PayoffTable([1 2; 3 0]),
                                 CredalSet([[1 // 2, 1 // 2], [1 // 4, 3 // 4]]))
        @test_throws BeliefFormError voi_profile(credal, Experiment([1 0; 0 1]))
    end

    @testset "expected variance reduction" begin
        rng = Xoshiro(72)
        for _ in 1:40
            n, k = rand(rng, 2:5), rand(rng, 2:4)
            problem = exact_problem(rng, 2, n)
            p = problem.belief.probabilities
            g = BigRat.(rand(rng, -9:9, n))
            e = Experiment(random_stochastic(rng, BigRat, k, n))
            r = expected_variance_reduction(problem, e, g)
            total = oracle_variance(p, g)
            remaining = big(0 // 1)
            for z in 1:k
                pz, posterior = oracle_posterior(p, e.likelihood, z)
                posterior === nothing || (remaining += pz * oracle_variance(posterior, g))
            end
            @test r.prior_variance == total
            @test r.expected_posterior_variance == remaining
            @test r.reduction == total - remaining >= 0
            @test iszero(total) || r.share == (total - remaining) / total
            # A partition is the experiment that reveals the cell; the finest one
            # leaves no variance and the coarsest removes none.
            cells = rand(rng, 1:2, n)
            by_cell = expected_variance_reduction(problem, cells, g)
            @test by_cell.reduction ==
                  expected_variance_reduction(problem,
                                              Experiment(partition_likelihood(cells,
                                                                              BigRat)),
                                              g).reduction
            @test expected_variance_reduction(problem, collect(1:n), g).reduction == total
            @test iszero(expected_variance_reduction(problem, fill(1, n), g).reduction)
            # A garbling never reduces the variance by more.
            G = random_stochastic(rng, BigRat, 2, k)
            @test expected_variance_reduction(problem, garble(e, G), g).reduction <=
                  r.reduction
        end
        problem = exact_problem(rng, 2, 3)
        @test_throws ArgumentError expected_variance_reduction(problem,
                                                               Experiment([1 0 0; 0 1 1]),
                                                               [1, 2])
        @test_throws ArgumentError expected_variance_reduction(problem, [1, 2], [1, 2, 3])

        # From a sample: the normal model, where learning p1 removes Var(p1) = 1 of the
        # variance 5 of p1 − p2, learning p2 removes 4, and learning both removes all.
        normal = voi_normal_sample(Xoshiro(73), 20_000)
        gain = normal.payoff.Q[2, :]
        r1 = expected_variance_reduction(normal, :p1; quantity=gain)
        @test r1.prior_variance ≈ 5 rtol = 0.05
        @test r1.reduction ≈ 1 rtol = 0.08
        @test expected_variance_reduction(normal, :p2; quantity=gain).share ≈ 4 / 5 rtol = 0.05
        @test expected_variance_reduction(normal, (:p1, :p2); quantity=gain).share ≈ 1 rtol = 1e-6
        @test expected_variance_reduction(normal, :p1; quantity=gain,
                                          method=StrongOakley()).reduction ≈ 1 rtol = 0.1
        # A parameter learned exactly has no variance left.
        @test expected_variance_reduction(normal, :p1; quantity=:p1).share ≈ 1 rtol = 1e-3
        @test expected_variance_reduction(normal, :p2; quantity=:p1).share < 0.01
        @test_throws ArgumentError expected_variance_reduction(normal, :p1; quantity=[1.0])
        @test_throws BeliefFormError expected_variance_reduction(problem, :p1;
                                                                 quantity=[1, 2, 3])
    end

    @testset "optimal design" begin
        rng = Xoshiro(74)
        for _ in 1:30
            m, n = rand(rng, 2:4), rand(rng, 2:4)
            problem = exact_problem(rng, m, n)
            Q, p = problem.payoff.Q, problem.belief.probabilities
            costs = BigRat.(rand(rng, 0:6, 3)) .// 4
            population = rand(rng, 1:3)
            designs = [Symbol("d", i) => Experiment(random_stochastic(rng, BigRat,
                                                                      rand(rng, 2:3), n);
                                                    cost=costs[i]) for i in 1:3]
            result = optimal_design(problem, designs; population)
            nets = [population * oracle_evsi(Q, p, e.likelihood) - e.cost
                    for (_, e) in designs]
            @test [d.net for d in result.designs] == nets
            @test [d.evsi for d in result.designs] ==
                  [oracle_evsi(Q, p, e.likelihood) for (_, e) in designs]
            @test [d.net for d in result.designs] ==
                  [enbs(problem, e; population) for (_, e) in designs]
            @test result.best == first(designs[first(argmax_set(nets))])
            @test result.tied == first.(designs[argmax_set(nets)])
            @test result.criterion === (all(iszero, costs) ? :evsi : :net_benefit)
            @test all(d.standard_error === nothing for d in result.designs)
            # Costs given apart override those of the experiments.
            other = BigRat.(rand(rng, 0:6, 3))
            override = optimal_design(problem, designs; costs=other)
            @test [d.cost for d in override.designs] == other
            named = optimal_design(problem, designs;
                                   costs=(; d3=other[3], d1=other[1], d2=other[2]))
            @test [d.net for d in named.designs] == [d.net for d in override.designs]
            # With no costs the most informative design wins: a garbling never does.
            e = last(first(designs))
            free = optimal_design(problem,
                                  (;
                                   garbled=garble(e,
                                                  random_stochastic(rng, BigRat, 2,
                                                                    size(e.likelihood, 1));
                                                  cost=0),
                                   original=Experiment(e.likelihood)))
            @test free.criterion === :evsi
            @test :original in free.tied
        end

        # Sample sizes of one study on the normal model: EVSI of n observations of p1
        # is E max(0, X) − 1 with X ~ N(1, n / (n + 1)), and each observation costs
        # `c` individuals. The best size is found by explicit search on the closed form.
        normal = voi_normal_sample(Xoshiro(75), 40_000)
        sizes = [1, 2, 5, 10, 20, 50, 100]
        study = GaussianApproximation(:p1; n0=1, n=sizes,
                                      regression=SplineRegression(; knots=0, degree=1))
        c = 0.002
        result = optimal_design(normal, study; costs=n -> c * n)
        truth = [voi_normal_value(sqrt(n / (n + 1))) - c * n for n in sizes]
        @test [d.name for d in result.designs] == [Symbol("n = ", n) for n in sizes]
        @test result.best == Symbol("n = ", sizes[argmax(truth)])
        @test result.criterion === :net_benefit
        for (d, t, n) in zip(result.designs, truth, sizes)
            @test d.cost == c * n
            @test abs(d.net - t) < 4 * d.net_standard_error
            @test d.net ≈ d.evsi - d.cost
        end
        curve = evsi(normal, study)
        @test [d.evsi for d in result.designs] == [e.value for e in curve]
        # Without costs the largest study wins.
        @test optimal_design(normal, study).best == Symbol("n = 100")
        # A named collection of studies, with costs as a vector.
        xbar = normal.belief.draws.p1 .+ randn(Xoshiro(76), 40_000) ./ sqrt(10)
        mixed = optimal_design(normal,
                               (; summary=SummaryStatistics((; xbar)),
                                approximation=GaussianApproximation(:p1; n0=1, n=10));
                               costs=[0.01, 0.02], population=2)
        @test mixed.designs[1].net ≈ 2 * mixed.designs[1].evsi - 0.01
        @test mixed.designs[1].net_standard_error ≈ 2 * mixed.designs[1].standard_error
        @test abs(mixed.designs[1].evsi - voi_normal_value(sqrt(10 / 11))) <
              4 * mixed.designs[1].standard_error

        small = DecisionProblem(PayoffTable([0 -100; -30 -30]), [3 // 5, 2 // 5])
        e = Experiment([4//5 1//5; 1//5 4//5])
        @test optimal_design(small, e).best === :design
        @test_throws ArgumentError optimal_design(small, (;))
        @test_throws ArgumentError optimal_design(small, (; e); population=-1)
        @test_throws ArgumentError optimal_design(small, (; e); costs=[1, 2])
        @test_throws ArgumentError optimal_design(small, (; e); costs=(; f=1))
        @test_throws ArgumentError optimal_design(small, (; e); costs=[Inf])
        @test_throws ArgumentError optimal_design(normal, (; a=study))
    end

    @testset "optional report sections" begin
        problem = DecisionProblem(PayoffTable([:wait, :act], [:low, :high],
                                              [0 -100; -30 -30]), [3 // 5, 2 // 5];
                                  baseline=:wait)
        gauge = Experiment([4//5 1//5; 1//5 4//5])
        plain = DecisionReport(problem; experiments=(; gauge))
        @test isempty(plain.profiles) && isempty(plain.estimation)
        @test !occursin("by outcome", sprint(show, MIME"text/plain"(), plain))
        report = DecisionReport(problem; experiments=(; gauge),
                                partitions=(; state=[:low, :high]), profile=true,
                                quantities=(; flow=[10, 60]))
        @test only(report.profiles).name === :gauge
        @test only(report.profiles).profile == voi_profile(problem, gauge)
        @test [(e.source, e.quantity) for e in report.estimation] ==
              [(:gauge, :flow), (:state, :flow)]
        @test report.estimation[1].reduction ==
              expected_variance_reduction(problem, gauge, [10, 60]).reduction
        @test report.estimation[2].share == 1
        for mime in (MIME"text/plain"(), MIME"text/markdown"(), MIME"text/html"())
            text = sprint(show, mime, report)
            @test occursin("by outcome", text) && occursin("variance reduction", text)
        end
        @test all(length(line) <= 80
                  for line in split(sprint(show, MIME"text/plain"(), report), '\n'))
        # Nothing else in the report changes.
        @test [r.chosen for r in report.results] == [r.chosen for r in plain.results]
        @test report.evpi == plain.evpi && report.experiments == plain.experiments
        @test_throws ArgumentError DecisionReport(problem; profile=true)
        scenario = DecisionProblem(problem.payoff, ScenarioSet([:low, :high]))
        @test_throws BeliefFormError DecisionReport(scenario; profile=true)
    end
end

# The value of waiting to learn (Lean: Finite/Waiting.lean).
@testset "value_of_waiting" begin
    states, actions = [:mild, :severe], [:none, :cull]
    now = PayoffTable(actions, states, [0 -100; -30 -30])
    late = PayoffTable(actions, states, [0 -100; -40 -55])
    problem = DecisionProblem(now, [3 // 5, 2 // 5]; baseline=:none)
    perfect = Experiment([1 0; 0 1]; signals=[:mild, :severe])

    @testset "a worked example" begin
        w = value_of_waiting(problem, late, perfect)
        @test w.act_now_value == -30 && w.act_now_action == :cull
        @test w.waiting_value == 3 // 5 * 0 + 2 // 5 * -55
        @test w.value == 8 && w.wait
        @test w.delay_cost == 10 && w.late_action == :none
        @test w.evsi_delayed == 18 && w.evsi_now == evsi(problem, perfect)
        @test w.policy == [:mild => :none, :severe => :cull]
        # Waiting costs nothing: the value of waiting is the EVSI.
        same = value_of_waiting(problem, now, perfect)
        @test same.value == same.evsi_now == same.evsi_delayed
        @test same.delay_cost == 0
        # Nothing is learned: waiting is worth minus the cost of the delay
        # (valueOfWaiting_constant).
        nothing_learned = value_of_waiting(problem, late, Experiment([1 1]))
        @test nothing_learned.value == -10 && !nothing_learned.wait
        @test nothing_learned.evsi_delayed == 0
        # A value of exactly zero is not a reason to wait.
        level = PayoffTable(actions, states, [0 -100; -40 -75])
        @test value_of_waiting(problem, level, perfect).value == 0
        @test !value_of_waiting(problem, level, perfect).wait
    end

    @testset "the instance proved in Lean" begin
        # valueOfWaiting_neg_example: EVSI 2 on the payoffs of acting now, and waiting
        # worth −1.
        p = DecisionProblem(PayoffTable([4 0; 0 4]), [1 // 2, 1 // 2])
        w = value_of_waiting(p, PayoffTable([1 -3; -3 1]), Experiment([1 0; 0 1]))
        @test w.evsi_now == 2 && w.value == -1 && !w.wait
    end

    @testset "against the definitions" begin
        rng = Random.Xoshiro(20261128)
        waits, above = 0, 0
        for trial in 1:200
            m, n, nz = rand(rng, 2:4), rand(rng, 2:4), rand(rng, 1:3)
            base = exact_problem(rng, m, n)
            p = probabilities(base.belief)
            L = random_stochastic(rng, BigRat, nz, n)
            e = Experiment(L)
            # Late payoffs: lower everywhere on even trials, unrelated on odd ones.
            Q = base.payoff.Q
            Qlate = iseven(trial) ? Q .- BigRat.(rand(rng, 0:6, m, n)) :
                    random_payoff_table(rng, m, n; T=BigRat).Q
            delayed = PayoffTable(base.payoff.actions, base.payoff.states, Qlate)
            w = value_of_waiting(base, delayed, e)
            @test w.act_now_value == maximum(oracle_expected_utility(Q, p))
            @test w.waiting_value == oracle_value_with_information(Qlate, p, L)
            @test w.value == w.waiting_value - w.act_now_value
            # valueOfWaiting_eq and valueOfWaiting_pos_iff.
            @test w.value == w.evsi_delayed - w.delay_cost
            @test w.wait == (w.evsi_delayed > w.delay_cost)
            @test w.evsi_delayed == oracle_evsi(Qlate, p, L) >= 0
            @test w.evsi_now == oracle_evsi(Q, p, L)
            @test w.delay_cost ==
                  w.act_now_value - maximum(oracle_expected_utility(Qlate, p))
            # valueOfWaiting_le_evsi, and that it needs its hypothesis.
            iseven(trial) && @test w.value <= w.evsi_now
            above += w.value > w.evsi_now
            waits += w.wait
            for (signal, action) in w.policy
                z = findfirst(==(signal), e.signals)
                scores = [sum(p[t] * L[z, t] * Qlate[a, t] for t in 1:n) for a in 1:m]
                @test scores[findfirst(==(action), base.payoff.actions)] == maximum(scores)
            end
            @test length(w.policy) == count(z -> sum(p .* L[z, :]) > 0, 1:nz)
        end
        @test 0 < waits < 200
        @test above > 0
    end

    @testset "feasibility" begin
        three = PayoffTable([:none, :cull, :vaccinate], states, [0 -100; -30 -30; -5 -20])
        three_late = PayoffTable([:none, :cull, :vaccinate], states,
                                 [0 -100; -40 -55; -5 -25])
        restricted = DecisionProblem(three, [3 // 5, 2 // 5]; feasible=[:none, :cull])
        w = value_of_waiting(restricted, three_late, perfect)
        @test w.value == value_of_waiting(problem, late, perfect).value
        @test all(a != :vaccinate for (_, a) in w.policy)
    end

    @testset "several lengths of wait" begin
        longer = PayoffTable(actions, states, [0 -100; -50 -80])
        noisy = Experiment([3//4 1//4; 1//4 3//4])
        r = value_of_waiting(problem,
                             [:week_noisy => (late, noisy),
                              :week => (late, perfect),
                              :month => (longer, perfect)])
        @test r.options == [:week_noisy, :week, :month]
        @test r.values == [-5 // 2, 8, -2]
        @test r.best == :week
        @test r.act_now_value == -30 && r.act_now_action == :cull
        @test r.results[2] == value_of_waiting(problem, late, perfect)
        @test value_of_waiting(problem, (; month=(longer, perfect))).best == :act_now
        @test value_of_waiting(problem, Dict(:week => (late, perfect))).best == :week
    end

    @testset "what is refused" begin
        @test_throws LabelMismatchError value_of_waiting(problem,
                                                         PayoffTable([:none, :treat],
                                                                     states, late.Q),
                                                         perfect)
        @test_throws LabelMismatchError value_of_waiting(problem,
                                                         PayoffTable(actions, [:a, :b],
                                                                     late.Q), perfect)
        @test_throws LabelMismatchError value_of_waiting(problem, late,
                                                         Experiment([1 0; 0 1];
                                                                    states=[:a, :b]))
        credal = DecisionProblem(now, CredalSet(states, [[0.6, 0.4], [0.5, 0.5]]))
        @test_throws BeliefFormError value_of_waiting(credal, late, perfect)
        @test_throws ArgumentError value_of_waiting(problem, Pair{Symbol,Any}[])
        @test_throws ArgumentError value_of_waiting(problem, [:week => late])
    end
end
