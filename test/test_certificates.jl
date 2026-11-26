# Certificates: export, re-read with an independent JSON parser, and recompute every
# claim from the data in the file with the oracles of helpers.jl.

function check_certificate(text)
    cert = JSON.parse(text)
    @test cert["schema"] == "decision-certificate/1" == CERTIFICATE_SCHEMA
    Q = rational_matrix(cert["payoff"])
    p = rational_vector(cert["prior"])
    @test size(Q) == (length(cert["actions"]), length(cert["states"]))
    @test length(p) == size(Q, 2) && sum(p) == 1
    claims = cert["claims"]
    eu = oracle_expected_utility(Q, p)
    R = oracle_regret(Q)
    @test rational_vector(claims["expected_utility"]) == eu
    @test parse_rational(claims["act_now_value"]) == maximum(eu)
    @test one_based(claims["bayes_actions"]) == argmax_set(eu)
    @test rational_matrix(claims["regret"]) == R
    @test rational_vector(claims["expected_regret"]) == oracle_expected_utility(R, p)
    worst = [maximum(R[a, :]) for a in axes(R, 1)]
    @test rational_vector(claims["max_regret"]) == worst
    @test one_based(claims["minimax_regret_actions"]) == argmin_set(worst)
    @test parse_rational(claims["evpi"]) == oracle_evpi(Q, p)
    if haskey(cert, "likelihood")
        L = rational_matrix(cert["likelihood"])
        @test size(L) == (length(cert["signals"]), size(Q, 2))
        @test parse_rational(claims["value_with_information"]) ==
              oracle_value_with_information(Q, p, L)
        @test parse_rational(claims["evsi"]) == oracle_evsi(Q, p, L)
    else
        @test !haskey(claims, "evsi") && !haskey(cert, "signals")
    end
    return cert
end

@testset "certificates" begin
    examples = load_examples()

    @testset "worked examples round trip" begin
        for ex in examples
            problem = fixture_problem(ex, identity)
            cert = check_certificate(sprint(export_decision_certificate, problem))
            @test cert["actions"] == ex["actions"] && cert["states"] == ex["states"]
            # The data in the certificate are the data of the fixture, string for string.
            @test cert["payoff"] == ex["payoff"] && cert["prior"] == ex["prior"]
            @test cert["claims"]["evpi"] == ex["expected"]["evpi"]
            @test cert["claims"]["regret"] == ex["expected"]["regret"]
            @test !haskey(cert, "baseline")       # the baseline is not part of the schema
            for (name, spec) in ex["experiments"]
                e = fixture_experiment(spec, identity)
                io = IOBuffer()
                @test export_decision_certificate(io, problem; experiment=e) === nothing
                cert = check_certificate(String(take!(io)))
                @test cert["likelihood"] == spec["likelihood"]
                @test cert["signals"] == spec["signals"]
                @test cert["claims"]["evsi"] == ex["expected"]["evsi"][name]
            end
        end
    end

    @testset "encoding" begin
        problem = fixture_problem(first(examples), identity)
        text = sprint(export_decision_certificate, problem)
        # Rationals are strings "n/d" in lowest terms or "n"; indices are bare integers
        # counted from zero.
        @test occursin("\"prior\": [\"1/2\", \"3/10\", \"3/20\", \"1/20\"]", text)
        @test occursin("\"expected_utility\": [\"-39/2\", \"-97/4\", \"-48\"]", text)
        @test occursin("\"bayes_actions\": [0]", text)
        @test occursin("\"minimax_regret_actions\": [1]", text)
        @test occursin("\"evpi\": \"207/20\"", text)
        keys_in_order = [m.captures[1] for m in eachmatch(r"^  \"(\w+)\":"m, text)]
        @test keys_in_order == ["schema", "actions", "states", "payoff", "prior", "claims"]
        # Ties are all listed.
        tied = DecisionProblem(PayoffTable([1 3; 3 1; 2 2]), [1 // 2, 1 // 2])
        cert = check_certificate(sprint(export_decision_certificate, tied))
        @test cert["claims"]["bayes_actions"] == [0, 1, 2]
        # Labels are escaped.
        odd = DecisionProblem(PayoffTable([Symbol("say \"no\""), Symbol("back\\slash")],
                                          [:s], reshape([1, 2], 2, 1)), [1 // 1])
        cert = check_certificate(sprint(export_decision_certificate, odd))
        @test cert["actions"] == ["say \"no\"", "back\\slash"]
        # Integer payoffs with a rational prior are exact and accepted.
        ints = DecisionProblem(PayoffTable([0 -100; -30 -30]), [3 // 5, 2 // 5])
        @test check_certificate(sprint(export_decision_certificate, ints))["claims"]["evpi"] ==
              "18"
        # Only feasible actions are written, and indices refer to them.
        filtered = DecisionProblem(problem.payoff, problem.belief;
                                   feasible=[:raise_levee, :relocate_assets])
        cert = check_certificate(sprint(export_decision_certificate, filtered))
        @test cert["actions"] == ["raise_levee", "relocate_assets"]
        @test cert["claims"]["bayes_actions"] == [0]
        # A known state.
        known = DecisionProblem(problem.payoff, PointBelief(:high))
        @test check_certificate(sprint(export_decision_certificate, known))["claims"]["evpi"] ==
              "0"
    end

    @testset "to a file" begin
        problem = fixture_problem(examples[2], identity)
        e = fixture_experiment(examples[2]["experiments"]["test"], identity)
        mktempdir() do dir
            path = joinpath(dir, "epidemic.json")
            @test export_decision_certificate(path, problem; experiment=e) == path
            cert = check_certificate(read(path, String))
            @test cert["claims"]["evsi"] == "44/5"
        end
    end

    @testset "random problems round trip" begin
        rng = Random.Xoshiro(42)
        for _ in 1:40
            m, n = rand(rng, 1:5), rand(rng, 1:5)
            problem = exact_problem(rng, m, n)
            e = random_experiment(rng, rand(rng, 1:4), n; T=BigRat)
            check_certificate(sprint(export_decision_certificate, problem))
            io = IOBuffer()
            export_decision_certificate(io, problem; experiment=e)
            check_certificate(String(take!(io)))
        end
    end

    @testset "inexact problems are refused" begin
        exact = fixture_problem(first(examples), identity)
        float = fixture_problem(first(examples), Float64)
        @test_throws InexactCertificateError sprint(export_decision_certificate, float)
        mixed = DecisionProblem(exact.payoff, Float64.(probabilities(exact.belief)))
        @test_throws InexactCertificateError sprint(export_decision_certificate, mixed)
        fe = Experiment([0.9 0.6 0.2 0.05; 0.1 0.4 0.8 0.95])
        @test_throws InexactCertificateError export_decision_certificate(IOBuffer(), exact;
                                                                         experiment=fe)
        err = try
            sprint(export_decision_certificate, float)
        catch e
            e
        end
        @test err isa DecisionAnalysisError
        @test occursin("payoff", sprint(showerror, err))
        credal = DecisionProblem(exact.payoff,
                                 CredalSet(exact.payoff.states,
                                           [probabilities(exact.belief)]))
        @test_throws BeliefFormError sprint(export_decision_certificate, credal)
    end
end

# Witnesses: what a checker of "garbling-witness/1" and "e-admissible-witness/1" must
# confirm, recomputed here from the file alone.
function check_garbling_witness(text)
    w = JSON.parse(text)
    @test w["schema"] == "garbling-witness/1" == GARBLING_WITNESS_SCHEMA
    L, Lg, G = rational_matrix.((w["experiment"], w["garbled"], w["garbling"]))
    @test size(L, 2) == size(Lg, 2)
    @test size(G) == (size(Lg, 1), size(L, 1))
    @test all(>=(0), L) && all(==(1), sum(L; dims=1))
    @test all(>=(0), G) && all(==(1), sum(G; dims=1))
    @test G * L == Lg
    return w
end

function check_eadmissible_witness(text)
    w = JSON.parse(text)
    @test w["schema"] == "e-admissible-witness/1" == EADMISSIBLE_WITNESS_SCHEMA
    Q = rational_matrix(w["payoff"])
    P = rational_matrix(w["priors"])
    weights = rational_vector(w["weights"])
    prior = rational_vector(w["prior"])
    a = w["action"] + 1
    @test size(P) == (length(weights), size(Q, 2)) && length(prior) == size(Q, 2)
    @test all(>=(0), P) && all(==(1), sum(P; dims=2))
    @test all(>=(0), weights) && sum(weights) == 1
    @test prior == [sum(weights[k] * P[k, t] for k in eachindex(weights))
                    for t in axes(Q, 2)]
    @test a in argmax_set(oracle_expected_utility(Q, prior))
    return w
end

# Independent checks of the two "no" witnesses, from the definitions with explicit loops.
function check_not_eadmissible_witness(text)
    w = JSON.parse(text)
    @test w["schema"] == "not-e-admissible-witness/1" == NOT_EADMISSIBLE_WITNESS_SCHEMA
    Q = rational_matrix(w["payoff"])
    P = rational_matrix(w["priors"])
    μ = rational_vector(w["mixed_action"])
    a = w["action"] + 1
    @test size(P, 2) == size(Q, 2) && length(μ) == size(Q, 1) && 1 <= a <= size(Q, 1)
    @test all(>=(0), P) && all(==(1), sum(P; dims=2))
    @test all(>=(0), μ) && sum(μ) == 1
    for k in axes(P, 1)
        E = [sum(P[k, t] * Q[b, t] for t in axes(Q, 2)) for b in axes(Q, 1)]
        @test sum(μ[b] * E[b] for b in axes(Q, 1)) > E[a]
    end
    return w
end

function check_not_garbling_witness(text)
    w = JSON.parse(text)
    @test w["schema"] == "not-garbling-witness/1" == NOT_GARBLING_WITNESS_SCHEMA
    L = rational_matrix(w["experiment"])
    Lt = rational_matrix(w["garbled"])
    Q = rational_matrix(w["payoff"])
    p = rational_vector(w["prior"])
    @test size(L, 2) == size(Lt, 2) == size(Q, 2) == length(p)
    @test all(>=(0), L) && all(==(1), sum(L; dims=1))
    @test all(>=(0), Lt) && all(==(1), sum(Lt; dims=1))
    @test all(>(0), p) && sum(p) == 1
    # A garbling is never worth more, so a strictly larger value refutes it.
    @test oracle_value_with_information(Q, p, Lt) > oracle_value_with_information(Q, p, L)
    return w
end

lean_available() = get(ENV, "DECISIONANALYSIS_LEAN", "") == "1"

@testset "witnesses" begin
    examples = load_examples()
    fx = first(examples)
    flood = fixture_problem(fx, identity)
    credal = CredalSet(flood.payoff.states, [rational_vector(q) for q in fx["credal"]])
    robust = DecisionProblem(flood.payoff, credal)
    gauge = fixture_experiment(fx["experiments"]["gauge"], identity)
    garbled = fixture_experiment(fx["experiments"]["garbled_gauge"], identity)

    @testset "garbling witness" begin
        io = IOBuffer()
        @test export_garbling_witness(io, garbled, gauge) === nothing
        w = check_garbling_witness(String(take!(io)))
        @test w["experiment"] == fx["experiments"]["gauge"]["likelihood"]
        @test w["garbled"] == fx["experiments"]["garbled_gauge"]["likelihood"]
        @test w["garbling"] == fx["experiments"]["garbled_gauge"]["garbling"]
        # Not a garbling: no witness, a typed error, and nothing written.
        io = IOBuffer()
        @test_throws WitnessError export_garbling_witness(io, gauge, garbled)
        @test isempty(take!(io))
        fgauge = fixture_experiment(fx["experiments"]["gauge"], Float64)
        @test_throws InexactCertificateError export_garbling_witness(IOBuffer(), garbled,
                                                                     fgauge)
        @test_throws InexactCertificateError export_garbling_witness(IOBuffer(), fgauge,
                                                                     gauge)
        mktempdir() do dir
            path = joinpath(dir, "garbling.json")
            @test export_garbling_witness(path, garbled, gauge) == path
            return check_garbling_witness(read(path, String))
        end
        rng = Random.Xoshiro(64)
        for _ in 1:25
            n, nz = rand(rng, 2:4), rand(rng, 2:3)
            e = random_experiment(rng, nz, n; T=BigRat)
            g = garble(e, random_stochastic(rng, BigRat, rand(rng, 1:3), nz))
            check_garbling_witness(sprint(export_garbling_witness, g, e))
            if is_garbling_of(e, g) === nothing
                @test_throws WitnessError sprint(export_garbling_witness, e, g)
            else
                check_garbling_witness(sprint(export_garbling_witness, e, g))
            end
        end
    end

    @testset "E-admissibility witness" begin
        result = decide(robust, EAdmissible())
        @test result.chosen == [:do_nothing, :raise_levee]
        # The result carries the hull weights that give each witness prior.
        for a in result.chosen
            λ, q = result.details.weights[a], result.details.witnesses[a]
            @test length(λ) == 3 && all(>=(0), λ) && sum(λ) == 1
            @test q == sum(λ[k] * extreme_points(credal)[k] for k in 1:3)
        end
        for (i, a) in enumerate(result.chosen)
            io = IOBuffer()
            @test export_eadmissible_witness(io, robust, a) === nothing
            w = check_eadmissible_witness(String(take!(io)))
            @test w["action"] == i - 1
            @test w["payoff"] == fx["payoff"] && w["priors"] == fx["credal"]
            @test rational_vector(w["weights"]) == result.details.weights[a]
        end
        @test check_eadmissible_witness(sprint(export_eadmissible_witness, robust,
                                               "raise_levee"))["action"] == 1
        io = IOBuffer()
        @test_throws WitnessError export_eadmissible_witness(io, robust, :relocate_assets)
        @test isempty(take!(io))
        @test_throws ArgumentError export_eadmissible_witness(IOBuffer(), robust, :swim)
        frobust = DecisionProblem(fixture_problem(fx, Float64).payoff,
                                  CredalSet(flood.payoff.states,
                                            [Float64.(q) for q in extreme_points(credal)]))
        @test_throws InexactCertificateError export_eadmissible_witness(IOBuffer(), frobust,
                                                                        :do_nothing)
        @test_throws BeliefFormError export_eadmissible_witness(IOBuffer(),
                                                                DecisionProblem(flood.payoff,
                                                                                ScenarioSet([:low])),
                                                                :do_nothing)
        # A single prior is a one-point set; a known state likewise.
        w = check_eadmissible_witness(sprint(export_eadmissible_witness, flood,
                                             :do_nothing))
        @test w["weights"] == ["1"] && w["prior"] == fx["prior"]
        @test_throws WitnessError sprint(export_eadmissible_witness, flood, :raise_levee)
        known = DecisionProblem(flood.payoff, PointBelief(:high))
        check_eadmissible_witness(sprint(export_eadmissible_witness, known, :raise_levee))
        # Only feasible actions are written, and the index refers to them.
        filtered = DecisionProblem(flood.payoff, credal;
                                   feasible=[:raise_levee, :relocate_assets])
        w = check_eadmissible_witness(sprint(export_eadmissible_witness, filtered,
                                             :raise_levee))
        @test w["action"] == 0 && length(w["payoff"]) == 2
        @test_throws ArgumentError sprint(export_eadmissible_witness, filtered, :do_nothing)
        # The interior case: the witness prior is at no generating point.
        hedge = DecisionProblem(PayoffTable([:left, :right, :hedge], [:s1, :s2],
                                            [1//1 0; 0 1; 3//5 3//5]),
                                CredalSet([:s1, :s2],
                                          [[9 // 10, 1 // 10], [1 // 10, 9 // 10]]))
        w = check_eadmissible_witness(sprint(export_eadmissible_witness, hedge, :hedge))
        @test all(x -> 0 < x < 1, rational_vector(w["weights"]))
        mktempdir() do dir
            path = joinpath(dir, "eadmissible.json")
            @test export_eadmissible_witness(path, hedge, :hedge) == path
            return check_eadmissible_witness(read(path, String))
        end
        rng = Random.Xoshiro(65)
        for _ in 1:30
            m, n, K = rand(rng, 2:5), rand(rng, 2:4), rand(rng, 1:4)
            problem = DecisionProblem(random_payoff_table(rng, m, n; T=BigRat),
                                      random_credal_set(rng, n, K; T=BigRat))
            chosen = decide(problem, EAdmissible()).chosen
            for a in feasible_actions(problem)
                if a in chosen
                    check_eadmissible_witness(sprint(export_eadmissible_witness, problem,
                                                     a))
                else
                    @test_throws WitnessError sprint(export_eadmissible_witness, problem, a)
                end
            end
        end
    end

    @testset "accepted by the Lean checker" begin
        proofs = joinpath(dirname(@__DIR__), "proofs")
        built = isdir(joinpath(proofs, ".lake", "build"))
        if !(lean_available() && Sys.which("lake") !== nothing && built)
            @info "Lean certificate check skipped: set DECISIONANALYSIS_LEAN=1, with " *
                  "`lake` on the PATH and `proofs/` built (`cd proofs && lake build`), " *
                  "to run `lake exe check_certificate` on exported files."
        else
            function exit_code(file)
                return run(Cmd(ignorestatus(`lake exe check_certificate $file`);
                               dir=proofs)).exitcode
            end
            mktempdir() do dir
                files = Dict("decision" => joinpath(dir, "flood_gauge.json"),
                             "garbling" => joinpath(dir, "garbling.json"),
                             "eadmissible" => joinpath(dir, "eadmissible.json"))
                export_decision_certificate(files["decision"], flood; experiment=gauge)
                export_garbling_witness(files["garbling"], garbled, gauge)
                export_eadmissible_witness(files["eadmissible"], robust, :raise_levee)
                for (kind, file) in files
                    @testset "$kind accepted" begin
                        @test exit_code(file) == 0
                    end
                end
                # One tampered copy of each must be refused.
                tampered = Dict("decision" =>
                                    ("\"evpi\": \"207/20\"" => "\"evpi\": \"10\""),
                                "garbling" => ("\"4/5\"" => "\"3/5\""),
                                "eadmissible" => ("\"action\": 1" => "\"action\": 2"))
                for (kind, change) in tampered
                    text = read(files[kind], String)
                    @test occursin(first(change), text)
                    bad = joinpath(dir, "tampered_$kind.json")
                    write(bad, replace(text, change))
                    @testset "tampered $kind rejected" begin
                        @test exit_code(bad) != 0
                    end
                end
            end
        end
    end
end
