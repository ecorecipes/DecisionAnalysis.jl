# Writes test/fixtures/voi/normal_sample.csv, the sample that `voi_reference.R` and the tests share: 2000
# draws of p1 ~ N(1, 1), p2 ~ N(0, 2²) and x3, ..., x6 ~ N(0, 1), independent, from
# Xoshiro(20261123). Run once (`julia validation/make_sample.jl`); the file is
# committed so that neither side depends on a random number generator's stream.
using Random
rng = Xoshiro(20261123)
n = 2000
columns = [1 .+ randn(rng, n), 2 .* randn(rng, n), randn(rng, n), randn(rng, n),
           randn(rng, n), randn(rng, n)]
open(joinpath(@__DIR__, "..", "test", "fixtures", "voi", "normal_sample.csv"), "w") do io
    println(io, "p1,p2,x3,x4,x5,x6")
    for i in 1:n
        println(io, join((repr(c[i]) for c in columns), ","))
    end
end
