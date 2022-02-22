"""
6.338 HW 2 -- Jeremiah DeGreeff
The previous submission was in a Pluto notebook, but multi-process support in that environment seems limited.
This submission is mostly code, so I decided that a jl file is probably more useful to the grader than a pdf.
"""

using Base.Threads

using BenchmarkTools
using Distributed
using LaTeXStrings
using Plots
using SharedArrays
using Test

if nworkers() == 1
    addprocs(6)
end

function printpart(n)
    println()
    println("----------")
    println("  PART $(n)  ")
    println("----------")
    println()
end

"""
###########################################
Problem 3: Bifurcating Data for Parallelism
###########################################
"""

""" Part 1 (from previous homework) """

printpart(1)

@everywhere function calc_attractor!(out, f, x₀; warmup = 400)
    x = x₀
    for _ in 1:warmup
        x = f(x)
    end
    for i in 1:length(out)
        x = f(x)
        out[i] = x
    end
end

@everywhere logistic(r) = x -> r * x * (1 - x)

out = zeros(150)
r = 2.9
x₀ = 0.25
calc_attractor!(out, logistic(r), x₀)

@test last(out) ≈ (r - 1) / r

""" Part 2 (from previous homework) """

printpart(2)

function calculate_bifurcation(rs, x₀, num_attract)
    data = Matrix{Float64}(undef, num_attract, length(rs))
    for (out, r) ∈ zip(eachcol(data), rs)
        calc_attractor!(out, logistic(r), x₀)
    end
    data
end

function plot_bifurcation(rs, data)
    # looks especially nice in Pluto dark mode
    scatter(
        fill(1, size(data, 1)) * collect(rs)',
        data,
        xlab = L"$r$",
        ylab = L"$x^*$",
        legend = false,
        bg = RGB(0x1F / 0xFF, 0x1F / 0xFF, 0x1F / 0xFF),
        foreground_color = :white,
        markersize = 1,
        markeralpha = 0.2,
        markercolor = :white,
        markerstrokewidth = 0
    )
end

rs = 2.4:0.001:4
num_attract = 500
data = calculate_bifurcation(rs, x₀, num_attract)
display(plot_bifurcation(rs, data))

""" Part 3 """

printpart(3)

function benchmark_bifurcation(f)
    println("\nBenchmarking ", Symbol(f), ":")
    b = @benchmark $f(rs, x₀, num_attract)
    display(b)
    time(median(b)) / 1e9
end

@show nthreads() # 6

function calculate_bifurcation_threads(rs, x₀, num_attract)
    data = Matrix{Float64}(undef, num_attract, length(rs))
    @threads for i ∈ 1:length(rs)
        calc_attractor!((@view data[:, i]), logistic(rs[i]), x₀)
    end
    data
end

@test calculate_bifurcation(rs, x₀, num_attract) == calculate_bifurcation_threads(rs, x₀, num_attract)

single_t = benchmark_bifurcation(calculate_bifurcation) # median: 3.2 ms
threads_t = benchmark_bifurcation(calculate_bifurcation_threads) # median: 0.79 ms
@show single_t / threads_t # speedup = 4.0

"""
On my laptop with 6 threads available to Julia, I observed a median time of 3.6 ms for the single-threaded implementation and 0.85 ms for the multi-threaded implementation.
This is a speedup of about 4.2x.
"""

""" Part 4 """

printpart(4)

@show length(workers()) # 6

@everywhere function calc_attractor(r, x₀, num_attract; num_warmup = 400)
    out = Vector{Float64}(undef, num_attract)
    calc_attractor!(out, logistic(r), x₀; warmup = num_warmup)
    out
end

function calculate_bifurcation_pmap(rs, x₀, num_attract)
    hcat(pmap(rs) do r
        calc_attractor(r, x₀, num_attract)
    end...)
end

@test calculate_bifurcation(rs, x₀, num_attract) == calculate_bifurcation_pmap(rs, x₀, num_attract)

pmap_t = benchmark_bifurcation(calculate_bifurcation_pmap) # median: 190 ms
@show single_t / pmap_t # speedup = 0.017


"""
On my laptop with 6 worker processes, I observed a median time of 190 ms for the pmap-hcat implementation.
This is about 60x slower than the single-threaded implementation.
"""

function calculate_bifurcation_distributed(rs, x₀, num_attract)
    @distributed hcat for r ∈ rs
        calc_attractor(r, x₀, num_attract)
    end
end

@test calculate_bifurcation(rs, x₀, num_attract) == calculate_bifurcation_distributed(rs, x₀, num_attract)

distributed_t = benchmark_bifurcation(calculate_bifurcation_distributed) # median: 120 ms
@show single_t / distributed_t # speedup = 0.026

"""
On my laptop with 6 worker processes, I observed a median time of 120 ms for the distributed-hcat implementation.
This is about 38x slower than the single-threaded implementation.
"""

function calculate_bifurcation_distributed_shared(rs, x₀, num_attract)
    data = SharedMatrix{Float64}((num_attract, length(rs)))
    @sync @distributed for i ∈ 1:length(rs)
        calc_attractor!((@view data[:, i]), logistic(rs[i]), x₀)
    end
    data
end

@test calculate_bifurcation(rs, x₀, num_attract) == calculate_bifurcation_distributed_shared(rs, x₀, num_attract)

distributed_shared_t = benchmark_bifurcation(calculate_bifurcation_distributed_shared) # median: 6.258
@show single_t / distributed_shared_t # speedup = 0.50

"""
On my laptop with 6 worker processes, I observed a median time of 120 ms for the distributed-hcat implementation.
This is about 2x slower than the single-threaded implementation.
"""

""" Part 5 """

printpart(5)

"""
The fastest method for this task was thread-based parallelism by a significant margin.
The individual calc_attractor! tasks are cheap, so the overhead to transfer and sync memory between processes dominates in all three distributed implmentations that I tried.
On the other hand, the threads are able to operate on the shared data directly, so the overhead in this implementation is much less significant.
"""
