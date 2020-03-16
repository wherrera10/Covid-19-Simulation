using DataFrames
using Distances
using Distributions
using LinearAlgebra
using Pathogen

function testcovid(n)
    n = 30_000
    risks = DataFrame(x = rand(Uniform(0, 15), n), y = rand(Uniform(0, 30), n), riskfactor1 = rand(Gamma(), n))
    
    # Precalculate Euclidean distances between individuals in a Population
    dists = [euclidean([risks[i, :x];
    risks[i, :y]],
    [risks[j, :x];
    risks[j, :y]]) for i = 1:n, j = 1:n]
    pop = Population(risks, dists)
    
    _constant(params::Vector{Float64}, pop::Population, i::Int64) = params[1]
    _one(params::Vector{Float64}, pop::Population, i::Int64) = 1.0
    _linear(params::Vector{Float64}, pop, i::Int64) = params[1] * pop.risks[i, :riskfactor1]
    
    function _powerlaw(params::Vector{Float64}, pop::Population, i::Int64, k::Int64)
        beta = params[1]
        d = pop.distances[k, i]
        return d^(-beta)
    end

    rf = RiskFunctions{SIR}(
        _constant, # sparks function
        _one, # susceptibility function
        _powerlaw, # infectivity kernel
        _one, # transmissibility function
        _linear) # removal function
    
    rparams = RiskParameters{SIR}(
        [0.0001], # sparks
        Float64[], # susceptibility
        [4.0], # infectivity
        Float64[], # transmissibility
        [0.1]) # removal
        
    starting_states = append!([State_I], fill(State_S, n-1))
    sim = Simulation(pop, starting_states, rf, rparams)
    simulate!(sim, tmax=200.0)

    p1 = plot(sim.events)
    p2 = plot(sim.transmission_network, sim.population, sim.events, 0.0, title="Time = 0")
    p6 = plot(sim.transmission_network, sim.population, sim.events, 100.0, title="Time = 200")
    
    
    l = @layout [a; b c d e f]
    plot(p1, p2, p3, p4, p5, p6, layout=l)
    
    obs = observe(sim, Uniform(0.5, 2.5), Uniform(0.5, 2.5), force=true)
    rpriors = RiskPriors{SIR}([Exponential(0.0001)], UnivariateDistribution[],
        [Uniform(1.0, 7.0)], UnivariateDistribution[], [Uniform(0.0, 1.0)])

    ee = EventExtents{SIR}(5.0, 5.0)
    mcmc = MCMC(obs, ee, pop, rf, rpriors)
    start!(mcmc, attempts=50000)
    
    iterate!(mcmc, 50000, 1.0, condition_on_network=true, event_batches=5)
    p1 = plot(1:20:50001, mcmc.markov_chains[1].risk_parameters, 
        yscale = :log10, title="TN-ILM parameters")
        
    p2 = plot(mcmc.markov_chains[1].events[10000], State_S, linealpha=0.01, title="S")
    
    for i=10020:20:50000
        plot!(p2, mcmc.markov_chains[1].events[i], State_S, linealpha=0.01)
    end
    plot!(p2, sim.events, State_S, linecolor=:black)
    l = @layout [a; [b c d]]
    plot(p1, p2, p3, p4, layout=l)
    
    p1 = plot(sim.transmission_network, sim.population, 
        title="True Transmission\nNetwork", framestyle=:box)
    tnp = TNPosterior(mcmc.markov_chains[1].transmission_network[10000:20:50000])
    p2 = plot(tnp, sim.population,
        title="Transmission Network\nPosterior Distribution", framestyle=:box)
    plot(p1, p2, layout=(1, 2))
    
    tracedata = convert(Array{Float64, 2}, mcmc.markov_chains[1].risk_parameters)
    tracesummary = vcat(mean(tracedata[10000:20:50000, :], dims=1),
        [quantile(tracedata[10000:20:50000, i], j) for j = [0.025, 0.975], i = 1:3])
    println("\nTracesummary:\n", tracesummary)
end

testcovid(30_000) # Kauai

testcovid(200_000) # Big Island


