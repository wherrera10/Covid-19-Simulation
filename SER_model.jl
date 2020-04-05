
# Hawaii County 2019
const μBI = (1480 / 200,983) / 365

const sigma, gamma = 0.25, 0.15

function seir_ode(dY, Y, p, t)
    β, σ, γ, μ = p[1], p[2], p[3], p[4]
    S, E, I = Y[1], Y[2], Y[3]
    dY[1] = μ * (1 - S) - β * S * I
    dY[2] = β * S * I - (σ + μ) * E
    dY[3] = σ * E - (γ + μ) * I
end

function seir(β, σ, γ, μ, tspan=(0.0, 200.0); S₀=0.99, E₀=0, I₀=0.01, R₀=0)
    par = [β, σ, γ, μ]
    init = [S₀, E₀, I₀]
    seir_prob = ODEProblem(seir_ode, init, tspan, par)
    sol = solve(seir_prob);
    R = ones(1, size(sol)[2]) - sum(sol, dims=1);
    hcat(sol.t, sol[1, :], sol[2, :], sol[3, :], R')
end

r0(β, σ, γ, μ) = (σ / (σ + μ)) * (β / (γ + μ))

plot(seir(0.4, sigma, gamma, μBI))
