# Problem Set 0 - ECON8853
# Federico Rodari

root = dirname(@__FILE__)
cd(root)
# activate project
using LinearAlgebra
using Random, Distributions
using Plots
using Optim
using Statistics
using LaTeXStrings

ps_n = "ps0"
figures_dir = string(root, "\\figures")


# Matlab linspace function
#function linspace(z_start::Real, z_end::Real, z_n::Int64)
#   return collect(range(z_start,stop=z_end,length=z_n))
#end

# OLS function

function OLS(X, y)
    # both arguments must be matrices, vector
    return inv(X'X) * X'y
end

Random.seed!(123)

# 1.1)
# Initialize parameter vector
β1, β2, δ, μ, σ_D, σ_S = 0.0, 2.0, 0.2, 0.0, 1.0, 1.0;

# Random draws
T = 50
# Number of simulations
S = 100

# ------------------------------------------------------------------------------
# SEM-ENDOGENEITY ISSUE
# ------------------------------------------------------------------------------

ϵ_D = rand(Normal(0, σ_D), T);
a = exp.(rand(Normal(0, σ_S), T));

# Generate log prices
log_p = (1 + δ * β2)^(-1) * (δ * β1 .+ μ .- log.(a) + δ * ϵ_D);
# Generate log quantities
log_q = β1 .- β2 * log_p + ϵ_D

histogram(exp.(log_p),
    xlabel=L"P_t",
    ylabel=L"\textrm{Density}",
    label="")
#save plot
savefig(string(figures_dir, "\\", ps_n, "_", "hist_p.pdf"))

histogram(exp.(log_q),
    xlabel=L"Q_t",
    ylabel=L"\textrm{Density}",
    label="")
#save plot
savefig(string(figures_dir, "\\", ps_n, "_", "hist_q.pdf"))

# Estimate OLS
β_ols = OLS([ones(T, 1) log_p], log_q)


# --------------------------------------------------------------------------------
# 2SLS ESTIMATION
# --------------------------------------------------------------------------------

# Initialize new parameters
σ_Z, γ = 1.0, 0.8

# --------------------------------------------------------------------------------
# ESTIMATION

# Simulate series for empirical moments

# Generate log_z
log_z = rand(Normal(0, σ_Z), T);
# Generate log_a
log_a = γ * log_z + rand(Normal(0, σ_S - γ^2 * σ_Z), T);
# Generate log prices
log_p = (1 + δ * β2)^(-1) * ((δ * β1 + μ) .- (log_a + δ * ϵ_D));
# Generate log quantities
log_q = β1 .- (β2 * log_p + ϵ_D)

# Empirical moment
E_g_emp = [dot(log_z, log_q) / T, dot(log_z, log_p) / T]


# Estimate 2SLS --------------------------------------

# First Stage
α0, α1 = OLS([ones(T, 1) log_z], log_p)
# Store fitted values
pi_hat = α0 * ones(T, 1) + α1 * log_z
# Second Stage
β_2SLS = OLS([ones(T, 1) pi_hat], log_q)


# --------------------------------------------------------------------------------
# CONSISTENCY

iter_grid = 100:100:100100;
vec_OLS = zeros(2, length(iter_grid));
vec_2SLS = zeros(2, length(iter_grid));


for t in eachindex(iter_grid)

    tmp = iter_grid[t]
    if iter_grid[t] % 10000 == 0
        print("Iteration $tmp...\n")
    end

    # Generate log_z
    tmp_log_z = rand(Normal(0, σ_Z), iter_grid[t])
    # Generate ϵ_D
    tmp_ϵ_D = rand(Normal(0, σ_D), iter_grid[t])
    # Generate log_a
    tmp_log_a = γ * tmp_log_z + rand(Normal(0, σ_S - γ^2 * σ_Z), iter_grid[t])
    # Generate log prices
    tmp_log_p = (1 + δ * β2)^(-1) * ((δ * β1 + μ) .- (tmp_log_a + δ * tmp_ϵ_D))
    # Generate log quantities
    tmp_log_q = β1 .- (β2 * tmp_log_p + tmp_ϵ_D)

    # Estimate OLS
    vec_OLS[:, t] = OLS([ones(iter_grid[t], 1) tmp_log_p], tmp_log_q)

    # Estimate 2SLS
    # First Stage
    tmp_α0, tmp_α1 = OLS([ones(iter_grid[t], 1) tmp_log_z], tmp_log_p)
    # Store fitted values
    tmp_pi_hat = tmp_α0 * ones(iter_grid[t], 1) + tmp_α1 * tmp_log_z
    # Second Stage
    vec_2SLS[:, t] = OLS([ones(iter_grid[t], 1) tmp_pi_hat], tmp_log_q)

end

plot(iter_grid, vec_OLS[2, :], label="OLS", xlabel=L"\textrm{T}", ylabel=L"\beta_2")
plot!(iter_grid, vec_2SLS[2, :], label="2SLS")
#save plot
savefig(string(figures_dir, "\\", ps_n, "_", "ols_2sls_consistency.pdf"))

# ------------------------------------------------------------------------------
# SIMULATED METHOD OF MOMENTS
# ------------------------------------------------------------------------------


# Shock simulations (needs to be fixed for each β2 candidate
#                    to not alter the objective function)
function simulate_shocks(g=γ, sZ=σ_Z, sS=σ_S, sD=σ_S, nobs=T, nsim=S)

    ϵ_D = zeros(nobs, nsim)
    a_u = zeros(nobs, nsim)
    log_z = zeros(nobs, nsim)

    # Simulate the model S times
    for s in 1:nsim
        # Random i.i.d shocks draws
        ϵ_D[:, s] = rand(Normal(0, sD), nobs)
        a_u[:, s] = rand(Normal(0, sS - g^2 * sZ), nobs)
        # Generate log_z
        log_z[:, s] = rand(Normal(0, sZ), nobs)
    end

    return (ϵ_D, a_u, log_z)
end

# Instantiate the shocks
sim_ϵ_D, sim_a_u, sim_log_z = simulate_shocks();

# Model simulation (computes simulated moment vector)
function simulation(ϵ_D, a_u, log_z, b2, b1=β1, m=μ,
    d=δ, g=γ, nsim=S, nobs=T)

    # Initialize the sum of the simulations
    S_sim = [0, 0]

    # Simulate the model S times
    for s in 1:nsim
        # Generate log_a
        log_a = g * log_z[:, s] + a_u[:, s]
        # Generate log prices
        log_p = (1 + d * b2)^(-1) * ((d * b1 + m) .- (log_a + d * ϵ_D[:, s]))
        # Generate log quantities
        log_q = b1 .- (b2 * log_p + ϵ_D[:, s])
        # Simulated moments (sum)
        S_sim += [dot(log_z[:, s], log_q) / nobs, dot(log_z[:, s], log_p) / nobs]
    end
    # compute average simulation
    E_sim = S_sim / nsim

    return (E_sim)
end

# ------------------------------------------------------------------------------
# Compare population-simulated moments for β2 = 2

# Compute simulated moments | β2 = 2
E_sim = simulation(sim_ϵ_D, sim_a_u, sim_log_z, 2.0)

# Compute population moments | β2 = 2
E_g1 = (1 + δ * β2)^(-1) * ((β1 - β2 * μ) * 0 + γ * β2 * σ_Z)
E_g2 = (1 + δ * β2)^(-1) * ((μ + δ * β1) * 0 - γ * σ_Z)
E_pop = [E_g1, E_g2]

# Comparison
E_pop - E_sim



# ------------------------------------------------------------------------------
# Minimization for simulated moments
function simulation_error(s_ϵ_D, s_a_u, s_log_z, empmom, b2, b1=β1, m=μ,
    d=δ, g=γ, sZ=σ_Z, nsim=S, nobs=T)

    # Initialize simulation sum
    S_sim = [0, 0]

    # Initialize simulation matrices
    for s in 1:nsim
        # Generate log_a
        s_log_a = g * s_log_z[:, s] + s_a_u[:, s]
        # Generate log prices
        s_log_p = (1 + d * b2)^(-1) * ((d * b1 + m) .- (s_log_a + d * s_ϵ_D[:, s]))
        # Generate log quantities
        s_log_q = b1 .- (b2 * s_log_p + s_ϵ_D[:, s])
        # Simulated moments (sum)
        S_sim += [dot(s_log_z[:, s], s_log_q) / nobs, dot(s_log_z[:, s], s_log_p) / nobs]
    end
    # compute average simulation
    E_sim = S_sim / S

    return (E_sim - empmom)
end


# Minimize using Nelder Mead
fsim(b) = sum(simulation_error(sim_ϵ_D, sim_a_u, sim_log_z, E_g_emp, b[1]) .^ 2)
θ_sim_optim = optimize(fsim, [0.0]).minimizer


# Minimize using naive β2 grid-search

# Initialize grid search
gridsize = 10000
grid = collect(-4:8/gridsize:4)
ssr_sim = Vector{Float64}(undef, size(grid)[1])
fill!(ssr_sim, NaN)

for b in eachindex(grid)
    ssr_sim[b] = sqrt(sum(simulation_error(sim_ϵ_D, sim_a_u, sim_log_z, E_g_emp, grid[b]) .^ 2))
end


θ_sim = round(grid[ssr_sim.==minimum(ssr_sim, dims=1)][1], digits=2)

# Plot loss function
plot(grid, ssr_sim, label="",
    ylabel=L"\left \| E^{\textrm{sim}}\left ( g(w,\theta) \right ) -  \widetilde{E}\left ( g(w,\theta) \right )\right \|",
    xlabel=L"θ")
scatter!([θ_sim], minimum(ssr_sim, dims=1), color="red",
    label=string(L"θ_{\textrm{sim}}", " = $θ_sim"), markersize=5)
savefig(string(figures_dir, "\\", ps_n, "_", "loss_e_sim.pdf"))



# ------------------------------------------------------------------------------
# Minimization for population moments

function moment_error(log_z, log_q, log_p, b2, b1=β1, m=μ, d=δ, g=γ,
    sZ=σ_Z, T=50)

    # Initialize population moments
    E_g1 = (1 + d * b2)^(-1) * ((b1 - b2 * m) * 0 + g * b2 * sZ)
    E_g2 = (1 + d * b2)^(-1) * ((m + d * b1) * 0 - g * sZ)
    E_g = [E_g1, E_g2]
    # Empirical moments
    E_g_emp = [dot(log_z, log_q) / T, dot(log_z, log_p) / T]

    return (E_g - E_g_emp)
end

# Minimize using Nelder Mead
fpop(b) = sum(moment_error(log_z, log_q, log_p, b[1]) .^ 2)
θ_optim = optimize(fpop, [0.0]).minimizer


# Minimize using naive β2 grid-search
ssr = Vector{Float64}(undef, size(grid)[1])
fill!(ssr, NaN)

for b in eachindex(grid)

    ssr[b] = sqrt(sum(moment_error(log_z, log_q, log_p, grid[b]) .^ 2))

end

θ = round(grid[ssr.==minimum(ssr, dims=1)][1], digits=2)

# Plot loss function
plot(grid, ssr, label="",
    ylabel=L"\left \| E\left ( g(w,\theta) \right ) -  \tilde{E}\left ( g(w,\theta) \right )\right \|",
    xlabel=L"θ")
scatter!([θ], minimum(ssr, dims=1), color="red",
    label=string(L"θ", " = $θ"), markersize=5)


savefig(string(figures_dir, "\\", ps_n, "_", "loss_e_pop.pdf"))
