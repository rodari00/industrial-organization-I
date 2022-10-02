##########

using Random, Distributions
using LinearAlgebra
using NLsolve

# Define parameters
J   = 3
M   = 100
β = [5, 1, 1]
β₁  = 5
β₂  = 1
β₃  = 1
α   = 1
σ_α = 1
γ₀  = 2
γ₁  = 1
γ₂  = 1
ns = 500 #100

Random.seed!(1);



X_1  = ones(J,M)
X_2  = rand(Float64, (J,M))
X_3  = randn(Float64, (J,M))
ξ    = randn(Float64, (J,M))
W = repeat(randn(Float64, (J)),
            outer = [1,M])
Z    = randn(Float64, (J,M))
ν_p  = rand(LogNormal(0,1), ns)
η    = randn(Float64, (J,M))
ϵ    = rand(Gumbel(0,1), (J,M))
Α    = α .+ σ_α*ν_p
MC   = γ₀ .+ γ₁*W .+ γ₂*Z .+ η

P_star    = zeros(J,M)
iter_matrix = zeros(J,M);
S    = zeros(J,M);



# good style
function fixedpointmap(f; iv, tolerance=1E-7, maxiter=1000)
    # setup the algorithm
    x_old = iv
    normdiff = Inf
    iter = 1
    while normdiff > tolerance && iter <= maxiter
        x_new = f(x_old) # use the passed in map
        normdiff = norm(x_new - x_old)
        x_old = x_new
        iter = iter + 1
    end
    return (value = x_old, normdiff=normdiff, iter=iter) # A named tuple
end


for m in 1:M

    print("Running Market $m...\n")

    # Initialize MC_m
    MC_m = Array{Float64}(undef, 0, 1)
    for j in 1:J
        MC_m = vcat(MC_m, γ₀ + γ₁*W[j,m] + γ₂*Z[j,m] + η[j,m]) 
    end

    # p_m = Array{Float64}(undef, 0, 1)
    # MC_m = Array{Float64}(undef, 0, 1)
    # Ω_m = Array{Float64}(undef, 0, J)
    # s_m = Array{Array}(undef, 0, 1)
    # δ_m = Array{Float64}(undef, 0, 1)

function f(p_m)
    δ1m = X_1[1,m]*β₁ .+ X_2[1,m]*β₂ .+ X_3[1,m]*β₃ .- Α*p_m[1] .+ ξ[1,m]
    δ2m = X_1[2,m]*β₁ .+ X_2[2,m]*β₂ .+ X_3[2,m]*β₃ .- Α*p_m[2] .+ ξ[2,m]
    δ3m = X_1[3,m]*β₁ .+ X_2[3,m]*β₂ .+ X_3[3,m]*β₃ .- Α*p_m[3] .+ ξ[3,m]

    den = 1 .+ exp.(δ1m) .+ exp.(δ2m) .+ exp.(δ3m)

    s_m = [mean((exp.(δ1m)) ./ den),
           mean((exp.(δ2m)) ./ den),
           mean((exp.(δ3m)) ./ den)]

     Ω_m =   diagm( [1/mean(Α .* s_m[1] .* (1-s_m[1])),
                     1/mean(Α .* s_m[2] .* (1-s_m[2])),
                     1/mean(Α .* s_m[3] .* (1-s_m[3]))
                     ])

    return(MC_m + Ω_m*s_m)

end

    P_star[:,m], normdiff, iter  = fixedpointmap(f,iv =ones(3))

end


for 
    
    m in 1:M

    δ = β
    δ1m = X_1[1,m]*β₁ .+ X_2[1,m]*β₂ .+ X_3[1,m]*β₃ .- Α*P_star[1,m] .+ ξ[1,m]
    δ2m = X_1[2,m]*β₁ .+ X_2[2,m]*β₂ .+ X_3[2,m]*β₃ .- Α*P_star[2,m]  .+ ξ[2,m]
    δ3m = X_1[3,m]*β₁ .+ X_2[3,m]*β₂ .+ X_3[3,m]*β₃ .- Α*P_star[3,m]  .+ ξ[3,m]

    den = 1 .+ exp.(δ1m) .+ exp.(δ2m) .+ exp.(δ3m)

    S[:,m] = [mean((exp.(δ1m)) ./ den),
                mean((exp.(δ2m)) ./ den),
                mean((exp.(δ3m)) ./ den)]

end





for m in 1:M

    δ = β
    δ1m = X_1[1,m]*β₁ .+ X_2[1,m]*β₂ .+ X_3[1,m]*β₃  .+ ξ[1,m]
    δ2m = X_1[2,m]*β₁ .+ X_2[2,m]*β₂ .+ X_3[2,m]*β₃   .+ ξ[2,m]
    δ3m = X_1[3,m]*β₁ .+ X_2[3,m]*β₂ .+ X_3[3,m]*β₃  .+ ξ[3,m]
    μijt = kron(P_star,A)

    
    den = 1 .+ exp.(δ1m) .+ exp.(δ2m) .+ exp.(δ3m)

    S[:,m] = [mean((exp.(δ1m)) ./ den),
                mean((exp.(δ2m)) ./ den),
                mean((exp.(δ3m)) ./ den)]

end





