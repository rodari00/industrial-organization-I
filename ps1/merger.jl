########## input: all parameters
########## output: P_merg and S_merg

using Random, Distributions, LinearAlgebra, NLsolve

P_merg    = zeros(J,M)
S_merg    = zeros(J,M)


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

function f(p_m)
    δμ1m = X_1[1,m]*β₁ .+ X_2[1,m]*β₂ .+ X_3[1,m]*β₃ .- A*p_m[1] .+ ξ[1,m]
    δμ2m = X_1[2,m]*β₁ .+ X_2[2,m]*β₂ .+ X_3[2,m]*β₃ .- A*p_m[2]  .+ ξ[2,m]
    δμ3m = X_1[3,m]*β₁ .+ X_2[3,m]*β₂ .+ X_3[3,m]*β₃ .- A*p_m[3]  .+ ξ[3,m]

    den = 1 .+ exp.(δμ1m) .+ exp.(δμ2m) .+ exp.(δμ3m)

    s_m = [mean((exp.(δμ1m)) ./ den),
        mean((exp.(δμ2m)) ./ den),
        mean((exp.(δμ3m)) ./ den)]

    Ω_m =   diagm( [1/(mean(A .* s_m[1] .* (1-s_m[1]))),
                1/mean(A .* s_m[2] .* (1-s_m[2])),
                1/mean(A .* s_m[3] .* (1-s_m[3]))])

    # off-diag entries of Ω_m
    Ω_m[1,2] = mean(A .* s_m[1] .* s_m[2])
    Ω_m[2,1] = mean(A .* s_m[2] .* s_m[1])
    
    return(max.(1E-12, MC_m + Ω_m*s_m))
end

P_merg[:,m], normdiff, iter  = fixedpointmap(f; iv = ones(3))

end

# Share matrices
for m in 1:M
    δμ1m = X_1[1,m]*β₁ .+ X_2[1,m]*β₂ .+ X_3[1,m]*β₃ .- A*P_merg[1,10] .+ ξ[1,m]
    δμ2m = X_1[2,m]*β₁ .+ X_2[2,m]*β₂ .+ X_3[2,m]*β₃ .- A*P_merg[2,10]  .+ ξ[2,m]
    δμ3m = X_1[3,m]*β₁ .+ X_2[3,m]*β₂ .+ X_3[3,m]*β₃ .- A*P_merg[3,10]  .+ ξ[3,m]

    den = 1 .+ exp.(δμ1m) .+ exp.(δμ2m) .+ exp.(δμ3m)

    S_merg[:,m] = [mean((exp.(δμ1m)) ./ den),
                mean((exp.(δμ2m)) ./ den),
                mean((exp.(δμ3m)) ./ den)]

end