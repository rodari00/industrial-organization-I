

function blpGMM(σ_α,ν_p,P,S,IV,X)

    # BLP inversion ---------------------------------------------------------
    J = 3
    M = 100
    # Find updated delta using the contraction mapping | alpha, beta, sigma
    # Random draw α, β, σ_αs
    # fixed for a given sigma a
    δbis= ones(J,M)
    α = 1
    β = ones(3)
    A = α .+ 0.6*ν_p; ######!!!!!!!!!!!!!!!!!!!!!CHANGE sigma 

    for m in 1:M
        print("Market $m...\n")
        function mapping(δm,ν_i,P,S,m)
            δμ1m = δm[1] .- 0.7*ν_i*P[1,m]
            δμ2m = δm[2] .- 0.7*ν_i*P[2,m]
            δμ3m = δm[3] .- 0.7*ν_i*P[3,m]
        
            den = 1 .+ exp.(δμ1m) .+ exp.(δμ2m) .+ exp.(δμ3m)
        
            s_m = [mean((exp.(δμ1m)) ./ den),
                   mean((exp.(δμ2m)) ./ den),
                   mean((exp.(δμ3m)) ./ den)]
            s_m = max.(1e-12, s_m) # avoid taking log of 0

            return(δm + (log.(S[:,m]) - log.(s_m)))
        end

        f(x) = mapping(x,ν_i,P,S,m); # wrap function to single input

        # Fixed point iteration
        δbis[:,m] = fixedpointmap(f; iv = ones(3), tolerance=1E-12)[1]
        
    end

    # First step for optimal matrix
    Φ = Matrix(1I,size(IV,2),size(IV,2));  # the n cancels out (inv(size(IV,1))*IV'*IV)
    
    # Recover linear parameters
    θ = inv(X'*IV *inv(Φ)*IV'*X)*X'IV*inv(Φ)*IV'*vec(δ) # alpha and betas
    # Residuals
    res =  vec(δ) -(X*θ) # must preserve the order of the coefficients!
   
    #Φ_new = inv(size(res,1))*IV'*res*res'*IV
    
    # Value function
    gn = inv(size(res,1))*IV'*vec(res)
    fval = gn'*inv(Φ_new)*gn;
    
    return(fval)
end








###############à FIXED POINT????

print("Running Market $m...\n")

# Initialize MC_m
MC_m = Array{Float64}(undef, 0, 1)
for j in 1:J
    MC_m = vcat(MC_m, γ₀ + γ₁*W[j,m] + γ₂*Z[j,m] + η[j,m]) 
end

function f(p_m)
    
    # product 1
    δ1m = X_1[1,m]*β[1] .+ X_2[1,m]*β[2]  .+ X_3[1,m]*β[3]  .- A*p_m[1] .+ ξ[1,m]
    # product 2
    δ2m = X_1[2,m]*β[1]  .+ X_2[2,m]*β[2]  .+ X_3[2,m]*β[3] .- A*p_m[2] .+ ξ[2,m]
    # product 3
    δ3m = X_1[3,m]*β[1]  .+ X_2[3,m]*β[2]  .+ X_3[3,m]*β[3] .- A*p_m[3] .+ ξ[3,m]

    den = 1 .+ exp.(δ1m) .+ exp.(δ2m) .+ exp.(δ3m)

    s_m = [mean((exp.(δ1m)) ./ den),
           mean((exp.(δ2m)) ./ den),
        mean((exp.(δ3m)) ./ den)]

    s_m =  max.(1e-12, s_m) 

    Ω_m =   diagm( [1/mean(A .* s_m[1] .* (1-s_m[1])),
                    1/mean(A .* s_m[2] .* (1-s_m[2])),
                    1/mean(A .* s_m[3] .* (1-s_m[3]))])

    return(MC_m + Ω_m*s_m)
end





MC = zeros(J,M)

for m in 1:M


    # Initialize MC_m
    MC_m = Array{Float64}(undef, 0, 1)
    for j in 1:J
        MC_m = vcat(MC_m, γ₀ + γ₁*W[j,m] + γ₂*Z[j,m] + η[j,m]) 
    end
    MC_m = max.(MC_m,0)
    MC[:,m] = MC_m

end




[minimum(MC,dims =2),
 maximum(MC,dims = 2),
 any(isnan,MC),
 any(isinf,MC)]