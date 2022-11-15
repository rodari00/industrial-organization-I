
#---- Load Packages -------------------------------------------------------
using Random, Distributions, CSV, DataFrames, BlackBoxOptim,
      LinearAlgebra, DelimitedFiles, Random, Optim

# Setup parent and children directories
ps_n = "ps2"

parent_dir = "C:\\Users\\feder\\Dropbox\\Github\\industrial-organization-I\\" # Federico
figures_dir = string(parent_dir,"\\",ps_n,"\\figures")
data_dir = string(parent_dir,"\\",ps_n,"\\data")

# Create figures folder
if isdir(figures_dir)
else  
    mkdir(figures_dir)
end

if isdir(data_dir)
else  
    mkdir(data_dir)
end



# -- Parameters Setup ---------------------------------------------------------
F = 3
(α, β, δ) = (1.0, 1.0, 1.0)
M = 100 # markets
T = 100 # sims

## import data
df = CSV.read(string(parent_dir,"\\",ps_n,"\\entryData.csv"), DataFrame, header = false)
Z = zeros(Float64, (F,M))
In = zeros(Int64, (F,M))
(X, Z[1,:], Z[2,:], Z[3,:], In[1,:], In[2,:], In[3,:]) = (df[:,1], df[:,2], df[:,3], df[:,4], df[:,5], df[:,6], df[:,7])
Nstar = In[1,:] .+ In[2,:] .+ In[3,:]
r = zeros(Float64, (F,M,F))

for f in 1:F, m in 1:M, N in 1:F
    r[f,m,N] = X[m] * β - δ * log(N) - Z[f,m] * α
end

## -- Simulated MLE ----------------------------------------------------------------------
function LL(θ)
    (μ, σ) = (θ[1], θ[2])
    p = zeros(Float64, (F, M, F, 2))
    for f in 1:F, m in 1:M, N in 1:F
        p[f, m, N, 1] = cdf(Normal(μ, σ^2), r[f,m,N]) # prob of entering
        p[f, m, N, 2] = 1 - cdf(Normal(μ, σ^2), r[f,m,N]) # prob of not entering
    end
    f = x -> max(x, eps(0.0))
    p = f.(p)

    P = zeros(Float64, M)
    for m in 1:M
        if Nstar[m] == 0
            P[m] = p[1, m, Nstar[m] + 1, 2] * p[2, m, Nstar[m] + 1, 2] * p[3, m, Nstar[m] + 1, 2]
        elseif Nstar[m] == 1
            P[m] = (p[1, m, Nstar[m], 1] * p[2, m, Nstar[m] + 1, 2] * p[3, m, Nstar[m] + 1, 2]) + (p[1, m, Nstar[m] + 1, 2] * p[2, m, Nstar[m], 1] * p[3, m, Nstar[m] + 1, 2]) + (p[1, m, Nstar[m] + 1, 2] * p[2, m, Nstar[m] + 1, 2] * p[3, m, Nstar[m], 1])
        elseif Nstar[m] == 2
            P[m] = (p[1, m, Nstar[m], 1] * p[2, m, Nstar[m], 1] * p[3, m, Nstar[m] + 1, 2]) + (p[1, m, Nstar[m], 1] * p[2, m, Nstar[m] + 1, 2] * p[3, m, Nstar[m], 1]) + (p[1, m, Nstar[m] + 1, 2] * p[2, m, Nstar[m], 1] * p[3, m, Nstar[m], 1])
        elseif Nstar[m] == 3
            P[m] = p[1, m, Nstar[m], 1] * p[2, m, Nstar[m], 1] * p[3, m, Nstar[m], 1]
        end
    end

    f = x -> max(x, eps(0.0))
    P = log.(f.(P))
    return(-sum(P)) # returns -log-likelihood
end


# Solve for the minimum
soln = bboptimize(LL, [0.0,0.0], SearchRange = [(-5.0, 5.0), (0, 10.0)], NumDimensions = 2)

# Save the parameters
writedlm( string(data_dir,"\\",ps_n,"_", "berry.csv"), best_candidate(soln), ',')
