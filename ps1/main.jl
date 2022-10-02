# ECON8853 Problem Set 1 #######################################
# Joe Jourden, Federico Rodari and Tuo Yang (2022) #############
################################################################


# Load Modules
using LinearAlgebra
using Random,Distributions
using Plots
using Optim
using Statistics
using LaTeXStrings
using NLsolve
using PlotlyJS
using DataFrames
using DelimitedFiles

# Setup parent and children directories
ps_n = "ps1"
parent_dir = "C:\\Users\\feder\\Dropbox\\Github\\industrial-organization-I"
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


# Setup seed
Random.seed!(2022);

# -----------------------------------------------------------------------------------
# Parameters setup

# Define parameters
# Number of products
J  = 3;
# Number of markets
M  = 100;
# Linear parameters
β = [5, 1, 1];
α   = 1;
# Normal shock
σ_α = 1;
# Number of consumers
ns = 1000; #100
γ₀,γ₁,γ₂ = 2,1,1;

# -----------------------------------------------------------------------------------

# Simulate Data
# product x market covariates
X_1  = ones(J,M);
X_2  = rand(Float64, (J,M));
X_3  = randn(Float64, (J,M));

# Demand shifter
ξ   = randn(Float64, (J,M));

# Cost shifters
W = repeat(randn(Float64, (J)),
            outer = [1,M]);
Z    = randn(Float64, (J,M));
ν_p  = rand(LogNormal(0,1), ns);
η    = randn(Float64, (J,M));
ϵ    = rand(Gumbel(0,1), (J,M));
A    = α .+ σ_α*ν_p; 
MC   = max.(γ₀ .+ γ₁*W .+ γ₂*Z .+ η,1e-12);
# Initialize price and shares matrices
P   = zeros(J,M)
S    = zeros(J,M);

# Fixed point function
function fixedpointmap(f; iv, tolerance=1E-7, maxiter=10000)
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

# -----------------------------------------------------------------------------------
# Equilibrium Prices

for m in 1:M

    print("Running Market $m...\n")

    function f!(F, p)
    
        # product 1
        δ1m = X_1[1,m]*β[1] .+ X_2[1,m]*β[2]  .+ X_3[1,m]*β[3]  .- A*p[1] .+ ξ[1,m]
        # product 2
        δ2m = X_1[2,m]*β[1]  .+ X_2[2,m]*β[2]  .+ X_3[2,m]*β[3] .- A*p[2] .+ ξ[2,m]
        # product 3
        δ3m = X_1[3,m]*β[1]  .+ X_2[3,m]*β[2]  .+ X_3[3,m]*β[3] .- A*p[3] .+ ξ[3,m]

        den = 1 .+ exp.(δ1m) .+ exp.(δ2m) .+ exp.(δ3m)

        s_m = [mean((exp.(δ1m)) ./ den),
               mean((exp.(δ2m)) ./ den),
               mean((exp.(δ3m)) ./ den)]

        s_m =  max.(1e-12, s_m) 

        Ω_m =   diagm( [1/mean(A .* s_m[1] .* (1-s_m[1])),
                        1/mean(A .* s_m[2] .* (1-s_m[2])),
                        1/mean(A .* s_m[3] .* (1-s_m[3]))])

        F[1] = p[1] - MC[1,m] - Ω_m[1]*s_m[1]
        F[2] = p[2] - MC[2,m] - Ω_m[2]*s_m[2]
        F[3] = p[3] - MC[3,m] - Ω_m[3]*s_m[3]
    end

    P[:,m]  = nlsolve(f!, rand(3)).zero

end


[minimum(P,dims =2),
 maximum(P,dims = 2),
 any(isnan,P),
 any(isinf,P)]

# -----------------------------------------------------------------------------------
# Share matrices

for m in 1:M

        δ1m = X_1[1,m]*β[1] .+ X_2[1,m]*β[2]  .+ X_3[1,m]*β[3]  .- A*P[1,m] .+ ξ[1,m]
        δ2m = X_1[2,m]*β[1] .+ X_2[2,m]*β[2]  .+ X_3[2,m]*β[3]  .- A*P[2,m]  .+ ξ[2,m]
        δ3m = X_1[3,m]*β[1] .+ X_2[3,m]*β[2]  .+ X_3[3,m]*β[3]  .- A*P[3,m]  .+ ξ[3,m]

        den = 1 .+ exp.(δ1m) .+ exp.(δ2m) .+ exp.(δ3m)

        S[:,m] = max.( 1e-12, [mean((exp.(δ1m)) ./ den),
                               mean((exp.(δ2m)) ./ den),
                               mean((exp.(δ3m)) ./ den)])         

end

[minimum(S,dims =2),
maximum(S,dims = 2),
sum(S.<0),
sum(S.>1)]



# -----------------------------------------------------------------------------------------------------
# Create BLP instruments (For X2 X3 and each product sum over other products for each market)

Z2= vcat(sum(X_2[1:end .!=1,:], dims = 1), # product 1 IV (sum over 2/3) over all markets
         sum(X_2[1:end .!=2,:], dims = 1), # product 2 IV (sum over 1/2) over all markets
         sum(X_2[1:end .!=3,:], dims = 1)); # product 3 IV (sum over 1/2) over all markets
         # vertical stack them to have a long structure
Z3= vcat(sum(X_3[1:end .!=1,:], dims = 1),
         sum(X_3[1:end .!=2,:], dims = 1),
         sum(X_3[1:end .!=3,:], dims = 1));


# Create vector of covariates
X = hcat(vec(X_1), vec(X_2), vec(X_3), vec(P));
IV = hcat(vec(X_1), vec(X_2), vec(X_3), vec(Z2), vec(Z3), vec(W), vec(Z));
ν_i  = rand(LogNormal(0,1), 5*ns);

# First step for optimal matrix
Φ = Matrix(1I,size(IV,2),size(IV,2));

# Compute shares (for a random δ)
function share(δm, P, σ, ν_i)
    # P must be a 3x1 vector, same for δm
        δ1m =  δm[1] .- σ*ν_i*P[1]
        δ2m =  δm[2] .- σ*ν_i*P[2] 
        δ3m =  δm[3] .- σ*ν_i*P[3] 
    
        den = 1 .+ exp.(δ1m) .+ exp.(δ2m) .+ exp.(δ3m)
    
        s = [mean((exp.(δ1m)) ./ den),
            mean((exp.(δ2m)) ./ den),
            mean((exp.(δ3m)) ./ den)]

        s =  max.(1e-12, s);
    
    return(s)
    
end

# Initialize linear parameters
θ2 = ones(4); # [β1 β2 β3 α]

# Data structure
data_struct = [];
push!(data_struct, P);
push!(data_struct, S);
push!(data_struct, X);
push!(data_struct, IV);
# [P,S,X,IV]

function GetRes(θ1,ν_i,data_struct,We) # better structure to handle inputs (θ2 as input?)
    # θ1: nonlinear parameters
    # θ2: linear parameters
    # BLP inversion ---------------------------------------------------------
    J = 3
    M = 100
    # Find updated delta using the contraction mapping | alpha, beta, sigma
    # Random draw α, β, σ_αs
    # fixed for a given sigma a
    δ= ones(J,M)

    # Export matrices from structure
    P = data_struct[1];
    S = data_struct[2];
    X = data_struct[3];
    IV = data_struct[4];

    for m in 1:M
        #print("Market $m...\n")

        opt(δ0) =  δ0  + (log.(S[:,m])  - log.(share(δ0 , P[:,m],θ1[1], ν_i))) # wrap function to single input 
        # Fixed point iteration
        δ[:,m] = fixedpointmap(opt; iv = ones(3), tolerance=1E-12)[1]
     
    end

    # Recover linear parameters
    θ2 = inv(X'*IV *inv(We)*IV'*X)*X'IV*inv(We)*IV'*vec(δ) # alpha and betas
    # Residuals
    res =  vec(δ) -(X[:,1]*θ2[1] + X[:,2]*θ2[2] +X[:,3]*θ2[3] +X[:,4]*θ2[4]) # must preserve the order of the coefficients!
   
    # Value function
    gn = inv(size(res,1))*IV'*vec(res)
    fval = gn'*inv(We)*gn;
    
    return(fval)
end

# Wrap function to pass to the optimizer
wrapped_nonlinear(x) = GetRes(x,ν_i,data_struct,Φ);

# -------------------------------------------------------
# (1) Search over non-linear parameters
results = optimize(wrapped_nonlinear, [0.5])
# Store minimizer
θ1_opt = Optim.minimizer(results)

# -------------------------------------------------------
# (2) Search over linear parameters

# initialize optimal δ
δ= ones(J,M)

# --------------------------------------------
# First Step
for m in 1:M
    #print("Market $m...\n")
    
    #δ0 = δ[:,m] 
    f(δ0) =  δ0  + log.(S[:,m])  - log.(share(δ0 , P[:,m],θ1_opt[1], ν_i)) # wrap function to single input
    # Fixed point iteration
    δ[:,m] = fixedpointmap(f; iv = ones(3), tolerance=1E-12)[1]
    
end

# Recover linear parameters
θ2 = inv(X'*IV *inv(Φ)*IV'*X)*X'IV*inv(Φ)*IV'*vec(δ); # alpha and betas

# Residuals
res =  vec(δ) -(X[:,1]*θ2[1] + X[:,2]*θ2[2] +X[:,3]*θ2[3] +X[:,4]*θ2[4]); # must preserve the order of the coefficients!

# --------------------------------------------
# Second Step
optW = inv(size(IV,1))*IV'*res*res'*IV; # Update weighting matrix
θ2_step2 = inv(X'*IV *inv(optW)*IV'*X)*X'IV*inv(optW)*IV'*vec(δ)

# Export relevant data
writedlm( string(data_dir,"\\",ps_n,"_", "theta1.csv"),  θ1_opt, ',')
writedlm( string(data_dir,"\\",ps_n,"_", "theta2.csv"),  θ2_step2, ',')
writedlm( string(data_dir,"\\",ps_n,"_", "delta.csv"),  δ, ',')
writedlm( string(data_dir,"\\",ps_n,"_", "prices.csv"),  P, ',')
writedlm( string(data_dir,"\\",ps_n,"_", "shares.csv"),  S, ',')


# -----------------------------------------------------------------------------------
# Compute Marginal Costs under different specifications

# We are measuring the effect of misspecification given the data we have

# Compute ds/dp
function compstat(j,k, δm, P, σ = θ1_opt[1] , α = -θ2_step2[4] , ν = ν_i)
    A    = α .+ σ*ν;  # Important, the minus on the input!
    # P must be a 3x1 vector, same for δm
    δim =  hcat(δm[1] .- σ*ν*P[1], # nsx1 vector
            δm[2] .- σ*ν*P[2],  # nsx1 vector
            δm[3] .- σ*ν*P[3])  # nsx1 vector

    den = 1 .+ exp.(δim[:,1]) .+ exp.(δim[:,2]) .+ exp.(δim[:,3])

    sjm = max(1e-12,mean((exp.(δim[:,j])) ./ den))
    skm = max(1e-12,mean((exp.(δim[:,k])) ./ den))

    if j == k  
        dsdp = -mean(A .* sjm*(1-sjm))
    else
        dsdp = mean(A .* sjm* skm)
    end
    
return(dsdp)

end

# -----------------------------------------------------------------------------------
# PERFECT COLLUSION 

# Initialize matrix
Δ_coll = zeros(J,J,M); # perfect collusion

# Fill matrix for perfect collusion
for m in 1:M

    Δ_coll[:,:,m] =vcat([compstat(1,1, δ[:,m], P[:,m]) compstat(1,2, δ[:,m], P[:,m]) compstat(1,3, δ[:,m], P[:,m])],
                   [compstat(2,1, δ[:,m], P[:,m]) compstat(2,2, δ[:,m], P[:,m]) compstat(2,3, δ[:,m], P[:,m])],
                   [compstat(3,1, δ[:,m], P[:,m]) compstat(3,2, δ[:,m], P[:,m]) compstat(3,3, δ[:,m], P[:,m])])

end

# Compute MC under perfect collusion
MC_collusion = zeros(J,M);

for m in 1:M

MC_collusion[:,m] = max.(1e-12, P[:,m] + inv(Δ_coll[:,:,m])*S[:,m]);

end

# -----------------------------------------------------------------------------------
# OLIGOPOLY 

# Initialize matrix
Δ_oli = zeros(J,J,M); # oligopoly

# Fill matrix for oligopoly
for m in 1:M

    Δ_oli[:,:,m] =vcat([compstat(1,1, δ[:,m], P[:,m]) 0 0],
                      [0 compstat(2,2, δ[:,m], P[:,m]) 0],
                      [0 0 compstat(3,3, δ[:,m], P[:,m])])

end

# Compute MC under oligopoly
MC_oli = zeros(J,M);

for m in 1:M

MC_oli[:,m] = max.(1e-12, P[:,m] + inv(Δ_oli[:,:,m])*S[:,m]);

end



# -----------------------------------------------------------------------------------
# Create boxplot under diffent specifications

# Create grouping variables

# Products
Prod = vcat(repeat(["Product 1"],
                    outer = [3*M,1]),
            repeat(["Product 2"],
                    outer = [3*M,1]),
            repeat(["Product 3"],
                    outer = [3*M,1]));

# Conducts
Cond = repeat(vcat(repeat(["Competition"],
                    outer = [M,1]),
            repeat(["Collusion"],
                    outer = [M,1]),
            repeat(["Oligopoly"],
                    outer = [M,1])),
                    outer = [3,1]);       

# Create dataframe
df = DataFrame( Product = vec(Prod),
                Conduct = vec(Cond),
                mc = vcat(P[1,:],MC_collusion[1,:], MC_oli[1,:],
                          P[2,:],MC_collusion[2,:], MC_oli[2,:],
                          P[3,:],MC_collusion[3,:], MC_oli[3,:]));


# Create Boxplot
boxplot_mc = PlotlyJS.plot(
                            df,
                            x=:Product, y=:mc, color=:Conduct,
                            quartilemethod="exclusive",
                            kind="box",
                            notched = true,
                            Layout(boxmode="group",
                            plot_bgcolor  = "#FFFFFF",
                            yaxis=attr(showgrid=true, zeroline=false,
                                      gridcolor="#F2F2F2")),
                            
                        );

# Save figure                        
PlotlyJS.savefig(
    boxplot_mc,
    string(figures_dir,"\\",ps_n,"_", "boxplot_mc.pdf"))




# -----------------------------------------------------------------------------------
# Merger analysis
# Here we compute prices from the start, as we are performing a counterfactual

# Initialize price and shares matrices
P_merger  = zeros(J,M)
S_merger  = zeros(J,M);


for m in 1:M

    print("Running Market $m...\n")

    function f!(F, p)
    
        # product 1
        δ1m = X_1[1,m]*β[1] .+ X_2[1,m]*β[2]  .+ X_3[1,m]*β[3]  .- A*p[1] .+ ξ[1,m]
        # product 2
        δ2m = X_1[2,m]*β[1]  .+ X_2[2,m]*β[2]  .+ X_3[2,m]*β[3] .- A*p[2] .+ ξ[2,m]
        # product 3
        δ3m = X_1[3,m]*β[1]  .+ X_2[3,m]*β[2]  .+ X_3[3,m]*β[3] .- A*p[3] .+ ξ[3,m]

        den = 1 .+ exp.(δ1m) .+ exp.(δ2m) .+ exp.(δ3m)

        s_m = [mean((exp.(δ1m)) ./ den),
               mean((exp.(δ2m)) ./ den),
               mean((exp.(δ3m)) ./ den)]

        s_m =  max.(1e-12, s_m) 

        Ω_m =   diagm( [1/mean(A .* s_m[1] .* (1-s_m[1])),
                        1/mean(A .* s_m[2] .* (1-s_m[2])),
                        1/mean(A .* s_m[3] .* (1-s_m[3]))])

        # Off-diagonal terms 
        Ω_m[1,2] = mean(A .* s_m[1] .* s_m[2])
        Ω_m[2,1] = mean(A .* s_m[2] .* s_m[1])                

        F[1] = p[1] - MC[1,m] - Ω_m[1]*s_m[1]
        F[2] = p[2] - MC[2,m] - Ω_m[2]*s_m[2]
        F[3] = p[3] - MC[3,m] - Ω_m[3]*s_m[3]
    end

    P_merger[:,m]  = nlsolve(f!, rand(3)).zero

end

 # Sanity check
[minimum(P_merger,dims =2),
 maximum(P_merger,dims = 2),
 any(isnan,P_merger),
 any(isinf,P_merger)]

 # Compute differences in prices
 diff_P = P_merger - P;


# -----------------------------------------------------------------------------------
# Create boxplot comparing price differences after merger

# Create grouping variables

# Products
Prod = vcat(repeat(["Product 1"],
                    outer = [M,1]),
            repeat(["Product 2"],
                    outer = [M,1]),
            repeat(["Product 3"],
                    outer = [M,1]));

# Create dataframe
df_merger = DataFrame( Product = vec(Prod),
                PriceDiff = vcat(diff_P[1,:],diff_P[2,:],diff_P[3,:]));


# Create Boxplot
boxplot_pdiff = PlotlyJS.plot(
                            df_merger,
                            x=:Product, y=:PriceDiff,
                            quartilemethod="exclusive",
                            kind="box",
                            notched = true,
                            Layout(plot_bgcolor  = "#FFFFFF",
                            yaxis=attr(showgrid=true, zeroline=false,
                                      gridcolor="#F2F2F2"),
                                      )
                            
                        );

# Save figure                        
PlotlyJS.savefig(
    boxplot_pdiff,
    string(figures_dir,"\\",ps_n,"_", "boxplot_deltap_merger.pdf"))

