#---- Load Packages -------------------------------------------------------
using Random, Distributions,LinearAlgebra, DelimitedFiles, Optim, DataFrames, CSV

# Setup parent and children directories
ps_n = "ps3"

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

# -- Structure Setup ---------------------------------------------------------

mutable struct RustModel
    β::Float64 # Discount Factor
    θ::Array{Float64} # Utility Function Parameters
    Vbar::Array{Float64} # Next Value Function Array
    k::Int32 # Size of State Space
 
   # Initialize Model Parameters
    function RustModel( β = 0.9, k = 5 )
       Vbar = zeros(k, 2)
       θ = [1,3] # μ, R
       return new( β, θ, Vbar, k )
    end
end;


# Instant Utility
function u(model::RustModel)
    a = collect(1:k) # Generate State Space range, i.e. [1, 2, 3, 4, ...]
    i_0 =  θ[1] * a               # Utility from not replacing
    i_1 =  θ[2] * ones(k,1) # Utility from replacing (constant)
    U = hcat(i_0, i_1) # Utility matrix
    return -U
end

# Transition probabilities
P = zeros(5,5,2);

P[:,:,1] = [ 0   1   0   0   0;  # i = 0
             0   0   1   0   0;
             0   0   0   1   0;
             0   0   0   0   1;
             0   0   0   0   1];
P[:,:,2] = hcat(ones(5,1), zeros(5,4)); # i = 1



# Construct the contraction mapping
# write conditional value function for each action, 2 vector 5x1, one constant and the other varying

function T(model::RustModel)::Matrix
    """Compute value function by Bellman iteration"""
    k = k                                   # Dimension of the state space
    U = u(model)                                  # Static utility
    index_A = Int[[2:k; k] 2*ones(k,1)];          # Deterministic age index [2:k; k]
    γ = Base.MathConstants.eulergamma             # Euler's gamma

    # Iterate the Bellman equation until convergence
    Vbar = Vbar;
    Vbar1 = Vbar;
    err = 1;
    iter = 0;
    while err>1e-8
        V = γ .+ log.(sum(exp.(Vbar), dims=2))     # Compute value V(x)
        Vbar1 =  U + β * V[index_A]                      # Compute v-specific (could use transition probabilities)
        #Vbar1 =  U + β * hcat(P[:,:,1]*V,P[:,:,2]*V)    # Compute v-specific (with transition probabilities)
        err = max(abs.(Vbar1 - Vbar)...);                      # Check distance
        iter += 1;
        #println("Error is ", err)
        Vbar = Vbar1                               # Update value function
    end
    return Vbar
end;

#=
function T(model::RustModel)::Matrix
    """Compute value function by Bellman iteration"""
    k = k                                   # Dimension of the state space
    U = u(model)                                  # Static utility
    index_A = Int[[2:k; k] ones(k,1)];            # Deterministic age index
    γ = Base.MathConstants.eulergamma             # Euler's gamma

    # Iterate the Bellman equation until convergence
    Vbar = Vbar;
    Vbar1 = Vbar;
    err = 1;
    iter = 0;
    while err>1e-8
        V = log.(sum(exp.( U + β*Vbar), dims=2))     # Compute value
        Vbar1 = V[index_A]                      # Compute v-specific (could use transition probabilities)
        #Vbar1 =  U + β * hcat(P[:,:,1]*V,P[:,:,2]*V)    # Compute v-specific (with transition probabilities)
        err = max(abs.(Vbar1 - Vbar)...);                      # Check distance
        iter += 1;
        #println("Error is ", err)
        Vbar = Vbar1                               # Update value function
    end
    return Vbar
end;
=#


# Solve
model = RustModel()
V_bar = T(model)
# V(x) = γ + Σ_d' v(x,d')*
# Conditional choice probabilities
ChoiceProb = exp.(V_bar) ./sum(exp.(V_bar), dims = 2)





# Simulate Data
#=
function simulate_data(model::RustModel, N::Int)::Tuple
    """Generate data from primitives"""
    Vbar = T(model)                             # Solve model
    ε = rand(Gumbel(0,1), N, 2)                 # Draw shocks(ε_0, ε_1)
    #a = collect(1:k)                     # Generate State Space range, i.e. [1, 2, 3, 4, ...] 
    St = zeros(Int, (N, 2))                     # Draw states
    A = zeros(Int, (N, 1))                      # Compute investment decisions (1 is i = 1)
    at = 1                                      # Initialize state

    for t in 1:N
    # is the discounted utility of replacement higher than not replacing?
    A[t] = convert(Int,Vbar[at,2] + ε[t,2] .> Vbar[at,1] + ε[t,1])  # Current action: (1 is i = 1)
    St[t,1] = at                                       # Current state
    # Deterministic future state
    if A[t] == 1
        St[t,2] = 1                                # Reset state
    else
        St[t,2] = min(5,at+1)                     # Advance state
    end  
    at = copy(St[t,2])                                       # Next state becomes current state for t+1
    end

    df = DataFrame(St = St[:,1], A = vec(A), St1 = St[:,2])         # Dataframe
    CSV.write(string(data_dir,"\\rust.csv"), df)
    return St, A
end;
=#

# Simulate Data

function simulate_data(model::RustModel, N::Int)::Tuple
    """Generate data from primitives"""
    Vbar = T(model)                             # Solve model
    ε = rand(Gumbel(0,1), N, 2)                 # Draw shocks(ε_0, ε_1)
    a = collect(1:k)                      # Generate State Space range, i.e. [1, 2, 3, 4, ...] 
    S = rand(a, N)                              # Draw states
    A = (((Vbar[S,:] + ε) * [-1;1]) .> 0)    # Compute investment decisions (1 is i = 1)
    #A = convert(Vector{Int}, Vbar[S,2] + ε[:,2] .> Vbar[S,1] + ε[:,1])
    Snext = min.((S .+ 1).*(A.==0) .+ (1*A.==1) , maximum(a))    # Compute future state
    df = DataFrame(St = S, A = vec(A), St1 = Snext)         # Dataframe
    CSV.write(string(data_dir,"\\rust.csv"), df)
    return S, A
end;


# here we see a change in policies due to unobservables, must have been a shock given that for the same age we might see different policies.
# No deterministic rule

N = 200000
St, A = simulate_data(model, N)

sum(A)/N

function logL(model::RustModel, St::Vector, A::BitVector, θc::Vector)::Number
    """Compute log-likelihood functionfor Rust problem"""
    # Compute value
    θ = θc
    Vbar = T(model)

    # Conditional choice probabilities
    ChoiceProb = exp.(Vbar ) ./sum(exp.(Vbar ), dims = 2)

    # Likelihood
    logL = sum(log.(ChoiceProb[St[A.==1]])) + sum(log.(1 .- ChoiceProb[St[A.==0]]))
    return -logL
end;

logL(model, St, A, [1,3])

# Select starting values
θstart = Float64[0,0];

# Optimize
θ_optim = optimize(x -> logL(model, St, A,x ), θstart)
θ_optim.minimizer 
print("Estimated thetas: $θ_R (true = $θ)")








