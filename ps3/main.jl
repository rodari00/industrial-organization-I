

# ECON8853 Problem Set 3 - Joe Jourden, Federico Rodari and Tuo Yang
# December 20, 2022

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

# -- Parameters and Functions Setup ---------------------------------------------------------
# Setup seed
Random.seed!(2022);
β = 0.9;
θ = [1,3];
k = 5;
a = collect(1:k);

# Instant Utility
function u(θ::Vector, s::Vector)
    a = collect(1:k)              # Generate State Space range, i.e. [1, 2, 3, 4, ...]
    i_0 =  θ[1] * a               # Utility from not replacing
    i_1 =  θ[2] * ones(k,1)       # Utility from replacing (constant)
    U = hcat(i_0, i_1)            # Utility matrix
    return -U
end


# Construct the contraction mapping
# write conditional value function for each action, 2 vector 5x1, one constant and the other varying

# Function that computes the Solution of the contraction mapping EV = T(EV)
function T(θ::Vector, β::Number, a::Vector)::Matrix
    k = length(a)                                      # State space dimensionality
    U = u(θ,a)                                         # Static utility
    Transition = Int[[2:k; k] ones(k,1)];              # Deterministic transition (degenerate transition probabilities)
    γ = Base.MathConstants.eulergamma                  # Euler's gamma

    # Iterate the Bellman equation until convergence
    Vbar = zeros(k, 2);
    Vbar1 = Vbar;
    err = 1;
    iter = 0;
    while err>1e-8
        V = γ .+ log.(sum(exp.(Vbar), dims=2))              # Compute value V(x)
        Vbar1 =  U + β * V[Transition]                      # Compute v-specific (without transition probabilities)
        #Vbar1 =  U + β * hcat(P[:,:,1]*V,P[:,:,2]*V)       # Compute v-specific (with transition probabilities)
        err = max(abs.(Vbar1 - Vbar)...);                   # Check distance (Chebychev row distance)
        iter += 1;
        #println("Error is ", err)
        Vbar = Vbar1                                        # Update value function
    end
    return Vbar
end;

# -- Solve Bellman Equation---------------------------------------------------------
V_bar = T(θ, β, a)
# Conditional choice probabilities
ChoiceProb = exp.(V_bar) ./sum(exp.(V_bar), dims = 2)

# Value of firm at state a = 4, ε0 = 1, ε1 = 1.5
ChoiceProb[4,1]*(V_bar[4,1] + 1) + ChoiceProb[4,2]*(V_bar[4,2] + 1.5)


# (a) Suppose a = 2. For what value of ε0 - ε1 is the firm indifferent between replacing or not?



# -- Simulate Data ---------------------------------------------------------

# Function that simulates the data
function simulate_data(θ::Vector, β::Number, a::Vector, N::Int)::Tuple

    Vbar = T(θ, β, a)                                            # Solve model
    ε = rand(Gumbel(0,1), N, 2)                                  # Draw shocks(ε_0, ε_1)
    a = collect(1:k)                                             # Generate State Space range, i.e. [1, 2, 3, 4, ...] 
    S = rand(a, N)                                               # Draw states S
    A = (((Vbar[S,:] + ε) * [-1;1]) .> 0)                        # Compute investment decisions (1 is i = 1)
    #A = convert(Vector{Int}, Vbar[S,2] + ε[:,2] .> Vbar[S,1] + ε[:,1])
    Snext = min.((S .+ 1).*(A.==0) .+ (1*A.==1) , maximum(a))    # Compute future state S'
    df = DataFrame(St = S, A = vec(A), St1 = Snext)              # Save Dataframe
    CSV.write(string(data_dir,"\\rust.csv"), df)
    return S, A
end;

# Simulate the data
N = 200000
St, A = simulate_data(θ, β, a, N)

println("We observe a replacements share of ",sum(A)/N)

# Comment: here we see a change in policies due to unobservables, must have been a shock given that
# for the same age we might see different policies (no deterministic rule from the econometrician pov)


# -- Compute ML Estimates ---------------------------------------------------------

# Function that computes the partial log-likelihood
function logL(β::Number, a::Vector, St::Vector, A::BitVector, θcand::Vector)::Number

    # Compute value
    Vbar = T(θcand, β, a)

    # Conditional choice probabilities
    ChoiceProb = exp.(Vbar ) ./sum(exp.(Vbar ), dims = 2)

    # Likelihood
    logL = sum(log.(ChoiceProb[St[A.==1],2])) + sum(log.(ChoiceProb[St[A.==0],1]))

    return -logL  # -logL to feed it into the minimizer
end;


# Initialize values for search
θstart = Float64[0,0];

# Optimize the function
θ_optim = optimize(x -> logL(β, a, St, A, x), θstart).minimizer 
print("Estimated thetas: $θ_optim (true = $θ)")

