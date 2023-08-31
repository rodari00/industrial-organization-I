# Set parameters
θ = [1,3];
λ = 1;
β = 0.9;

# State space
k = 5;
s = Vector(1:k);

function compute_U(θ::Vector, s::Vector)::Matrix
    """Compute static utility"""
    u1 = - θ[1]*s       # Utility of not investing
    u2 = - θ[2]*ones(size(s))       # Utility of investing
    U = [u1 u2]                     # Combine in a matrix
    return U
end;


function compute_Vbar(θ::Vector, β::Number, s::Vector)::Matrix
    """Compute value function by Bellman iteration"""
    k = length(s)                                 # Dimension of the state space
    U = compute_U(θ, s)                           # Static utility
    #index_λ = Int[1:k [2:k; k]];                  # Mileage index
    index_A = Int[[2:k; k] ones(k,1)];                   # Investment index
    γ = Base.MathConstants.eulergamma             # Euler's gamma

    # Iterate the Bellman equation until convergence
    Vbar = zeros(k, 2);
    Vbar1 = Vbar;
    dist = 1;
    iter = 0;
    while dist>1e-8
        V = γ .+ log.(sum(exp.(Vbar), dims=2))     # Compute value
        #expV = V[index_λ] * [1-λ; λ]               # Compute expected value
        Vbar1 =  U + β * V[index_A]             # Compute v-specific
        dist = max(abs.(Vbar1 - Vbar)...);         # Check distance
        iter += 1;
        Vbar = Vbar1                               # Update value function
    end
    return Vbar
end;


compute_U(θ,s)

# Compute value function
V_bar = compute_Vbar(θ, β, s)

function generate_data(θ::Vector, β::Number, s::Vector, N::Int)::Tuple
    """Generate data from primitives"""
    Vbar = compute_Vbar(θ, β, s)             # Solve model
    ε = rand(Gumbel(0,1), N, 2)                 # Draw shocks
    St = rand(s, N)                             # Draw states
    A = (((Vbar[St,:] + ε) * [-1;1]) .> 0)      # Compute investment decisions
    #δ = (rand(Uniform(0,1), N) .< λ)            # Compute mileage shock
    St1 = min.((St .+ 1).*(A.==0) .+ (1*A.==1) , maximum(s))    # Compute future state
    #St1 = min.(St .* (A.==0) + δ, max(s...))    # Compute neSr state
    df = DataFrame(St=St, A=A, St1=St1)         # Dataframe
    CSV.write(string(data_dir,"\\rust.csv"), df)
    return St, A, St1
end;

N = 200000;
St, A, St1 = generate_data(θ, β, s, N);

sum(A)/N

function logL_Rust(θ0::Vector, λ::Number, β::Number, s::Vector, St::Vector, A::BitVector)::Number
    """Compute log-likelihood functionfor Rust problem"""
    # Compute value
    Vbar = compute_Vbar(θ0, β, s)

    # Expected choice probabilities
    EP = exp.(Vbar[:,2]) ./ (exp.(Vbar[:,1]) + exp.(Vbar[:,2]))

    # Likelihood
    logL = sum(log.(EP[St[A.==1]])) + sum(log.(1 .- EP[St[A.==0]]))
    return -logL
end;

logL_trueθ = logL_Rust(θ, λ, β, s, St, A);


# Select starting values
θ0 = Float64[0,0];

# Optimize
θ_R = optimize(x -> logL_Rust(x, λ, β, s, St, A), θ0).minimizer;
print("Estimated thetas: $θ_R (true = $θ)")