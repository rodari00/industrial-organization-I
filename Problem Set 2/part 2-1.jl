
# Use part 1 to generate data, then:
R = 1000 # num of sims

# 8 market structures coded as follows
Y = zeros(Int64, M)
for m in 1:M
    if (In[:,m] == [0, 0, 0])
        Y[m] = 1
    elseif (In[:,m] == [1, 0, 0])
        Y[m] = 2
    elseif (In[:,m] == [1, 1, 0])
        Y[m] = 3
    elseif (In[:,m] == [1, 0, 1])
        Y[m] = 4
    elseif (In[:,m] == [1, 1, 1])
        Y[m] = 5
    elseif (In[:,m] == [0, 1, 0])
        Y[m] = 6
    elseif (In[:,m] == [0, 1, 1])
        Y[m] = 7
    else # if (In[:,m] == [0, 0, 1])
        Y[m] = 8
    end
end

# compute P(Y|X) as conditional mean for each X = (X, Z_1, Z_2, Z_3)
P = zeros(Int64, (M, 8))
for m in 1:M, y in 1:8
    P[m,y] = Y[m] == y 
end

############# simulation objective function
σ = 1
# errors centered at 0
u = rand(Normal(0,σ^2), (F,M,R))

# objective function
function calc_mi(mean, data)
    M = length(data[:,1])
    μ = mean[1]
    Z = zeros(Float64, (F,M))
    In = zeros(Int64, (F,M))
    (X, Z[1,:], Z[2,:], Z[3,:], In[1,:], In[2,:], In[3,:]) = (data[:,1], data[:,2], data[:,3], data[:,4], data[:,5], data[:,6], data[:,7])

    # potential profit for each f,m,N,r
    profit = zeros(Float64, (F,M,F,R))
    for f in 1:F, m in 1:M, N in 1:F, r in 1:R
        profit[f,m,N,r] = X[m] * β - δ * log(N) - Z[f,m] * α - μ - u[f,m,r]
    end

    # indicate which outcomes are equilibria, for each m,r
    Eqm = zeros(Int64, (M,8,R))
    for m in 1:M, r in 1:R
        Eqm[m,1,r] = (profit[1,m,1,r] < 0) * (profit[2,m,1,r] < 0) * (profit[3,m,1,r] < 0)
        Eqm[m,2,r] = (profit[1,m,1,r] >= 0) * (profit[2,m,2,r] < 0) * (profit[3,m,2,r] < 0)
        Eqm[m,3,r] = (profit[1,m,2,r] >= 0) * (profit[2,m,2,r] >= 0) * (profit[3,m,3,r] < 0)
        Eqm[m,4,r] = (profit[1,m,2,r] >= 0) * (profit[2,m,3,r] < 0) * (profit[3,m,2,r] >= 0)
        Eqm[m,5,r] = (profit[1,m,3,r] >= 0) * (profit[2,m,3,r] >= 0) * (profit[3,m,3,r] >= 0)
        Eqm[m,6,r] = (profit[1,m,2,r] < 0) * (profit[2,m,1,r] >= 0) * (profit[3,m,2,r] < 0)
        Eqm[m,7,r] = (profit[1,m,3,r] < 0) * (profit[2,m,2,r] >= 0) * (profit[3,m,2,r] >= 0)
        Eqm[m,8,r] = (profit[1,m,2,r] < 0) * (profit[2,m,2,r] < 0) * (profit[3,m,1,r] >= 0)
    end

    # identify which markets have unique eqm
    Unique = zeros(Int64, (M,R))
    for m in 1:M, r in 1:R
        Unique[m,r] = sum(Eqm[m,:,r]) == 1
    end

    # define lower estimate H₁ and upper estimate H₂
    H₁ = zeros(Int64, (M,8,R))
    H₂ = zeros(Int64, (M,8,R))
    for m in 1:M, s in 1:8, r in 1:R
        H₁[m,s,r] = (Eqm[m,s,r] == 1) * Unique[m,r]
        H₂[m,s,r] = (Eqm[m,s,r] == 1)
    end
    H₁ = (1 / 1000) * sum(H₁, dims = 3)[:,:,1]
    H₂ = (1 / 1000) * sum(H₂, dims = 3)[:,:,1]

    # define objective function Q
    Q = zeros(Float64, M)
    for m in 1:M
        below = (P[m,:] - H₁[m,:]) .* (P[m,:] - H₁[m,:] .<= 0)
        below = sum(abs2, below)
        above = (P[m,:] - H₂[m,:]) .* (P[m,:] - H₂[m,:] .>= 0)
        above = sum(abs2, above)
        Q[m] = above + below
    end
    Q = sum(Q)
    return(Q)
end

