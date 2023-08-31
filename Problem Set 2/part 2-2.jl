
#---- Load Packages ----
using Optim
#----------------------------------------------------------
#------- Construct Initial Confidence Region---------------
#----------------------------------------------------------

# Let min_mi be the minimized value of the objective calc_mi(μ; data) = a_n Q_n(μ, data)

res = optimize(μ -> calc_mi(μ, df), 1.5, 2.5)
min_mi = Optim.minimum(res)

# Define c0 as the 1.25*min_mi following Ciliberto-Tamer

c0 = 1.25 * min_mi

# Find initial confidence region by evaluating the obj function in a grid
# of 50 points (from -1 to 4.0)
calc_mi0 = μ -> calc_mi(μ, df)
MU = -1:0.1:4 
MU_I = MU[calc_mi0.(MU) .<= c0]
μ0_lb = minimum(MU_I)
μ0_ub = maximum(MU_I)


# Generate subsamples with a subsample size of M/4 following Ciliberto-Tamer
# Compute max value of the obj function of each subsample over initial 
# confidence region by finding the min of the negative obj function, and
# subtract from that the min of the obj function in the same region
# following Ciliberto-Tamer to correct for potential misspecification.

M = 100
B = 100

# Write a function subsample(data, size) to generate subsamples from data of a particular size
function subsample(data, size)
    size = Int(ceil(size))
    sub = sample(1:100, size, replace=false)
    return(data[sub,:])
end


function calc_mi_subsample(b) # arbitrary integer b
    sub_data = subsample(df, M/4)
    obj_values = map(μ -> calc_mi(μ, sub_data), MU_I) #Calculate a_n Q_n(μ, sub_data) for all μ in μ_I
    max_mi_sub = maximum(obj_values) # C_n = sup_{μ ∈ μ_I} a_n Q_n(μ, sub_data)
    min_mi_sub = minimum(obj_values) # to correct for misspecification
    return max_mi_sub - min_mi_sub 
end

# Take 1/4 the 95th percentile and set equal to c1 to compute 95% CI (1/4
# because #subsample=M/4)

c1_subsamples = map(calc_mi_subsample, 1:B)
c1 = 1/4 * quantile(c1_subsamples, 0.95)

# compute ci1 using Ciliberto and Tamer's estimator modified for
# misspecification
MU_I = MU[calc_mi0.(MU) .- min_mi  .<= c1] # update range of mu grid
μ1_lb = minimum(MU_I)
μ1_ub = maximum(MU_I)

#Last, repeat the subsampling procedure a few times to update the bounds
#further. In doing so, use a finer grid to obtain more accurate bounds (e.g.,
#MU = -1:0.025:4).


c2_subsamples = map(calc_mi_subsample, 1:B)
c2 = 1/4 * quantile(c2_subsamples, 0.95)
MU = -1:0.025:4 # make grid finer from here on
MU_I = MU[calc_mi0.(MU) .- min_mi  .<= c2] # update range of mu grid
μ2_lb = minimum(MU_I)
μ2_ub = maximum(MU_I)

c3_subsamples = map(calc_mi_subsample, 1:B)
c3 = 1/4 * quantile(c3_subsamples, 0.95)
MU_I = MU[calc_mi0.(MU) .- min_mi  .<= c3] # update range of mu grid
μ3_lb = minimum(MU_I)
μ3_ub = maximum(MU_I)

c4_subsamples = map(calc_mi_subsample, 1:B)
c4 = 1/4 * quantile(c4_subsamples, 0.95)
MU_I = MU[calc_mi0.(MU) .- min_mi  .<= c4] # update range of mu grid
μ4_lb = minimum(MU_I)
μ4_ub = maximum(MU_I)