# industrial-organization-I
Material for IO I



function simulation_error(ϵ_D, a_u, log_z, b2, b1 = β1, m = μ,
                             d = δ, g = γ, sZ = σ_Z, S = 100, T = 50)

    E_g1 = (1+d*b2)^(-1)*((b1 - b2*μ)*0+ g*b2*sZ)
    E_g2 = (1+d*b2)^(-1)*((m +d*b1)*0 - g*sZ)
    E_pop = [E_g1,E_g2]
    S_sim = [0,0];    

    # Initialize simulation matrices
    for s in 1:S 
        # Generate log_a
        log_a = g*log_z[:,s] + a_u[:,s];
        # Generate log prices
        log_p = (1+d*b2)^(-1)*((d*b1 + m) .- (log_a + d*ϵ_D[:,s]));
        # Generate log quantities
        log_q = b1 .- (b2*log_p + ϵ_D[:,s])
        # Simulated moments (sum)
        S_sim += [dot(log_z[:,s],log_q)/T, dot(log_z[:,s],log_p)/T]
    end
    # compute average simulation
    E_sim = S_sim/S

    # Empirical 
    E_g_emp = [dot(log_z,log_q)/T, dot(log_z,log_p)/T]
    
    return(E_pop- E_sim)
end
