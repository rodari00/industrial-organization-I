
using LinearAlgebra
using Random,Distributions
using Plots
using Optim
using Statistics

# Define parameters
# Number of products
J  = 3;
# Number of markets
M  = 100;
R = 100;
# -----------------------------------------------------------------------------------

# Simulate Data
# product x market covariates

ϵ  = rand(Float64, (J,R*M));

function makeindex(k)                                              
                                                                           
    total=2^k;                                                                 
    index=zeros(total,k);                                                      
    i=1;                                                                       
    for i=1:k                                                                  
        ii=1;                                                                  
        cc=1;                                                                  
        c=floor(Int,total/(2^i));                                                         
        while ii<=total                                                        
            if cc <=c                                                          
                index[ii,i]=1;                                                 
                cc=cc+1;                                                       
                ii=ii+1;                                                       
            else                                                               
                ii=ii+c;                                                       
                cc=1;                                                          
            end                                                                
        end                                                                    
    end      
    
    return(index)
                                                                            
end                                                                        


ϵ[:,1 + k*(i-1):k*(i-1) + k]

