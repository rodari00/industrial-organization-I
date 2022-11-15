
R = 100;
m = 100;
k = 3;
total = 2^k;
index=makeindex(k); 
epsi = randn(m,k*R);


indexheter=zeros(m*total,k);   
tempheter=reshape(1:k*m,m,k);                              %                                   %
for j=1:m                                                         %
    indexheter((j-1)*total+1:(j-1)*total+total,:)=repmat(tempheter(j,:),total,1);                                                       %
end     

repindex=repmat(index,m,1);      
diffdummies=repmat(reshape(1:total*k,total,k),m,1);                              % Assigns the market unobservable to each of the k firms.
 
% Will have # markets x # configurations bounds
sumupp=zeros(m,total);                                                  % Matrix for 'numerators' of upper bound.
sumlow=sumupp;   

totcount=R*ones(m,1);                                                   %
temp=zeros(m,1);   



i =1;

epsitemp=epsi(:,1+k*(i-1):k*(i-1)+k);    

epsitemp=epsitemp(indexheter);    
common = randn(m,1)*ones(1,k);
common = common(indexheter);


equiln=epsitemp+common;    

ispos = (equil>=0);
equil=(equiln>=0)==repindex; % check if those who enter make positive profits,
                            % and those who do not enter make negative
                            % profit.
sumequil=sum(equil,2);     % sum across firms
% for each configuration x market, check if all 3 are in equilibrium
vectorequil=(sumequil==k*ones(total*m,1));

cumsumvectorequil= cumsum(vectorequil);  

sumvectorequil=cumsumvectorequil(total:total:m*total);  % take the last one of every market

sumvectorequil(2:size(sumvectorequil,1),:)=diff(sumvectorequil); 
