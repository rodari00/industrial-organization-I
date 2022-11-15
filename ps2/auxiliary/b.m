
X = [ones(100,1),randn(100,19)];

param = randn(17,1);
%--------------------------------------------------------------------------%
% (1) DEFINE THE PARAMETERS                                                %
%--------------------------------------------------------------------------%
                                                                           %
paraconstant=param(1);                                                     % Constant
                                                                           %
paramX=param(2:9);                                                         % Exogenous Variables
                                                                           %
paraheterog1 = param(10);                                                  % Market Presence
                                                                           %
paraheterog2 = param(11);                                                  % Opportunity Cost
                                                                           %
parafirm1=param(12:17);                                                    % Competitive Effects
                                                                           %
%--------------------------------------------------------------------------%
% (2) COMPUTE THE EFFECT OF THE CONTROL VARIABLES                          %
%--------------------------------------------------------------------------%
                                                                           %
roleX=X(:,1:8)*paramX;                                                     %
                                                                           %
roleX=roleX*ones(1,k);                                                     %
                                                                           %
%--------------------------------------------------------------------------%
% (3) COMPUTE EFFECT OF HETEROGENEITY                                      %
%--------------------------------------------------------------------------%
                                                                           %
roleheter = X(:,9:14)*paraheterog1;                                        % Heterogeneity in Market Presence.
                                                                           %
roleheter=roleheter+X(:,15:20)*paraheterog2;                               % Heterogeneity in Opportunity Cost.
                                                                           %
%--------------------------------------------------------------------------%
% (4) COMPETITIVE EFFECTS                                                  %
%--------------------------------------------------------------------------%
                                                                           %
onothereffect=zeros(total*size(X,1),k);                                    % The sum of the competitive effects change by market structure.
                                                                           %     Dimension: (#markets*(2^#firms)) X #firms
                                                                           %                                          
for i=1:k                                                                  %
    effect=repindex(:,i)*parafirm1(i)*ones(1,k);                           %
    effect(:,i)=zeros(total*size(X,1),1);                                  % Makes sure that a firm does not negatively affect its own profit.
    onothereffect=onothereffect+effect;                                    % Sum over all the firms in the market.
end                                                                        %
clear effect                                                               %
                                                                           %
%--------------------------------------------------------------------------%
% (5) CONSTANT TERM                                                        %
%--------------------------------------------------------------------------%
                                                                           %
owneffect=[ones(total,k)*paraconstant];                                    %
                                                                           %
%--------------------------------------------------------------------------%
% (6) COMPUTES THE PART OF THE PROFIT THAT IS COMMON ACROSS SIMULATIONS    %
%--------------------------------------------------------------------------%
                                                                           %
common=roleX(indexheter)+roleheter(indexheter)+owneffect(diffdummies)+onothereffect;
