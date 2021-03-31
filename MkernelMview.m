function [result]=MkernelMview(K,label,alpha,beta)
%
% The code was developed by Dr. Huang (huangsd@std.uestc.edu.cn). 
% For any problem concerning the code, please feel free to contact Dr. Huang.
% You can run it at your own risk. 
%
% This package is free for academic usage. 
% For other purposes, please contact Prof. Zenglin Xu (zlxu@uestc.edu.cn)
%
% Input:
%   K: cell array, 1*view_num, each array is nSmp*nSmp.
%   label: the true class label.
%   alpha£ºtrade-off parameter
%   beta£ºparameter
%   num_view = size(K,2);
%   nSmp = size(K{1},1);
% Output: 
%   clustering result
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Ref:
% Shudong Huang, Zhao Kang, Ivor W. Tsang, Zenglin Xu. 
% Auto-weighted multi-view clustering via kernelized graph learning. 
% Pattern Recognition 88 (2019) 174¨C184. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
view_num = size(K,2);
[m,n]=size(K{1});
Z=eye(n);
c=length(unique(label));
%options=[];
%options = optimset( 'Algorithm','interior-point-convex','Display','off');

for iter = 1:20

    Zold=Z;
    Z= (Z+Z')/2;
    D = diag(sum(Z));
    L = D-Z;
    L(isnan(L)) = 0;
    [F, temp, ev]=eig1(L, c, 0);
    
     if iter == 1
        Wv = (1/view_num)*ones(1,view_num);
     end
  
   sumK = zeros(n,n);
   Kv = zeros(n,n);
   for v=1:view_num
       Kv=Wv(v)*K{v};
       sumK=sumK+Kv;
   end
   
    for ij=1:n
        for ji=1:n
            Fij(ji)=norm(F(ij,:)-F(ji,:))^2;
        end
        H=2*alpha*eye(n)+2*sumK;
        H=(H+H')/2;
%         sumKij = zeros(n,1);
%         Kvij = zeros(n,1);
%         for v=1:view_num
%             Kvij=Wv(v)*K{v}(:,ij);
%             sumKij=sumKij+Kvij;
%         end
        ff=0.5*beta*Fij'-2*sumK(:,ij);
% we use the free package to solve quadratic equation: http://sigpromu.org/quadprog/index.html
       % [Z(:,ij),err,lm] = qpas(H,ff,[],[],ones(1,n),1,zeros(n,1),ones(n,1));
% Z(:,ij)=quadprog(H,(beta/2*all'-2*K(:,ij))',[],[],ones(1,n),1,zeros(n,1),ones(n,1),Z(:,ij),options);
         Z(:,ij) = H\ff;
    end
     Z(Z<0)=0;
    % update Wv
    for v=1:view_num
        Wv(v) = 0.5/sqrt(trace(K{v}-2*K{v}*Z+Z'*K{v}*Z));
    end

    if iter>5 &&((norm(Z-Zold)/norm(Zold))<1e-3)
        break
    end

end
actual_ids= kmeans(F, c, 'emptyaction', 'singleton', 'replicates', 100, 'display', 'off');
[result] = ClusteringMeasure(actual_ids,label);
%
%     compute residue
%     tmp = lambda*trace(F'*LF*F)+ mu*trace(G'*LG*G)+ sum(sum((X-G*S*F').^2)) ;
%     residue = [residue tmp];

