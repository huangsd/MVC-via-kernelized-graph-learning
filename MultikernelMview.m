function [result]=MultikernelMview(A,label,alpha,beta)
%
% The code was developed by Dr. Huang (huangsd@std.uestc.edu.cn). 
% For any problem concerning the code, please feel free to contact Dr. Huang.
% You can run it at your own risk. 
%
% This package is free for academic usage. 
% For other purposes, please contact Prof. Zenglin Xu (zlxu@uestc.edu.cn)
%
% Input:
%   A: cell array, 1*view_num, each array has the dimension of n*n*, i.e., 9 kernels.
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
view_num = size(A,2);
[m,n,nm]=size(A{1});
Z=eye(n);
c=length(unique(label));
% e=1/12*ones(12,1);
for iter = 1:20
    if iter == 1
       e=1/9*ones(9,view_num);
       Wv = (1/view_num)*ones(1,view_num);
    end
    
    
    for i=1:view_num
        K{i}=zeros(n);
        for j=1:9
           K{i}=K{i}+e(j,i)*A{i}(:,:,j);
        end
    end
  
    Zold=Z;
    Z= (Z+Z')/2;
    D = diag(sum(Z));
    L = D-Z;
    L(isnan(L)) = 0;
    [F, temp, ev]=eig1(L, c, 0);
    
    sumK = zeros(n);
    Kv = zeros(n);
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
        ff=beta/2*Fij'-2*sumK(:,ij);
        % we use the free package to solve quadratic equation:
        % http://sigpromu.org/quadprog/index.html
        % [Z(:,ij),err,lm] = qpas(H,ff,[],[],ones(1,n),1,zeros(n,1),ones(n,1));
        Z(:,ij) = H\ff;
    end
    
    Z(Z<0)=0;
       
    % update  e and Wv
    for i=1:view_num
        h=zeros(9,1);
        for j=1:9
            h(j)=trace(A{i}(:,:,j)-2*A{i}(:,:,j)*Z+Z'*A{i}(:,:,j)*Z);
        end
        for j=1:9
            e(j,i)=(h(j)*sum(1./h))^(-2);
        end
        Wv(i) = 0.5/sqrt(trace(K{v}-2*K{v}*Z+Z'*K{v}*Z));
        % Wv(i) = 0.5/sqrt(trace(K{v}-2*K{v}*Z+Z'*K{v}*Z));
    end
  
    
    if iter>10 &&((norm(Z-Zold)/norm(Zold))<1e-3)   
        break
    end
    
end
actual_ids= kmeans(F, c, 'emptyaction', 'singleton', 'replicates', 100, 'display', 'off');
[result] = ClusteringMeasure( actual_ids,label);