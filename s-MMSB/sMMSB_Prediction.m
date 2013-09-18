function [gammahat, lastll,yl_pred,newclus] = sMMSB_Prediction(y, alpha, beta, gammahat, maxit,eta,sigma2,TruelabelFile,TestPiFile)
% y: N*N
% alpha: 1*K
% B: K*K
% gammahat: N*K

B = diag(beta);
N=size(y,1);
K=size(B,1);

N_train  = size(gammahat,1);
N_test = size(y,2) - N_train;

yl_gold = load(TruelabelFile);
NormY = max(yl_gold(1:N_train));

gammahat = [gammahat ; repmat(alpha,N_test,1)+(rand(N_test,K)-0.5).*0.1];
labelVec = [yl_gold(1:N_train) zeros(1,N_test)]/max(yl_gold(1:N_train)); 

lastll = 1; % invalid value, ensuring no convergence in 1st iteration
iter = 0;

while true
    iter=iter+1;

    newgamma=repmat(alpha,N,1);
    sumgamma=sum(gammahat,2);
    e1=exp(psi(gammahat) - repmat(psi(sumgamma),1,K));  % exp(E_q log gamma_i)
    
    ll = 0;
    %ll = ll + sum(log(gammahat),1)*(alpha-1)';
    %ll = ll - sum(log(sum(gammahat,2))).*(sum(alpha)-K);
    %ll = ll - (sum(log(gamma(alpha)))-log(gamma(sum(alpha))))*N;
    ll = ll - sum(log(sum(e1,2))) * (2*N-2); % common denum for delta

    e2 = e1';
    e3 = e2(:)';
    Blin = B(:);
    nBlin = 1-Blin;
    for i=1:N
        plabel = - (((labelVec(i) - eta).^2)' * (labelVec(i) - eta).^2) / 2*sigma2;
				plabel = plabel(:);
        delta =reshape(e2(:,i)*e3,[K*K,N])...
            .* (Blin*y(i,:)+nBlin*(1-y(i,:))); %.* repmat(exp(plabel),1,N) ;  % size K^2*N
        sum_delta = reshape(sum(delta,1),[N,1]);     % size N*1
        ll = ll + sum(log(sum_delta)) - log(sum_delta(i));
        rev_sum_delta = 1./sum_delta;
        rev_sum_delta(i) = 0;
        
        norm_delta = delta .* repmat(rev_sum_delta',K*K,1);
		sum_delta_v = reshape(sum(reshape(norm_delta,[K,K,N]),1),[K,N]);
		sum_delta_u = reshape(sum(reshape(norm_delta,[K,K,N]),2),[K,N]);
		phi(i,:,:) = sum_delta_v';
        inphi(i,:,:) = sum_delta_u';
        
        newgamma = newgamma+ reshape(sum(reshape(delta,[K,K,N]),1),[K,N])'...
            .* repmat(rev_sum_delta,[1,K]);     % newgamma(j,:)
        
        %newgamma = [gammahat(1:N_train,:); newgamma(N_train+1:end,:)]; %repmat(alpha,N_test,1)+(rand(N_test,K)-0.5).*0.1];
        %if i <= N_train
        %    continue
        %end
        
        delta2 = reshape(delta*rev_sum_delta,[K,K]);% size K*K
        newgamma(i,:) = newgamma(i,:)+sum(delta2,2)';
        
    end

    % update gammahat
    gammahat = newgamma;
    
    llchange = (ll-lastll)/lastll;
    lastll = ll;
    fprintf('[Prediction] iter=%d\t%f\t%f\n', iter, ll, llchange);
    if (abs(llchange) < 1e-4)
        break;
    end
    if (iter >= maxit)
        %fprintf('exceed iteration limit %d\n', maxit);
        break;
    end
    phi_bar = (1/2) * (reshape(mean(phi(:,:,:),2),[N,K]) + reshape(mean(inphi(:,:,:),2),[N,K])) ; 
    yl_pred = eta*phi_bar';
    labelVec = [labelVec(1:N_train) yl_pred(N_train+1:end)];
end
phi_bar = (1/2) * (reshape(mean(phi(:,:,:),2),[N,K]) + reshape(mean(inphi(:,:,:),2),[N,K])) ; 
yl_pred = eta*phi_bar';
yl_pred = yl_pred*NormY;
%yl_gold =  yl_gold / NormY;
TrainError = sum(abs((yl_gold(1:N_train) - yl_pred(1:N_train))))/N_train;
TestError = sum(abs((yl_gold(N_train+1:end) - yl_pred(N_train+1:end))))/(N_test);
fprintf('Training Data Labeling Error: %f\n', TrainError);
fprintf('Test Data Labeling Error: %f\n', TestError);

% find clusters for each node

[vals clus] = max(gammahat(N_train+1:end,:)');
pi = load(TestPiFile);
pi = pi(N_train+1:end,:);
[vals trueclus] = max(pi');
resClus = zeros(1,K);
clusStr = {};
mapping = zeros(1,K);
rev_mapping = zeros(1,K);
for i=1:K
	clusAss = clus(find(trueclus==i));
	%fprintf('Cluster assignments %d:\n=====================\n',i);
  maxClus = mode(clusAss);	numPoints = sum(clusAss==maxClus);
	%maxClus
	if(resClus(maxClus) < numPoints)
		resClus(maxClus) = numPoints;
		mapping(maxClus) = i;
		rev_mapping(i) = maxClus;
	end
	clusStr = [clusStr sprintf('clus-%d',i)];
end
%fprintf('\n\nAfter assigning to majority, the cluster sizes are:\n');
%disp(clusStr);
%disp(resClus);
%match = sum(resClus);
%fprintf('\nNumber of nodes in right groups/clusters: %d\n',match);
%clus
% compute Rand Index
for i=1:K
	if(resClus(i)==0)
		x = setdiff(1:K, mapping);
		mapping(i) = x(1);
		rev_mapping(x(1)) = i;
	end
end
%mapping
%resClus
%rev_mapping

newclus = zeros(1,N_test);
for i=1:K
	newclus(find(clus==rev_mapping(i))) = i;
end

newclus
trueclus
match=sum(newclus==trueclus);
fprintf('\nNumber of nodes in right groups/clusters: %d\n',match);

