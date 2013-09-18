function [] = MMSB_EM(adjMatrix,a,K,labelVec)

% AdjMatrix : NxN
% labelVec  : 1xN
% k : No of Clusters
% alpha : 1 x 1
% betahat  : 1 x k
% eta   : 1 x k
% sigma2 : 1

% true clusters
clus1 = [1 2 7 12 14 15 16];
clus2 = [3 13 17 18];
clus3 = [4 5 6 8 9 10 11];

% initialization
N = size(adjMatrix,1);
alpha = 0.5+(abs(rand(1,K)-0.5)*0.1); % a * ones(1,K);
%phi = 1/K * ones(N,N,K);
gammahat = repmat(alpha,N,1)+(rand(N,K)-0.5).*0.1;
phi = rand(N,N,K);
betahat = ones(1,K) - 1e-1;% rand(1,K);
eta=zeros(1,K);
sigma2=0;
eps = 0.001;
%oldLL = getLL();
oldLL = 0;

iter = 1;
conv = false;
while(~conv)
	[gammahat B ll] = MMSB_Learn_InnerLoop_Orig(adjMatrix, alpha, diag(betahat), gammahat, 5);
	betahat = diag(B)';
	disp('Finished E-step...');
	[alpha dummy] = MMSB_Mstep(adjMatrix,labelVec,K,phi,gammahat,alpha);
	disp('Finished M-step...');
	%newLL = getLL();
	newLL = ll;
	str = sprintf('[EM] Iter: %d Loglikelihood: %g',iter,newLL);
	disp(str);
	iter = iter+1;
	gammahat
	betahat
	alpha
	conv = (abs(newLL-oldLL) < eps);
	%conv = (iter > 5);
	oldLL = newLL;
end

% find clusters for each node
[vals clus] = max(gammahat');
idx1 = unique(clus(clus1));   count1 = hist(clus(clus1),idx1);  [v1 id1] = max(count1);
idx2 = unique(clus(clus2));   count2 = hist(clus(clus2),idx2);  [v2 id2] = max(count2);
idx3 = unique(clus(clus3));   count3 = hist(clus(clus3),idx3);  [v3 id3] = max(count3);
match = v1+v2+v3;
str = sprintf('Number of monks in right groups/clusters: %d',match);
disp(str);
clus(clus1)
clus(clus2)
clus(clus3)



function [logL] = getLL()
	% compute log-likelihood
	logL = 0;
	for p=1:1:N
		term1 = sum(sum(reshape(phi(p,:,:),N,K) .* reshape(phi(:,p,:),N,K) .* (adjMatrix(p,:)'*log(betahat) + ( 1 - adjMatrix(p,:))'*log(1 - betahat))));
		term2 = 2 * sum(sum(reshape(phi(p,:,:),N,K) .* repmat(psi(gammahat(p,:)) - psi(sum(gammahat(p,:))),N,1)));
		term3 = sum((alpha - 1) .* (psi(gammahat(p,:)) - psi(sum(gammahat(p,:)))));
		term4 = - log(gamma(sum(gammahat(p,:)))) + sum(log(gamma(gammahat(p,:))) - ((gammahat(p,:)-1) .* (psi(gammahat(p,:)) - psi(sum(gammahat(p,:)))))); 
		otherDim = setdiff(1:N,p);
		term5 = -2 * sum(sum(reshape(phi(p,otherDim,:),N-1,K) .* log(reshape(phi(otherDim,p,:),N-1,K))));
		logL = logL + term1 + term2 + term3 + term4 + term5;
	end
	
	constTerms = N * (log(gamma(sum(alpha))) - sum(log(gamma(alpha))));
	logL = logL + constTerms;
end

end
