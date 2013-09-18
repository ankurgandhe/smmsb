function [] = MMSB_EM(adjMatrix,a,K,labelVec)

% AdjMatrix : NxN
% labelVec  : 1xN
% k : No of Clusters
% alpha : 1 x 1
% betahat  : 1 x k
% eta   : 1 x k
% sigma2 : 1

% true clusters
trueclus = [1 1 2 3 3 3 1 3 3 3 3 1 2 1 1 1 2 2];
%clus1 = [1 2 7 12 14 15 16]; clus2 = [3 13 17 18]; clus3 = [4 5 6 8 9 10 11];

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
oldLL = 0;

iter = 1;
conv = false;
while(~conv)
	[gammahat B ll] = MMSB_Estep(adjMatrix, alpha, diag(betahat), gammahat, 5);
	betahat = diag(B)';
	disp('Finished E-step...');
	[alpha dummy] = MMSB_Mstep(adjMatrix,labelVec,K,phi,gammahat,alpha);
	disp('Finished M-step...');
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
resClus = zeros(1,K);
for i=1:K
  clusAss = find(trueclus==i);
  %idx1 = unique(clus(clusAss));   count1 = hist(clus(clusAss),idx1);  [v1 id1] = max(count1); idx1(id1)
  clus(clusAss)
  maxClus = mode(clus(clusAss));  numPoints = sum(clus(clusAss)==maxClus);
  if(resClus(maxClus) < numPoints)
    resClus(maxClus) = numPoints;
  end
end
resClus
match = sum(resClus);
str = sprintf('Number of monks in right groups/clusters: %d',match);
disp(str);


end
