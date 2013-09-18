function [alpha,betahat,gammahat] = MMSB_EM(adjMatrix,a,K,labelVec,filename,maxit)

% AdjMatrix : NxN
% labelVec  : 1xN
% k : No of Clusters
% alpha : 1 x 1
% betahat  : 1 x k
% eta   : 1 x k
% sigma2 : 1

% initialization
N = size(adjMatrix,1);
alpha = 0.5+(abs(rand(1,K)-0.5)*0.1); % a * ones(1,K);
phi = 1/K * ones(N,N,K);
gammahat = repmat(alpha,N,1)+(rand(N,K)-0.5).*0.1;

betahat = ones(1,K) - 1e-1;% rand(1,K);
eta=zeros(1,K);
sigma2=0;
eps = 1e-5;
oldLL = 0;
dlmwrite('../gamma.txt',gammahat);
dlmwrite('../alpha.txt',alpha);

iter = 1;
conv = false;
while(~conv)
	[gammahat B ll] = MMSB_Estep(adjMatrix, alpha, diag(betahat), gammahat, 5);
	betahat = diag(B)';
	%disp('Finished E-step...');
	[alpha dummy] = MMSB_Mstep(adjMatrix,labelVec,K,phi,gammahat,alpha);
	%disp('Finished M-step...');
	newLL = ll;
	str = sprintf('[EM] Iter: %d Loglikelihood: %g',iter,newLL);
	disp(str);
	iter = iter+1;
	%gammahat
	%betahat
	%alpha
	conv = (abs((newLL-oldLL)/oldLL) < eps) || (iter > maxit);
	oldLL = newLL;
end
gammahat
betahat
alpha
	
% find clusters for each node
[vals clus] = max(gammahat');
pi = load(filename);
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

newclus = zeros(1,N);
for i=1:K
	newclus(find(clus==rev_mapping(i))) = i;
end

%newclus
%trueclus
match=sum(newclus==trueclus);
fprintf('\nNumber of nodes in right groups/clusters: %d\n',match);
n11=0;
for i=1:K
	sz(i)=sum((newclus==trueclus) & (trueclus==i));
	n11 = n11 + sz(i)*(sz(i)-1)/2;
end
%sz
n00=0;
for i=1:K-1
	n00 = n00 + sz(i)*(sum(sz((i+1):K)));
end

%n11
%n00
randIdx = 2*(n00+n11)/(N*(N-1))

end
