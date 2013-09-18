function [alpha betahat] = MMSB_Mstep(adjMatrix,labelVec,K,phi,gamma,alpha)
% alpha : vector of size K

N=size(adjMatrix,1);
betahat = zeros(1,K);

% update betahat
%for i=1:1:K
%	phiphiT = reshape(phi(:,:,i),N,N) .* reshape(phi(:,:,i),N,N)';
%	betahat(i) = sum(sum(adjMatrix .* phiphiT)) / sum(sum(phiphiT));
%end

% newton rapson update for alpha
step = 0.005;
eps = 0.01;
conv = false;
iter = 1;
while(~conv)
	gradL = zeros(1,K);
	gradL = N * (psi(sum(alpha)) - psi(alpha)) + sum(psi(gamma) - repmat(psi(sum(gamma,2)),1,K));
	hessL = N * (diag(psi(1,alpha)) - psi(1,sum(alpha)));
	%inv(hessL)
	%gradL
	newalpha = alpha + step * gradL * inv(hessL);
	conv = (max(abs(newalpha - alpha)) < eps);
	%conv = (iter > 5);
	iter = iter+1;
	alpha = newalpha;
end
