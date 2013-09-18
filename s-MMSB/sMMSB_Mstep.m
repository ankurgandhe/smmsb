function [alpha ,eta, sigma2] = sMMSB_Mstep(adjMatrix,labelVec,K,phi,inphi,gammahat,alpha)
% alpha : vector of size K

N=size(adjMatrix,1);


% update betahat
%betahat = zeros(1,K);
%for i=1:1:K
%	phiphiT = reshape(phi(:,:,i),N,N) .* reshape(outphi(:,:,i),N,N);
%	betahat(i) = sum(sum(adjMatrix .* phiphiT)) / sum(sum(phiphiT));
%end

% newton rapson update for alpha
step = 0.0001;
eps = 0.001;
conv = false;
while(~conv)
	gradL = zeros(1,K);
	gradL = N * (psi(sum(alpha)) - psi(alpha)) + sum(psi(gammahat) - repmat(psi(sum(gammahat,2)),1,K));
	hessL = N * (diag(psi(1,alpha)) - psi(1,sum(alpha)));
	newalpha = alpha + step * gradL * inv(hessL);
	conv = (max(abs(newalpha - alpha)) < eps);
	alpha = newalpha;
    %conv=true;
end

E_A = (1/2) * (reshape(mean(phi(:,:,:),2),[N,K]) + reshape(mean(inphi(:,:,:),2),[N,K])) ; 
E_ATA = zeros(N,K,K);
for p =1:N
    phi_p = [reshape(phi(p,:,:) ,[N,K]) ; reshape(inphi(p,:,:),[N,K]) ];
    phi_pt = phi_p';
		E_ATAp = zeros(K,K);
    for j = 2:2*N
        phi_pt = [phi_pt(:,2:end), phi_pt(:,1)]; 
        E_ATAp = E_ATAp + ( phi_pt *  phi_p);
    end
		E_ATA(p,:,:) = 1/(4*N*N) * (E_ATAp + (diag(reshape(sum(phi(p,:,:),2),[1,K])) + diag(reshape(sum(inphi(p,:,:),2),[1,K]))));
    %( reshape(phi(p,:,:),[N,K])' *  reshape(phi(p,:,:),[N,K]) + reshape(mean(inphi(:,:,:),2),[N,K])) ; 
end
E_ATA = reshape(sum(E_ATA,1),[K,K]);
eta = (inv(E_ATA)*E_A' * labelVec')';
%eta = sum(gammahat,1)/ (sum(sum(gammahat))) %For multinomial 
sigma2 = 1/N * ((labelVec * labelVec') - (labelVec * E_A * eta'));
