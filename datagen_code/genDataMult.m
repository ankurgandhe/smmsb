function r = genDataMult(alpha,n,d)

a = alpha * ones(1,d);
eps = 0.005;% alpha/2;

pi = zeros(n,d);
i=1;	iter = 0;
while(i<=n)
	pi(i,:) = drchrnd(a,1);
  try
    z = mnrnd(1,pi(i,:),1); 
  catch err
		i = i-1;
  end
	i = i+1;
	iter = iter+1;
end
disp('done pi!');
iter


[prob clus] = max(pi');
%beta = [0.4 0.5 0.3 0.5 0.3 0.6 0.5 0.7];
%eta = [1 20 40 60 80 100 120 140];
beta = [0.4 0.5 0.6 0.5];
eta = [1 2 3 4]; % 40 60];
%beta = rand(1,d);
%beta = beta ./ sum(beta);

ptoq = zeros(n,n,d);
qtop = zeros(n,n,d);
A = zeros(n,n);
for i=1:1:n
	for j=1:1:n
		ptoq(i,j,:) = mnrnd(1,pi(i,:),1); % generate z(p->q)
		qtop(j,i,:) = mnrnd(1,pi(j,:),1); % generate z(p->q)
		p = eps;
		if(ptoq(i,j,:) == qtop(j,i,:))
			p = beta(find(ptoq(i,j,:) == 1));
		end
		A(i,j) = binornd(1,p);
	end
end

var = 1;
for i=1:1:n
	theta = (sum(reshape(ptoq(i,:,:),n,d)) + sum(reshape(qtop(:,i,:),n,d)))/(2*n);
	c = find(mnrnd(1,theta)>0);
	label(i) = eta(c);
end

%Anew = [];
%Ablk = [];
nodeOrder = [];
for i=1:1:d
  idx = find(clus==i);
	size(idx)
	nodeOrder = [nodeOrder idx];
	%Anew = [Anew ; A(idx,:)];
end

Ablk = (A([nodeOrder],[nodeOrder]));
sum(sum(Ablk))
imagesc(Ablk)
dlmwrite('../data/mmsb_model_0.05_100_4_mult_lessnoise_train.txt',A(1:100,1:100));
dlmwrite('../data/mmsb_model_0.05_100_4_mult_lessnoise.txt',A);
dlmwrite('../data/mmsb_model_blk_0.05_100_4_mult_lessnoise.txt',Ablk);
dlmwrite('../data/mmsb_model_0.05_100_4_mult_lessnoise_pi_train.txt',pi(1:100,:));
dlmwrite('../data/mmsb_model_0.05_100_4_mult_lessnoise_pi.txt',pi);
dlmwrite('../data/mmsb_model_0.05_100_4_mult_lessnoise_label.txt',label);
dlmwrite('../data/mmsb_model_0.05_100_4_mult_lessnoise_label_train.txt',label(1:100));
%spy(A)
