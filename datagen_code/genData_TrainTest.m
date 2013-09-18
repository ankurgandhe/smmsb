function r = genData_TrainTest(alpha,n,d)

TrainSize = n*4/5; 

a = alpha * ones(1,d)   ;
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
beta = [0.4 0.5 0.3 0.5 0.3 0.6 0.5 0.7];
eta = [1 20 40 60 80 100 120 140];
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
	mu = sum(eta .* (sum(reshape(ptoq(i,:,:),n,d)) + sum(reshape(qtop(:,i,:),n,d))))/(2*(n-1));
	label(i) = normrnd(mu,var);
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
dlmwrite(sprintf('../eval_data/N%d/mmsb_model_%0.1g_%d_%d_train.txt',n,alpha,n,d),A(1:TrainSize,1:TrainSize));
dlmwrite(sprintf('../eval_data/N%d/mmsb_model_%0.1g_%d_%d_test.txt',n,alpha,n,d),A);
dlmwrite(sprintf('../eval_data/N%d/mmsb_model_blk_%0.1g_%d_%d_train.txt',n,alpha,n,d),Ablk(1:TrainSize,1:TrainSize));
dlmwrite(sprintf('../eval_data/N%d/mmsb_model_blk_%0.1g_%d_%d_test.txt',n,alpha,n,d),Ablk);
dlmwrite(sprintf('../eval_data/N%d/mmsb_model_%0.1g_%d_%d_pi_train.txt',n,alpha,n,d),pi(1:TrainSize,:));
dlmwrite(sprintf('../eval_data/N%d/mmsb_model_%0.1g_%d_%d_pi_test.txt',n,alpha,n,d),pi);
dlmwrite(sprintf('../eval_data/N%d/mmsb_model_%0.1g_%d_%d_label_train.txt',n,alpha,n,d),label(1:TrainSize));
dlmwrite(sprintf('../eval_data/N%d/mmsb_model_%0.1g_%d_%d_label_test.txt',n,alpha,n,d),label);
%spy(A)
