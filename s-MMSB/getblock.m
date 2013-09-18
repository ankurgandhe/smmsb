function Ablk = getblock(A,clus,d)
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
end