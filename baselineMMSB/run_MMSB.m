function e = run_MMSB(AdjTrainFile,LabelTrainFile,AdjTestFile,LabelTestFile,TrainPiFile,a,K,runid,outdir)
    TrainadjMatrix = load(AdjTrainFile);
    
    TrainlabelVec = load(LabelTrainFile);
    
    TestadjMatrix = load(AdjTestFile);
    
    %TestlabelVec = load(LabelTestFile);
    t = cputime;
    [alphahat,betahat,gammahat] = MMSB_EM(TrainadjMatrix,a,K,TrainlabelVec,TrainPiFile,100);
    e = cputime-t;
    [Newgammahat, total_ll,yl_pred] = MMSB_Prediction(TestadjMatrix, alphahat, betahat, gammahat, 100,LabelTestFile);
    dlmwrite(sprintf('%s/gammahat_%s.txt',outdir,runid),Newgammahat);
    dlmwrite(sprintf('%s/LogL_%s.txt',outdir,runid),total_ll);
    dlmwrite(sprintf('%s/ypred_%s.txt',outdir,runid),yl_pred);
end