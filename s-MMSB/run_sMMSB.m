function [Newgammahat ,yl_pred,ClusterAssign] = run_sMMSB(AdjTrainFile,LabelTrainFile,AdjTestFile,LabelTestFile,TrainPiFile,TestPiFile,a,K,runid,outdir)
    TrainadjMatrix = load(AdjTrainFile);
    
    TrainlabelVec = load(LabelTrainFile);
    
    TestadjMatrix = load(AdjTestFile);
    
    %TestlabelVec = load(LabelTestFile);
    t = cputime;
    [alphahat,betahat,gammahat,etahat,sigma2hat] = MMSB_EM(TrainadjMatrix,a,K,TrainlabelVec,TrainPiFile,200); 
    e = cputime-t;
    [Newgammahat, total_ll,yl_pred, ClusterAssign] = sMMSB_Prediction(TestadjMatrix, alphahat, betahat, gammahat, 100, etahat, sigma2hat,LabelTestFile,TestPiFile);
    dlmwrite(sprintf('%s/gammahat_%d_.txt',outdir,runid),gammahat);
    dlmwrite(sprintf('%s/LogL_%d_.txt',outdir,runid),total_ll);
    dlmwrite(sprintf('%s/ypred_%d_.txt',outdir,runid),yl_pred);
end