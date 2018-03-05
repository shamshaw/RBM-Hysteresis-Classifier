function [w1,w_class,test_err,test_crerr,train_err,train_crerr,confusionMatrixTrain,...
    confusionMatrixTest,meanTestError,meanTrainError,...
    testPredProbs, trainPredProbs] = finetune_RBM_classifier...
    (DN,batchdata,testbatchdata,batchtargets,testbatchtargets)
% Version 1.000
%
% Code provided by Ruslan Salakhutdinov and Geoff Hinton
%
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our
% web page.
% The programs and documents are distributed without any warranty, express or
% implied.  As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own risk.

% This program fine-tunes an autoencoder with backpropagation.
% Weights of the autoencoder are going to be saved in mnist_weights.mat
% and trainig and test reconstruction errors in mnist_error.mat
% You can also set maxepoch, default value is 200 as in our paper.

% Version 1.1
%
% Updated by Scott D. Hamshaw
% University of Vermont
% shamshaw@uvm.edu
%
% Modified to fine-tune one layer RBM classifier model and output a
% confusion matrix
%


maxepoch=200;
fprintf(1,'\nTraining discriminative model by minimizing cross entropy error. \n');

hidrecbiases = DN.L{1,1}.hidbiases;
visbiases = DN.L{1,1}.visbiases;
vishid = DN.L{1,1}.vishid;

K = 15; % set number of classes

%%%% PREINITIALIZE WEIGHTS OF THE DISCRIMINATIVE MODEL%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

w1=[vishid; hidrecbiases];
w_class = 0.1*randn(size(w1,2)+1,K);
 

%%%%%%%%%% END OF PREINITIALIZATIO OF WEIGHTS  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

l1=size(w1,1)-1;
l2=size(w1,2);
l5=K; 
test_err=[];
train_err=[];


for epoch = 1:maxepoch

%%%%%%%%%%%%%%%%%%%% COMPUTE TRAINING MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
err_cr=0;
counter=0;
[numcases,~,numbatches]=size(batchdata);
N=numcases;

confusionMatrixTest = zeros(K+1,K+1);
confusionMatrixTrain = zeros(K+1,K+1);
trainPredProbs = [];
 for batch = 1:numbatches
  data = batchdata(:,:,batch);
  target = batchtargets(:,:,batch);
  data = [data ones(N,1)];
  w1probs = 1./(1 + exp(-data*w1)); w1probs = [w1probs  ones(N,1)];
  targetout = exp(w1probs*w_class);
  targetout = targetout./repmat(sum(targetout,2),1,K);
  trainPredProbs = [trainPredProbs; targetout];
  [I,J]=max(targetout,[],2);
  [I1,J1]=max(target,[],2);
  counter=counter+length(find(J==J1));
  err_cr = err_cr- sum(sum( target(:,1:end).*log(targetout))) ;
  % create matrix of error by class type
  for i = 1:numcases
     confusionMatrixTrain(J(i),J1(i)) = confusionMatrixTrain(J(i),J1(i))+1;
  end
  
 end
 

 confusionMatrixTrain(1:K,K+1) = diag(confusionMatrixTrain(1:K,1:K))./sum(confusionMatrixTrain(1:K,1:K),2);
 confusionMatrixTrain(K+1,1:K) = diag(confusionMatrixTrain(1:K,1:K))'./sum(confusionMatrixTrain(1:K,1:K),1);
 
 train_err(epoch)=(numcases*numbatches-counter);
 train_crerr(epoch)=err_cr/numbatches;

%%%%%%%%%%%%%% END OF COMPUTING TRAINING MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%% COMPUTE TEST MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
err_cr=0;
counter=0;
[testnumcases,~,testnumbatches]=size(testbatchdata);
N=testnumcases;
testPredProbs = [];
for batch = 1:testnumbatches
  data = testbatchdata(:,:,batch);
  target = testbatchtargets(:,:,batch);
  data = [data ones(N,1)];
  w1probs = 1./(1 + exp(-data*w1)); w1probs = [w1probs  ones(N,1)];
  targetout = exp(w1probs*w_class);
  targetout = targetout./repmat(sum(targetout,2),1,K);
  testPredProbs = [testPredProbs; targetout];
  [I,J]=max(targetout,[],2);
  [I1,J1]=max(target,[],2);
  counter=counter+length(find(J==J1));
  err_cr = err_cr- sum(sum( target(:,1:end).*log(targetout))) ;
  
   % create matrix of error by class type
 
    for i = 1:numcases
     confusionMatrixTest(J(i),J1(i)) = confusionMatrixTest(J(i),J1(i)) +1;
     end
 
end


 confusionMatrixTest(1:K,K+1) = diag(confusionMatrixTest(1:K,1:K))./sum(confusionMatrixTest(1:K,1:K),2);
 confusionMatrixTest(K+1,1:K) = diag(confusionMatrixTest(1:K,1:K))'./sum(confusionMatrixTest(1:K,1:K),1);
  
 test_err(epoch)=(testnumcases*testnumbatches-counter);
 test_crerr(epoch)=err_cr/testnumbatches;
 fprintf(1,'Before epoch %d Train # misclassified: %d (from %d). Test # misclassified: %d (from %d) \t \t \n',...
            epoch,train_err(epoch),numcases*numbatches,test_err(epoch),testnumcases*testnumbatches);

%%%%%%%%%%%%%% END OF COMPUTING TEST MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 for batch = 1:numbatches
 fprintf(1,'epoch %d batch %d\r',epoch,batch);
 
 data = batchdata(:,:,batch); 
 targets = batchtargets(:,:,batch);

%%%%%%%%%%%%%%% PERFORM CONJUGATE GRADIENT WITH 3 LINESEARCHES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  max_iter=3;

  if epoch<6  % First update top-level weights holding other weights fixed. 
    N = size(data,1);
    XX = [data ones(N,1)];
    w1probs = 1./(1 + exp(-XX*w1)); %w1probs = [w1probs  ones(N,1)];

    VV = (w_class(:)')';
    Dim = [l2; l5];
    [X, ~] = minimize(VV,'CG_CLASSIFY_INIT_RBM',max_iter,Dim,w1probs,targets,K);
    w_class = reshape(X,l2+1,l5);

  else
    VV = [w1(:)' w_class(:)']';
    Dim = [l1; l2; l5];
    [X, ~] = minimize(VV,'CG_CLASSIFY_RBM',max_iter,Dim,data,targets,K);

    w1 = reshape(X(1:(l1+1)*l2),l1+1,l2);
    xxx = (l1+1)*l2;
    w_class = reshape(X(xxx+1:xxx+(l2+1)*l5),l2+1,l5);

  end
%%%%%%%%%%%%%%% END OF CONJUGATE GRADIENT WITH 3 LINESEARCHES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 end
 
 save weights_RBM_Grayscale_select_25 w1 w_class
 save error_RBM_Grayscale_select_25 test_err test_crerr train_err train_crerr confusionMatrixTrain confusionMatrixTest;
 

end

meanTestError = 1-sum(diag(confusionMatrixTest(1:14,1:14)))/sum(sum(confusionMatrixTest(1:14,1:14)));
meanTrainError = 1-sum(diag(confusionMatrixTrain(1:14,1:14)))/sum(sum(confusionMatrixTrain(1:14,1:14)));
