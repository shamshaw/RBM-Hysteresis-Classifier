% Version 1.000 
%
% Code provided by Geoff Hinton and Ruslan Salakhutdinov 
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

% This program trains Restricted Boltzmann Machine in which
% visible, binary, stochastic pixels are connected to
% hidden, binary, stochastic feature detectors using symmetrically
% weighted connections. Learning is done with 1-step Contrastive Divergence.   
% The program assumes that the following variables are set externally:
% maxepoch  -- maximum number of epochs
% numhid    -- number of hidden units 
% batchdata -- the data that is divided into batches (numcases numdims numbatches)

% Version 1.100
%
% Updated by Computational Cognitive Neuroscience Lab
% University of Padova
% ccnl.psy.unipd.it
%
% Implementation on graphic processors (GPUs) using MATLAB Parallel Computing Toolbox

momentum_GPU    = gpuArray(init_momentum);

%%%%%%%%% START POSITIVE PHASE %%%%%%%%%
poshidprobs_GPU = 1./(1 + exp(-data_mb * vishid_GPU - repmat(hidbiases_GPU, numcases, 1)));
posprods_GPU    = data_mb' * poshidprobs_GPU;
poshidact_GPU   = sum(poshidprobs_GPU);
posvisact_GPU   = sum(data_mb);
%%%%%%%%% END OF POSITIVE PHASE  %%%%%%%%%
poshidstates_GPU = poshidprobs_GPU > rand(numcases, numhid);

%%%%%%%%% START NEGATIVE PHASE  %%%%%%%%%
negdata_GPU     = 1./(1 + exp(-poshidstates_GPU * vishid_GPU' - repmat(visbiases_GPU, numcases, 1)));
neghidprobs_GPU = 1./(1 + exp(-negdata_GPU * vishid_GPU       - repmat(hidbiases_GPU, numcases, 1)));
negprods_GPU    = negdata_GPU' * neghidprobs_GPU;
neghidact_GPU   = sum(neghidprobs_GPU);
negvisact_GPU   = sum(negdata_GPU);
%%%%%%%%% END OF NEGATIVE PHASE %%%%%%%%%

err = gather(sqrt(sum(sum((data_mb - negdata_GPU).^2))));
if epoch > 5,
    momentum_GPU = gpuArray(final_momentum);
end

%%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%
vishidinc_GPU  = momentum_GPU * vishidinc_GPU  + epsilonw_GPU*( (posprods_GPU-negprods_GPU)/numcases_GPU - weightcost_GPU * vishid_GPU);
visbiasinc_GPU = momentum_GPU * visbiasinc_GPU + (epsilonvb_GPU/numcases_GPU)*(posvisact_GPU-negvisact_GPU);
hidbiasinc_GPU = momentum_GPU * hidbiasinc_GPU + (epsilonhb_GPU/numcases_GPU)*(poshidact_GPU-neghidact_GPU);
vishid_GPU     = vishid_GPU + vishidinc_GPU;
visbiases_GPU  = visbiases_GPU + visbiasinc_GPU;
hidbiases_GPU  = hidbiases_GPU + hidbiasinc_GPU;
%%%%%%%%% END OF UPDATES %%%%%%%%%
