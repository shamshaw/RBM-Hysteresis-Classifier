function [DN] = deeptrain_GPU(batchdata,layers,batchsize)
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

% Version 1.100
%
% Updated by Computational Cognitive Neuroscience Lab
% University of Padova
% ccnl.psy.unipd.it
%
% Implementation on graphic processors (GPUs) using MATLAB Parallel Computing Toolbox


tic
% DEEP NETWORK SETUP
% (parameters and final network weights will be saved in structure DN)
DN.layersize   = layers;            % network architecture
DN.nlayers     = length(DN.layersize);
DN.maxepochs   = 200;                    % unsupervised learning epochs
DN.batchsize   = batchsize;                   % mini-batch size
sparsity       = 1;                     % set to 1 to encourage sparsity on third layer
spars_factor   = 0.05;                  % how much sparsity?
epsilonw_GPU   = gpuArray(0.1);         % learning rate (weights)
epsilonvb_GPU  = gpuArray(0.1);         % learning rate (visible biases)
epsilonhb_GPU  = gpuArray(0.1);         % learning rate (hidden biases)
weightcost_GPU = gpuArray(0.0002);      % decay factor
init_momentum  = 0.5;                   % initial momentum coefficient
final_momentum = 0.9;                   % momentum coefficient

% load training dataset
% fname = ['Hyst_Grayscale_select_b14.mat'];
% load(fname);
fprintf(1,'\nUnsupervised training of a deep belief net\n');
DN.err = zeros(DN.maxepochs, DN.nlayers, 'single');
tic();

for layer = 1:DN.nlayers
    
    % for the first layer, input data are raw images
    % for next layers, input data are preceding hidden activations
    fprintf(1,'Training layer %d...\n', layer);
    if layer == 1
        data_GPU = gpuArray(single(batchdata));
    else
        data_GPU  = batchposhidprobs;
    end
    
    % initialize weights and biases
    numhid  = DN.layersize(layer);
    [numcases, numdims, numbatches] = size(data_GPU);
    numcases_GPU     = gpuArray(numcases);
    vishid_GPU       = gpuArray(0.1*randn(numdims, numhid, 'single'));
    hidbiases_GPU    = gpuArray(zeros(1,numhid, 'single'));
    visbiases_GPU    = gpuArray(zeros(1,numdims, 'single'));
    vishidinc_GPU    = gpuArray(zeros(numdims, numhid, 'single'));
    hidbiasinc_GPU   = gpuArray(zeros(1,numhid, 'single'));
    visbiasinc_GPU   = gpuArray(zeros(1,numdims, 'single'));
    batchposhidprobs = gpuArray(zeros(DN.batchsize, numhid, numbatches, 'single'));
    
    for epoch = 1:DN.maxepochs
        errsum = 0;
        for mb = 1:numbatches
            data_mb = data_GPU(:, :, mb);  % select one slice (mini-batch)
            rbm_GPU;  % learn an RBM with 1-step contrastive divergence
            errsum = errsum + err;
            if epoch == DN.maxepochs
                batchposhidprobs(:, :, mb) = poshidprobs_GPU;
            end
            if sparsity && (layer == 3)
                poshidact = sum(poshidprobs_GPU);
                Q = poshidact/DN.batchsize;
                if mean(Q) > spars_factor
                    hidbiases_GPU = hidbiases_GPU - epsilonhb_GPU*(Q-spars_factor);
                end
            end
        end
        DN.err(epoch, layer) = errsum;
    end
    % save learned weights
    DN.L{layer}.hidbiases  = gather(hidbiases_GPU);
    DN.L{layer}.vishid     = gather(vishid_GPU);
    DN.L{layer}.visbiases  = gather(visbiases_GPU);
    DN.L{layer}.batchposhidprobs = gather(batchposhidprobs);
    
end

DN.learningtime = toc();
fprintf(1, '\nElapsed time: %d \n', DN.learningtime);
fname = 'DBN_Grayscale_select_14.mat';
% save final network and parameters
save (fname, 'DN');
hidrecbiases = DN.L{1,1}.hidbiases;
visbiases = DN.L{1,1}.visbiases;
vishid = DN.L{1,1}.vishid;
save mnistvhclassify vishid hidrecbiases visbiases;
penrecbiases = DN.L{1,2}.hidbiases;
hidgenbiases = DN.L{1,2}.visbiases;
hidpen = DN.L{1,2}.vishid;
save mnisthpclassify hidpen penrecbiases hidgenbiases;
penrecbiases2 = DN.L{1,3}.hidbiases;
hidgenbiases2 = DN.L{1,3}.visbiases;
hidpen2 = DN.L{1,3}.vishid;
save mnisthp2classify hidpen2 penrecbiases2 hidgenbiases2;
toc
end