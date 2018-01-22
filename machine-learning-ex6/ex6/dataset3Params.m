function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
 
v = [0.01 0.03 0.1 0.3 1 3 10 30];
m = size(v,2);
predicts = ones(m, 3);

#for i=1:m
#    for j=1:m
#        c_C = v(i);
#        c_sigma = v(j);
#        model= svmTrain(X, y, c_C, @(x1, x2) gaussianKernel(x1, x2, c_sigma)); 
#        c_predictions = svmPredict(model, Xval);
#        c_prediction = mean(double(c_predictions ~= yval));
#        predicts((i-1)*m+j, :) = [c_C c_sigma c_prediction];
        #if (i==1) and (j==1)
        #    prediction = c_prediction;
        #elseif c_prediction < prediction
        #    prediction = c_prediction;
        #    C = c_C;
        #    sigma = c_sigma;
        #endif
#    endfor
#endfor
#predicts
#[predict, i_p] = min(abs(predicts(:,3)))
#C = predicts(i_p,1)
#sigma = predicts(i_p, 2)
C = 1;
sigma = 0.1;
% =========================================================================

end