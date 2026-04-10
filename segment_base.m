% used to determine whether the model is being trained or tested
% set to true to retrain

trainModel = false;

% load images

imageDir = fullfile('cw_data', 'images');
labelDir = fullfile('cw_data', 'segmentation');


imageDS = imageDatastore(imageDir);

% Class and colour definitions

classNames = {'background', 'crop', 'weed'};
labelColours = [
    0,   0,   0;   % background — black
    0, 255,   0;   % crop — green
    255,  0,   0   % weed — red
];

labelDS = pixelLabelDatastore(labelDir, classNames, labelColours);

% Combining images and labels

combinedDS = combine(imageDS, labelDS);

numImages = 50;
numTrain = 40;
numTest = 10;

% Shuffle the order of the images
rng(42);
shuffledIdx = randperm(numImages);
trainIdx = shuffledIdx(1:numTrain);
testIdx = shuffledIdx(numTrain+1:numImages);
trainDS = subset(combinedDS, trainIdx);
testDS = subset(combinedDS, testIdx);

% Preprocessing

inputSize = [224 224];

% Transform

trainDS = transform(trainDS, @(data) preprocessingData(data, inputSize));
testDS = transform(testDS, @(data) preprocessingData(data, inputSize));

% ResNet-18 initialisation

net = imagePretrainedNetwork("resnet18");
lg = layerGraph(net);

lg = removeLayers(lg, 'pool5');
lg = removeLayers(lg, 'fc1000');
lg = removeLayers(lg, 'prob');

% Network decoder architecture

lg = addLayers(lg, transposedConv2dLayer(2, 256, 'Stride', 2, 'Cropping', 'same', 'Name', 'upsample1'));
lg = connectLayers(lg, 'res5b_relu', 'upsample1');
lg = addLayers(lg, depthConcatenationLayer(2, 'Name', 'concat1'));
lg = connectLayers(lg, 'upsample1', 'concat1/in1');
lg = connectLayers(lg, 'res4b_relu', 'concat1/in2');
lg = residualBlock(lg, 256, 'deBlock1', 'concat1');

lg = addLayers(lg, transposedConv2dLayer(2, 128, 'Stride', 2, 'Cropping', 'same', 'Name', 'upsample2'));
lg = connectLayers(lg, 'deBlock1_relu2', 'upsample2');
lg = addLayers(lg, depthConcatenationLayer(2, 'Name', 'concat2'));
lg = connectLayers(lg, 'upsample2', 'concat2/in1');
lg = connectLayers(lg, 'res3b_relu', 'concat2/in2');
lg = residualBlock(lg, 128, 'deBlock2', 'concat2');

lg = addLayers(lg, transposedConv2dLayer(2, 64, 'Stride', 2, 'Cropping', 'same', 'Name', 'upsample3'));
lg = connectLayers(lg, 'deBlock2_relu2', 'upsample3');
lg = addLayers(lg, depthConcatenationLayer(2,'Name','concat3'));
lg = connectLayers(lg, 'upsample3', 'concat3/in1');
lg = connectLayers(lg, 'res2b_relu', 'concat3/in2');
lg = residualBlock(lg, 64, 'deBlock3', 'concat3');

lg = addLayers(lg, transposedConv2dLayer(2, 32, 'Stride', 2, 'Cropping', 'same', 'Name', 'upsample4'));
lg = connectLayers(lg,'deBlock3_relu2', 'upsample4');
lg = addLayers(lg, depthConcatenationLayer(2,'Name','concat4'));
lg = connectLayers(lg, 'upsample4', 'concat4/in1');
lg = connectLayers(lg, 'conv1_relu', 'concat4/in2');
lg = residualBlock(lg, 32, 'deBlock4', 'concat4');

lg = addLayers(lg, transposedConv2dLayer(2, 16, 'Stride', 2, 'Cropping', 'same', 'Name', 'upsample5'));
lg = connectLayers(lg, 'deBlock4_relu2', 'upsample5');

lg = addLayers(lg, convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'finalRefine_conv1'));
lg = addLayers(lg, batchNormalizationLayer('Name', 'finalRefine_bn1'));
lg = addLayers(lg, reluLayer('Name', 'finalRefine_relu1'));
lg = addLayers(lg, convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'finalRefine_conv2'));
lg = addLayers(lg, batchNormalizationLayer('Name', 'finalRefine_bn2'));
lg = addLayers(lg, reluLayer('Name', 'finalRefine_relu2'));

lg = connectLayers(lg, 'upsample5', 'finalRefine_conv1');
lg = connectLayers(lg, 'finalRefine_conv1', 'finalRefine_bn1');
lg = connectLayers(lg, 'finalRefine_bn1', 'finalRefine_relu1');
lg = connectLayers(lg, 'finalRefine_relu1', 'finalRefine_conv2');
lg = connectLayers(lg, 'finalRefine_conv2', 'finalRefine_bn2');
lg = connectLayers(lg, 'finalRefine_bn2', 'finalRefine_relu2');

lg = addLayers(lg, convolution2dLayer(1, 3, 'Name', 'classConv'));
lg = connectLayers(lg, 'finalRefine_relu2', 'classConv');

net = dlnetwork(lg);

% means of viewing the network architecture

analyzeNetwork(lg)
%plot(lg)

numClasses = numel(classNames);

% Compute inverse-frequency class weights to address class imbalance
pixelCounts = zeros(1, numClasses);
for idx = 1:numTrain
    labelData = readimage(labelDS, trainIdx(idx));
    labelNumeric = double(labelData);
    for c = 1:numClasses
        pixelCounts(c) = pixelCounts(c) + sum(labelNumeric(:) == c);
    end
end

classFrequencies = pixelCounts / sum(pixelCounts);
medianFreq = median(classFrequencies);
classWeights = medianFreq ./ classFrequencies;

fprintf('Class weights - BG: %.4f, Crop: %.4f, Weed: %.4f\n', ...
    classWeights(1), classWeights(2), classWeights(3));

% Training options

options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-4, ...
    'MaxEpochs', 30, ...
    'ValidationData', testDS, ...
    'ValidationFrequency', 5, ...
    'Plots', 'training-progress', ...
    'Verbose', true, ...
    'MiniBatchSize', 4, ...
    'Shuffle', 'every-epoch');

% Train the network 

if trainModel
    numPixels = inputSize(1) * inputSize(2);
    lossFcn = @(Y, T) crossentropy(softmax(Y), T, single(classWeights), ...
        'WeightsFormat', 'C') / numPixels;
    netTrained = trainnet(trainDS, net, lossFcn, options);
    save('segmentnet_base.mat', 'netTrained');
else
    load('segmentnet_base.mat', 'netTrained');
end

% ---- Evaluation ----

% ImageNet normalisation constants (must match training preprocessing)
imagenetMean = reshape([0.485 0.456 0.406], 1, 1, 3);
imagenetStd  = reshape([0.229 0.224 0.225], 1, 1, 3);

% Separate test image and label datastores
testImageDS = subset(imageDS, numTrain+1:numImages);
testLabelDS = subset(labelDS, numTrain+1:numImages);

confusionMat = zeros(numClasses);
bfScores = zeros(numTest, numClasses);
visImages = cell(min(3, numTest), 3);

for i = 1:numTest
    % Read original-size image and ground truth label
    originalImg = readimage(testImageDS, i);
    groundTruthLabel = readimage(testLabelDS, i);
    originalSize = size(originalImg, [1 2]);

    % Preprocess to match training pipeline
    resizedImg = single(imresize(originalImg, inputSize)) / 255;
    resizedImg = (resizedImg - imagenetMean) ./ imagenetStd;
    inputDL = dlarray(resizedImg, 'SSCB');

    % Forward pass with softmax to get class probabilities
    predictions = predict(netTrained, inputDL);
    probabilities = extractdata(softmax(predictions));

    % Argmax for predicted class, resized back to original dimensions
    [~, predictedClassMap] = max(probabilities, [], 3);
    predictedClassMap = imresize(predictedClassMap, originalSize, 'nearest');

    % Convert ground truth categorical to numeric for comparison
    groundTruthNumeric = double(groundTruthLabel);

    % Accumulate confusion matrix
    for trueClass = 1:numClasses
        for predClass = 1:numClasses
            confusionMat(trueClass, predClass) = ...
                confusionMat(trueClass, predClass) + ...
                sum((groundTruthNumeric(:) == trueClass) & ...
                    (predictedClassMap(:) == predClass));
        end
    end

    % Compute boundary F1 score per class for this image
    for c = 1:numClasses
        predMask = predictedClassMap == c;
        gtMask = groundTruthNumeric == c;
        bfScores(i, c) = bfscore(predMask, gtMask);
    end

    % Store samples for visualisation
    if i <= 3
        visImages{i, 1} = originalImg;
        visImages{i, 2} = groundTruthNumeric;
        visImages{i, 3} = predictedClassMap;
    end
end

% Compute per-class IoU from confusion matrix
iouScores = zeros(numClasses, 1);
for c = 1:numClasses
    tp = confusionMat(c, c);
    fp = sum(confusionMat(:, c)) - tp;
    fn = sum(confusionMat(c, :)) - tp;
    iouScores(c) = tp / (tp + fp + fn);
end

% Compute per-class accuracy from confusion matrix
accuracyScores = zeros(numClasses, 1);
for c = 1:numClasses
    tp = confusionMat(c, c);
    totalClassPixels = sum(confusionMat(c, :));
    accuracyScores(c) = tp / totalClassPixels;
end

% Compute mean boundary F1 scores
meanBFScores = mean(bfScores, 1, 'omitnan');

% Display results
fprintf('\n--- Evaluation Results ---\n');
fprintf('%-12s %-10s %-10s %-10s\n', 'Class', 'Accuracy', 'IoU', 'BFScore');
for c = 1:numClasses
    fprintf('%-12s %-10.4f %-10.4f %-10.4f\n', ...
        classNames{c}, accuracyScores(c), iouScores(c), meanBFScores(c));
end
fprintf('%-12s %-10.4f %-10.4f %-10.4f\n', 'Mean', ...
    mean(accuracyScores), mean(iouScores), mean(meanBFScores));

% Visualise predictions against ground truth
colourMap = [0 0 0; 0 1 0; 1 0 0];
numVis = size(visImages, 1);
figure;
for v = 1:numVis
    subplot(numVis, 3, (v-1)*3 + 1);
    imshow(visImages{v, 1}); title(sprintf('Input %d', v));
    subplot(numVis, 3, (v-1)*3 + 2);
    imshow(label2rgb(visImages{v, 2}, colourMap)); title('Ground Truth');
    subplot(numVis, 3, (v-1)*3 + 3);
    imshow(label2rgb(double(visImages{v, 3}), colourMap)); title('Prediction');
end

