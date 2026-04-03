unzip('MerchData.zip');

imds = imageDatastore('MerchData','IncludeSubfolders',true,'LabelSource','foldernames');
[imdsTrain,imdsTest] = splitEachLabel(imds,0.7,'randomized');


numTrainImages = numel(imdsTrain.Labels);
idx = randperm(numTrainImages,16);
%figure
%for i = 1:16
%    subplot(4,4,i)
%    I = readimage(imdsTrain,idx(i));
%    imshow(I)
%end


% Residual block function

function lg = residualBlock(lg, numFilters, blockName, inputLayerName)
        
    lg = addLayers(lg, convolution2dLayer(3, numFilters, 'Padding','same','Name',[blockName '_conv1']));
    lg = addLayers(lg, batchNormalizationLayer('Name',[blockName '_bn1']));
    lg = addLayers(lg, reluLayer('Name',[blockName '_relu1']));
    lg = addLayers(lg, convolution2dLayer(3, numFilters, 'Padding','same','Name',[blockName '_conv2']));
    lg = addLayers(lg, batchNormalizationLayer('Name',[blockName '_bn2']));
    lg = addLayers(lg, additionLayer(2,'Name',[blockName '_add']));

    % Path A (Normal U-Net block)
    lg = connectLayers(lg, inputLayerName, [blockName '_conv1']);
    lg = connectLayers(lg, [blockName '_conv1'],[blockName '_bn1']);
    lg = connectLayers(lg, [blockName '_bn1'],[blockName '_relu1']);
    lg = connectLayers(lg, [blockName '_relu1'],[blockName '_conv2']);
    lg = connectLayers(lg, [blockName '_conv2'],[blockName '_bn2']);

    % Path B (Residual Shortcut)
    lg = addLayers(lg, convolution2dLayer(1, numFilters, 'Name', [blockName '_shortcut_conv']));
    lg = connectLayers(lg, inputLayerName, [blockName '_shortcut_conv']);

    % Path A and B addition
    lg = connectLayers(lg, [blockName '_bn2'], [blockName '_add/in1']);
    lg = connectLayers(lg, [blockName '_shortcut_conv'], [blockName '_add/in2']);

    % Final ReLU layer
    lg = addLayers(lg, reluLayer('Name',[blockName '_relu2']));
    lg = connectLayers(lg, [blockName '_add'], [blockName '_relu2']);

end




lg = layerGraph();
lg = addLayers(lg, imageInputLayer([480 640 3], 'Name', 'input'));

% Network encoder architecture

lg = residualBlock(lg, 32, 'enBlock1', 'input');
lg = addLayers(lg, maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1'));
lg = connectLayers(lg, 'enBlock1_relu2', 'pool1');


lg = residualBlock(lg, 64, 'enBlock2', 'pool1');
lg = addLayers(lg, maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2'));
lg = connectLayers(lg, 'enBlock2_relu2', 'pool2');

lg = residualBlock(lg, 128, 'enBlock3', 'pool2');
lg = addLayers(lg, maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool3'));
lg = connectLayers(lg, 'enBlock3_relu2', 'pool3');


lg = residualBlock(lg, 256, 'enBlock4', 'pool3');
lg = addLayers(lg, maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool4'));
lg = connectLayers(lg, 'enBlock4_relu2', 'pool4');

% Bottle neck 

lg = residualBlock(lg, 512, 'bottleNeck', 'pool4');

% Network decoder architecture

lg = addLayers(lg, transposedConv2dLayer(2, 256, 'Stride', 2, 'Cropping', 'same', 'Name', 'upsample1'));
lg = residualBlock(lg, 256, 'deBlock1', 'bottleNeck_relu2');
lg = connectLayers(lg, 'deBlock1_relu2', 'upsample1');

lg = addLayers(lg, transposedConv2dLayer(2, 128, 'Stride', 2, 'Cropping', 'same', 'Name', 'upsample2'));
lg = residualBlock(lg, 128, 'deBlock2', 'upsample1');
lg = connectLayers(lg, 'deBlock2_relu2', 'upsample2');

lg = addLayers(lg, transposedConv2dLayer(2, 64, 'Stride', 2, 'Cropping', 'same', 'Name', 'upsample3'));
lg = residualBlock(lg, 64, 'deBlock3', 'upsample2');
lg = connectLayers(lg, 'deBlock3_relu2', 'upsample3');

lg = addLayers(lg, transposedConv2dLayer(2, 32, 'Stride', 2, 'Cropping', 'same', 'Name', 'upsample4'));
lg = residualBlock(lg, 32, 'deBlock4', 'upsample3');
lg = connectLayers(lg, 'deBlock4_relu2', 'upsample4');



plot(lg)