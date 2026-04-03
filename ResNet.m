net = imagePretrainedNetwork("resnet18");


lg = layerGraph(net);

lg = removeLayers(lg, 'pool5');
lg = removeLayers(lg, 'fc1000');
lg = removeLayers(lg, 'prob');



analyzeNetwork(lg)