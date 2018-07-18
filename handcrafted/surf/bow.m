dataDir = '/mnt/6B7855B538947C4E/Dataset/JPEG_data/Hela_JPEG/';
featureDir = '/mnt/6B7855B538947C4E/handcraft_models/Hela_SURF';
categories = {'ActinFilaments',  'Endosome', 'ER', 'Golgi_gia', ...
    'Golgi_gpp', 'Lysosome', 'Microtubules', 'Mitochondria', 'Nucleolus', 'Nucleus'};
imds = imageDatastore(fullfile(dataDir, categories), 'LabelSource', 'foldernames'); 
[trainingSet, valSet, testSet] = splitEachLabel(imds,0.6, 0.2,'randomize');
bag = bagOfFeatures(trainingSet,'VocabularySize',1000, 'GridStep', [4,4],'BlockWidth', [32 64 96 128] );

categoryClassifier = trainImageCategoryClassifier(trainingSet,bag);
confValMatrix = evaluate(categoryClassifier, valSet);
mean(diag(confValMatrix));
confTestMatrix = evaluate(categoryClassifier,testSet);
mean(diag(confTestMatrix));
% for i=1:length(imds.Files)
%     im = readimage(imds, i);
%     feature = encode (bag, im); % create bow histogram
%     imagePath = string(imds.Files(i));
%     dirElements = strsplit(imagePath, '/');
%     % Create category folder if not exist
%     label = dirElements(length(dirElements)-1);
%     o6isssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssszzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzoip
%     p 
%     labelDir = strcat(featureDir, '/',label);
%     if ~ exist(labelDir,'dir')
%         mkdir(labelDir);
%     end
% 
%     % Write the feature to .mat file
%     imageName = dirElements(length(dirElements));
%     featurePath = strcat(featureDir, '/',label, '/', imageName, '.mat');
%     save(featurePath,'feature');
% end