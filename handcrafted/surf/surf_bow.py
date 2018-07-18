import cv2
import numpy as np
import os

class SURF_BOW:
    def __init__(self, num_of_words):
        self.detect = cv2.xfeatures2d.SURF_create(extended=False) # 128 dim instead of 64 dim descriptor
        self.extract = cv2.xfeatures2d.SURF_create(extended=False)
        # with SIFT/SURF, use algorithm=FLANN_INDEX_KDTREE=0, trees=5
        # see: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html
        self.flann_params = dict(algorithm=1, trees=5)
        self.matcher = cv2.FlannBasedMatcher(self.flann_params, {})
        self.num_of_words = num_of_words
        self.bow_train = cv2.BOWKMeansTrainer(num_of_words)
        self.bow_extract = cv2.BOWImgDescriptorExtractor(self.extract, self.matcher)

    # detect kp and extract features
    def extract_feature(self, filepath):
        image = cv2.imread(filepath, 0)  # read as grayscale
        return self.extract.compute(image, self.detect.detect(image))[1]

    def build_vocab(self, file_list):
        for filepath in file_list:
            features = self.extract_feature(filepath)
            # print ('*** Extracting SURF from ', filepath, ' ***')
            self.bow_train.add(features)

        print ('*** Now building BOW SURF vocab ***')
        voc = self.bow_train.cluster()
        self.bow_extract.setVocabulary(voc)
        print ("Bow vocab: ", np.shape(voc), voc)
        print ("--------------------------------")
        return

    def extract_bow(self, filepath):
        # print ("*** Extracting BOW hist for image ", filepath, " ***")
        image = cv2.imread(filepath, 0) # read as grayscale
        bow_hist= self.bow_extract.compute(image, self.detect.detect(image))
        return np.reshape(bow_hist, self.num_of_words)

    def extract_bow_hists(self, filelist):
        bow_hists = []
        for filepath in filelist:
            bow_hist = self.extract_bow(filepath)
            bow_hists.append(bow_hist)

        return np.asarray(bow_hists)

def get_filepath(directory):
    labels = os.listdir(directory)
    labels.sort()
    file_list = []

    for label in labels:
        for f in os.listdir(os.path.join(directory, label)):
            file_list.append(os.path.join(directory, label, f))
    return file_list

def main():
    filelist = get_filepath('/home/long/Desktop/test_images')
    surf_bow = SURF_BOW(1000)
    surf_bow.build_vocab(filelist)
    bow1= surf_bow.extract_bow(filelist[0])
    print (bow1.shape)
    print (bow1)


if __name__ == '__main__':
    main()
#TODO: log control