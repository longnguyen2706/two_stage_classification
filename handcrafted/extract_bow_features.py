import os

from handcrafted.bow.surf_bow import SURF_BOW
from split_data import print_split_report
from utils import dump_pickle, load_pickle

NUM_OF_WORDS = 5
def extract_BOW_features(train_files, val_files, test_files):
    surf_bow = SURF_BOW(num_of_words=NUM_OF_WORDS)
    surf_bow.build_vocab(train_files)
    train_bow_features = surf_bow.extract_bow_hists(train_files)
    val_bow_features = surf_bow.extract_bow_hists(val_files)
    test_bow_features = surf_bow.extract_bow_hists(test_files)
    return train_bow_features, val_bow_features, test_bow_features

def gen_bow_features_for_pool(pools, pool_idx, image_dir, feature_file_dir):
    train_files = []
    val_files = []
    test_files = []
    pool = pools['data'][str(pool_idx)]
    for path in pool['train_files']:
        full_path = os.path.join(image_dir, path)
        train_files.append(full_path)
    for path in pool['val_files']:
        full_path = os.path.join(image_dir, path)
        val_files.append(full_path)
    for path in pool['test_files']:
        full_path = os.path.join(image_dir, path)
        test_files.append(full_path)

    train_bow_features, val_bow_features, test_bow_features = extract_BOW_features(
        train_files, val_files, test_files)

    features = {
        'train_features': train_bow_features,
        'val_features': val_bow_features,
        'test_features': test_bow_features,
        'pool_idx': pool_idx
    }

    filepath = get_feature_file_path(pools, pool_idx, feature_file_dir)
    dump_pickle(features, filepath)
    return

def get_feature_file_path(pools, pool_idx, feature_file_dir):
    filename = pools['pool_name'] + '_' + str(pool_idx) + '_' + 'bow_features'
    filepath = os.path.join(feature_file_dir, filename)
    return filepath

def get_bow_features_for_pool(pools, pool_idx, feature_file_dir):
    if not os.path.exists(feature_file_dir):
        os.makedirs(feature_file_dir)

    filepath = get_feature_file_path(pools, pool_idx, feature_file_dir) + '.pickle'

    try:
        features = load_pickle(filepath)
        if not (str(features['pool_idx']) == str(pool_idx)):
            raise ValueError
        return features
    except:
        pass


def main():
    # modify this to load the
    data_pools = load_pickle('/home/long/Desktop/Hela_split_30_2018-07-19.pickle')
    pool = data_pools['data']['0']
    print(pool['data_name'])

    # gen_bow_features_for_pool(data_pools, 0, image_dir='/mnt/6B7855B538947C4E/Dataset/JPEG_data/Hela_JPEG',
    #                           feature_file_dir='/home/long/Desktop/bow')
    features = get_bow_features_for_pool(data_pools, 0, '/home/long/Desktop/bow')
    print (features['train_features'].shape)
if __name__ == "__main__":
    main()