import os

from handcrafted.bow.surf_bow import SURF_BOW
from utils import dump_pickle

NUM_OF_WORDS = 1000
def extract_BOW_features(train_files, val_files, test_files):
    surf_bow = SURF_BOW(num_of_words=NUM_OF_WORDS)
    surf_bow.build_vocab(train_files)
    train_bow_features = surf_bow.extract_bow_hists(train_files)
    val_bow_features = surf_bow.extract_bow_hists(val_files)
    test_bow_features = surf_bow.extract_bow_hists(test_files)

    return train_bow_features, val_bow_features, test_bow_features


def gen_bow_features_for_pool(pools, pool_idx, image_dir):
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
    filename = pools['pool_name']+'_'+str(pool_idx)+'_'+'bow_features'
    dump_pickle(features, filename)
    return

def get_bow_features_for_pool(feature_file_dir, pools, pool_idx):


