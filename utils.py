
import pickle
from datetime import datetime

def current_date(now):
    return now.strftime('%Y-%m-%d')

def current_time(now):
    return now.strftime('%Y-%m-%d_%H:%M:%S')


def dump_pickle(dict, path):
    filepath = path + '.pickle'
    with open(filepath, 'wb') as handle:
        pickle.dump(dict, handle)
    return filepath


def load_pickle(filepath):
    with open(filepath, 'rb') as handle:
        dict = pickle.load(handle)
    return dict
