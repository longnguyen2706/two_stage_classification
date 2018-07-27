import argparse

import os

import datetime

import sys

from keras_finetune import train
import numpy as np
import tensorflow as tf

from split_data import print_split_report
from utils import current_date, current_time, load_pickle, dump_pickle

sgd_hyper_params = {
    'learning_rates': [0.05, 0.1, 0.15, 0.2, 0.25],
    'lr_decays': [0, 1e-3, 1e-6],
    'momentums': [0.0], #[0.8, 0.9],
    'nesterovs' : [False] #[True, False]
}


#TODO: flags - pickle dir, splits no to train, image_dir
FLAGS = None

'''
Train a single pool with hyper tuning
The model will be trained multiple times with different params setting and record the result
The best params then chosen based on val acc. 
The model will be train again using this params. Model will be saved as .h5 and .pb file. Tensorboard log also be saved
Returns:
    dict: results of all train with different hyper params and the final train result with best hyper params
'''
def train_single_pool(pool_split, image_dir, log_path, architecture, save_model_path, train_batch, test_batch):
    results = {}
    print('architecture: ', architecture)
    # hyper tuning and record result
    for lr in sgd_hyper_params['learning_rates']:
        for lr_decay in sgd_hyper_params['lr_decays']:
            for momentum in sgd_hyper_params['momentums']:
                for nesterov in sgd_hyper_params['nesterovs']:
                    hyper_params = {'lr': lr, 'lr_decay': lr_decay, 'momentum': momentum,  'nesterov': nesterov }
                    train_score, val_score, test_score = train(pool_split, image_dir, architecture, hyper_params,
                                                  train_batch=train_batch, test_batch=test_batch)
                    result = {
                        'hyper_params': hyper_params,
                        'train_score': train_score,
                        'test_score': test_score,
                        'val_score': val_score
                    }
                    results['hyper_tuning_result'].append(result)

    # for debug
    print('all results: ', results)

    # choosing the best params
    val_accuracies = []
    for result in results:
        test_accuracy = result['val_score']['acc']
        val_accuracies.append(test_accuracy)

    val_accuracies = np.asarray(val_accuracies)
    best_val_acc_index = np.argmax(val_accuracies)
    print ('best val acc: ', val_accuracies[best_val_acc_index])
    # for debug
    print ('best result: ', results[best_val_acc_index])

    # retrain the model with the best params and save the model to .h5 and .pb
    best_hyper_params = results[best_val_acc_index]['hyper_params']
    final_train_score, final_val_score, final_test_score = train(pool_split, image_dir, log_path, architecture, hyper_params,
                                              save_model_path= save_model_path, log_path=log_path,
                                              train_batch=train_batch, test_batch=test_batch)
    final_result = {
        'hyper_params': best_hyper_params,
        'train_score': final_train_score,
        'test_score': final_test_score,
        'val_score': final_val_score
    }
    results['final_result']=final_result
    return results

'''
    train models with given pools and architecture
    record result to .pickle file 
'''
def train_pools(_):
    pools= load_pickle(FLAGS.pool_dir)
    start_pool_idx = int(FLAGS.start_pool)
    end_pool_idx = int(FLAGS.end_pool)

    now = datetime.datetime.now()
    time = current_time(now)

    trained_models_info = {}

    for idx in range(start_pool_idx, end_pool_idx+1):
        pool = pools['data'][str(idx)]
        print ('pool idx: ', idx)
        print ('****************')
        print_split_report('train', pool['train_report'])
        print_split_report('val', pool['val_report'])
        print_split_report('test', pool['test_report'])
        print('-----------------')

        name = pools['pool_name']+'_'+str(idx)
        log_path = os.path.join(FLAGS.log_dir, name)
        save_model_path = os.path.join(FLAGS.save_model_dir, name+'_'+str(FLAGS.architecture))

        results = train_single_pool(pool, FLAGS.image_dir, log_path, FLAGS.architecture,
                          save_model_path, FLAGS.train_batch, FLAGS.test_batch)
        model_info = {
            'hyper_param_setting':sgd_hyper_params,
            'pool_idx': str(idx),
            'pool_name': pool['data_name'],
            'time': time,
            'architecture': FLAGS.architecture,
            'train_batch': FLAGS.train_batch,
            'test_batch': FLAGS.test_batch,
            'log_path': log_path,
            'save_model_path': save_model_path,
            'results': results,
            'final_results': results['final_result']
        }
        trained_models_info.append(model_info)

    # save result to .pickle
    trained_models_info_pickle_name = pools['pool_name']+'_'+str(start_pool_idx)+'_'+str(end_pool_idx)
    dump_pickle(trained_models_info, os.path.join(FLAGS.result_dir, trained_models_info_pickle_name))
    return trained_models_info


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--pool_dir',
        type=str,
    )

    parser.add_argument(
        '--image_dir',
        type=str,
    )

    parser.add_argument(
        '--architecture',
        type=str
    )

    parser.add_argument(
        '--start_pool',
        type=int
    )

    parser.add_argument(
        '--end_pool',
        type=int
    )

    parser.add_argument(
        '--log_dir',
        type=str,
    )
    parser.add_argument(
        '--save_model_dir',
        type=str,
    )
    parser.add_argument(
        '--result_dir',
        type=str,
    )

    parser.add_argument(
        '--train_batch',
        default=8,
        type=int
    )
    parser.add_argument(
        '--test_batch',
        default=16,
        type=int
    )

    FLAGS, unparsed = parser.parse_known_args()
    print(FLAGS)
    tf.app.run(main=train_pools, argv=[sys.argv[0]] + unparsed)

