## method to load a pkl file

## print INFORMATIVE FORMATTED LOGS THROUGHOUT, PRIORITIZE SHAPES

from mulooc.models.utils.task_metrics import giantsteps_metrics, get_tempo_metrics
import pandas as pd

import numpy as np
import torch

import pickle
import time

def load_pkl(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data


## method to get nearest neighbours 

def get_nn(query, embeddings, k=5):
    dists = np.dot(embeddings, query)/np.linalg.norm(embeddings, axis=1)
    idxs = np.argsort(dists)[::-1][:k].T
    
    return idxs, dists[idxs]


def get_nn(queries, embeddings, k=5):
    dists = np.dot(embeddings, queries.T)/np.linalg.norm(embeddings, axis=1).reshape(-1,1)
    idxs = np.argsort(dists, axis=0)[::-1][1:k+1].T
    
    return idxs, dists[idxs]




def transpose_key(key,k):
    
    key_cycle = ['Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B', 'C', 'Db', 'D']
    letter, colour = key.split(' ')
    idx = key_cycle.index(letter)
    
    new_idx = (idx + k) 
    new_idx = int(np.round(new_idx))
    new_idx = new_idx % 12
    transpose = key_cycle[new_idx]
    
    new_key = transpose + ' ' + colour
    
    return new_key 


def shift_labels_key(labels, k):
    idx2class = {0: 'Eb minor', 1: 'A major', 2: 'F minor', 3: 'D minor', 4: 'G minor', 5: 'C minor', 6: 'A minor', 7: 'B minor', 8: 'Db minor', 9: 'D major', 10: 'E minor', 11: 'Bb major', 12: 'Ab minor', 13: 'C major', 14: 'Db major', 15: 'Ab major', 16: 'E major', 17: 'G major', 18: 'B major', 19: 'Gb minor', 20: 'Gb major', 21: 'Bb minor', 22: 'F major', 23: 'Eb major'}
    new_labels = np.zeros(labels.shape)
    for idx in range(labels.shape[0]):
        key = idx2class[np.argmax(labels[idx])]
        new_key = transpose_key(key, k[idx])
        new_labels[idx, list(idx2class.values()).index(new_key)] = 1
        
    return labels,new_labels, None

def shift_labels_tempo(labels, tau):
    new_labels = np.zeros(labels.shape)
    for idx in range(labels.shape[0]):
        new_tempo = int(np.round(np.argmax(labels[idx]) * tau[idx]))
        if new_tempo >= labels.shape[-1] or new_tempo < 0:
            pass 
        else:
            new_labels[idx, new_tempo] = 1
            
    null_indices = new_labels.sum(axis=1) == 0
    new_labels = new_labels[~null_indices]
    labels = labels[~null_indices]
    
    return labels, new_labels, null_indices
    
def get_data(path, task, probing_space = None, embed_dim = 1728):
    
    data = load_pkl(path)
    
    mean_clean_embeddings_all = []
    mean_audio_embeddings_all = []
    mean_transform_parameters_all = []
    mean_labels_all = []
    
    
    for split in ['train', 'val', 'test']:
        
    
        split_data = data[split]
        
        mean_clean_embeddings = np.load(split_data['mean_clean_audio_embeddings_path'])
        mean_audio_embeddings = np.load(split_data['mean_audio_embeddings_path'])
        mean_transform_parameters = split_data['mean_transform_parameters']
        mean_labels = split_data['mean_labels']
        
        mean_clean_embeddings_all.append(mean_clean_embeddings)
        mean_audio_embeddings_all.append(mean_audio_embeddings)
        mean_transform_parameters_all.append(mean_transform_parameters)
        mean_labels_all.append(mean_labels)
        
    merged = merge_data({
        'mean_clean_embeddings_all': mean_clean_embeddings_all,
        'mean_audio_embeddings_all': mean_audio_embeddings_all,
        'mean_transform_parameters_all': mean_transform_parameters_all,
        'mean_labels_all': mean_labels_all
    })
    
    if probing_space is not None:
        
        range_ = torch.arange(0,embed_dim)
        range_ = torch.cat([range_ + i * embed_dim for i in probing_space])
        
        merged['clean'] = merged['clean'][...,range_]
        merged['audio'] = merged['audio'][...,range_]
        
        
        
    return add_new_labels(merged, task)
    

def merge_data(data):
    
    mean_clean_embeddings_all = np.concatenate(data['mean_clean_embeddings_all'], axis=0)
    mean_audio_embeddings_all = np.concatenate(data['mean_audio_embeddings_all'], axis=0)
    mean_labels_all = np.concatenate(data['mean_labels_all'], axis=0)
    
    new_params = {}
    
    for key in data['mean_transform_parameters_all'][0].keys():
        for key2 in data['mean_transform_parameters_all'][0][key].keys():
            new_params[key2] = np.concatenate([x[key][key2] for x in data['mean_transform_parameters_all']])
            
    return {
        'clean': mean_clean_embeddings_all,
        'audio': mean_audio_embeddings_all,
        'labels': mean_labels_all,
        'params': new_params
    }
    
def add_new_labels(data, task):
    
    
    if task == 'key':
        params = data['params']['semitones']
        labels, new_labels, null_indices = shift_labels_key(data['labels'], params)
        
    elif task == 'tempo':
        params = data['params']['stretch_rates']
        labels, new_labels, null_indices = shift_labels_tempo(data['labels'], params)
        
    data['labels'] = labels
    data['new_labels'] = new_labels
    
    if null_indices is not None:
    
        data['clean'] = data['clean'][~null_indices]
        data['audio'] = data['audio'][~null_indices]
        
        for param in data['params'].keys():
            data['params'][param] = data['params'][param][~null_indices]
        
    assert data['labels'].shape[0] == data['clean'].shape[0] == data['audio'].shape[0], 'Shapes do not match'    
    
    return data



def get_nn_for_data(data, k=5):
    
    nn = {
        'clean_query': {},
        'transformed_query': {}
    }
    

    
    nn['clean_query']['transformed_neighbours'] = get_nn(data['clean'], data['audio'], k)[0]
    nn['clean_query']['clean_neighbours'] = get_nn(data['clean'], data['clean'], k)[0]
    nn['transformed_query']['transformed_neighbours'] = get_nn(data['audio'], data['audio'], k)[0]
    nn['transformed_query']['clean_neighbours'] = get_nn(data['audio'], data['clean'], k)[0]
    
    return nn

def get_labels_from_nn(nn, data, query = 'clean_query', neighbours = 'clean_neighbours'):
    
    queries = []
    retrieved = []
    
    nn_index = nn[query][neighbours]
    
    query_label_key = 'new_labels' if 'transformed' in query else 'labels'
    neighbour_label_key = 'new_labels' if 'transformed' in neighbours else 'labels'
    
    for query_idx in range(nn_index.shape[0]):
        neighbour_idx = nn_index[query_idx]
        
        original_label = data[query_label_key][query_idx]
        neighbour_label = data[neighbour_label_key][neighbour_idx]
        
        
        original_label = np.tile(original_label, (neighbour_label.shape[0], 1))
        
        
        queries.append(original_label)
        retrieved.append(neighbour_label)
        
    queries  = np.concatenate(queries, axis=0)
    retrieved = np.concatenate(retrieved, axis=0)
        
    return queries, retrieved
    
    
def get_metrics_from_nn(nn,data,query='clean_query',neighbours='clean_neighbours', task = 'key'):
    
    queries, retrieved = get_labels_from_nn(nn, data, query, neighbours)
    
    if task == 'key':
        metrics = giantsteps_metrics(queries, retrieved, n_classes=24)
    else:
        metrics = get_tempo_metrics(queries, retrieved)
        
    # print(metrics)
    
    return metrics




def run_single_experiment(task, head, subspace, pretraining, applied, k, query, neighbour):
    
    
    try:
        arg_task = 'key' if task == 'giantsteps' else 'tempo'
        if os.path.exists(f'/import/research_c4dm/jpmg86/MuLOOC/experiments/embeddings/{task}/{head}/{pretraining}/{applied}/output.pkl'):
            config = {
                'task': task,
                'head': head,
                'pretraining': pretraining,
                'applied': applied,
                'k': k,
                'subspace': subspace,
                'query': query,
                'neighbour': neighbour
            }
            data = get_data(
                f'/import/research_c4dm/jpmg86/MuLOOC/experiments/embeddings/{task}/{head}/{pretraining}/{applied}/output.pkl',
                task=arg_task,probing_space=subspace)
            nn = get_nn_for_data(data,k=k)
            metrics = get_metrics_from_nn(nn, data, task=arg_task, query=query, neighbours=neighbour)
            
            out_ =  {
                'config': config,
                'metrics': metrics,
            }
            
            
            return out_
            
            
    except Exception as e:
        print(e)
        pass
    
def run_single_experiment_singlethread(task, head, subspace, pretraining, applied, k):
    
    out_, nn, data = None, None, None
    
    try:
        arg_task = 'key' if task == 'giantsteps' else 'tempo'
        if os.path.exists(f'/import/research_c4dm/jpmg86/MuLOOC/experiments/embeddings/{task}/{head}/{pretraining}/{applied}/output.pkl'):
            config = {
                'task': task,
                'head': head,
                'pretraining': pretraining,
                'applied': applied,
                'k': k,
                'subspace': subspace
            }
            data = get_data(
                f'/import/research_c4dm/jpmg86/MuLOOC/experiments/embeddings/{task}/{head}/{pretraining}/{applied}/output.pkl',
                task=arg_task,probing_space=subspace)
            nn = get_nn_for_data(data,k=k)
            
            out_ =  {
                'config': config,
            }
            
            return out_, nn, data
            
            
    except Exception as e:
        print(e)
        pass
    
    return out_, nn, data

def run_experiment(grid, processes=1):
    
    
    
    num_experiments = len(grid['task']) * sum([len(x) for x in grid['heads'].values()]) * len(grid['pretraining']) * len(grid['applied']) * len(grid['k']) * len(grid['queries']) * len(grid['neighbours'])
    progress_bar = tqdm(total=num_experiments)
    
    
    
    path = '/import/research_c4dm/jpmg86/MuLOOC/experiments/retrieval/results.pkl'
    previous_results = None
    # if the file exists, get all the configs in the file
    if os.path.exists(path):
        with open(path, 'rb') as f:
            previous_results = pickle.load(f)
            configs = [x['config'] for x in previous_results]
            shorter_configs = [{k:v for k,v in x.items() if k != 'query' and k != 'neighbour'} for x in configs]
    
    
    
    
    temp_configs = []
    
    for task in grid['task']:
        for head, subspaces in grid['heads'].items():
            for subspace in subspaces:
                for pretraining in grid['pretraining']:
                    for applied in grid['applied']:
                        for k in grid['k']:
                            
                            temp_config = {
                                'task': task,
                                'head': head,
                                'pretraining': pretraining,
                                'applied': applied,
                                'k': k,
                                'subspace': subspace
                            }
                            
                            temp_configs.append(temp_config)
                           
    
    # print(previous_results) 
    if previous_results is not None:
        # print the number of experiments that have already been run
        count = len([x for x in temp_configs if x in shorter_configs])
        print(f'{count} experiments have already been run')
    
    
    results = []
    for temp_config in temp_configs:
        
        if temp_config not in shorter_configs:
            
            task = temp_config['task']
            head = temp_config['head']
            subspace = temp_config['subspace']
            pretraining = temp_config['pretraining']
            applied = temp_config['applied']
            k = temp_config['k']
        
            config, nn, data  = run_single_experiment_singlethread(task, head, subspace, pretraining, applied, k)

            if config is not None:

                for query in grid['queries']:
                    for neighbour in grid['neighbours']:
                        arg_task = 'key' if task == 'giantsteps' else 'tempo'

                        metrics = get_metrics_from_nn(nn, data, task=arg_task, query=query, neighbours=neighbour)
                        
                        
                        
                        new_config = {k : v for k,v in config['config'].items()}
                        new_config = {**new_config, 'query': query, 'neighbour': neighbour}
                        
                        
                        out_ =  {
                            'config': new_config,
                            'metrics': metrics,
                        }
                        
                        new_config = None
                        
                        
                        results.append(out_)
        
        progress_bar.update(1)
                                
    return results


if __name__ == '__main__':
    
    import os
    from tqdm import tqdm
    import wandb 
    import pickle
    from multiprocessing import Pool



    wandb_project = 'MuLOOC-Retrieval'


    grid = {
        'task': ['giantsteps', 'hainsworth_vs_all_tempo'],
        'heads': {
            'singlehead' : [None],
            'multihead': [None],
            'singlehead_plusplus': [
                [0,1,2]
                ],
            'multihead_plusplus': [
                    [0],
                    [1],
                    [2],
                    [0,1,2]
                    ]
            },
        'pretraining': [
            'pretrain_base',
            'pretrain_ps',
            'pretrain_psts',
            'pretrain_mixup',
            'pretrain_ts'
            ],
        'applied': ['ps', 'ts', 'psts'],
        'k': [
            1,
            3,
            5,
            10,
            100
            ],
        'queries': [
            'clean_query',
            'transformed_query'
            ],
        'neighbours': [
            'clean_neighbours',
            'transformed_neighbours'
            ]
    }

    log = True

    count = 0



    new_results = run_experiment(grid, processes=0)
    
    path = '/import/research_c4dm/jpmg86/MuLOOC/experiments/retrieval/results.pkl'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    if os.path.exists(path):
        with open(path, 'rb') as f:
            previous_results = pickle.load(f)
            results = previous_results + new_results
    
    
    with open(path, 'wb') as f:
        pickle.dump(results, f)
        
    if log:
        for result in tqdm(new_results):
            
            # check if the exact config is in the wandb run configs
            
            # runs = wandb.Api().runs(f'jul-guinot/{wandb_project}')
            # run_configs = [run.config for run in runs]
            
            save = True
            # for key in result['config'].keys():
            #     for run_config in run_configs:
            #         # if there is a run config containing all the keys in the result config and the values are the same
            #         if all([key in run_config.keys() for key in result['config'].keys()]) and all([result['config'][key] == run_config[key] for key in result['config'].keys()]):
            #             save = False
            #             break
            if save:
                wandb.init(project=wandb_project)
                wandb.config.update(result['config'])
                wandb.log(result['metrics'])
                wandb.finish()
