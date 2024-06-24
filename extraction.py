from mulooc.dataloading.datamodule import AudioDataModule
from mulooc.models.encoders.frontend import Melgram
from mulooc.models.mulooc import MuLOOC
from mulooc.models.encoders.nfnet import NFNet, NFNetPlus 

import pickle

import torch
import yaml
import os
from tqdm import tqdm
import numpy as np

def extract_representations(config):

    data_config = config['data']
    if data_config['frontend']:
        frontend = Melgram(**data_config['frontend'])
    data_config['frontend'] = frontend
        
    mode = config['aug_mode']
            
    dm = AudioDataModule(**data_config)
    dm.setup()

    datasets = [dm.test_dataset, dm.val_dataset, dm.train_dataset]

    for dataset in datasets:
        dataset.transform = True
        dataset.train = True
        dataset.augmentations = dm.aug_chain
        dataset.return_tfm_parameters = True
        dataset.return_clean_audio = True
        dataset.extract_features = True
        dataset.return_full = True

    config['aug_mode'] = mode

    epoch = config['epoch']
    model_config = config['model']    
    head_dims = model_config['head_dims']
    ckpt_path = model_config['ckpt_path']
    plusplus = model_config['plusplus']
    feat_extract_head = model_config['feat_extract_head']

    encoder_class = NFNetPlus if plusplus else NFNet

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if config['device'] is None else torch.device(f'cuda:{config["device"]}' if config['device'].isdigit() else config['device'])
    print(f'Using device: {device}')

    mulooc = MuLOOC(encoder=encoder_class(), head_dims=head_dims, feat_extract_head=feat_extract_head)
    mulooc.to(device)

    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location=device)
        mulooc.load_state_dict(ckpt['state_dict'], strict=True)
        print(f'Loaded model from {ckpt_path}')

    mulooc.eval()
    for param in mulooc.parameters():
        param.requires_grad = False
        
    def get_data_and_save_embeddings(split, dataloader, annotations, save_path, mulooc, epoch, device, data_config, model_config):
        data = get_embeddings(dataloader, mulooc, epoch=epoch, device=device, data_config=data_config, model_config=model_config, split=split)
        data['annotations'] = annotations
        data['save'].update({
            'audio_embeddings_path': f'{save_path}/{split}/audio_embeddings.npy',
            'mean_audio_embeddings_path': f'{save_path}/{split}/mean_audio_embeddings.npy',
            'clean_audio_embeddings_path': f'{save_path}/{split}/clean_audio_embeddings.npy',
            'mean_clean_audio_embeddings_path': f'{save_path}/{split}/mean_clean_audio_embeddings.npy',
        })
        return data

    def save_embeddings(data, split, save_path):
        if data is not None:
            os.makedirs(f'{save_path}/{split}', exist_ok=True)
            with open(data['save']['audio_embeddings_path'], 'wb') as f:
                np.save(f, data['audio_embeddings'])
            with open(data['save']['mean_audio_embeddings_path'], 'wb') as f:
                np.save(f, data['mean_audio_embeddings'])
                
            with open(data['save']['clean_audio_embeddings_path'], 'wb') as f:
                np.save(f, data['clean_audio_embeddings'])
            
            with open(data['save']['mean_clean_audio_embeddings_path'], 'wb') as f:
                np.save(f, data['mean_clean_audio_embeddings'])

    dm.set_test_batch_size(dm.train_dataloader().batch_size)

    save_path = f'/import/research_c4dm/jpmg86/MuLOOC/experiments/embeddings/{config["name"]}'
    splits = ['train', 'val', 'test']
    dataloaders = [dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()]
    annotations = [dm.train_dataset.annotations, dm.val_dataset.annotations, dm.test_dataset.annotations]
    for dl in dataloaders:
        dl.dataset.set_aug_mode(mode)

    data_splits = {split: get_data_and_save_embeddings(split, dataloader, annotation, save_path, mulooc, epoch, device, data_config, model_config) 
                for split, dataloader, annotation in zip(splits, dataloaders, annotations)}

    output = {split: data['save'] if data is not None else None for split, data in data_splits.items()}

    if config['save']:
        os.makedirs(save_path, exist_ok=True)
        with open(f'{save_path}/output.pkl', 'wb') as f:
            pickle.dump(output, f)
        with open(f'{save_path}/config.yaml', 'w') as f:
            yaml.dump(config, f)
        for split, data in data_splits.items():
            save_embeddings(data, split, save_path)

def get_embeddings(dataloader, model, epoch = 1, device = 'cuda', data_config = None, model_config = None, split = 'train'):
    dataloader_iter = iter(dataloader)
    audio_embeddings = []
    clean_audio_embeddings = []
    mean_audio_embeddings = []
    mean_clean_audio_embeddings = []
    labels = []
    mean_labels = []
    indices = []
    transform_params = {}
    mean_transform_parameters = {}
    
    # epoch can be a float which gives a fraction of the dataset to use
    
    dataset_len = len(dataloader)*dataloader.batch_size
    dataset_frac = int(dataset_len * epoch)
    dataloader_frac = int(dataset_frac/dataloader.batch_size)
    
    index = 0
    
          
    for i in tqdm(range(dataloader_frac)):
        
        try:
            data = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            data = next(dataloader_iter)
            
        
        audio = data['audio'].to(device)
        clean_audio = data['clean_audio'].to(device)
        if 'labels' in data.keys():
            mean_labels += data['labels'].cpu()
        
        
        transform_parameters = data['transform_parameters']
        
        bsz, chunks, channels, _, _ = audio.shape
        
        
        audio_embedding = model.extract_features(audio)['encoded'].cpu()
        clean_audio_embedding = model.extract_features(clean_audio)['encoded'].cpu()
        
        # explode batch to list
        audio_embedding = [audio_embedding[i] for i in range(audio_embedding.shape[0])]
        
        # update indices using bsz and chunks
        batch_indices = [[index]*chunks for index in range(index, index+bsz)]
        batch_indices = [item for sublist in batch_indices for item in sublist]
        indices += batch_indices
        index += bsz
        
        
        clean_audio_embedding = [clean_audio_embedding[i] for i in range(clean_audio_embedding.shape[0])]
        
        audio_embeddings += audio_embedding
        clean_audio_embeddings += clean_audio_embedding
        
        # extend transform_params by recursively going through keys and appending lists when they exist
        for augmentation in transform_parameters.keys():
            if augmentation not in transform_params.keys():
                transform_params[augmentation] = {}
            for key in transform_parameters[augmentation].keys():
                # if key=='transpositions':
                #     transform_parameters[augmentation][key] = [float(12.0 * log2(ratio)) for ratio in transform_parameters[augmentation][key]]
                if key not in transform_params[augmentation].keys():
                    transform_params[augmentation][key] = []
                transform_params[augmentation][key].append(transform_parameters[augmentation][key].contiguous().view(-1))
    
    
    
    # concat the embeddings but create an index tensor to keep track of the original index
    audio_embeddings = torch.stack(audio_embeddings, dim = 0)
    clean_audio_embeddings = torch.stack(clean_audio_embeddings, dim = 0)
    
    
    ## pretty print the 
    
    # flatten each key of the transform parameters
    for augmentation in transform_params.keys():
        for key in transform_params[augmentation].keys():
            transform_params[augmentation][key] = torch.cat(transform_params[augmentation][key], dim = 0)
            # print(f'{augmentation} {key} shape: {transform_params[augmentation][key].shape}')
    
    
    # mean audio embeddings for each index
    indices = torch.tensor(indices)
    
    
    for i in range(indices.max()+1):
        mean_audio_embeddings.append(audio_embeddings[indices == i].mean(axis = 0))
        mean_clean_audio_embeddings.append(clean_audio_embeddings[indices == i].mean(axis = 0))
        labels += torch.stack([mean_labels[i]]*indices[indices == i].shape[0])
        # print(labels[-1].shape)
        
        
        for key in transform_params.keys():
            if key not in mean_transform_parameters.keys():
                mean_transform_parameters[key] = {}
            for key2 in transform_params[key].keys():
                if key2 != 'should_apply':
                    if key2 not in mean_transform_parameters[key].keys():
                        mean_transform_parameters[key][key2] = []
                    mean_transform_parameters[key][key2].append(transform_params[key][key2][indices == i].mean(axis = 0))
    
    mean_audio_embeddings = torch.stack(mean_audio_embeddings, dim = 0).numpy()
    mean_clean_audio_embeddings = torch.stack(mean_clean_audio_embeddings, dim = 0).numpy()
    mean_labels = torch.stack(mean_labels, dim = 0).numpy()
    labels = torch.stack(labels, dim = 0).numpy()
    
    
    for key in mean_transform_parameters.keys():
        for key2 in mean_transform_parameters[key].keys():
            mean_transform_parameters[key][key2] = torch.stack(mean_transform_parameters[key][key2], dim = 0).numpy()
    
    audio_embeddings = audio_embeddings.numpy()
    clean_audio_embeddings = clean_audio_embeddings.numpy()
    
    data_config['frontend'] = None
    
    out_ = {
        'audio_embeddings': audio_embeddings,
        'mean_audio_embeddings': mean_audio_embeddings,
        'clean_audio_embeddings': clean_audio_embeddings,
        'mean_clean_audio_embeddings': mean_clean_audio_embeddings,
        'save' :{
        'data_config': data_config,
        'model_config' : model_config,
        'indices': indices, # this is a tensor of the same length as the embeddings, each element is the index of the original audio file
        'transform_parameters': transform_params,
        'mean_transform_parameters': mean_transform_parameters,
        'mean_labels': mean_labels,
        'labels': labels}
    }
    
    # print shapes
    print(f'Audio embeddings shape: {audio_embeddings.shape}')
    print(f'Mean audio embeddings shape: {mean_audio_embeddings.shape}')
    print(f'Clean audio embeddings shape: {clean_audio_embeddings.shape}')
    print(f'Mean clean audio embeddings shape: {mean_clean_audio_embeddings.shape}')
    print(f'Mean labels shape: {mean_labels.shape}')
    print(f'Labels shape: {labels.shape}')
    
    # recursively print the transform parameters shapes
    for augmentation in transform_params.keys():
        for key in transform_params[augmentation].keys():
            print(f'{augmentation} {key} shape: {transform_params[augmentation][key].shape}')
            try:
                print(f'{augmentation} {key} mean: {mean_transform_parameters[augmentation][key].shape}')
            except:
                pass
    
    
    # pretty print the output
    
    
    
    return out_


if __name__ == '__main__':
    ## argparse with added name and save arguments that change the config
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config', type=str, help='Path to the config file')
    parser.add_argument('--name', type=str, help='Name of the output', default = 'output')
    # save as an argument
    parser.add_argument('--save', type=bool, help='Whether to save the output', default = False)
    # device as an argument
    parser.add_argument('--device', type=str, help='Device to use', default = None)
    # argument for checkpoint path
    parser.add_argument('--ckpt_path', type=str, help='Path to the checkpoint', default = None)
    parser.add_argument('--epoch', type=float, help='Fraction of the dataset to use', default = 1.0)
    parser.add_argument('--aug_mode', type=str, help='Augmentation mode', default = 'per_example')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    if args.name:
        config['name'] = args.name
    config['save'] = bool(args.save)
    config['device'] = args.device
    if args.ckpt_path:
        config['model']['ckpt_path'] = args.ckpt_path
    config['epoch'] = args.epoch
    config['aug_mode'] = args.aug_mode
    
    extract_representations(config)
    