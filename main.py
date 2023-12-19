import torch
from torchvision.utils import make_grid, save_image
from datetime import datetime
from modules import VectorQuantizedVAE
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import json
from torch.nn.functional import normalize
import numpy as np
import pandas as pd
import cv2
from classifier import *
import shutil
from testing_adv_paths import *

def generate_samples(images, org_labels, predict_labels, model, classify_model, args, concept_id):
    # generate samples for the choosen concept
    total_change = []
    sample_stack = None
    # generate sample for batch size 100
    for i in range(int(len(images)/100)):
        curimages = images[i*100:(i+1)*100]
        with torch.no_grad():
            curimages = curimages.to(args.device)
            x_tildes = model.generate_with_mask(curimages, concept_id)
            probas_test = classify_model((x_tildes+1)/2)
            _,test_predicted_labels = torch.max(probas_test, axis=1)
            for curimage_id in range(len(curimages)):
                if org_labels[i*100+curimage_id].item() != predict_labels[i*100+curimage_id]:
                    # the orignal image has wrong prediction
                    continue
                if (test_predicted_labels[len(curimages)*49+curimage_id] != org_labels[i*100+curimage_id].item()):
                    # the orignal reconstructed image has wrong prediction
                    continue
                for j in range(49):
                    if test_predicted_labels[len(curimages)*j+curimage_id] != org_labels[i*100+curimage_id]:
                        if sample_stack is not None :
                            sample_stack = torch.concatenate(( sample_stack,x_tildes[len(curimages)*j+curimage_id].cpu()))
                        else:
                            sample_stack = x_tildes[len(curimages)*j+curimage_id].cpu()
                        
                        total_change.append({
                            'image_id': i*100 + curimage_id,
                            'org_label': org_labels[i*100+curimage_id].item(),
                            'masked_label': test_predicted_labels[len(curimages)*j+curimage_id].item(),
                            'concept_using': concept_id,
                            'concept_location': j
                        })
                        
    
    return total_change, sample_stack

def generate_sampe(args):
    weights_kmeans = None
    model = VectorQuantizedVAE(1, args.hidden_size_vae, args.k, weights_kmeans).to(args.device)
    with open(args.model, 'rb') as f:
        state_dict = torch.load(f)
        model.load_state_dict(state_dict)
        
    classify_model = load_model(args.classifier_model, args.device)

    transform_adv = transforms.Compose([transforms.Normalize([0.5], [0.5])])
    # Define the train dataSets
    images = None
    labels = None
    if args.dataset == "MNIST":
        train_dataset = datasets.MNIST(root='data',  train=True, download=True, transform=transforms.ToTensor())
        images = train_dataset.train_data / 255
        labels = train_dataset.train_labels
        images = images.reshape(len(images), 1, 28, 28)
        
    # get original predict label
    probas_test = classify_model(images)
    _,predict_labels = torch.max(probas_test, axis=1)
    print(f"model acc: {torch.sum(predict_labels==labels)/len(predict_labels)}")
    
    
    images = transform_adv(images)
    images = images[:1000]
    if args.mode == 'gs_c':
        total_change, sample_stack = generate_samples(images, labels, predict_labels, model, classify_model, args, args.concept)
        np.save(f'aug_sample/concept_{args.concept}.npy', sample_stack.numpy())
        df = pd.DataFrame(total_change)
        df.to_csv(f'aug_sample/concept_{args.concept}.csv', index=False)
        print(f"Finished concept {args.concept}")
    
    else:
        for i in range(512):
            total_change, sample_stack = generate_samples(images, labels, predict_labels, model, classify_model, args, i)
            np.save(f'aug_sample/concept_{i}.npy', sample_stack.numpy())
            df = pd.DataFrame(total_change)
            df.to_csv(f'aug_sample/concept_{i}.csv', index=False)
            print(f"Finished concept {i}")
            
    


def predict_test(attack,classify_model):
    # Define the train dataSets

    adv_images = np.load(attack['images_numpy']).astype(np.float32)
    if np.max(adv_images) > 1:
        adv_images = adv_images / 255

    adv_images = np.transpose(adv_images,[0,3,1,2])
    adv_images = torch.from_numpy(adv_images)
    labels = np.load(attack['labels'])
    labels = torch.from_numpy(labels)
    probas = classify_model(adv_images)
    _,predict_labels = torch.max(probas, axis=1)
    indices = [i for i, (label1, label2) in enumerate(zip(list(predict_labels), list(labels))) if label1 != label2]
    return indices

def model_finetune(args):
    
    images = None
    labels = None
    if args.dataset == "MNIST":
        # Define the train & test dataSets
        train_dataset =  datasets.MNIST(root='data', 
                            train=True, 
                            transform=transforms.ToTensor())
        test_dataset = datasets.MNIST(root='data', 
                            train=False, 
                            transform=transforms.ToTensor())
        images = test_dataset.test_data / 255
        labels = test_dataset.test_labels
        images = images.reshape(len(images), 1, 28, 28)
    features = None
    targets = None
    if args.mode == 'ft':
        features = np.load('aug_sample/concatenated_images.npy')
        file_path = 'aug_sample/concatenated_csv.csv'
        df = pd.read_csv(file_path)
        targets = np.array(list(df['org_label']))
        features = np.expand_dims(features, axis=1)
        features = features + 1 / 2
        x_train = np.concatenate([(train_dataset.train_data / 255).reshape(len(train_dataset.train_data), 1, 28, 28).cpu().numpy()[:int(args.org_train_ratio*len(features))],features], axis=0)
        y_train = np.concatenate([train_dataset.train_labels[:int(args.org_train_ratio*len(features))], targets], axis=0)
    elif args.mode == 'ft_c':
        features = np.load(f'aug_sample/concept_{args.concept}.npy')
        file_path = f'aug_sample/concept_{args.concept}.csv'
        df = pd.read_csv(file_path)
        targets = np.array(list(df['org_label']))
        features = np.expand_dims(features, axis=1)
        features = features + 1 / 2
        x_train = np.concatenate([(train_dataset.train_data / 255).reshape(len(train_dataset.train_data), 1, 28, 28).cpu().numpy()[:int(args.org_train_ratio*len(features))],features], axis=0)
        y_train = np.concatenate([train_dataset.train_labels[:int(args.org_train_ratio*len(features))], targets], axis=0)
    elif args.mode == 'ft_m_c':
        
        features = np.load(f'aug_sample/concept_{args.concept}.npy')
        file_path = f'aug_sample/concept_{args.concept}.csv'
        df = pd.read_csv(file_path)
        targets = np.array(list(df['org_label']))
        ids = np.array(list(df['image_id']))
        features = np.expand_dims(features, axis=1)
        features = features + 1 / 2
        used_image_id = list(set(ids))
        features2 = np.load(args.adv_path)
        features2 = np.transpose(features2, [0,3,1,2])
        features2 = torch.from_numpy(features2)
        features2 = features2[used_image_id]
        
        targets2 = np.load(args.adv_label)
        targets2 = torch.from_numpy(targets2)
        targets2 = targets2[used_image_id]

        x_train = np.concatenate([(train_dataset.train_data / 255).reshape(len(train_dataset.train_data), 1, 28, 28).cpu().numpy()[:int(args.org_train_ratio*len(features))],features,features2], axis=0)
        y_train = np.concatenate([train_dataset.train_labels[:int(args.org_train_ratio*len(features))], targets,targets2], axis=0)

    
    
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    features = torch.from_numpy(features)
    targets = torch.from_numpy(targets)
    train_dataset = CustomDataset(x_train, y_train)
    train_loader = DataLoader(dataset=train_dataset, 
                            batch_size=128, 
                            shuffle=True)
    

    classify_model = load_model(args.classifier_model, args.device)
    probas_test = classify_model(images)
    _,predict_labels = torch.max(probas_test, axis=1)
    print(f"model_preformance: {torch.sum(predict_labels==labels)/len(predict_labels)}")




    attacks = get_adv_path()
    # get sample that attacked successfully
    unequal_id = {}
    for attack in attacks:
        indices = predict_test(attack,classify_model)
        unequal_id[attack['name']] = indices

    
    print("-----------------------------------------------------------------------")
    source_path = args.classifier_model
    destination_path = args.classifier_model[:-4] + "_temp.pth"

    # Copy the file
    shutil.copy(source_path, destination_path)
    
    classify_model = load_model(destination_path, args.device)
    print("Start Finetuning")
    finetune(classify_model, train_loader, args.device, 3, 0.005, args.classifier_model[:-4] + "_new.pth")
    print("Finetuning Finshed")
    print("-----------------------------------------------------------------------")

    classify_model = load_model(args.classifier_model[:-4] + "_new.pth", args.device)
    
    probas_test = classify_model(images)
    _,predict_labels = torch.max(probas_test, axis=1)
    print('model_preformance_after_finetune:', torch.sum(predict_labels == labels).item()/len(images))

    for attack in attacks:
        indices_new = predict_test(attack,classify_model)
        indices_old = unequal_id[attack['name']]
        print("defence_performance for" ,attack['name'], ":",1-len(set(indices_new).intersection(set(indices_old)))/len(indices_old))

    probas_test = classify_model(features)
    _,predict_labels = torch.max(probas_test, axis=1)
    print('model_aug_preformance_after_finetune:', torch.sum(predict_labels == targets).item()/len(features))

    
    
if __name__ == '__main__':
    import argparse
    import os
    parser = argparse.ArgumentParser(description='Generated Image')
    
    parser.add_argument('--mode', type=str,
                        help='gs, gs_c, ft, ft_c, ft_m, rt, rt_c')
    
    parser.add_argument('--dataset', type=str, default='MNIST',
                        help='MNIST, CIFAR10, CELEBA')
     
    # Model weights
    parser.add_argument('--model', type=str,
                        help='filename containing the vqvae model')
    
    parser.add_argument('--classifier-model', type=str,
                        help='filename containing the classifier model')
    
    # If do not using any concept, use all concept
    parser.add_argument('--concept', type=int, help='concept using, < k')
    
    # Finetune 

    parser.add_argument('--org-train-ratio', type=float, default=1.0, help='The proption of training sample usage while fintuning')
    
    parser.add_argument('--adv-path', type=str, help='adv sample for training dataset')
    parser.add_argument('--adv-label', type=str, help='adv sample label for training dataset')
    
    
    # VQ-VAE
    parser.add_argument('--hidden-size-vae', type=int, default=40,
                        help='size of the latent vectors')
    parser.add_argument('--k', type=int, default=512,
                        help='number of latent vectors (default: 512)')
    
    parser.add_argument('--device', type=str, default='cpu',
                        help='set the device (cpu or cuda, default: cpu)')
    
    parser.add_argument('--codebook-index-path', type=str,
                        help='file path of codebook index json')
    args = parser.parse_args()

    # Device
    args.device = torch.device(args.device
                            if torch.cuda.is_available() else 'cpu')

        
    if args.mode == 'gs' or  args.mode == 'gs_c':
        generate_sampe(args)
    elif args.mode == 'ft' or args.mode == 'ft_c' or args.mode == 'ft_m_c':
        model_finetune(args)
    else:
        print('No such mode')

        
        
    
    