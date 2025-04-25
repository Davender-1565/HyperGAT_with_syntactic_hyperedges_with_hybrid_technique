import argparse
import pickle
import time
import numpy as np
from utils import split_validation, Data
from preprocess import *
from model import *
from sklearn.utils import class_weight
import random
import warnings
import os
import torch

warnings.filterwarnings('ignore') 

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Tweet', help='Dataset name')
parser.add_argument('--batchSize', type=int, default=16, help='Batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='Hidden size')
parser.add_argument('--initialFeatureSize', type=int, default=300, help='Feature size')
parser.add_argument('--epoch', type=int, default=20, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
parser.add_argument('--lr_dc', type=float, default=0.1, help='LR decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='LR decay steps')
parser.add_argument('--l2', type=float, default=1e-6, help='L2 penalty')
parser.add_argument('--valid_portion', type=float, default=0.1, help='Validation split')
parser.add_argument('--rand', type=int, default=1234, help='Random seed')
parser.add_argument('--normalization', action='store_true', help='Use normalization')
parser.add_argument('--use_LDA', action='store_true', help='Enable LDA edges')
parser.add_argument('--syn_threshold', type=float, default=0.4, help='Syntax similarity threshold')
parser.add_argument('--syn_weight', type=float, default=0.7, help='Syntax edge weight')
parser.add_argument('--syn_edge_limit', type=int, default=500, help='Max syntax edges')

args = parser.parse_args()
print(args)

# Seed everything
SEED = args.rand
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

def main():
    # Load and preprocess data
    data = read_file(args.dataset, args.use_LDA)
    doc_content_list, doc_train_list, doc_test_list, vocab_dic, labels_dic, max_num_sentence, keywords_dic, class_weights = data

    # Load embeddings
    pre_trained_weight = None
    if args.dataset in ['mr', 'Tweet']:
        glove_path = 'data/glove.6B.300d.txt'
        if not os.path.exists(glove_path):
            raise FileNotFoundError(f"GloVe embeddings not found at {glove_path}")
        pre_trained_weight = loadGloveModel(glove_path, vocab_dic, len(vocab_dic)+1)

    # Create datasets
    train_data, valid_data = split_validation(doc_train_list, args.valid_portion, SEED)
    test_data = split_validation(doc_test_list, 0.0, SEED)

    num_categories = len(labels_dic)
    
    # Initialize Data objects with proper edge parameters
    train_data = Data(train_data, max_num_sentence, keywords_dic, num_categories, args.use_LDA)
    valid_data = Data(valid_data, max_num_sentence, keywords_dic, num_categories, args.use_LDA)
    test_data = Data(test_data, max_num_sentence, keywords_dic, num_categories, args.use_LDA)
    
    # Set edge parameters from arguments
    train_data.syn_edge_limit = args.syn_edge_limit
    train_data.syntax_weight = args.syn_weight
    valid_data.syn_edge_limit = args.syn_edge_limit
    valid_data.syntax_weight = args.syn_weight
    test_data.syn_edge_limit = args.syn_edge_limit
    test_data.syntax_weight = args.syn_weight

    # Initialize model
    if args.dataset in ['mr', 'Tweet']:
        model = trans_to_cuda(DocumentGraph(
            args, pre_trained_weight, class_weights,
            len(vocab_dic)+2, len(labels_dic), vocab_dic))
    else:
        model = trans_to_cuda(DocumentGraph(
            args, None, class_weights,
            len(vocab_dic)+2, len(labels_dic)))

    model.reset_parameters()

    # Training loop
    best_acc = 0
    for epoch in range(args.epoch):
        print(f'\nEpoch {epoch+1}/{args.epoch}')
        train_model(model, train_data, args)
        
        # Validation
        _, val_acc = test_model(model, valid_data, args)
        _, test_acc = test_model(model, test_data, args)
        
        print(f'Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')

    print(f'Best validation accuracy: {best_acc:.4f}')

if __name__ == '__main__':
    main()