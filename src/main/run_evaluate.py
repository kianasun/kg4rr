import argparse
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
import csv
import torch
import numpy as np
import os
import pandas as pd
import random
import copy
from models import EvaluateModel
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from utils import *
from collections import Counter
import scipy.stats as stats
import json

from models import CombineLiteral, TriplesLiteralsFactory
from pykeen.models import RotatE, TransE, ConvE, MuRE, DistMA, ComplEx, DistMult

class CustomDataset(Dataset):
     def __init__(self, X, y):
         super(Dataset, self).__init__()
         self.X = X
         self.y = y

     def __len__(self):
         return len(self.X)

     def __getitem__(self, index):
        return self.X[index], self.y[index]

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training Knowledge Graph Embedding Models',
        usage='run.py [<args>] [-h | --help]'
    )

    parser.add_argument('--device', type=str, default="cpu", help='set device')
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--valid_node', type=str, default=None)
    parser.add_argument('--model', default=None, type=str)
    parser.add_argument('--content', default=None, type=str)
    parser.add_argument('--numeric', default=None, type=str)
    parser.add_argument('--dim', default=400, type=int)
    parser.add_argument('--method', default="node", type=str)
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--eval_type', default="regression", type=str)
    return parser.parse_args(args)


def main(args):
    print("args", args)
    set_seed(args.seed)

    id2ent_path = os.path.join(args.data_path, "entities.dict")
    id2rel_path = os.path.join(args.data_path, "relations.dict")
    triples_path = os.path.join(args.data_path, "triples.txt")
    ent2id = read_id2obj(id2ent_path)
    rel2id = read_id2obj(id2rel_path)
    mapped_triplets = read_triples(triples_path, ent2id, rel2id)

    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    df = pd.read_csv(os.path.join(args.data_path, args.valid_node))

    print(len(ent2id), len(rel2id), mapped_triplets.shape)


    if args.content:
        # preprocess numbers
        text_emb = torch.tensor(np.load(args.content)).float()
    else:
        text_emb = None

    if args.numeric:
        num_emb = torch.tensor(np.load(args.numeric)).float()
        for k in range(num_emb.shape[0]):
            num_emb[k] = convert_num(num_emb[k])
    else:
        num_emb = None

    triples = TriplesLiteralsFactory(
        triples=mapped_triplets,
        literal_embedding=get_literal_emb(text_emb, num_emb, args),
        entity_to_id=checkpoint['entity_to_id_dict'],
        relation_to_id=checkpoint['relation_to_id_dict']
    )

    base_model = process_model(args)(triples_factory=triples,
                                    **process_kwargs(args),
                                    preferred_device="cpu")


    base_model.load_state_dict(checkpoint["model_state_dict"])

    X, y = df["Index"].values, df["label"].values
    # 5-fold CV
    kf = KFold(n_splits=5)

    all_result = []

    for fold_idx, (train_index, test_index) in enumerate(kf.split(X)):
        #print("train", train_index[:10], "test", test_index[:10])
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        temp_model = copy.deepcopy(base_model)

        model = EvaluateModel(temp_model, text_emb.clone(), num_emb.clone(), args.model, args.method)

        model = model.to(args.device)
        model.ent_emb = model.ent_emb.to(args.device)
        model.content_emb = model.content_emb.to(args.device)

        # Set training configuration
        current_learning_rate = 1e-5

        #if model.dim_reduction:
        #    model.dim_reduction.requires_grad = False
        optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=current_learning_rate
        )

        train_dataloader = DataLoader(
            CustomDataset(X_train, y_train),
            batch_size=4,
        )
        test_dataloader = DataLoader(
            CustomDataset(X_test, y_test),
            batch_size=4,
        )

        model.train()
        for epoch in range(100):
            total_loss = 0
            for _i, (_x, _y) in enumerate(train_dataloader):

                optimizer.zero_grad()

                _x = _x.to(args.device)
                _y = _y.to(args.device).float()

                _, loss = model.forward(_x, _y)

                total_loss += loss.item()
                loss.backward()
                optimizer.step()

            if epoch % 10 == 0:
                print("epoch {}: loss: {}".format(epoch, total_loss / len(train_dataloader.dataset)))

        model.eval()
        all_pred, all_gt = [], []
        for (_x, _y) in test_dataloader:
            _x = _x.to(args.device)
            pred = model.forward(_x, None)
            all_pred += pred
            all_gt += _y.cpu().numpy().tolist()

        tau, _p = stats.kendalltau(all_gt, all_pred)

        res = mean_squared_error(all_gt, all_pred, squared=False)
        all_result.append((res, tau))

        print("fold: {}\t train_size: {}\t test_size: {}\trmse: {}\t kendalltau: {}".format(fold_idx, len(train_dataloader.dataset),
                                                                        len(test_dataloader.dataset), res, tau))
    print("Average Result for RMSE: {}; Kendaltau: {}".format(
            sum([v[0] for v in all_result]) / len(all_result),
            sum([v[1] for v in all_result]) / len(all_result)))

if __name__ == '__main__':
    main(parse_args())
