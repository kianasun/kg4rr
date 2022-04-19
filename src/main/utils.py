
import torch

from torch.nn.functional import embedding
from pykeen.nn.combinations import DistMultCombination, GatedCombination
from pykeen.nn.modules import DistMultInteraction, RotatEInteraction, TransEInteraction, DistMAInteraction
import sys
import numpy as np
import random
from models import CombineLiteral
from pykeen.models import RotatE, TransE, ConvE, MuRE, DistMA
import csv

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    # if you are suing GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def read_triples(path: str, entity2id: dict, relation2id: dict):
    triples = []
    with open(path) as fin:
        reader = csv.reader(fin, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            h, r, t = row
            triples.append([entity2id[h], relation2id[r], entity2id[t]])
    return torch.tensor(triples, dtype=torch.long)

def read_id2obj(path: str):
    dic = {}
    with open(path, "r") as f:
        for line in f:
            (id, obj) = line.strip().split("\t")
            dic[obj] = int(id)
    return dic


def process_model(args):

    model_maps = {"TransE": TransE, "RotatE": RotatE,
                  "ConvE": ConvE, "MuRE": MuRE, "DistMA": DistMA}

    if "CombineLiteral" in args.model:
        return CombineLiteral
    elif "CombineGatedLiteral" in args.model:
        return CombineLiteral
    elif args.model in model_maps:
        return model_maps[args.model]
    else:
        return args.model

def get_literal_emb(text_emb, num_emb, args):
    if "Literal" in args.model:
        if "All" in args.model:
            return torch.cat((text_emb, num_emb), dim=1).clone()
        elif "Text" in args.model:
            return text_emb.clone()
        else:
            return num_emb.clone()
    else:
        return torch.cat((text_emb, num_emb), dim=1).clone()

def convert_num(num):
    new_num = [-10 if float(v.item()) <= 0 else int(torch.log(v).item())
                for v in num]
    return torch.tensor(new_num).float()

def process_kwargs(args):
    if "CombineLiteral" in args.model:
        _c = args.model.split("_")
        if len(_c) == 0:
            return dict(embedding_dim=args.dim)
        elif _c[1] == "TransE":
            return dict(embedding_dim=args.dim,
                        base_interaction=TransEInteraction,
                        combine_interaction=DistMultCombination,
                        interact_kwargs=dict(p=2))
        elif _c[1] == "DistMA":
            return dict(embedding_dim=args.dim,
                        base_interaction=DistMAInteraction,
                        combine_interaction=DistMultCombination,
                        interact_kwargs=dict())
        elif _c[1] == "RotatE":
            return dict(embedding_dim=args.dim,
                        base_interaction=RotatEInteraction,
                        interact_kwargs=dict())
        elif _c[1] == "ComplEx":
            return dict(embedding_dim=args.dim,
                        interact_kwargs=dict())
        elif _c[1] == "DistMult":
            return dict(embedding_dim=args.dim,
                        base_interaction=DistMultInteraction,
                        combine_interaction=DistMultCombination,
                        interact_kwargs=dict())
        else:
            print("Error model")
            sys.exit(0)
    elif "CombineGatedLiteral" in args.model:
        _c = args.model.split("_")
        if _c[1] == "TransE":
            return dict(embedding_dim=args.dim,
                        base_interaction=TransEInteraction,
                        combine_interaction=GatedCombination,
                        interact_kwargs=dict(p=2))
        elif _c[1] == "DistMult":
            return dict(embedding_dim=args.dim,
                        base_interaction=DistMultInteraction,
                        combine_interaction=GatedCombination,
                        interact_kwargs=dict())
        else:
            print("Error model")
            sys.exit(0)
    else:
        return dict(embedding_dim=args.dim)
