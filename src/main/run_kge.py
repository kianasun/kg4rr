import argparse
import sys

from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
import csv
import torch
import numpy as np

from utils import *
from models import TriplesLiteralsFactory

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training Knowledge Graph Embedding Models',
        usage='run_kge.py [<args>] [-h | --help]'
    )

    parser.add_argument('--device', type=str, default="cuda", help='set device')
    parser.add_argument('--triplets', type=str, default=None)
    parser.add_argument('--id2ent', type=str, default=None)
    parser.add_argument('--id2rel', type=str, default=None)
    parser.add_argument('--model', type=str)
    parser.add_argument('--dim', default=200, type=int)
    parser.add_argument('--batch_size', default=2048, type=int)
    parser.add_argument('--neg_sample_size', default=16, type=int)
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--content', default=None, type=str)
    parser.add_argument('--numeric', default=None, type=str)

    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--epoch', default=3, type=int)
    parser.add_argument('--save', default=None, type=str)
    return parser.parse_args(args)

def main(args):
    set_seed(args.seed)

    ent2id = read_id2obj(args.id2ent)
    rel2id = read_id2obj(args.id2rel)
    mapped_triplets = read_triples(args.triplets, ent2id, rel2id)

    print(len(ent2id), len(rel2id), mapped_triplets.shape)
    print(args.model)

    if args.content:
        # preprocess numbers
        text_emb = torch.tensor(np.load(args.content)).float()
    else:
        text_emb = None

    if args.numeric:
        num_emb = torch.tensor(np.load(args.numeric)).float()
        # number converter for easier learning
        for k in range(num_emb.shape[0]):
            num_emb[k] = convert_num(num_emb[k])
    else:
        num_emb = None

    triples = TriplesLiteralsFactory(
        triples=mapped_triplets,
        literal_embedding=get_literal_emb(text_emb, num_emb, args),
        entity_to_id=ent2id,
        relation_to_id=rel2id
    )

    # dummy testing for running the pipeline
    # todo: remove testing
    _, testing = triples.split(0.99999)
    del _

    result = pipeline(
        training=triples,
        testing=testing,
        model=process_model(args),
        model_kwargs=process_kwargs(args),
        optimizer_kwargs=dict(lr=args.lr),
        negative_sampler_kwargs=dict(num_negs_per_pos=args.neg_sample_size),
        device=args.device,
        random_seed=args.seed,
        training_kwargs=dict(num_epochs=args.epoch,
                            batch_size=args.batch_size,
                            checkpoint_directory=args.save,
                            checkpoint_name='checkpoint.pt',
                            checkpoint_frequency=1),
    )
    result.save_to_directory(args.save)

if __name__ == '__main__':
    main(parse_args())
