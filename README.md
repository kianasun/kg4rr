# Assessing Scientific Research Papers with Knowledge Graphs

## Input Data Format
1. entities.dict: each line represents an entity including an index and entity name separated by "\t".
2. relations.dict: each line represents a relation including an index and relation name separated by "\t".
3. triplets.txt: each line represents a triplet including an entity name, a relation name and another entity name separated by "\t".
4. content_emb.npy: a numpy array with the shape [number of entities, dim of text embeddings].
5. num_feats.npy: a numpy array with the shape [number of entities, dim of numeric features].
6. node2lab.csv: a csv file with labels for paper entities including entity indices (Index) and their scores (label).

## Run LiteralE-based KG embedding models

```
bash scripts/run_kge_rpp.sh 0 CombineLiteralAll_X 100 10 # 100 dimension and 10 epochs
```
X is a base KGE model (e.g. TransE, DistMult, ComplEx).

## Reproducibility Evaluation
```
bash scripts/run_rr_rpp.sh CombineLiteralAll_X 100 # 100 dimension
```

## Run on your own data
1. Convert the data in the required format (e.g. entities.dict, relations.dict, etc.).
2. Create a directory under "datasets/" and put all files under the new directory.
3. Create scripts follow the sample scripts "scripts/run_**.sh".
4. Run the scripts.

## Acknowledgements
Our code is mainly based on [PyKEEN](https://github.com/pykeen/pykeen). Thanks to the organizers for developing and sharing the library!