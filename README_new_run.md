# Correct and Smooth (C&S) OGB submissions

Paper: https://arxiv.org/abs/2010.13993

This directory contains OGB submissions. All hyperparameters were tuned on the validation set with optuna, except for products, which was hand tuned. All experiments were run with a RTX 2080 TI with 11GB.

## Some Tips 
- In general, the more complex and "smooth" your GNN is, the less likely it'll be that applying the "Correct" portion helps performance. In those cases, you may consider just applying the "smooth" portion, like we do on the GAT. In almost all cases, applying the "smoothing" component will improve performance. For Linear/MLP models, applying the "Correct" portion is almost always essential for obtaining good performance.

- In a similar vein, an improvement of performance of your model may not correspond to an improvement after applying C&S. Considering that C&S learns no parameters over your data, our intuition is that C&S "levels" the playing field, allowing models that learn interesting features to shine (as opposed to learning how to be smooth).
     - Even though GAT (73.57) is outperformed by GAT + labels (73.65), when we apply C&S, we see that GAT + C&S (73.86) performs better than GAT + labels + C&S (~73.70) , 
     - Even though a 6 layer GCN performs on par with a 2 layer GCN with Node2Vec features, C&S improves performance of the 2 layer GCN with Node2Vec features substantially more.
     - Even though MLP + Node2Vec outperforms MLP + Spectral in both arxiv and products, the performance ordering flips after we apply C&S.
     - On Products, the MLP (74%) is substantially outperformed by ClusterGCN (80%). However, MLP + C&S (84.1%) substantially outperforms ClusterGCN + C&S (82.4%).

- In general, autoscale works more reliably than fixedscale, even though fixedscale may make more sense...

## Arxiv

### Label Propagation (0 params):  --> Done
```
python run_experiments.py --dataset arxiv --method lp
Valid acc:  0.7018356320681902
Test acc: 0.683496903483324



python run_experiments_2.py --dataset cora --method lp	D
Valid acc:  0.744
Test acc: 0.7858823529411765

Valid acc: 0.7013658176448874
Test acc: 0.6832294302820814
```




### Plain Linear + C&S (5160 params, 52.5% base accuracy) --> Done
```
python gen_models.py --dataset arxiv --model plain --epochs 1000
All runs:
Highest Train: 55.2123 ± 0.0029
Highest Valid: 55.0351 ± 0.0116
  Final Train: 55.1803 ± 0.0123
   Final Test: 52.5145 ± 0.0250
   
       
python run_experiments.py --dataset arxiv --method plain
Valid acc -> Test acc
Args []: 72.99 ± 0.01 -> 71.26 ± 0.01

python gen_models_2.py --dataset cora --model plain --epochs 1000  D
All runs:
Highest Train: 13.2265 ± 0.0000
Highest Valid: 13.2000 ± 0.0000
  Final Train: 13.2265 ± 0.0000
   Final Test: 12.8235 ± 0.0000
   
   
python run_experiments_2.py --dataset cora --method plain	D
Valid acc -> Test acc
Args []: 13.20 ± 0.00 -> 12.82 ± 0.00


Valid acc -> Test acc
Args []: 73.00 ± 0.01 -> 71.26 ± 0.01
```

### Linear + C&S (15400 params, 70.11% base accuracy)  --> Done
```
python gen_models.py --dataset arxiv --model linear --use_embeddings --epochs 1000 
All runs:
Highest Train: 74.6781 ± 0.0157
Highest Valid: 71.4017 ± 0.0323
  Final Train: 74.0235 ± 0.1915
   Final Test: 70.0743 ± 0.0262
   
   
python run_experiments.py --dataset arxiv --method linear
Valid acc -> Test acc
Args []: 73.67 ± 0.03 -> 72.21 ± 0.04



python gen_models_2.py --dataset cora --model linear --use_embeddings --epochs 1000	D
All runs:
Highest Train: 13.2265 ± 0.0000
Highest Valid: 13.2000 ± 0.0000
  Final Train: 13.2265 ± 0.0000
   Final Test: 12.8235 ± 0.0000		
python run_experiments_2.py --dataset cora --method linear	D
Valid acc -> Test acc
Args []: 13.20 ± 0.00 -> 12.82 ± 0.00


Valid acc -> Test acc
Args []: 73.68 ± 0.04 -> 72.22 ± 0.02;
```

### MLP + C&S (175656 params, 71.44% base accuracy) --> Done
```
python gen_models.py --dataset arxiv --model mlp --use_embeddings
All runs:
Highest Train: 61.0914 ± 0.2815
Highest Valid: 62.0014 ± 0.3730
  Final Train: 61.0914 ± 0.2815
   Final Test: 60.3395 ± 0.5196
   
python run_experiments.py --dataset arxiv --method mlp
Valid acc -> Test acc
Args []: 70.36 ± 0.21 -> 69.10 ± 0.23


python gen_models_2.py --dataset cora --model mlp --use_embeddings	D
All runs:
Highest Train: 13.2265 ± 0.0000
Highest Valid: 13.2000 ± 0.0000
  Final Train: 13.2265 ± 0.0000
   Final Test: 12.8235 ± 0.0000
   
python run_experiments_2.py --dataset cora --method mlp	D
Valid acc -> Test acc
Args []: 13.20 ± 0.00 -> 12.82 ± 0.00

Valid acc -> Test acc
Args []: 73.91 ± 0.15 -> 73.12 ± 0.12
```

### GAT + C&S (1567000 params, 73.56% base accuracy)
```
cd gat && python gat.py --use-norm --cpu
cd .. && python run_experiments.py --dataset arxiv --method gat

cd gat && python gat.py --use-norm --cpu
cd .. && python run_experiments.py --dataset cora --method gat

Valid acc -> Test acc
Args []: 74.84 ± 0.07 -> 73.86 ± 0.14
```

### Notes
As opposed to the paper's results, which only use spectral embeddings, here we use spectral *and* diffusion embeddings, which we find improves Arxiv performance.

## Products

### Label Propagation (0 params): --> killed
```
python run_experiments.py --dataset products --method lp 

Valid acc:  0.9090608549703736
Test acc: 0.7434145274640762
```

### Plain Linear + C&S (4747 params, 47.73% base accuracy) --> killed
```
python gen_models.py --dataset products --model plain --epochs 1000 --lr 0.1
python run_experiments.py --dataset products --method plain

Valid acc -> Test acc
Args []: 91.03 ± 0.01 -> 82.54 ± 0.03
```

### Linear + C&S (10763 params, 50.05% base accuracy) --> killed
```
python gen_models.py --dataset products --model linear --use_embeddings --epochs 1000 --lr 0.1
python run_experiments.py --dataset products --method linear

Valid acc -> Test acc
Args []: 91.34 ± 0.01 -> 83.01 ± 0.01
```

### MLP + C&S (96247 params, 63.41% base accuracy) --> killed
```
python gen_models.py --dataset products --model mlp --hidden_channels 200 --use_embeddings
python run_experiments.py --dataset products --method mlp

Valid acc -> Test acc
Args []: 91.47 ± 0.09 -> 84.18 ± 0.07
```
