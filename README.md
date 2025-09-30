
# Learning to Relative Expression under Batch Effects and Stochastic Noise in Spatial Transcriptomics
This repository is the official implementation of [Learning to Relative Expression under Batch Effects and Stochastic Noise in Spatial Transcriptomics]
<!-- (https://arxiv.org/abs/2030.12345).  -->

![Concept](./figures/concept.png)

## Requirements

- pytorch
- doten

## Training and evaluation on synthetic data
To train the model(s) in the paper, run this command:

```train and evaluation
cd toy_example
bash ./scripts/run_comparisons.sh
```

## Training and evaluation on real dataset
Data preparation:
Our experimetns used [HEST-1k](https://github.com/mahmoodlab/HEST/tree/main) dataset. Before execute our experiments, you should get permittion. 

Execute all experiments

```eval
python ./scripts/execute_all_exps.py
```

## Separately running code

'''
# download dataset
# Put HEST_v1_1_0.csv file into dataset/hest1k/ 
python ./preprocessing/download_hest_benchmarks.py st_v2
python ./preprocessing/download_hest_benchmarks.py task_1
python ./preprocessing/download_hest_benchmarks.py st_v3

# preprocessing
python ./preprocessing/make_paired.py ./dataset/hest1k/task_1
python ./preprocessing/feature_extraction.py --model_name conch_v1 \
    --save_dir ./dataset/hest1k/task_1/feat/conch_v1 \
    --input_dir ./dataset/hest1k/task_1/paired_data

python ./scripts/export_highly_variable.py \
    --data_dir ./dataset/hest1k/task_1/feat/conch_v1 \
    --output_path ./dataset/hest1k/task_1/opts/comp/highly_variable_genes_50.txt \
    --ntop_genes 50

# run benchmark
# download dataset
python ./preprocessing/download_hest_benchmarks.py

# preprocessing
python ./preprocessing/make_paired.py ./dataset/hest1k/task_1
python ./preprocessing/feature_extraction.py --model_name conch_v1 \
    --save_dir ./dataset/hest1k/task_1/feat/conch_v1 \
    --input_dir ./dataset/hest1k/task_1/paired_data
python ./scripts/export_highly_variable.py \
    --data_dir ./dataset/hest1k/task_1/feat/conch_v1 \
    --output_path ./dataset/hest1k/task_1/opts/comp/highly_variable_genes.txt \
    --ntop_genes 50

# run benchmark
python ./strank/train.py \
        --data_dir  ./dataset/hest1k/task_1/feat/conch_v1\
        --param_path ./dataset/hest1k/task_1/opts/comp/stranklist/opt_param.pt \
        --test_sample_ids NCBI783  \
        --val_sample_ids TENX95 \
        --log_dir ./dataset/hest1k/task_1/opts/comp/logs \
        --loss stranklist \
        --model linear \
        --max_epochs 1000 \
        --use_gene ./dataset/hest1k/task_1/opts/comp/highly_variable_genes.txt \
        --ngpu 1

python ./strank/evaluation.py \
        --data_dir ./dataset/hest1k/task_1/feat/conch_v1\
        --param_path ./dataset/hest1k/task_1/opts/comp/stranklist/opt_param.pt \
        --sample_ids NCBI783  \
        --model linear \
        --loss stranklist \
        --batch_size 1024 \
        --use_gene ./dataset/hest1k/task_1/opts/comp/highly_variable_genes.txt \
        --output_csv {save_path}
'''


## Results

Our model achieves the following performance on :


**Table: Real dataset from Hest 1k. Bold = best performance, Underline = second-best. Ave. = average performance.**

|           | Loss        | IDC   | PRAD  | PAAD  | COAD  | READ  | ccRCC | IDC-L | Ave.  |
|-----------|-------------|-------|-------|-------|-------|--------|--------|--------|--------|
| **Point** | MSE         | 0.393 | 0.484 | 0.307 | 0.556 | 0.140 | 0.093 | 0.168 | 0.306 |
|           | Po          | 0.314 | _0.485_ | 0.336 | 0.524 | **0.172** | 0.091 | 0.134 | 0.293 |
|           | NB          | 0.199 | **0.491** | 0.119 | 0.538 | _0.160_ | 0.075 | 0.126 | 0.244 |
| **Pair**  | Rank        | 0.317 | 0.317 | 0.181 | 0.566 | 0.047 | 0.059 | 0.110 | 0.228 |
|           | PairSTrank  | _0.494_ | 0.458 | **0.346** | _0.613_ | 0.136 | **0.127** | _0.228_ | _0.343_ |
| **List**  | PCC         | 0.472 | 0.459 | 0.307 | **0.640** | 0.105 | 0.102 | 0.198 | 0.326 |
|           | ListSTrank  | **0.510** | 0.459 | _0.343_ | 0.597 | 0.140 | _0.125_ | **0.238** | **0.345** |



<!-- ## Contributing -->


## Acknowledgement
- We used [HEST-1k](https://github.com/mahmoodlab/HEST) dataset.
- For the feature extractor, we implemented the code based on [CLAM](https://github.com/mahmoodlab/CLAM).





LOSS_NAMES=(mse nb poisson strankg pearsona stranklist ranking)
LOSS_NAMES=(stranklist_v2)
GENE_NUMS=(250 1000)
for num_gene in "${GENE_NUMS[@]}"
do
for loss in "${LOSS_NAMES[@]}"
do 

python ./strank/train.py \
        --data_dir  ./dataset/hest1k/st_v2/feat/conch_v1\
        --param_path ./dataset/hest1k/st_v2/opts/comp/${loss}/opt_param_stnet.pt \
        --test_sample_ids SPA142 SPA141 SPA140 SPA139 SPA138 SPA137  \
        --val_sample_ids SPA136 SPA135 SPA134 SPA133 SPA132 SPA131 \
        --log_dir ./dataset/hest1k/st_v2/opts/comp/logs \
        --loss ${loss} \
        --model linear \
        --max_epochs 500 \
        --use_gene ./dataset/hest1k/st_v2/opts/comp/highly_variable_genes_${num_gene}.txt \
        --ngpu 1

python ./strank/evaluation.py \
        --data_dir ./dataset/hest1k/st_v2/feat/conch_v1\
        --param_path ./dataset/hest1k/st_v2/opts/comp/${loss}/opt_param.pt \
        --sample_ids SPA142 SPA141 SPA140 SPA139 SPA138 SPA137  \
        --model linear \
        --loss ${loss} \
        --batch_size 1024 \
        --use_gene ./dataset/hest1k/st_v2/opts/comp/highly_variable_genes_${num_gene}.txt \
        --output_csv output/stnet/${num_gene}/st_${loss}_exp.csv
done
done



########## memo
LOSS_NAMES=(mse nb poisson ranking strankg pearsona stranklist)
GENE_NUMS=(50 250)
for num_gene in "${GENE_NUMS[@]}"
do
python ./scripts/export_highly_variable.py \
    --data_dir ./dataset/hest1k/st_v3/feat/conch_v1 \
    --output_path ./dataset/hest1k/st_v3/opts/comp/highly_variable_genes_${num_gene}.txt \
    --ntop_genes ${num_gene}

for loss in "${LOSS_NAMES[@]}"
do 

python ./strank/train.py \
        --data_dir  ./dataset/hest1k/st_v3/feat/conch_v1\
        --param_path ./dataset/hest1k/st_v3/opts/comp/${loss}/opt_param.pt \
        --test_sample_ids SPA142 SPA141 SPA140 SPA139 SPA138 SPA137  \
        --val_sample_ids SPA136 SPA135 SPA134 SPA133 SPA132 SPA131 \
        --log_dir ./dataset/hest1k/st_v3/opts/comp/logs \
        --loss ${loss} \
        --model linear \
        --max_epochs 500 \
        --use_gene ./dataset/hest1k/st_v3/opts/comp/highly_variable_genes_${num_gene}.txt \
        --ngpu 1

python ./strank/evaluation.py \
        --data_dir ./dataset/hest1k/st_v3/feat/conch_v1\
        --param_path ./dataset/hest1k/st_v3/opts/comp/${loss}/opt_param.pt \
        --sample_ids SPA142 SPA141 SPA140 SPA139 SPA138 SPA137  \
        --model linear \
        --loss ${loss} \
        --batch_size 1024 \
        --use_gene ./dataset/hest1k/st_v3/opts/comp/highly_variable_genes_${num_gene}.txt \
        --output_csv output/st3/${num_gene}/st_${loss}_exp.csv
done
done

python ./strank/train.py \
        --data_dir  ./dataset/hest1k/task_1/feat/conch_v1\
        --param_path ./dataset/hest1k/task_1/opts/comp/${loss}/opt_param_${num_gene}.pt \
        --test_sample_ids NCBI783  \
        --val_sample_ids TENX95 \
        --log_dir ./dataset/hest1k/task_1/opts/comp/logs \
        --loss ${loss} \
        --model linear \
        --max_epochs 1000 \
        --use_gene ./dataset/hest1k/task_1/opts/comp/highly_variable_genes_${num_gene}.txt \
        --ngpu 1

python ./strank/evaluation.py \
        --data_dir ./dataset/hest1k/task_1/feat/conch_v1\
        --param_path ./dataset/hest1k/task_1/opts/comp/${loss}/opt_param_${num_gene}.pt \
        --sample_ids NCBI783  \
        --model linear \
        --loss ${loss} \
        --batch_size 1024 \
        --use_gene ./dataset/hest1k/task_1/opts/comp/highly_variable_genes_${num_gene}.txt \
        --output_csv output/all_generesult_${loss}_${num_gene}.csv
done
done

$$$$$$$$$$$$$$
LOSS_NAMES=(strankg pearsona stranklist)
GENE_NUMS=(50 250)
for num_gene in "${GENE_NUMS[@]}"
do
python ./scripts/export_highly_variable.py \
    --data_dir ./dataset/hest1k/st_v3/feat/conch_v1 \
    --output_path ./dataset/hest1k/st_v3/opts/comp/highly_variable_genes_${num_gene}.txt \
    --ntop_genes ${num_gene}

for loss in "${LOSS_NAMES[@]}"
do 

python ./strank/train.py \
        --data_dir  ./dataset/hest1k/st_v3/feat/densenet121\
        --param_path ./dataset/hest1k/st_v3/opts/comp/${loss}/opt_param.pt \
        --test_sample_ids SPA118 SPA117 SPA116  \
        --val_sample_ids SPA115 SPA114 SPA113 \
        --log_dir ./dataset/hest1k/st_v3/opts/comp/logs \
        --loss ${loss} \
        --model linear \
        --max_epochs 500 \
        --use_gene ./dataset/hest1k/st_v3/opts/comp/highly_variable_genes_${num_gene}.txt \
        --ngpu 1

python ./strank/evaluation.py \
        --data_dir ./dataset/hest1k/st_v3/feat/conch_v1\
        --param_path ./dataset/hest1k/st_v3/opts/comp/${loss}/opt_param.pt \
        --sample_ids SPA118 SPA117 SPA116  \
        --model linear \
        --loss ${loss} \
        --batch_size 1024 \
        --use_gene ./dataset/hest1k/st_v3/opts/comp/highly_variable_genes_${num_gene}.txt \
        --output_csv output/st3/${num_gene}/st_${loss}_exp.csv
done
done
st
['SPA118', 'SPA117', 'SPA116']
['SPA115', 'SPA114', 'SPA113']
$$$$$$$$$$$$$$$


num_gene=50
LOSS_NAMES=(mse nb poisson ranking strankg pearsona stranklist)
for loss in "${LOSS_NAMES[@]}"
do 

python ./strank/train_hist.py \
        --data_dir  ./dataset/hest1k/st_v2/paired_data\
        --param_path ./dataset/hest1k/st_v2/opts/comp/${loss}/opt_param_his_${num_gene}.pt \
        --test_sample_ids SPA136 SPA135 SPA134 SPA133 SPA132 SPA131  \
        --val_sample_ids  SPA142 SPA141 SPA140 SPA139 SPA138 SPA137\
        --log_dir ./dataset/hest1k/st_v2/opts/comp/logs_his \
        --loss ${loss} \
        --model linear \
        --max_epochs 500 \
        --use_gene ./dataset/hest1k/st_v2/opts/comp/highly_variable_genes_${num_gene}.txt \
        --ngpu 1

python ./strank/evaluation_his2gene.py \
        --data_dir ./dataset/hest1k/st_v2/paired_data\
        --param_path ./dataset/hest1k/st_v2/opts/comp/${loss}/opt_param_his_${num_gene}.pt \
        --sample_ids SPA136 SPA135 SPA134 SPA133 SPA132 SPA131 \
        --model linear \
        --loss ${loss} \
        --batch_size 1024 \
        --use_gene ./dataset/hest1k/st_v2/opts/comp/highly_variable_genes_${num_gene}.txt \
        --output_csv output2/his2gene_${num_gene}/st_${loss}_exp.csv
done


# to evaluate method on max number of genes
LOSS_NAMES=(mse nb poisson ranking strankg pearsona stranklist_v2)
GENE_NUMS=(500)
for num_gene in "${GENE_NUMS[@]}"
do
python ./scripts/export_highly_variable.py \
    --data_dir ./dataset/hest1k/task_1/feat/conch_v1 \
    --output_path ./dataset/hest1k/task_1/opts/comp/highly_variable_genes_${num_gene}.txt \
    --ntop_genes ${num_gene}

for loss in "${LOSS_NAMES[@]}"
do 

python ./strank/train.py \
        --data_dir  ./dataset/hest1k/st_task_1v3/feat/conch_v1\
        --param_path ./dataset/hest1k/task_1/opts/comp/${loss}/opt_param.pt \
        --test_sample_ids SPA142 SPA141 SPA140 SPA139 SPA138 SPA137  \
        --val_sample_ids SPA136 SPA135 SPA134 SPA133 SPA132 SPA131 \
        --log_dir ./dataset/hest1k/task_1/opts/comp/logs \
        --loss ${loss} \
        --model linear \
        --max_epochs 500 \
        --use_gene ./dataset/hest1k/task_1/opts/comp/highly_variable_genes_${num_gene}.txt \
        --ngpu 1

python ./strank/evaluation.py \
        --data_dir ./dataset/hest1k/task_1/feat/conch_v1\
        --param_path ./dataset/hest1k/task_1/opts/comp/${loss}/opt_param.pt \
        --sample_ids SPA142 SPA141 SPA140 SPA139 SPA138 SPA137  \
        --model linear \
        --loss ${loss} \
        --batch_size 1024 \
        --use_gene ./dataset/hest1k/task_1/opts/comp/highly_variable_genes_${num_gene}.txt \
        --output_csv output/st3/${num_gene}/st_${loss}_exp.csv
done
done

# strank 10000 genes
LOSS_NAMES=(mse nb poisson ranking strankg pearsona stranklist_v2)
LOSS_NAMES=(strankg pearsona stranklist_v2)
GENE_NUMS=(10000)
for num_gene in "${GENE_NUMS[@]}"
do
python ./scripts/export_highly_variable.py \
    --data_dir ./dataset/hest1k/st_v2/feat/conch_v1 \
    --output_path ./dataset/hest1k/st_v2/opts/comp/highly_variable_genes_${num_gene}.txt \
    --ntop_genes ${num_gene}

for loss in "${LOSS_NAMES[@]}"
do 

python ./strank/train.py \
        --data_dir  ./dataset/hest1k/st_v2/feat/conch_v1\
        --param_path ./dataset/hest1k/st_v2/opts/comp/${loss}/opt_param_${num_gene}.pt \
        --test_sample_ids SPA142 SPA141 SPA140 SPA139 SPA138 SPA137  \
        --val_sample_ids SPA136 SPA135 SPA134 SPA133 SPA132 SPA131 \
        --log_dir ./dataset/hest1k/st_v2/opts/comp/logs \
        --loss ${loss} \
        --model linear \
        --max_epochs 500 \
        --use_gene ./dataset/hest1k/st_v2/opts/comp/highly_variable_genes_${num_gene}.txt \
        --ngpu 1

python ./strank/evaluation.py \
        --data_dir ./dataset/hest1k/st_v2/feat/conch_v1\
        --param_path ./dataset/hest1k/st_v2/opts/comp/${loss}/opt_param_${num_gene}.pt \
        --sample_ids SPA142 SPA141 SPA140 SPA139 SPA138 SPA137  \
        --model linear \
        --loss ${loss} \
        --batch_size 1024 \
        --use_gene ./dataset/hest1k/st_v2/opts/comp/highly_variable_genes_${num_gene}.txt \
        --output_csv output/${num_gene}/st_${loss}_exp.csv
done
done

## batch sampling change
LOSS_NAMES=(stranklist)
SAMPLING=(single pat)
GENE_NUMS=(250 1000)
for num_gene in "${GENE_NUMS[@]}"; 
do 
for sampling in "${SAMPLING[@]}"; 
do 
for loss in "${LOSS_NAMES[@]}"; 
do  

python ./strank/evaluation.py  \
        --data_dir ./dataset/hest1k/st_v2/feat/conch_v1  \
        --param_path ./dataset/hest1k/st_v2/opts/comp/${loss}/opt_param_${num_gene}_${sampling}.pt  \
        --sample_ids SPA142 SPA141 SPA140 SPA139 SPA138 SPA137  \
        --model linear  \
        --loss ${loss}  \
        --batch_size 1024  \
        --use_gene ./dataset/hest1k/st_v2/opts/comp/highly_variable_genes_${num_gene}.txt  \
        --output_csv output/sampling/${num_gene}_${sampling}/st_${loss}_exp.csv
done
done
done

python ./strank/train.py        \
        --data_dir  ./dataset/hest1k/st_v2/feat/conch_v1   \
        --param_path ./dataset/hest1k/st_v2/opts/comp/${loss}/opt_param_${num_gene}_${sampling}.pt  \
        --test_sample_ids SPA142 SPA141 SPA140 SPA139 SPA138 SPA137  \
        --val_sample_ids SPA136 SPA135 SPA134 SPA133 SPA132 SPA131  \
        --log_dir ./dataset/hest1k/st_v2/opts/comp/logs \
        --loss ${loss} --model linear --max_epochs 500 \
        --use_gene ./dataset/hest1k/st_v2/opts/comp/highly_variable_genes_${num_gene}.txt --ngpu 1 \
        --sampling_strategy ${sampling}


## List loss evaluation
LOSS_NAMES=(stranklist_v2)
GENE_NUMS=(250 1000)
for num_gene in "${GENE_NUMS[@]}"; 
do 
for loss in "${LOSS_NAMES[@]}"; 
do  
python ./strank/train.py         --data_dir  ./dataset/hest1k/st_v2/feat/conch_v1        --param_path ./dataset/hest1k/st_v2/opts/comp/${loss}/opt_param_${num_gene}.pt         --test_sample_ids SPA142 SPA141 SPA140 SPA139 SPA138 SPA137          --val_sample_ids SPA136 SPA135 SPA134 SPA133 SPA132 SPA131         --log_dir ./dataset/hest1k/st_v2/opts/comp/logs         --loss ${loss}         --model linear         --max_epochs 500         --use_gene ./dataset/hest1k/st_v2/opts/comp/highly_variable_genes_${num_gene}.txt         --ngpu 1 

python ./strank/evaluation.py         --data_dir ./dataset/hest1k/st_v2/feat/conch_v1        --param_path ./dataset/hest1k/st_v2/opts/comp/${loss}/opt_param_${num_gene}.pt         --sample_ids SPA142 SPA141 SPA140 SPA139 SPA138 SPA137          --model linear         --loss ${loss}         --batch_size 1024         --use_gene ./dataset/hest1k/st_v2/opts/comp/highly_variable_genes_${num_gene}.txt         --output_csv output/${num_gene}/st_${loss}_exp.csv
done
done

## 500 gene xenium target sample を変更すること

LOSS_NAMES=(mse nb poisson ranking strankg pearsona stranklist_v2)
GENE_NUMS=(500)
for num_gene in "${GENE_NUMS[@]}"
do
python ./scripts/export_highly_variable.py \
    --data_dir ./dataset/hest1k/task_1/feat/conch_v1 \
    --output_path ./dataset/hest1k/task_1/opts/comp/highly_variable_genes_${num_gene}.txt \
    --ntop_genes ${num_gene}

for loss in "${LOSS_NAMES[@]}"
do 

python ./strank/train.py \
        --data_dir  ./dataset/hest1k/task_1/feat/conch_v1\
        --param_path ./dataset/hest1k/task_1/opts/comp/${loss}/opt_param_${num_gene}.pt \
        --test_sample_ids SPA142 SPA141 SPA140 SPA139 SPA138 SPA137  \
        --val_sample_ids SPA136 SPA135 SPA134 SPA133 SPA132 SPA131 \
        --log_dir ./dataset/hest1k/task_1/opts/comp/logs \
        --loss ${loss} \
        --model linear \
        --max_epochs 500 \
        --use_gene ./dataset/hest1k/task_1/opts/comp/highly_variable_genes_${num_gene}.txt \
        --ngpu 1

python ./strank/evaluation.py \
        --data_dir ./dataset/hest1k/task_1/feat/conch_v1\
        --param_path ./dataset/hest1k/task_1/opts/comp/${loss}/opt_param_${num_gene}.pt \
        --sample_ids SPA142 SPA141 SPA140 SPA139 SPA138 SPA137  \
        --model linear \
        --loss ${loss} \
        --batch_size 1024 \
        --use_gene ./dataset/hest1k/task_1/opts/comp/highly_variable_genes_${num_gene}.txt \
        --output_csv output/${num_gene}/st_${loss}_exp.csv
done
done
