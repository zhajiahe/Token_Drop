# Token_Drop
This repository contains the original implementation of the token drop methods presented in   
[Token Drop mechanism for Neural Machine Translation](todo) (COLING 2020)  
code based on fairseq-v0.9.0
## Environment
- python=3.8
- pytorch=1.4
## Installation
`git clone https://github.com/zhajiahe/Token_Drop.git`

`cd Token_Drop && pip intall --editable .`
## Prepare data (EN-RO)
#### Download data
Download WMT16'EN-RO preprocessed by [Lee et al. 2018](https://arxiv.org/abs/1802.06901) : [Data](https://drive.google.com/file/d/1YrAwCEuktG-iDVxtEW-FE72uFTLc5QMl/view?usp=sharing)
#### Binarize data
`bash preprocess.sh`
## Training model
We conduct our experiment on 2 Tesla V100 by defalut.
#### Baseline system
```
CUDA_VISIBLE_DEVICES=0,1 python train.py data-bin/wmt16.en_ro --optimizer adam --criterion label_smoothed_cross_entropy --lr-scheduler inverse_sqrt \
                        --arch transformer --min-lr  1e-09 --lr  0.0003 --label-smoothing 0.1 --dropout 0.3 --adam-betas '(0.9, 0.98)' \
                        --warmup-init-lr 1e-07 --warmup-updates 4000 --keep-last-epochs 10 \
                        --share-all-embeddings --fp16 --max-tokens 7500 \
                        --eval-bleu --eval-bleu-remove-bpe \
                        --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
                        -s en -t ro --save-dir baseline_model --max-epoch 100 \
                        --src-drop 0.0 --tgt-drop 0.0 --drop-method 'none' \
```
#### Token drop system
```
CUDA_VISIBLE_DEVICES=0,1 python train.py data-bin/wmt16.en_ro --optimizer adam --criterion label_smoothed_cross_entropy --lr-scheduler inverse_sqrt \
                        --arch transformer --min-lr  1e-09 --lr  0.0003 --label-smoothing 0.1 --dropout 0.3 --adam-betas '(0.9, 0.98)' \
                        --warmup-init-lr 1e-07 --warmup-updates 4000 --keep-last-epochs 10 \
                        --share-all-embeddings --fp16 --max-tokens 7500 \
                        --eval-bleu --eval-bleu-remove-bpe \
                        --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
                        -s en -t ro --save-dir tokendrop_model --max-epoch 200 \
                        --src-drop 0.15 --tgt-drop 0.3 --drop-method 'unk_tag' --DTP --RTD \
```
## Evaluation
#### Checkpoint average
```
python scripts/average_checkpoints.py --inputs checkpoints/ --output baseline_model/checkpoint_avg10.pt --num-epoch-checkpoint 10
```
#### Generate
```
python generate.py data-bin/wmt16.en_ro  --path baseline_model/checkpoint_avg10.pt --gen-subset test --beam 4 --remove-bpe
```
