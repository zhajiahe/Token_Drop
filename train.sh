CUDA_VISIBLE_DEVICES=0,1 python train.py data-bin/wmt16.en_ro --optimizer adam --criterion label_smoothed_cross_entropy --lr-scheduler inverse_sqrt \
                        --arch transformer --min-lr  1e-09 --lr  0.0003 --label-smoothing 0.1 --dropout 0.3 --adam-betas '(0.9, 0.98)' \
                        --warmup-init-lr 1e-07 --warmup-updates 4000 --keep-last-epochs 10 \
                        --share-all-embeddings --fp16 --max-tokens 7500 \
                        --eval-bleu --eval-bleu-remove-bpe \
                        --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
                        -s en -t ro --save-dir baseline_model --max-epoch 100 \
                        --src-drop 0.0 --tgt-drop 0.0 --drop-method 'none' \
