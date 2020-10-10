data_path=path/to/data
python preprocess.py --source-lang en --target-lang ro \
    --trainpref $data_path/train.bpe \
    --validpref $data_path/dev.bpe \
    --testpref $data_path/test.bpe \
    --destdir data-bin/wmt16.en_ro \
    --workers 40 \
    --join-dictionary