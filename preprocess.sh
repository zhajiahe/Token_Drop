train_dir=path/to/wmt16/en-ro/train
valid_dir=path/to/wmt16/en-ro/dev
test_dir=path/to/wmt16/en-ro/test
python preprocess.py --source-lang en --target-lang ro \
    --trainpref $train_dir/corpus.bpe \
    --validpref $valid_dir/dev.bpe \
    --testpref $test_dir/test.bpe \
    --destdir data-bin/wmt16.en_ro \
    --workers 40 \
    --join-dictionary
