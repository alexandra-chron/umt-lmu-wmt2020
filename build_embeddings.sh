#!/usr/bin/env bash
# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
set -e

# N_THREADS=`nproc`    # number of threads in data preprocessing
N_THREADS=40    # number of threads in data preprocessing
GPUS=0    # comma separated list of GPU indices or empty for running on CPUS
PYTHON=python

# Read arguments

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"
case $key in
  --src)
    SRC="$2"; shift 2;;
  --tgt)
    TGT="$2"; shift 2;;
  *)
  POSITIONAL+=("$1")
  shift
  ;;
esac
done
set -- "${POSITIONAL[@]}"

# Check parameters
if [ "$SRC" == "" ]; then echo "--src not provided"; exit; fi
if [ "$TGT" == "" ]; then echo "--tgt not provided"; exit; fi
if [ "$SRC" == "$TGT" ]; then echo "source and target cannot be identical"; exit; fi

MAIN_PATH=$PWD
TOOLS_PATH=$PWD/tools
MONOSES_DIR=$TOOLS_PATH/monoses
FASTTEXT_DIR=$TOOLS_PATH/fastText
VECMAP_DIR=$MONOSES_DIR/third-party/vecmap

DATA_PATH=./data/$SRC-wmt
PROC_PATH=./data/$SRC-$TGT-wmt
OUTPUT_DIR=./models/SMT/step3
BWE_OUTPUT_DIR=./models/SMT/step4

SRC_TRAIN=$DATA_PATH/train_raw.$SRC
SRC_TRAIN_TOK=$SRC_TRAIN.tok
SRC_TRAIN_TC=$SRC_TRAIN_TOK.tc

TGT_TRAIN=$PROC_PATH/train_raw.$TGT
TGT_TRAIN_TOK=$TGT_TRAIN.tok
TGT_TRAIN_TC=$TGT_TRAIN_TOK.tc

N_UNIGRAMS=200000
N_BIGRAMS=400000
N_TRIGRAMS=400000

mkdir -p $OUTPUT_DIR
mkdir -p $BWE_OUTPUT_DIR

#############################################################################
echo "Building embeddings for $SRC..."
#############################################################################
if [ ! -f $OUTPUT_DIR/$SRC.phrase_counts ]; then
    echo "Looking for frequent n-grams..."
    $PYTHON $MONOSES_DIR/training/extract-ngrams.py --min-order 2 --max-order 3 --min-count 10 -i $SRC_TRAIN_TC | sort -nr --parallel $N_THREADS > $OUTPUT_DIR/$SRC.phrase_counts
fi

if [ ! -f $OUTPUT_DIR/$SRC.phrases ]; then
    echo "Filtering most frequent n-grams..."
    egrep "^[0-9]+[[:space:]][^ ]+( [^ ]+){1}$" $OUTPUT_DIR/$SRC.phrase_counts | cut -f 2 | head -n $N_BIGRAMS > $OUTPUT_DIR/$SRC.phrases
    egrep "^[0-9]+[[:space:]][^ ]+( [^ ]+){2}$" $OUTPUT_DIR/$SRC.phrase_counts | cut -f 2 | head -n $N_TRIGRAMS >> $OUTPUT_DIR/$SRC.phrases
fi

if [ ! -f $OUTPUT_DIR/$SRC.replaced ]; then
    echo "Merging n-grams in the corpus..."
        $PYTHON ngram_replacer.py -i $SRC_TRAIN_TC -o $OUTPUT_DIR/$SRC.replaced -sp '_' -n $OUTPUT_DIR/$SRC.phrases -keep-orig 1 --threads $N_THREADS
fi

if [ ! -f $OUTPUT_DIR/emb.src.vec ]; then
    echo "Running fasttext..."
    $FASTTEXT_DIR/fasttext skipgram -input $OUTPUT_DIR/$SRC.replaced -output $OUTPUT_DIR/emb.src -dim 300 -ws 5 -neg 5 -loss ns -thread $N_THREADS
fi

if [ ! -f $OUTPUT_DIR/emb.src ]; then
    echo "Filtering vocabulary..."
    tail -n+2 $OUTPUT_DIR/emb.src.vec | egrep "^[^_ ]+ " | head -n $N_UNIGRAMS > $OUTPUT_DIR/emb.src.tmp
    tail -n+2 $OUTPUT_DIR/emb.src.vec | egrep "^[^_ ]+(_[^_ ]+){1} " | head -n $N_BIGRAMS >> $OUTPUT_DIR/emb.src.tmp
    tail -n+2 $OUTPUT_DIR/emb.src.vec | egrep "^[^_ ]+(_[^_ ]+){2} " | head -n $N_TRIGRAMS >> $OUTPUT_DIR/emb.src.tmp

    echo "`wc -l $OUTPUT_DIR/emb.src.tmp | cut -f 1 -d ' '` 300" > $OUTPUT_DIR/emb.src
    cat $OUTPUT_DIR/emb.src.tmp | sed "s/\(.\)_\+\(.\)/\1\&\#32;\2/g" >> $OUTPUT_DIR/emb.src

    rm $OUTPUT_DIR/emb.src.tmp
fi
############################################################################

############################################################################
echo "Building embeddings for $TGT..."
#############################################################################
if [ ! -f $OUTPUT_DIR/$TGT.phrase_counts ]; then
    echo "Looking for frequent n-grams..."
    $PYTHON $MONOSES_DIR/training/extract-ngrams.py --min-order 2 --max-order 3 --min-count 10 -i $TGT_TRAIN_TC | sort -nr --parallel $N_THREADS > $OUTPUT_DIR/$TGT.phrase_counts
fi

if [ ! -f $OUTPUT_DIR/$TGT.phrases ]; then
    echo "Filtering most frequent n-grams..."
    egrep "^[0-9]+[[:space:]][^ ]+( [^ ]+){1}$" $OUTPUT_DIR/$TGT.phrase_counts | cut -f 2 | head -n $N_BIGRAMS > $OUTPUT_DIR/$TGT.phrases
    egrep "^[0-9]+[[:space:]][^ ]+( [^ ]+){2}$" $OUTPUT_DIR/$TGT.phrase_counts | cut -f 2 | head -n $N_TRIGRAMS >> $OUTPUT_DIR/$TGT.phrases
fi

if [ ! -f $OUTPUT_DIR/$TGT.replaced ]; then
    echo "Merging n-grams in the corpus..."
        $PYTHON ngram_replacer.py -i $TGT_TRAIN_TC -o $OUTPUT_DIR/$TGT.replaced -sp '_' -n $OUTPUT_DIR/$TGT.phrases -keep-orig 1 --threads $N_THREADS
fi

if [ ! -f $OUTPUT_DIR/emb.trg.vec ]; then
    echo "Running fasttext..."
    $FASTTEXT_DIR/fasttext skipgram -input $OUTPUT_DIR/$TGT.replaced -output $OUTPUT_DIR/emb.trg -dim 300 -ws 5 -neg 5 -loss ns -thread $N_THREADS
fi

if [ ! -f $OUTPUT_DIR/emb.trg ]; then
    echo "Filtering vocabulary..."
    tail -n+2 $OUTPUT_DIR/emb.trg.vec | egrep "^[^_ ]+ " | head -n $N_UNIGRAMS > $OUTPUT_DIR/emb.trg.tmp
    tail -n+2 $OUTPUT_DIR/emb.trg.vec | egrep "^[^_ ]+(_[^_ ]+){1} " | head -n $N_BIGRAMS >> $OUTPUT_DIR/emb.trg.tmp
    tail -n+2 $OUTPUT_DIR/emb.trg.vec | egrep "^[^_ ]+(_[^_ ]+){2} " | head -n $N_TRIGRAMS >> $OUTPUT_DIR/emb.trg.tmp

    echo "`wc -l $OUTPUT_DIR/emb.trg.tmp | cut -f 1 -d ' '` 300" > $OUTPUT_DIR/emb.trg
    cat $OUTPUT_DIR/emb.trg.tmp | sed "s/\(.\)_\+\(.\)/\1\&\#32;\2/g" >> $OUTPUT_DIR/emb.trg

    rm $OUTPUT_DIR/emb.trg.tmp
fi
############################################################################
echo "Running VecMap..."
#############################################################################
if [[ ! -f $BWE_OUTPUT_DIR/emb.src || ! -f $BWE_OUTPUT_DIR/emb.trg ]]; then
    CUDA_VISIBLE_DEVICES=$GPUS $PYTHON $VECMAP_DIR/map_embeddings.py --identical --orthogonal `[ -z $GPUS ] && echo '' || echo '--cuda'` -v $OUTPUT_DIR/emb.src $OUTPUT_DIR/emb.trg $BWE_OUTPUT_DIR/emb.src $BWE_OUTPUT_DIR/emb.trg
fi
############################################################################
