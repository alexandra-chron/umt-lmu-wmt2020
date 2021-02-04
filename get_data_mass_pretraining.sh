#!/usr/bin/env bash
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
set -e

# Data preprocessing configuration

CODES=32000     # number of BPE codes
N_THREADS=1    # number of threads in data preprocessing

# Read arguments

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"
case $key in
  --src)
    SRC="$2"; shift 2;;
  *)
  POSITIONAL+=("$1")
  shift
  ;;
esac
done
set -- "${POSITIONAL[@]}"

# Check parameters
if [ "$SRC" == "" ]; then echo "--src not provided"; exit; fi

# Initialize tools and data paths

# main paths
MAIN_PATH=$PWD
TOOLS_PATH=$PWD/tools
DATA_PATH=./data/$SRC-wmt

# create paths
mkdir -p $TOOLS_PATH

# moses
MOSES=$TOOLS_PATH/mosesdecoder
REPLACE_UNICODE_PUNCT=$MOSES/scripts/tokenizer/replace-unicode-punctuation.perl
NORM_PUNC=$MOSES/scripts/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$MOSES/scripts/tokenizer/remove-non-printing-char.perl
TOKENIZER=$MOSES/scripts/tokenizer/tokenizer.perl
INPUT_FROM_SGM=$MOSES/scripts/ems/support/input-from-sgm.perl
TRUE_CASE_TRAIN=$MOSES/scripts/recaser/train-truecaser.perl
TRUE_CASE_APPLY=$MOSES/scripts/recaser/truecase.perl

# fastBPE
FASTBPE_DIR=$TOOLS_PATH/fastBPE
FASTBPE=$TOOLS_PATH/fastBPE/fast

# Sennrich's WMT16 scripts for Romanian pre-processing
WMT16_SCRIPTS=$TOOLS_PATH/wmt16-scripts
NORMALIZE_ROMANIAN=$WMT16_SCRIPTS/preprocess/normalise-romanian.py
REMOVE_DIACRITICS=$WMT16_SCRIPTS/preprocess/remove-diacritics.py

# train/ valid / test file raw data
SRC_TRAIN=$DATA_PATH/train_raw.$SRC
SRC_VALID=$DATA_PATH/valid_raw.$SRC
SRC_TEST=$DATA_PATH/test_raw.$SRC

# tokenized files
SRC_TRAIN_TOK=$SRC_TRAIN.tok
SRC_VALID_TOK=$SRC_VALID.tok
SRC_TEST_TOK=$SRC_TEST.tok

# truecased files
SRC_TRAIN_TC=$SRC_TRAIN_TOK.tc
SRC_VALID_TC=$SRC_VALID_TOK.tc
SRC_TEST_TC=$SRC_TEST_TOK.tc
MODEL_TC=$DATA_PATH/$SRC.tc.model

# train / valid / test monolingual BPE data
SRC_TRAIN_BPE=$DATA_PATH/train.$SRC
SRC_VALID_BPE=$DATA_PATH/valid.$SRC
SRC_TEST_BPE=$DATA_PATH/test.$SRC

# BPE / vocab files
BPE_CODES=$DATA_PATH/codes.$SRC
SRC_VOCAB=$DATA_PATH/vocab.$SRC


# install tools
./install-tools.sh

#
# Download monolingual data
#

# preprocessing commands - special case for Romanian
if [ "$SRC" == "ro" ]; then
  SRC_PREPROCESSING="$REPLACE_UNICODE_PUNCT | $NORM_PUNC -l $SRC | $REM_NON_PRINT_CHAR | $NORMALIZE_ROMANIAN | $REMOVE_DIACRITICS | $TOKENIZER -l $SRC -no-escape -threads $N_THREADS"
elif [ "$SRC" == "hsb" ]; then
  SRC_PREPROCESSING="$REPLACE_UNICODE_PUNCT | $NORM_PUNC -l cs | $REM_NON_PRINT_CHAR |                                            $TOKENIZER -l cs -no-escape -threads $N_THREADS"
else
  SRC_PREPROCESSING="$REPLACE_UNICODE_PUNCT | $NORM_PUNC -l $SRC | $REM_NON_PRINT_CHAR |                                            $TOKENIZER -l $SRC -no-escape -threads $N_THREADS"
fi

# tokenize data
if ! [[ -f "$SRC_TRAIN_TOK" ]]; then
  echo "Tokenize $SRC monolingual data..."
  eval "cat $SRC_TRAIN | $SRC_PREPROCESSING > $SRC_TRAIN_TOK"
fi

echo "$SRC monolingual data tokenized in: $SRC_TRAIN_TOK"

# truecasing
if ! [[ -f "$MODEL_TC" ]]; then
    echo "Training truecase model for $SRC..."
    eval "$TRUE_CASE_TRAIN --model $MODEL_TC --corpus $SRC_TRAIN_TOK"
fi

if ! [[ -f "$SRC_TRAIN_TC" ]]; then
    echo "Truecase $SRC monolingual data..."
    eval "$TRUE_CASE_APPLY --model $MODEL_TC < $SRC_TRAIN_TOK > $SRC_TRAIN_TC"
fi

# learn BPE codes
if [ ! -f "$BPE_CODES" ]; then
  echo "Learning BPE codes..."
  $FASTBPE learnbpe $CODES $SRC_TRAIN_TC > $BPE_CODES
fi
echo "BPE learned in $BPE_CODES"

# apply BPE codes
if ! [[ -f "$SRC_TRAIN_BPE" ]]; then
  echo "Applying $SRC BPE codes..."
  $FASTBPE applybpe $SRC_TRAIN_BPE $SRC_TRAIN_TC $BPE_CODES
fi
echo "BPE codes applied to $SRC in: $SRC_TRAIN_BPE"

# extract source vocabulary
if ! [[ -f "$SRC_VOCAB" ]]; then
  echo "Extracting vocabulary..."
  $FASTBPE getvocab $SRC_TRAIN_BPE > $SRC_VOCAB
fi
echo "$SRC vocab in: $SRC_VOCAB"

# binarize data
if ! [[ -f "$SRC_TRAIN_BPE.pth" ]]; then
  echo "Binarizing $SRC data..."
  $MAIN_PATH/preprocess.py $SRC_VOCAB $SRC_TRAIN_BPE
fi

echo "$SRC binarized data in: $SRC_TRAIN_BPE.pth"


echo "Tokenizing valid and test data..."
# tokenize data
if ! [[ -f "$SRC_VALID_TOK" ]]; then
  eval "cat $SRC_VALID | $SRC_PREPROCESSING > $SRC_VALID_TOK"
fi

if ! [[ -f "$SRC_TEST_TOK" ]]; then
  eval "cat $SRC_TEST | $SRC_PREPROCESSING > $SRC_TEST_TOK"
fi

echo "Truecasing valid and test data..."
# truecase data
if ! [[ -f "$SRC_VALID_TC" ]]; then
    echo "Truecase $SRC monolingual data..."
    eval "$TRUE_CASE_APPLY --model $MODEL_TC < $SRC_VALID_TOK > $SRC_VALID_TC"
fi

if ! [[ -f "$SRC_TEST_TC" ]]; then
    echo "Truecase $SRC monolingual data..."
    eval "$TRUE_CASE_APPLY --model $MODEL_TC < $SRC_TEST_TOK > $SRC_TEST_TC"
fi

echo "Applying BPE to valid and test files..."
$FASTBPE applybpe $SRC_VALID_BPE "$SRC_VALID_TC" $BPE_CODES $SRC_VOCAB
$FASTBPE applybpe $SRC_TEST_BPE  "$SRC_TEST_TC"  $BPE_CODES $SRC_VOCAB

echo "Binarizing data..."
rm -f $SRC_VALID_BPE.pth $SRC_TEST_BPE.pth
$MAIN_PATH/preprocess.py $SRC_VOCAB $SRC_VALID_BPE
$MAIN_PATH/preprocess.py $SRC_VOCAB $SRC_TEST_BPE


# Summary

echo ""
echo "===== Data summary"
echo "Monolingual training data:"
echo "    $SRC: $SRC_TRAIN_BPE.pth"
echo "Monolingual validation data:"
echo "    $SRC: $SRC_VALID_BPE.pth"
echo "Monolingual test data:"
echo "    $SRC: $SRC_TEST_BPE.pth"
