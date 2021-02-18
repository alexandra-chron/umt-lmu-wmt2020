#!/usr/bin/env bash
# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
set -e

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
  --lang)
    LANG="$2"; shift 2;;
  --input)
    INP="$2"; shift 2;;
  --output)
    OUTP="$2"; shift 2;;
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
if [ "$INP" == "" ]; then echo "--input not provided"; exit; fi
if [ "$OUTP" == "" ]; then echo "--output not provided"; exit; fi

# Initialize tools and data paths

PROC_PATH=./data/$SRC-$TGT-wmt
TOOLS_PATH=$PWD/tools
DATA_PATH=./data/$SRC-wmt
FASTBPE=$TOOLS_PATH/fastBPE/fast
BPE_JOINT_CODES=$PROC_PATH/codes.$TGT-$SRC
BPE_CODES_HMR=$DATA_PATH/codes.$SRC


if [ "$LANG" == "de" ];
then $FASTBPE applybpe $OUTP $INP $BPE_CODES_HMR ; fi


if [ "$LANG" == "hsb" ];
then $FASTBPE applybpe $OUTP $INP $BPE_JOINT_CODES ; fi


