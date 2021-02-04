# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# NOTICE FILE in the root directory of this source tree.
#

set -e

lg=$1  # input language

# data path
MAIN_PATH=$PWD
TOOLS_PATH=$PWD/tools

# tools
MOSES_DIR=$TOOLS_PATH/mosesdecoder
FASTBPE_DIR=$TOOLS_PATH/fastBPE
FASTBPE=$FASTBPE_DIR/fast
WMT16_SCRIPTS=$TOOLS_PATH/wmt16-scripts
FASTTEXT_DIR=$TOOLS_PATH/fastText
MONOSES_DIR=$TOOLS_PATH/monoses
MONOSES_THIRD_PARTY_DIR=$MONOSES_DIR/third-party
MONOSES_MOSES_DIR=$MONOSES_THIRD_PARTY_DIR/mosesdecoder
MONOSES_FAST_ALIGN_DIR=$MONOSES_THIRD_PARTY_DIR/fast_align
MONOSES_VECMAP_DIR=$MONOSES_THIRD_PARTY_DIR/vecmap

# tools path
mkdir -p $TOOLS_PATH

#
# Download and install tools
#

cd $TOOLS_PATH

# Download Moses
if [ ! -d "$MOSES_DIR" ]; then
  echo "Cloning Moses from GitHub repository..."
  git clone https://github.com/moses-smt/mosesdecoder.git
fi

# Download fastBPE
if [ ! -d "$FASTBPE_DIR" ]; then
  echo "Cloning fastBPE from GitHub repository..."
  git clone https://github.com/glample/fastBPE
fi

# Compile fastBPE
if [ ! -f "$FASTBPE" ]; then
  echo "Compiling fastBPE..."
  cd fastBPE
  g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast
  cd ..
fi

# Download Sennrich's tools
if [ ! -d "$WMT16_SCRIPTS" ]; then
  echo "Cloning WMT16 preprocessing scripts..."
  git clone https://github.com/rsennrich/wmt16-scripts.git
fi

# Download WikiExtractor
if [ ! -d $TOOLS_PATH/wikiextractor ]; then
    echo "Cloning WikiExtractor from GitHub repository..."
    git clone https://github.com/attardi/wikiextractor.git
fi

cd ..

# Download fastText
if [ ! -d $FASTTEXT_DIR ]; then
    echo "Cloning fastText from GitHub repository..."
    git clone https://github.com/facebookresearch/fastText.git $FASTTEXT_DIR
fi
if [ ! -f $FASTTEXT_DIR/fasttext ]; then
    echo "Compiling fastText"
    pushd $FASTTEXT_DIR
    make
    popd
fi

# Download monoses
if [ ! -d $MONOSES_DIR ]; then
    echo "Cloning Monoses from GitHub repository..."
    git clone https://github.com/artetxem/monoses.git $MONOSES_DIR
    mkdir -p $MONOSES_THIRD_PARTY_DIR
fi

# Download monoses requirements
if [ ! -d $MONOSES_MOSES_DIR ]; then
    echo "Linking moses to monoses"
    ln -s $MOSES_DIR $MONOSES_MOSES_DIR
fi

if [ ! -d $MONOSES_FAST_ALIGN_DIR ]; then
    echo "Cloning fast_align from GitHub repository..."
    git clone 'https://github.com/clab/fast_align.git' $MONOSES_FAST_ALIGN_DIR
fi
if [ ! -f $MONOSES_FAST_ALIGN_DIR/build/fast_align ]; then
    echo "Compiling fast_align"
    pushd $MONOSES_FAST_ALIGN_DIR
    mkdir build
    cd build
    cmake ..
    make
    popd
fi

if [ ! -d $MONOSES_VECMAP_DIR ]; then
    echo "Cloning vecmap from GitHub repository..."
    git clone 'https://github.com/artetxem/vecmap.git' $MONOSES_VECMAP_DIR
fi

# # Chinese segmenter
# if ! ls $TOOLS_PATH/stanford-segmenter-* 1> /dev/null 2>&1; then
#   echo "Stanford segmenter not found at $TOOLS_PATH/stanford-segmenter-*"
#   echo "Please install Stanford segmenter in $TOOLS_PATH"
#   exit 1
# fi
# 
# # Thai tokenizer
# if ! python -c 'import pkgutil; exit(not pkgutil.find_loader("pythainlp"))'; then
#   echo "pythainlp package not found in python"
#   echo "Please install pythainlp (pip install pythainlp)"
#   exit 1
# fi
# 
