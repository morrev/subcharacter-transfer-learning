#!/bin/sh

# Download and prepare livedoor dataset using train/test split in Aoki et al (2020)
# Script adapted from https://github.com/IyatomiLab/GDCE-SSA/blob/master/script/make_datasets.sh
GDCEfolder="data/GDCE-SSA"
GDCEurl="https://github.com/IyatomiLab/GDCE-SSA.git"
if ! git clone "${GDCEurl}" "${GDCEfolder}" 2>/dev/null && [ -d "${GDCEfolder}" ] ; then
    echo "Clone ignored because the folder ${GDCEfolder} exists"
fi

# Curl livedoor data if it does not already exist locally
cd data/GDCE-SSA/data/livedoor

if [ -e "ldcc-20140209.tar.gz" ]; then
    echo 'Livedoor data file already exists' >&2
else
    curl -OL https://www.rondhuit.com/download/ldcc-20140209.tar.gz
fi
tar -zxf ldcc-20140209.tar.gz
python make_data.py

cd ..
python split_train_test.py
cd ../../../

# Download JWE radical and subcomponent mapping files
JWEfolder="data/JWE"
JWEurl="https://github.com/HKUST-KnowComp/JWE.git"
if ! git clone "${JWEurl}" "${JWEfolder}" 2>/dev/null && [ -d "${JWEfolder}" ] ; then
    echo "Clone ignored because the folder ${JWEfolder} exists"
fi

# Download word similarity intrinsic evaluation dataset
mkdir -p data/JWSAN
cd data/JWSAN

if [ -e "jwsan-1400.csv" ]; then
    echo 'jwsan-1400 data file already exists' >&2
else
    curl -OL http://www.utm.inf.uec.ac.jp/JWSAN/en/jwsan-1400.csv
fi

cd ../../

# Download wikipedia data if it does not already exist locally
mkdir -p data/Wikipedia_title_dataset
cd data/Wikipedia_title_dataset

if [ -e "ja_raw.txt" ]; then
    echo 'ja_raw.txt (Wikipedia) data file already exists' >&2
else
    curl -OL https://raw.githubusercontent.com/frederick0329/Wikipedia-Title-Dataset/master/acl2017_data/ja_raw.txt
fi

cd ../../

# Need to run 'git lfs install' first
CBERTfolder="data/ChineseBERT-large"
CBERTurl="https://huggingface.co/ShannonAI/ChineseBERT-large"
if ! git clone "${CBERTurl}" "${CBERTfolder}" 2>/dev/null && [ -d "${CBERTfolder}" ] ; then
    echo "Clone ignored because the folder ${CBERTfolder} exists"
fi

CBERTfolder2="data/ChineseBert"
CBERTurl="https://github.com/ShannonAI/ChineseBert.git"
if ! git clone "${CBERTurl}" "${CBERTfolder2}" 2>/dev/null && [ -d "${CBERTfolder2}" ] ; then
    echo "Clone ignored because the folder ${CBERTfolder2} exists"
fi

cp data/modeling_glycebert.py "${CBERTfolder2}"/models/
