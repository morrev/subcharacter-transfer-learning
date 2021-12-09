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
