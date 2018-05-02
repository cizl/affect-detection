#!/bin/bash
# Setup script - installs requirements, setups up virtual 
# environment, downloads and organizes all the data.
# **************************************************************** 


# Create virtual environment and install requirements.
# **************************************************************** 
virtualenv -p python3 v
source v/bin/activate
pip install -r requirements.txt


# Get GloVe embeddings.
# **************************************************************** 
mkdir glove && cd glove
wget 'http://nlp.stanford.edu/data/glove.twitter.27B.zip'
unzip '*.zip' && rm *.zip
cd ..


# Get datasets.
# **************************************************************** 
mkdir data && cd data

# EI-reg (Emotion Intensity - regression)
# English
wget 'rttp://www.saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/EI-reg/English/EI-reg-En-train.zip'
wget 'http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/EI-reg/English/2018-EI-reg-En-dev.zip'
wget 'http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/AIT2018-TEST-DATA/semeval2018englishtestfiles/2018-EI-reg-En-test.zip'
# Arabic
wget 'http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/EI-reg/Arabic/2018-EI-reg-Ar-train.zip'
wget 'http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/EI-reg/Arabic/2018-EI-reg-Ar-dev.zip'
wget 'http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/AIT2018-TEST-DATA/semeval2018arabictestfiles/2018-EI-reg-Ar-test.zip'
# Spanish
wget 'http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/EI-reg/Spanish/2018-EI-reg-Es-train.zip'
wget 'http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/EI-reg/Spanish/2018-EI-reg-Es-dev.zip'
wget 'http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/AIT2018-TEST-DATA/semeval2018spanishtestfiles/2018-EI-reg-Es-test.zip'

# EI-oc (Emotion Intensity - ordinal clssification)
# English
wget 'http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/EI-oc/English/EI-oc-En-train.zip'
wget 'http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/EI-oc/English/2018-EI-oc-En-dev.zip'
wget 'http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/AIT2018-TEST-DATA/semeval2018englishtestfiles/2018-EI-oc-En-test.zip'
# Arabic
wget 'http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/EI-oc/Arabic/2018-EI-oc-Ar-train.zip'
wget 'http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/EI-oc/Arabic/2018-EI-oc-Ar-dev.zip'
wget 'http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/AIT2018-TEST-DATA/semeval2018arabictestfiles/2018-EI-oc-Ar-test.zip'
# Spanish
wget 'http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/EI-oc/Spanish/2018-EI-oc-Es-train.zip'
wget 'http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/EI-oc/Spanish/2018-EI-oc-Es-dev.zip'
wget 'http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/AIT2018-TEST-DATA/semeval2018spanishtestfiles/2018-EI-oc-Es-test.zip'

# V-reg (Valence - regression)
# English
wget 'http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/V-reg/English/2018-Valence-reg-En-train.zip'
wget 'http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/V-reg/English/2018-Valence-reg-En-dev.zip'
wget 'http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/AIT2018-TEST-DATA/semeval2018englishtestfiles/2018-Valence-reg-En-test.zip'
# Arabic
wget 'http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/V-reg/Arabic/2018-Valence-reg-Ar-train.zip'
wget 'http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/V-reg/Arabic/2018-Valence-reg-Ar-dev.zip'
wget 'http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/AIT2018-TEST-DATA/semeval2018arabictestfiles/2018-Valence-reg-Ar-test.zip'
# Spanish
wget 'http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/V-reg/Spanish/2018-Valence-reg-Es-train.zip'
wget 'http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/V-reg/Spanish/2018-Valence-reg-Es-dev.zip'
wget 'http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/AIT2018-TEST-DATA/semeval2018spanishtestfiles/2018-Valence-reg-Es-test.zip'

# V-oc (Valence - ordinal classification)
# English
wget 'http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/V-oc/English/2018-Valence-oc-En-train.zip'
wget 'http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/V-oc/English/2018-Valence-oc-En-dev.zip'
wget 'http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/AIT2018-TEST-DATA/semeval2018englishtestfiles/2018-Valence-oc-En-test.zip'
# Arabic
wget 'http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/V-oc/Arabic/2018-Valence-oc-Ar-train.zip'
wget 'http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/V-oc/Arabic/2018-Valence-oc-Ar-dev.zip'
wget 'http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/AIT2018-TEST-DATA/semeval2018arabictestfiles/2018-Valence-oc-Ar-test.zip'
# Spanish
wget 'http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/V-oc/Spanish/2018-Valence-oc-Es-train.zip'
wget 'http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/V-oc/Spanish/2018-Valence-oc-Es-dev.zip'
wget 'http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/AIT2018-TEST-DATA/semeval2018spanishtestfiles/2018-Valence-oc-Es-test.zip'

# E-c (Emotion - classification)
# English
wget 'http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/E-c/English/2018-E-c-En-train.zip'
wget 'http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/E-c/English/2018-E-c-En-dev.zip'
wget 'http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/AIT2018-TEST-DATA/semeval2018englishtestfiles/2018-E-c-En-test.zip'
# Arabic
wget 'http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/E-c/Arabic/2018-E-c-Ar-train.zip'
wget 'http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/E-c/Arabic/2018-E-c-Ar-dev.zip'
wget 'http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/AIT2018-TEST-DATA/semeval2018arabictestfiles/2018-E-c-Ar-test.zip'
# Spanish
wget 'http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/E-c/Spanish/2018-E-c-Es-train.zip'
wget 'http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/E-c/Spanish/2018-E-c-Es-dev.zip'
wget 'http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/AIT2018-TEST-DATA/semeval2018spanishtestfiles/2018-E-c-Es-test.zip'

unzip '*.zip' && rm *.zip
cd ..
