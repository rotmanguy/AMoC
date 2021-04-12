mkdir -p data/MNLI/raw_data
cd data/MNLI/raw_data
wget https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip
unzip -q multinli_1.0.zip
rm multinli_1.0.zip
wget https://cims.nyu.edu/~sbowman/multinli/snli_1.0.zip
unzip -q snli_1.0.zip
rm snli_1.0.zip
