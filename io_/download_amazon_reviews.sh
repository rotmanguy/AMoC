mkdir -p data/Amazon_Reviews/raw_data
cd data/Amazon_Reviews/raw_data
for domain in Amazon_Instant_Video Beauty Digital_Music Musical_Instruments Sports_and_Outdoors Video_Games
do
  wget "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_${domain}""_5.json.gz"
done

for domain in Amazon_Instant_Video Beauty Digital_Music Musical_Instruments Sports_and_Outdoors Video_Games
do
  gzip -d reviews_${domain}_5.json.gz
done

