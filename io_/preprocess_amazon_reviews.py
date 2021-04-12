import os
import json
import pandas as pd
from datasets_utils import output_datasets, split_data

AMAZON_RAW_DATA_DIR = 'data/Amazon_Reviews/raw_data/'
AMAZON_DATA_DIR = 'data/Amazon_Reviews'
test_size = 0.2

def main():
    files = [f for f in os.listdir(AMAZON_RAW_DATA_DIR)
                 if os.path.isfile(os.path.join(AMAZON_RAW_DATA_DIR, f))]
    for file in files:
        domain = file.split('reviews_')[1].split('.json')[0].split('_5')[0]
        domain_path = os.path.join(AMAZON_DATA_DIR, domain)
        if not os.path.exists(domain_path):
            os.mkdir(domain_path)
        reviews = []
        with open(os.path.join(AMAZON_RAW_DATA_DIR, file), 'r') as f:
            for review in f:
                reviews.append(json.loads(review))

        df = pd.DataFrame.from_dict(pd.json_normalize(reviews), orient='columns')
        df = df[df['overall'] != 3.0]
        df['sentiment'] = (df['overall'] > 3).astype(int)
        df = df[['reviewText', 'sentiment']]
        df = df.groupby('sentiment')
        df = pd.DataFrame(df.apply(lambda x: x.sample(df.size().min()).reset_index(drop=True)))
        print(file, len(df))
        split_data(df, domain_path, test_size=test_size)
        print('DONE with: ' + domain)

if __name__ == "__main__":
    main()