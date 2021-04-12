Official code for the paper ["Model Compression for Domain Adaptation through Causal Effect Estimation"](https://arxiv.org/abs/2101.07086).
Please cite our paper in case you are using this code.

# Requirements
To install the requirement simply run the following command:
```
pip install -r requirements.txt
```
# Data
## Amazon Reviews
To download the Amazon Reviews dataset please run the following command (or run the commands appearing in the bash file manually): \
`bash io_/download_amazon_reviews.sh`

To preprocess the dataset please run the following command:
`python io_/preprocess_amazon_reviews.py`

## MultiNLI
To download MultiNLI dataset please run the following command (or run the commands appearing in the bash file manually): \
`bash io_/download_mnli.sh`

To preprocess the dataset please run the following command:
`python io_/preprocess_mnli_reviews.py`

# Running the large base model
We provide here examples where the large base model is BERT-base.

## Amazon Reviews
An example for training the base model on the Beauty domain from the Amazon Reviews dataset:
```
python sentence_classification.py --dataset_name Amazon_Reviews --src_domain Beauty --tgt_domains Amazon_Instant_Video Beauty Digital_Music Musical_Instruments Sports_and_Outdoors Video_Games --bert_model bert-base-cased --task_name sentiment_cnn --cnn_window_size 9 --cnn_out_channels 16 --train_batch_size 32  --combine_layers mix --layer_dropout 0.1 --save_best_weights True --num_train_epochs 10 --warmup_proportion 0.0 --learning_rate 1e-4 --weight_decay 0.01 --bert_model_type default --output_dir saved_models/Amazon_Reviews/Beauty/original_model --model_name pytorch_model.bin --do_train --do_eval
```

To train the model on a different domain simply modify the src_domain and the output_dir accordingly.

## MultiNLI
An example for training the base model on the Fiction domain from the MultiNLI dataset:
```
python sentence_classification.py --dataset_name MNLI --src_domain fiction --tgt_domains captions fiction government slate telephone travel --bert_model bert-base-cased --task_name sentiment_cnn --cnn_window_size 9 --cnn_out_channels 16 --train_batch_size 32  --combine_layers mix --layer_dropout 0.1 --save_best_weights True --num_train_epochs 10 --warmup_proportion 0.0 --learning_rate 1e-4 --weight_decay 0.01 --bert_model_type default --output_dir saved_models/MNLI/fiction/original_model --model_name pytorch_model.bin --do_train --do_eval
```

To train the model on a different domain simply modify the src_domain and the output_dir accordingly.

# Running a compressed AMoC model
Once training the base model, we can now compress it to a smaller model by removing a subset of its layers

## Amazon Reviews
An example for training a compressed model on the Beauty domain from the Amazon Reviews dataset by removing all the odd layers:
```
python sentence_classification.py --dataset_name Amazon_Reviews --src_domain Beauty --tgt_domains Amazon_Instant_Video Beauty Digital_Music Musical_Instruments Sports_and_Outdoors Video_Games --bert_model bert-base-cased --task_name sentiment_cnn --cnn_window_size 9 --cnn_out_channels 16 --train_batch_size 32  --combine_layers mix --layer_dropout 0.1 --save_best_weights True --layers_to_prune 1 3 5 7 9 11 --num_train_epochs 1 --warmup_proportion 0.0 --learning_rate 1e-4 --bert_model_type layer_pruning --output_dir saved_models/Amazon_Reviews/Beauty/counterfactual_models/freeze_layers_1+3+5+7+9+11 --load_model_path saved_models/Amazon_Reviews/Beauty/original_model/pytorch_model.bin --model_name pytorch_model.bin --do_train --do_eval
```

To remove a different set of layers simply modify the layers_to_prune and the output_dir accordingly.

## MultiNLI
An example for training a compressed model on the Fiction domain from the MultiNLI dataset by removing all the odd layers:
```
python sentence_classification.py --dataset_name MNLI --src_domain fiction --tgt_domains captions fiction government slate telephone travel --bert_model bert-base-cased --task_name sentiment_cnn --cnn_window_size 9 --cnn_out_channels 16 --train_batch_size 32  --combine_layers mix --layer_dropout 0.1 --save_best_weights True --layers_to_prune 1 3 5 7 9 11 --num_train_epochs 1 --warmup_proportion 0.0 --learning_rate 1e-4 --bert_model_type layer_pruning --output_dir saved_models/MNLI/fiction/counterfactual_models/freeze_layers_1+3+5+7+9+11 --load_model_path saved_models/MNLI/fiction/original_model/pytorch_model.bin --model_name pytorch_model.bin --do_train --do_eval
```

To remove a different set of layers simply modify the layers_to_prune and the output_dir accordingly.

# Citation
```
@article{rotman2021model,
  title={Model Compression for Domain Adaptation through Causal Effect Estimation},
  author={Rotman, Guy and Feder, Amir and Reichart, Roi},
  journal={arXiv preprint arXiv:2101.07086},
  year={2021}
}
```