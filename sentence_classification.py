from __future__ import absolute_import, division, print_function

import argparse
import json
from log import Logger
import logging
import math
import numpy as np
import os
import pandas as pd
import random
from sklearn.metrics import f1_score
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, Parameter
from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from transformers import BertTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from models import CNNBertForSequenceClassification

logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, id=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string or float. The label of the example.
            id: (Optional) string. Unique example id.

        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.id = id

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, sample_id, multi_label=[]):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.sample_id = sample_id
        # for auxilary task
        if len(multi_label) > 0:
            self.multi_label = multi_label


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_examples(self, data_dir):
        """Gets a collection of `InputExample`s."""
        raise NotImplementedError()

class SentimentProcessor(DataProcessor):
    """Processor for the Domain Adaptation Sentiment data set."""
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.labels = self.get_labels()

    def get_examples(self, dataset):
        data_path = os.path.join(self.data_dir, dataset + '.csv')
        logger.info("Loading {} data from {}".format(dataset, data_path))
        df = pd.read_csv(data_path)
        text = df['reviewText'].tolist()
        labels = df['sentiment'].tolist()
        ids = df['index'].tolist()
        return self._create_examples(text, labels, ids, dataset)

    def get_labels(self):
        """See base class."""
        data_path = os.path.join(self.data_dir, 'train.csv')
        df = pd.read_csv(data_path)
        labels = sorted(df['sentiment'].unique())
        return labels

    def _create_examples(self, x, label, id, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, data_point) in enumerate(zip(x, label, id)):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = data_point[0]
            text_b = None
            label = data_point[1]
            id = data_point[2]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, id=id))
        return examples

class MNLIProcessor(DataProcessor):
    """Processor for the Domain Adaptation Sentiment data set."""
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.labels = self.get_labels()

    def get_examples(self, dataset):
        data_path = os.path.join(self.data_dir, dataset + '.csv')
        logger.info("Loading {} data from {}".format(dataset, data_path))
        df = pd.read_csv(data_path)
        text_a = df['sentence1'].tolist()
        text_b = df['sentence2'].tolist()
        labels = df['gold_label'].tolist()
        ids = df['index'].tolist()
        return self._create_examples(text_a, text_b, labels, ids, dataset)

    def get_labels(self):
        """See base class."""
        data_path = os.path.join(self.data_dir, 'train.csv')
        df = pd.read_csv(data_path)
        labels = sorted(df['gold_label'].unique())
        return labels

    def _create_examples(self, x_a, x_b, label, id, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, data_point) in enumerate(zip(x_a, x_b, label, id)):
            guid = "%s-%s" % (set_type, i)
            text_a = data_point[0]
            text_b = data_point[1]
            label = data_point[2]
            id = data_point[3]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, id=id))
        return examples

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        if type(example.text_a) != str:
            continue
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            if type(example.text_b) != str:
                continue
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # If unlabeled example - mask pivots
        output_multi_label = []

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if isinstance(example.label, list):
            label_id = example.label
        else:
            label_id = label_map[example.label]
        id = int(example.id)

        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("sample id: %s" % (example.id))
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %s)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          sample_id=id,
                          multi_label=output_multi_label))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.p = 0.5
            m.train()

def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(args, preds, labels):
    acc = simple_accuracy(preds, labels)
    if args.dataset_name == 'Mantis':
        f1 = f1_score(y_true=labels, y_pred=preds, average="samples")
    else:
        f1 = f1_score(y_true=labels, y_pred=preds, average="macro")
    return {
        "acc": acc,
        "f1": f1
    }


def evaluate(args, eval_dataloader, model):
    model.eval()
    id_to_pred = {}
    results = {}
    eval_loss = 0
    nb_eval_steps = 0
    preds = []
    sample_ids = []

    for eval_element in tqdm(eval_dataloader):
        input_ids, input_mask, segment_ids, label_ids, ids = eval_element[:5]
        input_ids = input_ids.to(args.device)
        input_mask = input_mask.to(args.device)
        segment_ids = segment_ids.to(args.device)
        label_ids = label_ids.to(args.device)
        # create eval loss and other metric required by the task

        with torch.no_grad():
            outputs = model(input_ids, segment_ids, input_mask)
            logits = outputs

            if args.dataset_name == 'Mantis':
                 loss_fct = BCEWithLogitsLoss()
                 tmp_eval_loss = loss_fct(logits, label_ids.type(logits.dtype))

            else:
                loss_fct = CrossEntropyLoss()
                tmp_eval_loss = loss_fct(logits.view(-1, args.num_labels), label_ids.view(-1))

        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
            all_label_ids = label_ids.detach().cpu().numpy()
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)
            all_label_ids = np.append(all_label_ids, label_ids.detach().cpu().numpy(), axis=0)
        sample_ids.append(ids)

    eval_loss = eval_loss / nb_eval_steps
    sample_ids = torch.cat(sample_ids).numpy()
    preds = preds[0]

    if args.dataset_name == 'Mantis':
        prob_preds = torch.sigmoid(torch.FloatTensor(preds)).detach().cpu().numpy()
        preds = (prob_preds > 0.5).astype(np.int64)
    else:
        prob_preds = torch.softmax(torch.FloatTensor(preds), axis=1).detach().cpu().numpy()
        preds = np.argmax(preds, axis=1)

    for sample_id, prob_pred in zip(sample_ids, prob_preds):
        id_to_pred[sample_id] = prob_pred

    res = acc_and_f1(args, preds, all_label_ids)

    if len(results.keys()) == 0:
        for k, v in res.items():
            results[k] = [v]
    else:
        for k, v in res.items():
            results[k].append(v)

    model.train()
    return res, eval_loss, id_to_pred


def make_DataLoader(processor, tokenizer, max_seq_length, batch_size=6,
                    local_rank=-1, split_set="train", N=-1):
    examples = processor.get_examples(split_set)
    if N > 0:
        examples = examples[:N]
        examples = examples[:N]
    features = convert_examples_to_features(
        examples, processor.labels, max_seq_length, tokenizer)
    logger.info("Num examples = %d", len(examples))
    logger.info("Batch size = %d", batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_sample_ids = torch.tensor([f.sample_id for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_sample_ids)

    if split_set == "train":
        if local_rank == -1:
            sampler = RandomSampler(data)
        else:
            sampler = DistributedSampler(data)
    else:
        # Run prediction for full data
        sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size, num_workers=8)

    return dataloader

def prune_layers(args, model):
    if 'top_models' in args.output_dir or 'median_models' in args.output_dir:
        params_to_freeze = ['layer.' + str(i - 1) + '.' for i in args.layers_to_prune]
        for n, p in model.named_parameters():
            p.requires_grad = not any(nd in n for nd in params_to_freeze)
    else:
        params_to_unfreeze = ['layer.' + str(i - 2) + '.' if i >= 2 else '.embeddings.' for i in args.layers_to_prune
                             if i - 1 not in args.layers_to_prune]
        for n, p in model.named_parameters():
            p.requires_grad = any(nd in n for nd in params_to_unfreeze) or 'bert' not in n

    for i in args.layers_to_prune:
        model._scalar_mix.scalar_parameters[i] = Parameter(torch.FloatTensor([-1e20]), requires_grad=False)
    return model

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--dataset_name",
                        default='Amazon_Reviews',
                        type=str,
                        required=True,
                        help="The name of the dataset.")
    parser.add_argument("--src_domain",
                        type=str,
                        required=True,
                        help="The source data name")
    parser.add_argument("--tgt_domains",
                        type=str,
                        nargs='+',
                        required=True,
                        help="The target data names.")
    parser.add_argument("--bert_model", default='bert-base-cased', type=str, required=False,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default='sentiment_cnn',
                        type=str,
                        required=False,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    ## Other parameters
    parser.add_argument("--log_dir",
                        default='log/',
                        type=str,
                        help="The log output dir.")
    parser.add_argument("--model_name",
                        default=None,
                        type=str,
                        help="The name of the model to load, relevant only in case that load_model is positive.")
    parser.add_argument("--load_model_path",
                        default='',
                        type=str,
                        help="Path to directory containing fine-tuned model.")
    parser.add_argument("--save_on_epoch_end",
                        action='store_true',
                        help="Whether to save the weights each time an epoch ends.")
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--N_train",
                        type=int,
                        default=-1,
                        help="number of training examples")
    parser.add_argument("--N_dev",
                        type=int,
                        default=-1,
                        help="number of development examples")
    parser.add_argument("--cnn_window_size",
                        type=int,
                        default=9,
                        help="CNN 1D-Conv window size")
    parser.add_argument("--cnn_out_channels",
                        type=int,
                        default=16,
                        help="CNN 1D-Conv out channels")
    parser.add_argument("--combine_layers",
                        type=str,
                        default='last',
                        help="Choose how to combine encoded layers from bert (last/mix)")
    parser.add_argument("--layer_dropout",
                        type=float,
                        default=0.0,
                        help="dropout probability per bert layer")
    parser.add_argument("--layers_to_prune",
                        type=int,
                        nargs='+',
                        default=[],
                        help="choose which bert layers to freeze")
    parser.add_argument("--bert_model_type",
                        type=str,
                        default='default',
                        choices=['default', 'layer_pruning'],
                        help="Bert model type")
    parser.add_argument("--save_best_weights",
                        type=bool,
                        default=False,
                        help="saves model weight each time epoch accuracy is maximum")
    parser.add_argument("--write_log_for_each_epoch",
                        type=bool,
                        default = False,
                        help = "whether to write log file at the end of every epoch or not")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--unfreeze_bert",
                        type=bool,
                        default=True,
                        help="Unfreeze bert weights if true.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=2.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs",
                        default=5,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--optimizer',
                        type=str,
                        default='adam',
                        choices=['adam', 'sgd'],
                        help="which optimizer model to use: adam or sgd")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()
    args.src_domain_data_dir = os.path.join('data', args.dataset_name, args.src_domain)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    args_path = os.path.join(args.output_dir, 'args.json')
    with open(args_path, 'w') as f:
        json.dump(vars(args), f)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count() if not args.no_cuda else 0
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    args.device = device
    args.n_gpu = n_gpu

    logging.basicConfig(filename=os.path.join(args.output_dir, args.src_domain + '_log.txt'),
                        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    fh = logging.FileHandler(os.path.join(args.output_dir, args.src_domain + '_log.txt'))
    logger.setLevel(logging.getLevelName('DEBUG'))
    logger.addHandler(fh)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    logger.info("learning rate: {}, batch size: {}".format(
        args.learning_rate, args.train_batch_size))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if args.dataset_name == 'Amazon_Reviews':
        _processor = SentimentProcessor
    elif args.dataset_name == 'MNLI':
        _processor = MNLIProcessor
    else:
        raise ValueError('Dataset %s is not supported' % args.dataset_name)

    processor = _processor(args.src_domain_data_dir)

    label_list = processor.labels
    num_labels = len(label_list)
    args.num_labels = num_labels

    # Load a trained model and vocabulary that you have fine-tuned
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                                   'distributed_{}'.format(args.local_rank))
    model = CNNBertForSequenceClassification.from_pretrained(args.bert_model,
                                                         cache_dir=cache_dir,
                                                         num_labels=args.num_labels,
                                                         hidden_size=768,
                                                         max_seq_length=args.max_seq_length,
                                                         filter_size=args.cnn_window_size,
                                                         out_channels=args.cnn_out_channels,
                                                         combine_layers=args.combine_layers,
                                                         layer_dropout=args.layer_dropout,
                                                         layers_to_prune=args.layers_to_prune,
                                                         output_hidden_states=True,
                                                         bert_model_type=args.bert_model_type)

    if args.load_model_path != '':
        print("--- Loading pytorch_model:", args.load_model_path)
        model.load_state_dict(torch.load(args.load_model_path, map_location=args.device), strict=True)

        for param in model.bert.parameters():
            param.requires_grad = args.unfreeze_bert

        if args.bert_model_type == 'layer_pruning':
            model = prune_layers(args, model)

    else:
        for param in model.bert.parameters():
            param.requires_grad = args.unfreeze_bert

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    model.to(args.device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    # elif args.n_gpu > 1:
    #     model = torch.nn.DataParallel(model)

    dataloaders = {}
    # prepare dev-set evaluation DataLoader
    for dataset in ['train', 'dev', 'test']:
        if dataset == 'train':
            batch_size_ = args.train_batch_size
            N_ = args.N_train
        else:
            batch_size_ = args.eval_batch_size
            N_ = args.N_dev

        processor = _processor(args.src_domain_data_dir)
        dataloaders[dataset] = make_DataLoader(processor=processor,
                                               tokenizer=tokenizer,
                                               max_seq_length=args.max_seq_length,
                                               batch_size=batch_size_,
                                               local_rank=args.local_rank,
                                               split_set=dataset,
                                               N=N_)
    num_train_optimization_steps = int(
        len(dataloaders['train'].dataset) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    param_optimizer = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    if args.optimizer == 'adam':
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=args.learning_rate,
                          eps=args.adam_epsilon)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=args.learning_rate, weight_decay=1e-2)
    num_warmup_steps = math.ceil(args.warmup_proportion * num_train_optimization_steps)
    warmup_linear = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    if args.do_train:
        global_step = 0
        best_epoch = 0
        # creat results logger
        log_dir_path = os.path.join(args.log_dir, os.path.basename(args.output_dir))
        print("\nsaving logs to {}\n".format(log_dir_path))
        os.makedirs(log_dir_path, exist_ok=1)
        results_logger = Logger(log_dir_path)
        os.chmod(log_dir_path, 0o775)
        os.chmod(args.log_dir, 0o775)

        if args.dataset_name == 'Mantis':
            loss_fct = BCEWithLogitsLoss()
        else:
            loss_fct = CrossEntropyLoss(ignore_index=-1)

        model.train()

        # main training loop
        best_dev_f1 = -1.0
        src_domain_best = {}
        src_domain_best['dev_acc'] = 0.0
        src_domain_best['test_acc'] = 0.0
        src_domain_best['dev_f1'] = 0.0
        src_domain_best['test_f1'] = 0.0

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            logger.info("***** Running Epoch %d *****" % epoch)
            tr_loss = 0
            tr_res = {"f1": 0, "acc": 0}

            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(dataloaders['train'], desc="Iteration")):
                batch = tuple(t.to(args.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, ids = batch[:5]

                # define a new function to compute loss values
                logits = model(input_ids, segment_ids, input_mask)
                if args.dataset_name == 'Mantis':
                    loss = loss_fct(logits, label_ids.type(logits.dtype))
                    preds = (torch.sigmoid(logits) > 0.5).detach().cpu().numpy().astype(np.int64)
                else:
                    loss = loss_fct(logits.view(-1, args.num_labels), label_ids.view(-1))
                    preds = logits.detach().cpu().numpy()
                    preds = np.argmax(preds, axis=1)

                eval_res = acc_and_f1(args, preds, label_ids.detach().cpu().numpy())
                tr_res["acc"] += eval_res["acc"]
                tr_res["f1"] += eval_res["f1"]

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                tr_loss += loss.item()

                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    optimizer.step()
                    warmup_linear.step()
                    model.zero_grad()
                    global_step += 1

            # train-set loss
            tr_loss /= nb_tr_steps
            tr_res["acc"] /= nb_tr_steps
            tr_res["f1"] /= nb_tr_steps

            # run evaluation on dev set
            # dev-set loss
            dev_res, dev_loss, _ = evaluate(args=args,
                                            eval_dataloader=dataloaders['dev'],
                                            model=model)


            test_res, test_loss, _ = evaluate(args=args,
                                              eval_dataloader=dataloaders['test'],
                                              model=model)

            # print and save results
            result = {"train_f1": tr_res["f1"], "train_acc": tr_res["acc"], "train_loss": tr_loss,
                      "dev_f1": dev_res["f1"], "dev_acc": dev_res['acc'], "dev_loss": dev_loss,
                      "test_f1": test_res["f1"],"test_acc": test_res["acc"], "test_loss": test_loss}

            results_logger.log_training(tr_loss, tr_res["f1"], tr_res["acc"], epoch)
            results_logger.log_validation(dev_loss, dev_res["f1"], dev_res["acc"], test_loss,
                                          test_res["f1"], test_res["acc"], epoch)
            results_logger.close()


            print('Epoch {}'.format(epoch + 1))
            for key, val in result.items():
                print("{}: {}".format(key, val))

            if args.write_log_for_each_epoch:
                output_eval_file = os.path.join(args.output_dir, "eval_results_Epoch_{}.txt".format(epoch + 1))
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Evaluation results *****")
                    for key in sorted(result.keys()):
                        logger.info("  %s = %s", key, str(result[key]))
                        writer.write("%s = %s\n" % (key, str(result[key])))
            else:
                logger.info("***** Evaluation results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))

            # Save model, configuration and tokenizer on the first epoch
            # If we save using the predefined names, we can load using `from_pretrained`
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            if epoch == 0:
                output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
                model_to_save.config.to_json_file(output_config_file)
                tokenizer.save_vocabulary(args.output_dir)

            if args.save_on_epoch_end:
                # Save a trained model
                if args.model_name is not None:
                    os.path.join(args.output_dir, args.model_name + '.Epoch_{}'.format(epoch + 1))
                else:
                    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME + '.Epoch_{}'.format(epoch+1))
                torch.save(model_to_save.state_dict(), output_model_file)

            # save model with best performance on dev-set
            if dev_res["f1"] >= best_dev_f1:
                src_domain_best['dev_f1'] = dev_res["f1"]
                src_domain_best['dev_acc'] = dev_res["acc"]
                src_domain_best['test_f1'] = test_res["f1"]
                src_domain_best['test_acc'] = test_res["acc"]
                best_epoch = epoch

                if args.save_best_weights:
                    print("Saving model, F1 improved from {} to {}".format(best_dev_f1, dev_res["f1"]))
                    if args.model_name is not None:
                        output_model_file = os.path.join(args.output_dir, args.model_name)
                    else:
                        output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)

                best_dev_f1 = src_domain_best['dev_f1']

            logger.info('Best results in domain: Dev F1 - {}'.format(src_domain_best['dev_f1']))
            logger.info('Best results in domain: Dev Acc - {}'.format(src_domain_best['dev_acc']))
            logger.info('Best results in domain: Test F1 - {}'.format(src_domain_best['test_f1']))
            logger.info('Best results in domain: Test Acc - {}'.format(src_domain_best['test_acc']))

    logger.info("***** Done Training *****")

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        if args.do_train:
            if args.save_best_weights:
                if args.model_name is not None:
                    output_model_file = os.path.join(args.output_dir, args.model_name)
                else:
                    output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
            elif args.save_on_epoch_end:
                if args.model_name is not None:
                    epoch = int(args.num_train_epochs) - 1
                    os.path.join(args.output_dir, args.model_name + '.Epoch_{}'.format(epoch + 1))
                else:
                    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME + '.Epoch_{}'.format(epoch+1))
            if hasattr(model, 'module'):
                model.module.load_state_dict(torch.load(output_model_file, map_location=args.device), strict=True)
            else:
                model.load_state_dict(torch.load(output_model_file, map_location=args.device), strict=True)

        for domain in args.tgt_domains:
            domain_eval_dir = os.path.join(args.output_dir, 'results', domain)
            if not os.path.isdir(domain_eval_dir):
                os.makedirs(domain_eval_dir)
            domain_data_dir = os.path.join(os.path.split(args.src_domain_data_dir)[0], domain)
            results = {'dev': {}, 'test': {}}
            for dataset in ['dev', 'test']:
                if dataset == 'train':
                    batch_size_ = args.train_batch_size
                    N_ = args.N_train
                else:
                    batch_size_ = args.eval_batch_size
                    N_ = args.N_dev

                processor = _processor(domain_data_dir)
                dataloader = make_DataLoader(processor=processor,
                                             tokenizer=tokenizer,
                                             max_seq_length=args.max_seq_length,
                                             batch_size=batch_size_,
                                             local_rank=args.local_rank,
                                             split_set=dataset,
                                             N=N_)

                logger.info("Evaluating on %s set" % dataset)
                res, loss, id_to_pred = evaluate(args=args,
                                                 eval_dataloader=dataloader,
                                                 model=model)
                results[dataset]['f1'] = res["f1"]
                results[dataset]['acc'] = res["acc"]
                results[dataset]['loss'] = loss
                # print results
                logger.info('{} F1: {}'.format(dataset, res["f1"]))
                logger.info('{} Accuracy: {}'.format(dataset, res["acc"]))

                id_to_pred_file = os.path.join(domain_eval_dir, 'preds_' + dataset + '.csv')
                with open(id_to_pred_file, 'w') as f:
                    for id, pred in id_to_pred.items():
                        f.write(("%s,%s\n" % (id, ','.join([str(p) for p in pred]))))

            # write results
            eval_file = os.path.join(domain_eval_dir, 'eval.txt')
            with open(eval_file, "w") as writer:
                for dataset in ['dev', 'test']:
                    writer.write("%s_f1 = %s\n" % (dataset, str(results[dataset]['f1'])))
                    writer.write("%s_acc = %s\n" % (dataset, str(results[dataset]['acc'])))
                    writer.write('\n')

if __name__ == "__main__":
    main()
