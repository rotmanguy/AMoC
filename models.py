import torch
from torch import nn
from torch.nn import ParameterList
from transformers.modeling_bert import BertModel, BertPreTrainedModel, BertEncoder
from torch.nn.parameter import Parameter

class CNNBertForSequenceClassification(BertPreTrainedModel):
    """BERT model for classification and masked Pivot feature prediction.
    This module is composed of the BERT model with two linear layer on top of
    the pooled output - one for classification task and the other for multi-label classification task.
    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.
        `num_aux_labels`: the number of classes for the auxiliary task classifier. Default = 500.
    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary. Items in the batch should begin with the special "CLS" token.
             (see the tokens preprocessing logic in the scripts `extract_features.py`, `run_classifier.py` and
              `run_squad.py`). When training for the auxiliary task the input pivot features are masked.
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].
        `multy_class_labels`: multy-class labels for the auxiliary task classification output:
        torch.LongTensor of shape [batch_size, num_of_pivots] with indices selected in [0, 1].
    Outputs:
        if `labels` is not `None` and multy_class_labels is not `None`:
            Outputs the CrossEntropy classification loss for labeled data + CrossEntropy Multi Class Binary
            classification loss for unlabeled data.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].
    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, -1, 1], [-1, 1, 0]]) # '-1' for unlabeled data
    token_aux_ids = torch.LongTensor([[0, 0, ..., 1, ..., 0, 1, ... , 0], [0, 1, ..., 0, ..., 0, 0, ... , 1]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    num_labels = 2
    num_aux_labels = 500
    model = BertForSequenceClassificationWithAux(config, num_labels, num_aux_labels)
    logits = model(input_ids, token_type_ids, token_aux_ids, input_mask)
    ```
    """
    def __init__(self, config, hidden_size=768, filter_size=9, out_channels=16, max_seq_length=128,
                 padding=True, combine_layers='mix', layer_dropout=0.0, layers_to_prune=[], bert_model_type='default'):
        super(CNNBertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        config.layers_to_prune = layers_to_prune
        if bert_model_type == 'default' or bert_model_type == 'neuron_pruning':
            self.bert = BertModel(config)
        elif bert_model_type == 'layer_pruning':
            print('Performing Layer Pruning')
            self.bert = BertModelLayerFreezing(config)
        else:
            raise ValueError('bert_model_type should be chosen from ["default", "skip_connection", "layer_pruning"]')

        self.combine_layers = combine_layers
        if self.combine_layers == "mix":
            self._scalar_mix = ScalarMixWithDropout(self.bert.config.num_hidden_layers + 1,
                                                    do_layer_norm=False,
                                                    layer_dropout=layer_dropout,
                                                    layers_to_prune=layers_to_prune)
        else:
            self._scalar_mix = None
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        padding_size = int((filter_size-1)/2) if padding else 0
        self.conv1 = nn.Conv1d(in_channels=hidden_size,
                               out_channels=out_channels,
                               kernel_size=filter_size, padding=padding_size)

        # self.max_pool = nn.MaxPool1d(kernel_size=2)
        self.max_pool = nn.AvgPool1d(kernel_size=2)
        classifier_in_size = int(out_channels*max_seq_length/2) if padding else \
            int((out_channels*(max_seq_length-filter_size+1))/2)
        self.classifier = nn.Linear(classifier_in_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        enc_sequence = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        if self._scalar_mix is not None:
            enc_sequence = self._scalar_mix(torch.stack(enc_sequence[-1]), attention_mask)
        elif self.combine_layers == "last":
            enc_sequence = enc_sequence[-1][-1]
            enc_sequence = self.dropout(enc_sequence)

        enc_seq_shape = enc_sequence.shape
        enc_sequence = enc_sequence.reshape(enc_seq_shape[0], enc_seq_shape[2], enc_seq_shape[1])
        features = self.conv1(enc_sequence)
        features_shape = features.shape
        features = features.reshape(features_shape[0], features_shape[2], features_shape[1])
        final_features_shape = features.shape
        final_features = features.reshape(final_features_shape[0], final_features_shape[2], final_features_shape[1])
        final_features = self.max_pool(final_features)
        final_features_shape = final_features.shape
        flat = final_features.reshape(-1, final_features_shape[1]*final_features_shape[2])
        logits = self.classifier(flat)
        return logits


class BertModelLayerFreezing(BertModel):
    def __init__(self, config):
        super(BertModelLayerFreezing, self).__init__(config)
        self.encoder = BertEncoderLayerFreezing(config)

class BertEncoderLayerFreezing(BertEncoder):
    def __init__(self, config):
        super(BertEncoderLayerFreezing, self).__init__(config)
        self.layers_to_prune = config.layers_to_prune
        self.layer_dropout = 0.5

    def forward(self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        output_hidden_states=False):

        all_hidden_states = ()
        all_attentions = ()
        if self.layers_to_prune is not None and len(self.layers_to_prune) > 0:
            untouched_layers = [i + 1 for i in range(len(self.layer)) if i + 1 not in self.layers_to_prune]
            graph_connections = {untouched_layers[i]: 0 if i == 0 else untouched_layers[i - 1] for i in range(len(untouched_layers))}
        else:
            probs = torch.rand(12)
            untouched_layers = [i + 1 for i in range(len(self.layer)) if self.layer_dropout < probs[i]]
            graph_connections = {untouched_layers[i]: 0 if i == 0 else untouched_layers[i - 1] for i in range(len(untouched_layers))}

        for i, layer_module in enumerate(self.layer):
            all_hidden_states = all_hidden_states + (hidden_states,)
            if i + 1 in graph_connections:
                prev_hidden_states = all_hidden_states[graph_connections[i + 1]]
            else:
                prev_hidden_states = all_hidden_states[-1]
            layer_outputs = layer_module(prev_hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask)
            hidden_states = layer_outputs[0]
            if i + 1 not in graph_connections:
                hidden_states = torch.zeros_like(hidden_states)
            if output_attentions:
                if i + 1 not in graph_connections:
                    all_attentions = all_attentions + (torch.zeros_like(layer_outputs[1]),)
                else:
                    all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        all_hidden_states = all_hidden_states + (hidden_states, )

        if len(self.layer) not in graph_connections:
            if len(graph_connections) == 0:
                last_input_layer = -1
            else:
                last_input_layer = max([x + 1 for x in range(len(self.layer)) if x + 1 in graph_connections])
            outputs = (all_hidden_states[last_input_layer],)
        else:
            outputs = (hidden_states,)

        if output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)

class ScalarMixWithDropout(torch.nn.Module):
    """
    Computes a parameterised scalar mixture of N tensors, ``mixture = gamma * sum(s_k * tensor_k)``
    where ``s = softmax(w)``, with ``w`` and ``gamma`` scalar parameters.

    If ``do_layer_norm=True`` then apply layer normalization to each tensor before weighting.

    If ``dropout > 0``, then for each scalar weight, adjust its softmax weight mass to 0 with
    the dropout probability (i.e., setting the unnormalized weight to -inf). This effectively
    should redistribute dropped probability mass to all other weights.
    """
    def __init__(self,
                 mixture_size,
                 do_layer_norm,
                 initial_scalar_parameters=None,
                 trainable=True,
                 layer_dropout=None,
                 dropout_value=-1e20,
                 layers_to_prune=[]):
        super(ScalarMixWithDropout, self).__init__()
        self.mixture_size = mixture_size
        self.do_layer_norm = do_layer_norm
        self.layer_dropout = layer_dropout
        self.layers_to_prune = layers_to_prune

        if initial_scalar_parameters is None:
            initial_scalar_parameters = [0.0] * mixture_size
        elif len(initial_scalar_parameters) != mixture_size:
            raise ValueError("Length of initial_scalar_parameters {} differs "
                                     "from mixture_size {}".format(
                                             initial_scalar_parameters, mixture_size))
        for i in self.layers_to_prune:
            initial_scalar_parameters[i] = -1e20

        self.scalar_parameters = ParameterList(
                [Parameter(torch.FloatTensor([initial_scalar_parameters[i]]),
                           requires_grad=trainable) for i
                 in range(mixture_size)])

        for i in self.layers_to_prune:
            self.scalar_parameters[i].requires_grad = False

        self.gamma = Parameter(torch.FloatTensor([1.0]), requires_grad=trainable)

        if self.layer_dropout:
            dropout_mask = torch.zeros(mixture_size)
            dropout_fill = torch.empty(mixture_size).fill_(dropout_value)
            self.register_buffer("dropout_mask", dropout_mask)
            self.register_buffer("dropout_fill", dropout_fill)

    def forward(self, tensors, mask=None):
        """
        Compute a weighted average of the ``tensors``.  The input tensors an be any shape
        with at least two dimensions, but must all be the same shape.

        When ``do_layer_norm=True``, the ``mask`` is required input.  If the ``tensors`` are
        dimensioned  ``(dim_0, ..., dim_{n-1}, dim_n)``, then the ``mask`` is dimensioned
        ``(dim_0, ..., dim_{n-1})``, as in the typical case with ``tensors`` of shape
        ``(batch_size, timesteps, dim)`` and ``mask`` of shape ``(batch_size, timesteps)``.

        When ``do_layer_norm=False`` the ``mask`` is ignored.
        """
        if len(tensors) != self.mixture_size:
            raise ValueError("{} tensors were passed, but the module was initialized to "
                                     "mix {} tensors.".format(len(tensors), self.mixture_size))

        def _do_layer_norm(tensor, broadcast_mask, num_elements_not_masked):
            tensor_masked = tensor * broadcast_mask
            mean = torch.sum(tensor_masked) / num_elements_not_masked
            variance = torch.sum(((tensor_masked - mean) * broadcast_mask)**2) / num_elements_not_masked
            return (tensor - mean) / torch.sqrt(variance + 1E-12)

        weights = torch.cat([parameter for parameter in self.scalar_parameters])

        if self.layer_dropout:
            weights = torch.where(self.dropout_mask.uniform_() > self.layer_dropout, weights, self.dropout_fill)

        normed_weights = torch.nn.functional.softmax(weights, dim=0)
        normed_weights = torch.split(normed_weights, split_size_or_sections=1)
        if not self.do_layer_norm:
            pieces = []
            for weight, tensor in zip(normed_weights, tensors):
                pieces.append(weight * tensor)
            return self.gamma * sum(pieces)

        else:
            mask_float = mask.float()
            broadcast_mask = mask_float.unsqueeze(-1)
            input_dim = tensors[0].size(-1)
            num_elements_not_masked = torch.sum(mask_float) * input_dim

            pieces = []
            for weight, tensor in zip(normed_weights, tensors):
                pieces.append(weight * _do_layer_norm(tensor,
                                                      broadcast_mask, num_elements_not_masked))
            return self.gamma * sum(pieces)