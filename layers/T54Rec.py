from typing import Optional, Tuple
from numpy import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration, T5Stack
from transformers.models.t5.configuration_t5 import T5Config
from transformers.modeling_outputs import Seq2SeqLMOutput

from util.util import Trie, get_prefix_allowed_tokens_fn


class ManaulT5Stack(T5Stack):

    def __init__(self, config, embed_tokens=None):
        super().__init__(config, embed_tokens)
    
    def _forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        return_dict=None,
    ):
        # when: input_ids and inputs_embeds，inputs_embeds is soft_probs
        if input_ids is not None and inputs_embeds is not None:
            hard_embeds = self.embed_tokens(input_ids)                             
            soft_embeds = torch.matmul(inputs_embeds, self.embed_tokens.weight)
            inputs_embeds = (hard_embeds - soft_embeds).detach() + soft_embeds
        elif input_ids is not None:
            assert (
                self.embed_tokens is not None
            ), "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)
        elif inputs_embeds is not None:
            pass
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds"
            )

        return self.forward(
            input_ids=None,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            inputs_embeds=inputs_embeds,
            return_dict=return_dict,
        )




class T54Rec(T5ForConditionalGeneration):

    def __init__(self, config: T5Config):
        super().__init__(config)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = ManaulT5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = ManaulT5Stack(decoder_config, self.shared)

        self.loss_func = nn.CrossEntropyLoss()

        self.post_init()

    def _forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,                   # [batch_size, seq_len + 1]
        attention_mask: Optional[torch.FloatTensor] = None,             # [batch_size, seq_len + 1]
        decoder_input_ids: Optional[torch.LongTensor] = None,           # [batch_size, 1 + seq_len_tgt]
        inputs_embeds: Optional[torch.FloatTensor] = None,              # [batch_size, seq_len, hidden_size]
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,      # [batch_size, 1 + seq_len_tgt, hidden_size]
        labels: Optional[torch.LongTensor] = None,                      # [batch_size, seq_len_tgt + 1]
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        assert labels is not None, "labels are required"

        encoder_outputs = self.encoder._forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            return_dict=return_dict,
        ) # (last_hidden_state)

        hidden_states = encoder_outputs[0] # [batch_size, seq_len, hidden_size]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels) # [batch_size, 1 + seq_len_tgt]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder._forward(
            input_ids=decoder_input_ids,
            inputs_embeds=decoder_inputs_embeds,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            return_dict=return_dict,
        ) # (last_hidden_state)

        sequence_output = decoder_outputs[0] # [batch_size, seq_len_tgt + 1, hidden_size]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output) # [batch_size, seq_len_tgt + 1, vocab_size]

        loss = None
        # move labels to correct device to enable PP
        labels = labels.to(lm_logits.device)
        loss = self.loss_func(lm_logits.view(-1, lm_logits.shape[-1]), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output)

        return loss

    def _inference(
        self,
        input_ids,                   
        attention_mask,  
        item_indices,
        beam_size      
    ):
        item_indices = [[0] + x + [1] for x in item_indices.tolist()]
        trie = Trie(item_indices)
        prefix_allowed_tokens_fn = get_prefix_allowed_tokens_fn(trie)

        output = self.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=10,
            num_beams=beam_size,
            num_return_sequences=beam_size,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            early_stopping=True,
            return_dict_in_generate=True,
        )

        batch_size = input_ids.shape[0]
        output_ids = output["sequences"]
        output_ids = torch.stack([x[1:-1] for x in output_ids])
        output_ids = torch.reshape(
            output_ids, [batch_size, beam_size, -1]
        )
        return output_ids
