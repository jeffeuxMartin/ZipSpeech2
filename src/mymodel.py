import torch.nn as nn
from transformers import BartModel
from transformers.models.bart.modeling_bart import (
    shift_tokens_right)
from transformers.modeling_outputs import (
    BaseModelOutput,
)

# from ..torch_cif import cif_function  # FIXME
import torch_cif  # FIXME

from .mydataclasses import WithLossAccOutput
from .utils import mask_generator

class BartHuBertAutoEncoder(BartModel):
    def __init__(self, *a, **k):
        if "PAD_TOKEN" in k:
            PAD_TOKEN = k['PAD_TOKEN']
            k.pop('PAD_TOKEN')
        else:
            PAD_TOKEN = None
        super().__init__(*a, **k)
        self.alpha_predictor = nn.Linear(768, 1)  # XXX
        self.LMHead = nn.Linear(768, 504)  # XXX
        if PAD_TOKEN is not None:
            k['PAD_TOKEN'] = PAD_TOKEN
        self.PAD_TOKEN = k.get("PAD_TOKEN", -100)
        # print('SELF.PAD_TOKEN =', self.PAD_TOKEN, "(line 20)")
        self.crit = nn.CrossEntropyLoss(ignore_index=self.PAD_TOKEN)
        
    @classmethod
    def from_pretrained(
        cls, 
        # PAD_TOKEN=-100, 
        *a, 
        **k,
      ):
        obj = super().from_pretrained(*a, **k)
        # print("=== Reinitialized PAD_TOKEN = {} ===".format(obj.PAD_TOKEN))
        return obj
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        word_lengths=None,
        
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # print(input_ids.shape)

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            
            importances = self.alpha_predictor(encoder_outputs[0])
            importances = importances.squeeze(-1)
            importances = importances.softmax(-1)
            cif_out = torch_cif.cif_function(
                encoder_outputs[0], 
                importances,
                padding_mask=~(attention_mask.bool()),
                # target_lengths=(
                #     torch.tensor(word_lengths).long()),
                target_lengths=word_lengths,
            )
            [shortened] = cif_out['cif_out']
            [new_word_lengths] = cif_out['cif_lengths']
            [_check1] = cif_out['alpha_sum']
            
            out_mask = mask_generator(word_lengths)
            
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            
            encoder_hidden_states=shortened,
            # encoder_attention_mask=attention_mask,  ## XXX
            encoder_attention_mask=out_mask, ## Fix by `mask_generator`
            
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = self.LMHead(decoder_outputs.last_hidden_state)
        
        loss = self.crit(
            logits.transpose(1, 2),
            input_ids,
        )
        same_place = (logits.argmax(-1) == input_ids) * attention_mask  # B x S
        acc = same_place.sum(1) / attention_mask.sum(1)  # B
        acc = acc.mean()
        
        

        if not return_dict:
            # return decoder_outputs + encoder_outputs
            return shortened + logits

        return WithLossAccOutput(
            loss=loss,
            acc=acc,
            last_hidden_state=decoder_outputs.last_hidden_state,
            logits=logits,  # reconstructed logits
            
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            
            # encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_last_hidden_state=shortened,

            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

