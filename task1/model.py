from transformers import DebertaPreTrainedModel, DebertaModel
from transformers import ElectraPreTrainedModel, ElectraModel
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss


class DebertaReader(DebertaPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.deberta = DebertaModel(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        labels=None,
    ):
        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )


class DebertaReaderfortask(DebertaReader):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 12
        self.qa_outputs = nn.Linear(config.hidden_size, 12)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        labels=None,
    ):
        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)
        total_loss = None
        if labels is not None:

            loss = CrossEntropyLoss()
            _loss = []
            for i in range(12):
                _loss.append(loss(logits[:, :, i], labels[:, i]))
            total_loss = sum(_loss) / 12

        return {
            "loss": total_loss,
            "logits": logits
        }



if __name__ == '__main__':
    qa_model = DebertaReaderfortask.from_pretrained('WENGSYX/Deberta-Chinese-Large')
    # qa_model.to(torch.device())