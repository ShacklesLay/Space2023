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


class DebertaReaderfortask1(DebertaReader):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 4
        self.qa_outputs = nn.Linear(config.hidden_size, 4)

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

        # return logits
        # span A start/end B start/end
        # A_start, A_end, B_start, B_end

        A_start_logits, A_end_logits, B_start_logits, B_end_logits = logits[:, :, 0], logits[:, :, 1], logits[:, :, 2], logits[:, :, -1]
        A_start_logits, A_end_logits, B_start_logits, B_end_logits = A_start_logits.contiguous(), A_end_logits.contiguous(), B_start_logits.contiguous(), B_end_logits.contiguous()

        total_loss = None
        if labels is not None:
            A_start_postion, A_end_postion, B_start_postion, B_end_position = labels[:, 0], labels[:, 1], labels[:, 2], labels[:, -1]
            loss = CrossEntropyLoss()

            # assert(A_end_logits.size(1) > torch.max(A_end_postion))

            A_start_loss = loss(A_start_logits, A_start_postion)
            A_end_loss = loss(A_end_logits, A_end_postion)
            B_start_loss = loss(B_start_logits, B_start_postion)
            B_end_loss = loss(B_end_logits, B_end_position)
            total_loss = (A_start_loss + A_end_loss + B_start_loss + B_end_loss) / 4

        return {
            "loss": total_loss,
            "A_start_logits": A_start_logits,
            "A_end_logits": A_end_logits,
            "B_start_logits": B_start_logits,
            "B_end_logits": B_end_logits
        }


class DebertaReaderfortask2(DebertaReader):
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


class DebertaReaderfortask2MLP(DebertaReader):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 12
        self.qa_output = nn.Linear(config.hidden_size, 12)
        self.transfer = \
            nn.Sequential(
                nn.Linear(self.num_labels, self.num_labels),
                nn.ReLU(),
                nn.Linear(self.num_labels, self.num_labels),
                nn.ReLU(),
                nn.Linear(self.num_labels, self.num_labels),
                nn.ReLU())

        # self.relu = nn.ReLU()

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
        logits = self.qa_output(sequence_output)

        '''
        first_logits = self.first_output(sequence_output)
        second_logits = self.second_output(sequence_output)
        logits = torch.cat((first_logits, self.transfer(first_logits) + second_logits), dim=-1)
        '''

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

class DebertaReaderfortask3(DebertaReader):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 6
        self.qa_outputs = nn.Linear(config.hidden_size, 6)

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

        # return logits
        # S_start, S_end, P_start, P_end, E_start, E_end
        # logits: [bsz, len_s, 6]
        # label : [bsz, 6]
        # assert(logits.size(-1) == labels.size(-1))


        total_loss = None
        if labels is not None:

            loss = CrossEntropyLoss()
            S_start = loss(logits[:, :, 0], labels[:, 0])
            S_end = loss(logits[:, :, 1], labels[:, 1])
            P_start = loss(logits[:, :, 2], labels[:, 2])
            P_end = loss(logits[:, :, 3], labels[:, 3])
            E_start = loss(logits[:, :, 4], labels[:, 4])
            E_end = loss(logits[:, :, 5], labels[:, 5])
            total_loss = (S_start + S_end + P_start + P_end + E_start + E_end) / 6

        return {
            "loss": total_loss,
            "S_start_logits": logits[:, :, 0],
            "S_end_logits": logits[:, :, 1],
            "P_start_logits": logits[:, :, 2],
            "P_end_logits": logits[:, :, 3],
            "E_start_logits": logits[:, :, 4],
            "E_end_logits": logits[:, :, 5]
        }


if __name__ == '__main__':
    qa_model = DebertaReaderfortask1.from_pretrained('../pretrained_model/chinese-deberta-large')
    # qa_model.to(torch.device())