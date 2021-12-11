import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import SequenceClassifierOutput

# Related references:
# https://stackoverflow.com/questions/64156202/add-dense-layer-on-top-of-huggingface-bert-model
# https://github.com/huggingface/transformers/blob/v4.5.0/src/transformers/models/bert/modeling_bert.py#L1515
class CustomPooledModel(nn.Module): #CustomPooledModel(nn.Module):
    def __init__(self, bert, embeddings, num_labels, component_pad_idx):
        super().__init__()
        self.num_labels = num_labels
        self.component_pad_idx = component_pad_idx
        self.component_embedding_dim = embeddings.shape[1]
        
        self.bert = bert
        self.dropout = nn.Dropout(0.1)
        self.subcomponent_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embeddings), 
                                                                   padding_idx = -1)
        self.classifier = nn.Linear(bert.config.hidden_size + self.component_embedding_dim, self.num_labels)
        # dummy parameter to store device: 
        # https://stackoverflow.com/questions/58926054/how-to-get-the-device-type-of-a-pytorch-module-conveniently
        self.dummy_param = nn.Parameter(torch.empty(0)) 
        
    def forward(self, input_ids=None, attention_mask=None, labels=None, subcomponent_ids=None):
        device = self.dummy_param.device
        # get pooled output from bert base model
        outputs = self.bert(input_ids, attention_mask = attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        
        # obtain averaged subcomponent vector for non-pad entries
        subcomponent_lengths = torch.sum(subcomponent_ids != self.component_pad_idx, dim = -1)
        tmp = torch.stack([torch.ones((1,self.component_embedding_dim), device = device)*max([x, 1]) 
                           for x in subcomponent_lengths]).squeeze()
        subcomponent_vectors = self.subcomponent_embedding(subcomponent_ids)
        sum_subcomponent_vectors = torch.sum(subcomponent_vectors, dim = 1)
        averaged_subcomponent_vectors = torch.div(sum_subcomponent_vectors, tmp)
        
        # concatenate pooled bert output and averaged subcomponent vector
        combined_output = torch.cat((pooled_output, averaged_subcomponent_vectors), dim=-1)
        logits = self.classifier(combined_output)
        
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        
class CustomUnpooledModel(torch.nn.Module):
    
    def __init__(self, lstm_input_size: int, hidden_size: int, output_size: int, padding_idx: int, bertconfig, dropout=0.5):
        super().__init__()
        self.lstm_input_size = lstm_input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.padding_idx = padding_idx
        self.bert = BertModel.from_pretrained(bertconfig)
        self.rnn = nn.LSTM(self.lstm_input_size, self.hidden_size, 1, bidirectional=True)
        self.fc = Linear(4*self.hidden_size, self.output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        glyph_embeddings=None,
        lens=None
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        unpooled_outputs = outputs['last_hidden_state'][:,1:,:]
        combined_output = torch.concat([unpooled_outputs, glyph_embeddings], axis=-1)

        #LSTM architecture
        X = pack_padded_sequence(combined_output, lens, batch_first=True, enforce_sorted=False)
        output, (hn, cn) = self.rnn(X)
        X = torch.cat([*hn, *cn], dim=-1).unsqueeze(dim=0)
        X = self.dropout(X)
        X = self.fc(X).squeeze()
        return X

def train_loop(dataloader, model, optimizer, device):
    model.train()
    train_loss = 0; correct = 0;
    num_batches = len(dataloader)
    size = len(dataloader.dataset)

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        component_ids = batch['subcomponent_ids'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels, 
                        subcomponent_ids = component_ids)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()
        # compute accuracy
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        correct += (predictions == labels).type(torch.float).sum().item()

    train_loss /= num_batches; correct /= size
    print(f"Train Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {train_loss:>8f} \n")
    return train_loss, correct

def test_loop(dataloader, model, lr_scheduler, device):
    model.eval()
    test_loss = 0; correct = 0;
    num_batches = len(dataloader)
    size = len(dataloader.dataset)

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            component_ids = batch['subcomponent_ids'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels, subcomponent_ids = component_ids)

            test_loss += outputs.loss.item()
            # compute accuracy
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).type(torch.float).sum().item()

    test_loss /= num_batches; correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, correct
