import math
import torch

def last_hidden_output(batch, model):
  model.eval()
  with torch.no_grad():
      # Get inputs
      input_ids = batch['input_ids'].to(device)
      attention_mask = batch['attention_mask'].to(device)
      glyph_embeddings = batch['glyph_embeddings'].to(device)
      seq_lengths = batch['seq_lengths']

      # Get last output of BiLSTM (size 800)
      outputs = model.bert(input_ids, attention_mask=attention_mask)
      unpooled_outputs = outputs['last_hidden_state'][:,1:,:]
      combined_output = torch.concat([unpooled_outputs, glyph_embeddings], axis=-1)
      X = pack_padded_sequence(combined_output, seq_lengths, batch_first=True, enforce_sorted=False)
      output, (hn, cn) = model.rnn(X)
      X = torch.cat([*hn, *cn], dim=-1).unsqueeze(dim=0)
  return X

def get_intermediate_outputs(batch, model):
    with torch.no_grad():
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        glyph_embeddings = batch['glyph_embeddings'].to(device)
        seq_lengths = batch['seq_lengths'].to(device)
        model(input_ids, attention_mask=attention_mask, glyph_embeddings= glyph_embeddings, lens=seq_lengths)
        _pooled_output = activation['pooler']
        combined_output =_pooled_output
        return combined_output

def cos_sim(v1, v2):
  dt = torch.dot(v1, v2)
  return (dt / (torch.norm(v1) * torch.norm(v2))).item()

def euclidean(v1, v2):
  return math.sqrt(torch.sum((v1 - v2) ** 2))

def manhattan(v1, v2):
  return torch.sum(v1 - v2).item()
