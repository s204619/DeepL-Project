# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import numpy as np
# from tqdm.auto import tqdm
from tqdm import tqdm

# class AttLayer2(nn.Module):
#     """Attention layer for NRMS.
    
#         Attention layer with:
#         - Linear transformation
#         - Tanh activation 
#         - Linear projection to scalar
#         - Softmax to get attention weights
#          """
#     def __init__(self, attention_hidden_dim, seed=None):
#         super().__init__()
#         if seed is not None:
#             torch.manual_seed(seed)
        
#         self.attention = nn.Sequential(
#             nn.Linear(attention_hidden_dim, attention_hidden_dim),
#             nn.Tanh(),
#             nn.Linear(attention_hidden_dim, 1, bias=False)
#         )

#     def forward(self, x):
#         attention_weights = self.attention(x)
#         attention_weights = torch.softmax(attention_weights, dim=1)
#         return torch.sum(x * attention_weights, dim=1)

# class NRMSWrapper:
#     def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
#         """
#         Initializes the NRMSWrapper class.
#         Handles:
#         - Model placement on CPU/GPU
#         - Optimizer (Adam)
#         - Loss function (BCE)
#         - Training loop
#         - Validation
#          """
        
        
#         print("NRMSWrapper init")
#         print(torch.cuda.is_available())
#         self.model = model
#         self.device = device
#         self.model.to(device)
#         self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#         self.criterion = nn.BCELoss()
#         self.scorer = self  # Add this line
        
#     def fit(self, train_dataloader, validation_data=None, epochs=1, callbacks=None):
#         best_val_loss = float('inf')
#         for epoch in range(epochs):
            
#             # Training phase
#             self.model.train()
#             train_loss = 0
#             train_iter = tqdm(enumerate(train_dataloader), 
#                          desc=f'Epoch {epoch+1}/{epochs} [Train]',
#                          total=len(train_dataloader),
#                          position=0, 
#                          leave=True)
            
#             for batch_idx, (inputs, labels) in train_iter:
#                 # train_iter = tqdm(train_dataloader, desc=f'Batch {batch_idx+1}/{len(train_dataloader)} [Train]', position=0, leave=True)  
#                 history, candidates = inputs
                
#                 # Convert numpy arrays to torch tensors
#                 history = torch.from_numpy(history).to(self.device)
#                 candidates = torch.from_numpy(candidates).to(self.device)
#                 labels = torch.from_numpy(labels).float().to(self.device)
                
#                 self.optimizer.zero_grad()
#                 outputs = self.model(history, candidates, training=True)
#                 loss = self.criterion(outputs, labels)
                
#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
#                 self.optimizer.step()
                
#                 train_loss += loss.item()
#                 train_iter.set_postfix({'loss': f'{train_loss/(batch_idx+1):.4f}'})
            
#             avg_train_loss = train_loss / len(train_dataloader)
            
            
#             # Validation phase
#             if validation_data is not None:
#                 self.model.eval()
#                 val_loss = 0
#                 val_iter = tqdm(validation_data,
#                                             desc=f'Epoch {epoch+1}/{epochs} [Valid]',
#                                             total=len(validation_data),
#                                             position=0,
#                                             leave=True)
#                 with torch.no_grad():
#                     for inputs, labels in val_iter:
#                         history, candidates = inputs
#                         history = torch.from_numpy(history).to(self.device)
#                         candidates = torch.from_numpy(candidates).to(self.device)
#                         labels = torch.from_numpy(labels).float().to(self.device)
                        
#                         outputs = self.model(history, candidates, training=False)
#                         val_loss += self.criterion(outputs, labels).item()
                        
#                         val_iter.set_postfix({'loss': f'{val_loss/(batch_idx+1):.4f}'})
                
#                 avg_val_loss = val_loss / len(validation_data)
                
#                 print(f'Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
                
#                 # Handle callbacks
#                 if callbacks is not None:
#                     logs = {'loss': avg_train_loss, 'val_loss': avg_val_loss}
#                     for callback in callbacks:
#                         if hasattr(callback, 'on_epoch_end'):
#                             callback.on_epoch_end(epoch, logs)
        
#         return self
    
#     def predict(self, dataloader):
#         self.model.eval()
#         predictions = []
        
#         pred_iter = tqdm(dataloader, desc='Predicting',
#                          total=len(dataloader),
#                          position=0, 
#                          leave=True)
        
            
        
#         with torch.no_grad():
#             for inputs, _ in pred_iter:
#                 history, candidates = inputs
#                 history = torch.from_numpy(history).to(self.device)
#                 candidates = torch.from_numpy(candidates).to(self.device)
#                 outputs = self.model(history, candidates, training=False)
#                 predictions.append(outputs.cpu().numpy())
                
#                 # pred_iter.set_postfix(XXXXX)
                
#         return np.concatenate(predictions)

#     def save_weights(self, filepath):
#         torch.save(self.model.state_dict(), filepath)
        
#     def load_weights(self, filepath):
#         self.model.load_state_dict(torch.load(filepath))

# class SelfAttention(nn.Module):

#     """
#     Multi-head self attention layer.
#     - Projects input to a specified dimension if necessary.
#     - Computes query, key, and value matrices.
#     - Applies scaled dot-product attention.
#     - Outputs the context vector.
#     """
#     def __init__(self, head_num, head_dim, input_dim=None, seed=None):
#         super().__init__()
#         if seed is not None:
#             torch.manual_seed(seed)
            
#         self.head_num = head_num
#         self.head_dim = head_dim
#         self.output_dim = head_num * head_dim
        
#         # Add input projection if input_dim is provided and different from output_dim
#         if input_dim is not None and input_dim != self.output_dim:
#             self.input_proj = nn.Linear(input_dim, self.output_dim)
#         else:
#             self.input_proj = nn.Identity()
        
#         self.wq = nn.Linear(self.output_dim, self.output_dim)
#         self.wk = nn.Linear(self.output_dim, self.output_dim)
#         self.wv = nn.Linear(self.output_dim, self.output_dim)
        
#     def forward(self, x):
#         # Project input if necessary
#         x = self.input_proj(x)
        
#         q = self.wq(x).view(-1, x.size(1), self.head_num, self.head_dim)
#         k = self.wk(x).view(-1, x.size(1), self.head_num, self.head_dim)
#         v = self.wv(x).view(-1, x.size(1), self.head_num, self.head_dim)
        
#         q = q.transpose(1, 2)  # (batch, head_num, seq_len, head_dim)
#         k = k.transpose(1, 2)
#         v = v.transpose(1, 2)
        
#         scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
#         attention_weights = torch.softmax(scores, dim=-1)
        
#         context = torch.matmul(attention_weights, v)
#         context = context.transpose(1, 2).contiguous()
#         output = context.view(-1, x.size(1), self.output_dim)
#         return output

# class NRMSModel(nn.Module):
#     """
    
#     """ 
#     def __init__(
#         self,
#         hparams: dict,
#         word2vec_embedding: np.ndarray = None,
#         word_emb_dim: int = 300,
#         vocab_size: int = 32000,
#         seed: int = None,
#     ):
#         super().__init__()
#         self.hparams = hparams
        
#         if seed is not None:
#             torch.manual_seed(seed)
#             np.random.seed(seed)
        
#         # Initialize word embeddings
#         if word2vec_embedding is None:
#             self.word2vec_embedding = nn.Embedding(vocab_size, word_emb_dim)
#         else:
#             self.word2vec_embedding = nn.Embedding.from_pretrained(
#                 torch.FloatTensor(word2vec_embedding),
#                 freeze=False
#             )
#             word_emb_dim = word2vec_embedding.shape[1]
        
#         self.word_emb_dim = word_emb_dim
#         self.newsencoder = self._build_newsencoder()
#         self.userencoder = self._build_userencoder()
#         self.dropout = nn.Dropout(self.hparams.dropout)
        
#     def _build_newsencoder(self):
#         return nn.Sequential(
#             SelfAttention(
#                 head_num=self.hparams.head_num,
#                 head_dim=self.hparams.head_dim,
#                 input_dim=self.word_emb_dim
#             ),
#             AttLayer2(self.hparams.head_num * self.hparams.head_dim)
#         )
    
#     def _build_userencoder(self):
#         return nn.Sequential(
#             SelfAttention(
#                 head_num=self.hparams.head_num,
#                 head_dim=self.hparams.head_dim,
#                 input_dim=self.hparams.head_num * self.hparams.head_dim
#             ),
#             AttLayer2(self.hparams.head_num * self.hparams.head_dim)
#         )
    
#     def forward(self, his_input_title, pred_input_title, training=True):
#         # Process history
#         batch_size = his_input_title.size(0)
#         his_input_title = his_input_title.view(-1, self.hparams.title_size)
#         his_embedded = self.word2vec_embedding(his_input_title)
#         if training:
#             his_embedded = self.dropout(his_embedded)
#         his_encoded = self.newsencoder(his_embedded)
#         his_encoded = his_encoded.view(batch_size, self.hparams.history_size, -1)
#         user_present = self.userencoder(his_encoded)
        
#         # Process candidate news
#         pred_input_title = pred_input_title.view(-1, self.hparams.title_size)
#         pred_embedded = self.word2vec_embedding(pred_input_title)
#         if training:
#             pred_embedded = self.dropout(pred_embedded)
#         news_present = self.newsencoder(pred_embedded)
#         news_present = news_present.view(batch_size, -1, news_present.size(-1))
        
#         # Calculate scores - Modified dot product and scoring
#         scores = torch.bmm(news_present, user_present.unsqueeze(-1)).squeeze(-1)
#         return torch.sigmoid(scores)  # Changed from softmax to sigmoid for binary classification
    
#     def get_news_embedding(self, news_title):
#         news_embedded = self.word2vec_embedding(news_title)
#         return self.newsencoder(news_embedded)
    
#     def get_user_embedding(self, his_input_title):
#         batch_size = his_input_title.size(0)
#         his_input_title = his_input_title.view(-1, self.hparams.title_size)
#         his_embedded = self.word2vec_embedding(his_input_title)
#         his_encoded = self.newsencoder(his_embedded)
#         his_encoded = his_encoded.view(batch_size, self.hparams.history_size, -1)
#         return self.userencoder(his_encoded)

class NRMSModel(nn.Module):
    def __init__(
        self,
        hparams: dict,
        bert_model,
        seed: int = None,
    ):
        super().__init__()
        self.hparams = hparams
        self.bert_model = bert_model  # Use the pre-trained BERT model
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        self.dropout = nn.Dropout(self.hparams.dropout)
    
    def encode_news(self, input_ids, attention_mask):
        outputs = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Use the [CLS] token as the representation
        news_rep = outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size, hidden_size)
        return news_rep
    
    def forward(self, his_input_ids, his_attention_mask, pred_input_ids, pred_attention_mask, training=True):
        batch_size = his_input_ids.size(0)
        
        # Encode user history
        his_input_ids = his_input_ids.view(-1, his_input_ids.size(-1))
        his_attention_mask = his_attention_mask.view(-1, his_attention_mask.size(-1))
        his_encoded_news = self.encode_news(his_input_ids, his_attention_mask)
        his_encoded_news = his_encoded_news.view(batch_size, self.hparams.history_size, -1)
        
        # Aggregate user history (e.g., average pooling)
        user_rep = torch.mean(his_encoded_news, dim=1)
        if training:
            user_rep = self.dropout(user_rep)
        
        # Encode candidate news
        pred_input_ids = pred_input_ids.view(-1, pred_input_ids.size(-1))
        pred_attention_mask = pred_attention_mask.view(-1, pred_attention_mask.size(-1))
        pred_encoded_news = self.encode_news(pred_input_ids, pred_attention_mask)
        pred_encoded_news = pred_encoded_news.view(batch_size, -1, pred_encoded_news.size(-1))
        if training:
            pred_encoded_news = self.dropout(pred_encoded_news)
        
        # Compute scores
        scores = torch.bmm(pred_encoded_news, user_rep.unsqueeze(-1)).squeeze(-1)
        return torch.sigmoid(scores)


class NRMSWrapper:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # Smaller LR for BERT
        self.criterion = nn.BCELoss()
        
    def fit(self, train_dataloader, validation_data=None, epochs=1, callbacks=None):
        for epoch in range(epochs):
            # Training loop remains largely the same
            self.model.train()
            for batch_idx, (inputs, labels) in enumerate(train_dataloader):
                history_input_ids, history_attention_mask, candidate_input_ids, candidate_attention_mask = inputs
                labels = labels.float().to(self.device)
                
                history_input_ids = history_input_ids.to(self.device)
                history_attention_mask = history_attention_mask.to(self.device)
                candidate_input_ids = candidate_input_ids.to(self.device)
                candidate_attention_mask = candidate_attention_mask.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(
                    history_input_ids,
                    history_attention_mask,
                    candidate_input_ids,
                    candidate_attention_mask,
                    training=True
                )
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
            # Validation loop similar adjustments

    def predict(self, dataloader):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for inputs, _ in dataloader:
                history_input_ids, history_attention_mask, candidate_input_ids, candidate_attention_mask = inputs
                
                history_input_ids = history_input_ids.to(self.device)
                history_attention_mask = history_attention_mask.to(self.device)
                candidate_input_ids = candidate_input_ids.to(self.device)
                candidate_attention_mask = candidate_attention_mask.to(self.device)
                
                outputs = self.model(
                    history_input_ids,
                    history_attention_mask,
                    candidate_input_ids,
                    candidate_attention_mask,
                    training=False
                )
                predictions.append(outputs.cpu().numpy())
        return np.concatenate(predictions)
    
    def save_weights(self, filepath):
        torch.save(self.model.state_dict(), filepath)
        
    def load_weights(self, filepath):
        self.model.load_state_dict(torch.load(filepath))
    
    # Methods for saving/loading weights remain the same

# _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
# New bert thing
# _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
# replace word embeddings with what we get from the genreral tranformer
# replace the self attention with the one from the transformer since thats build in
# theory is that its a better architecture i guss
# _-_-_-_-_-_-_-_-_
#Claudes explanation:
    #Key changes made:
        # Removed the word embedding layer since BERT handles this internally
        # Replaced the SelfAttention layers with BERT's transformer layers
        # Updated the forward pass to use BERT's input format (input_ids and attention_mask)
        # Modified the attention mechanism to work with BERT's hidden size
        # Kept the user-level attention mechanism for aggregating history



# from transformers import AutoTokenizer, AutoModel

# class NRMSModel(nn.Module):
#     def __init__(
#         self,
#         hparams: dict,
#         model__name: str = "FacebookAI/xlm-roberta-large",
#         seed: int = None,
#     ):
#         super().__init__()
#         self.hparams = hparams
        
#         if seed is not None:
#             torch.manual_seed(seed)
#             np.random.seed(seed)
        
#         # Initialize BERT model
#         self.tranformer_model = AutoModel.from_pretrained(model__name)
#         self.dropout = nn.Dropout(self.hparams.dropout)
        
#         # matches transformer hidden size (Bubble-bee type stuff)
#         self.news_attention = AttLayer2(self.tranformer_model.config.hidden_size)
#         self.user_attention = AttLayer2(self.tranformer_model.config.hidden_size)
        
        
#     def encode_news(self, input_ids, attention_mask=None, training=True):
#         """
#         Newsencorder type stuff???
#         """
        
#         # Get BERT outputs
#         tranformer_model_outputs = self.tranformer_model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             return_dict=True
#         )
        
#         sequence_output = tranformer_model_outputs.last_hidden_state
#         if training:
#             sequence_output = self.dropout(sequence_output)
            
#         # Apply attention to get news representation
#         news_vector = self.news_attention(sequence_output)
#         return news_vector
    
#     def forward(self, his_input_ids, pred_input_ids, his_attention_mask=None, pred_attention_mask=None, training=True):
#         """
#         unchangd from pytorch verison
#         """
#         batch_size = his_input_ids.size(0)
        
#         # Process history
#         his_input_ids = his_input_ids.view(-1, self.hparams.title_size)
#         if his_attention_mask is not None:
#             his_attention_mask = his_attention_mask.view(-1, self.hparams.title_size)
            
#         his_encoded = self.encode_news(his_input_ids, his_attention_mask, training)
#         his_encoded = his_encoded.view(batch_size, self.hparams.history_size, -1)
#         user_present = self.user_attention(his_encoded)
        
#         # Process candidate news
#         pred_input_ids = pred_input_ids.view(-1, self.hparams.title_size)
#         if pred_attention_mask is not None:
#             pred_attention_mask = pred_attention_mask.view(-1, self.hparams.title_size)
            
#         news_present = self.encode_news(pred_input_ids, pred_attention_mask, training)
#         news_present = news_present.view(batch_size, -1, news_present.size(-1))
        
#         # Calculate scores
#         scores = torch.bmm(news_present, user_present.unsqueeze(-1)).squeeze(-1)
#         return torch.sigmoid(scores)
    
#     def get_news_embedding(self, news_ids, attention_mask=None):
#         """
#         unchangd from pytorch verison
#         """
#         return self.encode_news(news_ids, attention_mask, training=False)
    
#     def get_user_embedding(self, his_input_ids, attention_mask=None):
#         """
#         unchangd from pytorch verison
#         """
#         batch_size = his_input_ids.size(0)
#         his_input_ids = his_input_ids.view(-1, self.hparams.title_size)
#         if attention_mask is not None:
#             attention_mask = attention_mask.view(-1, self.hparams.title_size)
            
#         his_encoded = self.encode_news(his_input_ids, attention_mask, training=False)
#         his_encoded = his_encoded.view(batch_size, self.hparams.history_size, -1)
#         return self.user_attention(his_encoded)
    
    
    
    
# # class NRMSModelWithBERT(nn.Module):
# #     def __init__(self, transformer_model_name, hparams, seed=None):
# #         super().__init__()
# #         if seed is not None:
# #             torch.manual_seed(seed)
        
# #         # Load pre-trained transformer model
# #         self.bert = AutoModel.from_pretrained(transformer_model_name)
# #         self.tokenizer = AutoTokenizer.from_pretrained(transformer_model_name)
        
# #         # Parameters
# #         self.hparams = hparams
# #         self.hidden_size = self.bert.config.hidden_size
# #         self.dropout = nn.Dropout(hparams.dropout)
# #         self.output_layer = nn.Linear(self.hidden_size, 1)
        
# #     def encode_news(self, news_titles):
# #         # Tokenize and encode input
# #         inputs = self.tokenizer(
# #             news_titles,
# #             max_length=self.hparams.title_size,
# #             padding=True,
# #             truncation=True,
# #             return_tensors="pt"
# #         ).to(self.bert.device)
# #         outputs = self.bert(**inputs)
# #         return outputs.last_hidden_state[:, 0, :]  # Use [CLS] token output
    
# #     def forward(self, his_input_titles, pred_input_titles, training=True):
# #         batch_size = len(his_input_titles)
        
# #         # Encode user history
# #         history_embeddings = [
# #             self.encode_news(his) for his in his_input_titles
# #         ]
# #         user_embedding = torch.stack(history_embeddings, dim=1).mean(dim=1)  # Aggregate
        
# #         # Encode candidate news
# #         candidate_embeddings = self.encode_news(pred_input_titles)
        
# #         # Compute scores
# #         scores = torch.matmul(candidate_embeddings, user_embedding.unsqueeze(-1)).squeeze(-1)
# #         scores = torch.sigmoid(self.output_layer(scores))  # Binary classification
# #         return scores
