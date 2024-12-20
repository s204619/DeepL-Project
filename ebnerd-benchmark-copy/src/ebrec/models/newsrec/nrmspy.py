# # Copyright (c) Microsoft Corporation. All rights reserved.
# # Licensed under the MIT License.
# import torch
# import torch.nn as nn
# import numpy as np

# class AttLayer2(nn.Module):
#     """Attention layer for NRMS."""
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

# # class NRMSWrapper:
# #     def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
# #         print(torch.cuda.is_available())
# #         self.model = model
# #         self.device = device
# #         self.model.to(device)
# #         self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# #         self.criterion = nn.BCELoss()
# #         self.scorer = self  # Add this line
        
# #     def fit(self, train_dataloader, validation_data=None, epochs=1, callbacks=None):
# #         for epoch in range(epochs):
# #             self.model.train()
# #             for batch_idx, (inputs, labels) in enumerate(train_dataloader):
# #                 history, candidates = inputs
                
# #                 # Convert numpy arrays to torch tensors
# #                 history = torch.from_numpy(history).to(self.device)
# #                 candidates = torch.from_numpy(candidates).to(self.device)
# #                 labels = torch.from_numpy(labels).float().to(self.device)
                
# #                 self.optimizer.zero_grad()
# #                 outputs = self.model(history, candidates)
# #                 loss = self.criterion(outputs, labels)
                
# #                 loss.backward()
# #                 self.optimizer.step()
                
# #             # Handle validation if provided
# #             if validation_data is not None:
# #                 self.model.eval()
# #                 val_loss = 0
# #                 with torch.no_grad():
# #                     for inputs, labels in validation_data:
# #                         history, candidates = inputs
# #                         history = torch.from_numpy(history).to(self.device)
# #                         candidates = torch.from_numpy(candidates).to(self.device)
# #                         labels = torch.from_numpy(labels).float().to(self.device)
                        
# #                         outputs = self.model(history, candidates)
# #                         val_loss += self.criterion(outputs, labels).item()
                
# #                 # Handle callbacks
# #                 if callbacks is not None:
# #                     for callback in callbacks:
# #                         if hasattr(callback, 'on_epoch_end'):
# #                             callback.on_epoch_end(epoch, {'val_loss': val_loss})
        
# #         return self
    
# #     def predict(self, dataloader):
# #         self.model.eval()
# #         predictions = []
# #         with torch.no_grad():
# #             for inputs, _ in dataloader:
# #                 history, candidates = inputs
# #                 history = torch.from_numpy(history).to(self.device)
# #                 candidates = torch.from_numpy(candidates).to(self.device)
# #                 outputs = self.model(history, candidates)
# #                 predictions.append(outputs.cpu().numpy())
# #         return np.concatenate(predictions)

# #     def save_weights(self, filepath):
# #         torch.save(self.model.state_dict(), filepath)
        
# #     def load_weights(self, filepath):
# #         self.model.load_state_dict(torch.load(filepath))

# # class SelfAttention(nn.Module):
# #     """Multi-head self attention layer."""
# #     def __init__(self, head_num, head_dim, input_dim=None, seed=None):
# #         super().__init__()
# #         if seed is not None:
# #             torch.manual_seed(seed)
            
# #         self.head_num = head_num
# #         self.head_dim = head_dim
# #         self.output_dim = head_num * head_dim
        
# #         # Add input projection if input_dim is provided and different from output_dim
# #         if input_dim is not None and input_dim != self.output_dim:
# #             self.input_proj = nn.Linear(input_dim, self.output_dim)
# #         else:
# #             self.input_proj = nn.Identity()
        
# #         self.wq = nn.Linear(self.output_dim, self.output_dim)
# #         self.wk = nn.Linear(self.output_dim, self.output_dim)
# #         self.wv = nn.Linear(self.output_dim, self.output_dim)
        
# #     def forward(self, x):
# #         # Project input if necessary
# #         x = self.input_proj(x)
        
# #         q = self.wq(x).view(-1, x.size(1), self.head_num, self.head_dim)
# #         k = self.wk(x).view(-1, x.size(1), self.head_num, self.head_dim)
# #         v = self.wv(x).view(-1, x.size(1), self.head_num, self.head_dim)
        
# #         q = q.transpose(1, 2)  # (batch, head_num, seq_len, head_dim)
# #         k = k.transpose(1, 2)
# #         v = v.transpose(1, 2)
        
# #         scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
# #         attention_weights = torch.softmax(scores, dim=-1)
        
# #         context = torch.matmul(attention_weights, v)
# #         context = context.transpose(1, 2).contiguous()
# #         output = context.view(-1, x.size(1), self.output_dim)
# #         return output

# # class NRMSModel(nn.Module):
# #     def __init__(
# #         self,
# #         hparams: dict,
# #         word2vec_embedding: np.ndarray = None,
# #         word_emb_dim: int = 300,
# #         vocab_size: int = 32000,
# #         seed: int = None,
# #     ):
# #         super().__init__()
# #         self.hparams = hparams
# #         # self.model = self.build_model(hparams, word2vec_embedding, seed)
        
# #         if seed is not None:
# #             torch.manual_seed(seed)
# #             np.random.seed(seed)
        
# #         # Initialize word embeddings
# #         if word2vec_embedding is None:
# #             self.word2vec_embedding = nn.Embedding(vocab_size, word_emb_dim)
# #         else:
# #             self.word2vec_embedding = nn.Embedding.from_pretrained(
# #                 torch.FloatTensor(word2vec_embedding),
# #                 freeze=False
# #             )
# #             word_emb_dim = word2vec_embedding.shape[1]
        
# #         self.word_emb_dim = word_emb_dim
# #         self.newsencoder = self._build_newsencoder()
# #         self.userencoder = self._build_userencoder()
# #         self.dropout = nn.Dropout(self.hparams.dropout)
        
# #     def _build_newsencoder(self):
# #         return nn.Sequential(
# #             SelfAttention(
# #                 head_num=self.hparams.head_num,
# #                 head_dim=self.hparams.head_dim,
# #                 input_dim=self.word_emb_dim
# #             ),
# #             AttLayer2(self.hparams.head_num * self.hparams.head_dim)
# #         )
    
# #     def _build_userencoder(self):
# #         return nn.Sequential(
# #             SelfAttention(
# #                 head_num=self.hparams.head_num,
# #                 head_dim=self.hparams.head_dim,
# #                 input_dim=self.hparams.head_num * self.hparams.head_dim
# #             ),
# #             AttLayer2(self.hparams.head_num * self.hparams.head_dim)
# #         )
    
# #     def forward(self, his_input_title, pred_input_title, training=True):
# #         # Process history
# #         batch_size = his_input_title.size(0)
# #         his_input_title = his_input_title.view(-1, self.hparams.title_size)
# #         his_embedded = self.word2vec_embedding(his_input_title)
# #         if training:
# #             his_embedded = self.dropout(his_embedded)
# #         his_encoded = self.newsencoder(his_embedded)
# #         his_encoded = his_encoded.view(batch_size, self.hparams.history_size, -1)
# #         user_present = self.userencoder(his_encoded)
        
# #         # Process candidate news
# #         pred_input_title = pred_input_title.view(-1, self.hparams.title_size)
# #         pred_embedded = self.word2vec_embedding(pred_input_title)
# #         if training:
# #             pred_embedded = self.dropout(pred_embedded)
# #         news_present = self.newsencoder(pred_embedded)
# #         news_present = news_present.view(batch_size, -1, news_present.size(-1))
        
# #         # Calculate scores
# #         scores = torch.matmul(news_present, user_present.unsqueeze(-1)).squeeze(-1)
# #         return torch.softmax(scores, dim=-1)
    
# #     def get_news_embedding(self, news_title):
# #         news_embedded = self.word2vec_embedding(news_title)
# #         return self.newsencoder(news_embedded)
    
# #     def get_user_embedding(self, his_input_title):
# #         batch_size = his_input_title.size(0)
# #         his_input_title = his_input_title.view(-1, self.hparams.title_size)
# #         his_embedded = self.word2vec_embedding(his_input_title)
# #         his_encoded = self.newsencoder(his_embedded)
# #         his_encoded = his_encoded.view(batch_size, self.hparams.history_size, -1)
# #         return self.userencoder(his_encoded)
# class NRMSWrapper:
#     def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
#         print(torch.cuda.is_available())
#         self.model = model
#         self.device = device
#         self.model.to(device)
#         self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#         self.criterion = nn.BCELoss()
#         self.scorer = self
        
#     def save_weights(self, filepath):
#         """Save model weights to file"""
#         torch.save(self.model.state_dict(), filepath)
        
#     def load_weights(self, filepath):
#         """Load model weights from file"""
#         state_dict = torch.load(filepath, map_location=self.device)
#         self.model.load_state_dict(state_dict)
#         self.model.to(self.device)
        
#     def fit(self, train_dataloader, validation_data=None, epochs=1, callbacks=None):
#         best_val_loss = float('inf')
        
#         for epoch in range(epochs):
#             # Training
#             self.model.train()
#             total_loss = 0
#             for batch_idx, (inputs, labels) in enumerate(train_dataloader):
#                 history, candidates = inputs
                
#                 # Convert and verify inputs
#                 history = torch.from_numpy(history).to(self.device)
#                 candidates = torch.from_numpy(candidates).to(self.device)
#                 labels = torch.from_numpy(labels).float().to(self.device)
                
#                 self.optimizer.zero_grad()
#                 outputs = self.model(history, candidates)
                
#                 # Ensure outputs are proper probabilities
#                 outputs = torch.clamp(outputs, min=1e-7, max=1-1e-7)
                
#                 loss = self.criterion(outputs, labels)
#                 loss.backward()
#                 self.optimizer.step()
                
#                 total_loss += loss.item()
                
#                 if batch_idx % 100 == 0:
#                     print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
#                     print(f'Output range: {outputs.min().item():.4f} to {outputs.max().item():.4f}')
            
#             # Validation
#             if validation_data is not None:
#                 val_loss = self._validate(validation_data)
#                 print(f'Epoch {epoch}: Train Loss = {total_loss/len(train_dataloader):.4f}, Val Loss = {val_loss:.4f}')
                
#                 if callbacks is not None:
#                     for callback in callbacks:
#                         callback.on_epoch_end(epoch, {'val_loss': val_loss})
    
#     def _validate(self, validation_data):
#         self.model.eval()
#         total_val_loss = 0
#         with torch.no_grad():
#             for inputs, labels in validation_data:
#                 history, candidates = inputs
#                 history = torch.from_numpy(history).to(self.device)
#                 candidates = torch.from_numpy(candidates).to(self.device)
#                 labels = torch.from_numpy(labels).float().to(self.device)
                
#                 outputs = self.model(history, candidates)
#                 outputs = torch.clamp(outputs, min=1e-7, max=1-1e-7)
#                 loss = self.criterion(outputs, labels)
#                 total_val_loss += loss.item()
                
#         return total_val_loss / len(validation_data)
    
#     def predict(self, dataloader):
#         """Generate predictions for the given dataloader"""
#         self.model.eval()
#         predictions = []
        
#         with torch.no_grad():
#             for inputs, _ in dataloader:
#                 history, candidates = inputs
                
#                 # Move to device
#                 history = torch.from_numpy(history).to(self.device)
#                 candidates = torch.from_numpy(candidates).to(self.device)
                
#                 # Get predictions
#                 outputs = self.model(history, candidates)
#                 predictions.append(outputs.cpu().numpy())
                
#         return np.concatenate(predictions, axis=0)

# class NRMSModel(nn.Module):
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
#             nn.Dropout(self.hparams.dropout),
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

#     def forward(self, his_input_title, pred_input_title, pred_input_title_one=None):
#         """
#         Forward pass handling both training and scoring modes
#         """
#         # Process user history
#         batch_size = his_input_title.size(0)
#         his_input_title = his_input_title.view(-1, self.hparams.title_size)
#         his_embedded = self.dropout(self.word2vec_embedding(his_input_title))
#         his_encoded = self.newsencoder(his_embedded)
#         his_encoded = his_encoded.view(batch_size, self.hparams.history_size, -1)
#         user_present = self.userencoder(his_encoded)

#         # Scoring mode
#         if pred_input_title_one is not None:
#             pred_title_one = pred_input_title_one.view(-1, self.hparams.title_size)
#             pred_embedded_one = self.dropout(self.word2vec_embedding(pred_title_one))
#             news_present_one = self.newsencoder(pred_embedded_one)
#             pred_one = torch.matmul(news_present_one, user_present.unsqueeze(-1))
#             return torch.sigmoid(pred_one.squeeze(-1))
            
#         # Training mode
#         pred_input_title = pred_input_title.view(-1, self.hparams.title_size)
#         pred_embedded = self.dropout(self.word2vec_embedding(pred_input_title))
#         news_present = self.newsencoder(pred_embedded)
#         news_present = news_present.view(batch_size, -1, news_present.size(-1))
        
#         scores = torch.matmul(news_present, user_present.unsqueeze(-1)).squeeze(-1)
#         return torch.softmax(scores, dim=-1)

# class SelfAttention(nn.Module):
#     def __init__(self, head_num, head_dim, input_dim=None, seed=None):
#         super().__init__()
#         if seed is not None:
#             torch.manual_seed(seed)
            
#         self.head_num = head_num
#         self.head_dim = head_dim
#         self.output_dim = head_num * head_dim
        
#         # Input projection if needed
#         if input_dim is not None and input_dim != self.output_dim:
#             self.input_proj = nn.Linear(input_dim, self.output_dim)
#         else:
#             self.input_proj = nn.Identity()
        
#         # Q, K, V projections
#         self.wq = nn.Linear(self.output_dim, self.output_dim)
#         self.wk = nn.Linear(self.output_dim, self.output_dim)
#         self.wv = nn.Linear(self.output_dim, self.output_dim)

#     def forward(self, inputs):
#         # Handle triple input case like TF version
#         if isinstance(inputs, list):
#             q, k, v = [self.input_proj(x) for x in inputs]
#         else:
#             x = self.input_proj(inputs)
#             q = k = v = x

#         # Multi-head attention
#         q = self.wq(q).view(-1, q.size(1), self.head_num, self.head_dim)
#         k = self.wk(k).view(-1, k.size(1), self.head_num, self.head_dim)
#         v = self.wv(v).view(-1, v.size(1), self.head_num, self.head_dim)
        
#         q = q.transpose(1, 2)
#         k = k.transpose(1, 2)
#         v = v.transpose(1, 2)
        
#         scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
#         attention_weights = torch.softmax(scores, dim=-1)
        
#         context = torch.matmul(attention_weights, v)
#         context = context.transpose(1, 2).contiguous()
#         output = context.view(-1, q.size(2), self.output_dim)
#         return output

##### SPLITTTT
#####
####

# # nrmspy.py
# import torch
# import torch.nn as nn
# import numpy as np
# from torch.utils.tensorboard import SummaryWriter

# class AttLayer2(nn.Module):
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

# class SelfAttention(nn.Module):
#     def __init__(self, head_num, head_dim, input_dim=None, seed=None):
#         super().__init__()
#         if seed is not None:
#             torch.manual_seed(seed)
            
#         self.head_num = head_num
#         self.head_dim = head_dim
#         self.output_dim = head_num * head_dim
        
#         if input_dim is not None and input_dim != self.output_dim:
#             self.input_proj = nn.Linear(input_dim, self.output_dim)
#         else:
#             self.input_proj = nn.Identity()
        
#         self.wq = nn.Linear(self.output_dim, self.output_dim)
#         self.wk = nn.Linear(self.output_dim, self.output_dim)
#         self.wv = nn.Linear(self.output_dim, self.output_dim)

#     def forward(self, inputs):
#         if isinstance(inputs, list):
#             q, k, v = [self.input_proj(x) for x in inputs]
#         else:
#             x = self.input_proj(inputs)
#             q = k = v = x
        
#         q = self.wq(q).view(-1, q.size(1), self.head_num, self.head_dim)
#         k = self.wk(k).view(-1, k.size(1), self.head_num, self.head_dim)
#         v = self.wv(v).view(-1, v.size(1), self.head_num, self.head_dim)
        
#         q = q.transpose(1, 2)
#         k = k.transpose(1, 2)
#         v = v.transpose(1, 2)
        
#         scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
#         attention_weights = torch.softmax(scores, dim=-1)
#         context = torch.matmul(attention_weights, v)
        
#         context = context.transpose(1, 2).contiguous()
#         return context.view(-1, q.size(2), self.output_dim)

# class NRMSModel(nn.Module):
#     def __init__(self, hparams, word2vec_embedding=None, word_emb_dim=300, vocab_size=32000, seed=None):
#         super().__init__()
#         self.hparams = hparams
        
#         if seed is not None:
#             torch.manual_seed(seed)
#             np.random.seed(seed)
        
#         # Word embeddings
#         if word2vec_embedding is None:
#             self.word_embedding = nn.Embedding(vocab_size, word_emb_dim)
#         else:
#             self.word_embedding = nn.Embedding.from_pretrained(
#                 torch.FloatTensor(word2vec_embedding),
#                 freeze=False
#             )
        
#         # Build encoders
#         self.newsencoder = self._build_newsencoder()
#         self.userencoder = self._build_userencoder()
#         self.dropout = nn.Dropout(hparams.dropout)

#     def _build_newsencoder(self):
#         return nn.Sequential(
#             nn.Dropout(self.hparams.dropout),
#             SelfAttention(
#                 self.hparams.head_num,
#                 self.hparams.head_dim,
#                 input_dim=self.word_embedding.embedding_dim
#             ),
#             nn.Dropout(self.hparams.dropout),
#             AttLayer2(self.hparams.head_num * self.hparams.head_dim)
#         )
    
#     def _build_userencoder(self):
#         return nn.Sequential(
#             SelfAttention(
#                 self.hparams.head_num,
#                 self.hparams.head_dim,
#                 input_dim=self.hparams.head_num * self.hparams.head_dim
#             ),
#             AttLayer2(self.hparams.head_num * self.hparams.head_dim)
#         )

#     def forward(self, his_input_title, pred_input_title, pred_input_title_one=None):
#         # Encode user history
#         batch_size = his_input_title.size(0)
#         his_embedded = self.word_embedding(his_input_title.view(-1, self.hparams.title_size))
#         his_encoded = self.newsencoder(his_embedded)
#         his_encoded = his_encoded.view(batch_size, self.hparams.history_size, -1)
#         user_vector = self.userencoder(his_encoded)
        
#         # Scoring mode
#         if pred_input_title_one is not None:
#             news_one = self.word_embedding(pred_input_title_one.view(-1, self.hparams.title_size))
#             news_vector_one = self.newsencoder(news_one)
#             pred_one = torch.matmul(news_vector_one, user_vector.unsqueeze(-1))
#             return torch.sigmoid(pred_one.squeeze(-1))
            
#         # Training mode
#         pred_embedded = self.word_embedding(pred_input_title.view(-1, self.hparams.title_size))
#         pred_vectors = self.newsencoder(pred_embedded)
#         pred_vectors = pred_vectors.view(batch_size, -1, pred_vectors.size(-1))
        
#         scores = torch.matmul(pred_vectors, user_vector.unsqueeze(-1)).squeeze(-1)
#         return torch.softmax(scores, dim=-1)

# class NRMSWrapper:
#     def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
#         self.model = model
#         self.device = device
#         self.model.to(device)
#         self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#         self.criterion = nn.CrossEntropyLoss()
        
#     def fit(self, train_dataloader, validation_data=None, epochs=1, callbacks=None, writer=None):
#         writer = SummaryWriter(LOG_DIR)
        
#         for epoch in range(epochs):
#             self.model.train()
#             total_loss = 0
            
#             for batch_idx, (inputs, labels) in enumerate(train_dataloader):
#                 history, candidates = inputs
#                 history = torch.from_numpy(history).to(self.device)
#                 candidates = torch.from_numpy(candidates).to(self.device)
#                 labels = torch.from_numpy(labels).to(self.device)
                
#                 self.optimizer.zero_grad()
#                 outputs = self.model(history, candidates)
#                 loss = self.criterion(outputs, labels)
                
#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
#                 self.optimizer.step()
                
#                 total_loss += loss.item()
                
#             # Validation
#             if validation_data is not None:
#                 val_loss = self._validate(validation_data)
#                 writer.add_scalar('Loss/val', val_loss, epoch)
                
#                 if callbacks:
#                     for callback in callbacks:
#                         callback.on_epoch_end(epoch, {'val_loss': val_loss})
                        
#         writer.close()
#         return self
    
#     def _validate(self, validation_data):
#         self.model.eval()
#         total_loss = 0
#         with torch.no_grad():
#             for inputs, labels in validation_data:
#                 history, candidates = inputs
#                 history = torch.from_numpy(history).to(self.device)
#                 candidates = torch.from_numpy(candidates).to(self.device)
#                 labels = torch.from_numpy(labels).to(self.device)
                
#                 outputs = self.model(history, candidates)
#                 loss = self.criterion(outputs, labels)
#                 total_loss += loss.item()
#         return total_loss / len(validation_data)
    
#     def predict(self, dataloader):
#         self.model.eval()
#         predictions = []
#         with torch.no_grad():
#             for inputs, _ in dataloader:
#                 history, candidates = inputs
#                 history = torch.from_numpy(history).to(self.device)
#                 candidates = torch.from_numpy(candidates).to(self.device)
#                 outputs = self.model(history, candidates)
#                 predictions.append(outputs.cpu().numpy())
#         return np.concatenate(predictions)

#     def save_weights(self, filepath):
#         torch.save(self.model.state_dict(), filepath)
        
#     def load_weights(self, filepath):
#         self.model.load_state_dict(torch.load(filepath, map_location=self.device))

##### SPLITTTT
#####
####

# nrmspy.py
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
from torch.utils.tensorboard import SummaryWriter


class AttLayer2(nn.Module):
    def __init__(self, attention_hidden_dim: int, seed: Optional[int] = None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        
        self.attention = nn.Sequential(
            nn.Linear(attention_hidden_dim, attention_hidden_dim),
            nn.Tanh(),
            nn.Linear(attention_hidden_dim, 1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.attention(x)
        weights = torch.softmax(weights, dim=1)
        return torch.sum(x * weights, dim=1)

class SelfAttention(nn.Module):
    def __init__(self, head_num: int, head_dim: int, input_dim: Optional[int] = None, seed: Optional[int] = None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
            
        self.head_num = head_num
        self.head_dim = head_dim
        self.output_dim = head_num * head_dim
        
        if input_dim is not None and input_dim != self.output_dim:
            self.input_proj = nn.Linear(input_dim, self.output_dim)
        else:
            self.input_proj = nn.Identity()
        
        self.wq = nn.Linear(self.output_dim, self.output_dim)
        self.wk = nn.Linear(self.output_dim, self.output_dim)
        self.wv = nn.Linear(self.output_dim, self.output_dim)
        
    def forward(self, inputs):
        if isinstance(inputs, list):
            x = inputs[0]  # Use first input for self-attention
        else:
            x = inputs
            
        x = self.input_proj(x)
        
        q = self.wq(x).view(-1, x.size(1), self.head_num, self.head_dim)
        k = self.wk(x).view(-1, x.size(1), self.head_num, self.head_dim)
        v = self.wv(x).view(-1, x.size(1), self.head_num, self.head_dim)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, v)
        
        context = context.transpose(1, 2).contiguous()
        return context.view(-1, x.size(1), self.output_dim)

class NRMSModel(nn.Module):
    def __init__(self, hparams, word2vec_embedding=None, word_emb_dim=300, vocab_size=32000, seed=None):
        super().__init__()
        self.hparams = hparams
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
        # Word embeddings
        if word2vec_embedding is None:
            self.word_embedding = nn.Embedding(vocab_size, word_emb_dim)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(
                torch.FloatTensor(word2vec_embedding),
                freeze=False
            )
            
        self.newsencoder = self._build_newsencoder()
        self.userencoder = self._build_userencoder()
        self.dropout = nn.Dropout(hparams.dropout)

    def _build_newsencoder(self):
        return nn.Sequential(
            nn.Dropout(self.hparams.dropout),
            SelfAttention(
                self.hparams.head_num,
                self.hparams.head_dim,
                input_dim=self.word_embedding.embedding_dim,
                seed=getattr(self.hparams, 'seed', None)
            ),
            nn.Dropout(self.hparams.dropout),
            AttLayer2(self.hparams.head_num * self.hparams.head_dim)
        )
    
    def _build_userencoder(self):
        return nn.Sequential(
            SelfAttention(
                self.hparams.head_num,
                self.hparams.head_dim,
                input_dim=self.hparams.head_num * self.hparams.head_dim,
                seed=getattr(self.hparams, 'seed', None)
            ),
            AttLayer2(self.hparams.head_num * self.hparams.head_dim)
        )

    def forward(self, his_input_title, pred_input_title, pred_input_title_one=None):
        # Encode user history
        batch_size = his_input_title.size(0)
        his_embedded = self.word_embedding(his_input_title.view(-1, self.hparams.title_size))
        print(f"his_embedded shape: {his_embedded.shape}")
        
        his_encoded = self.newsencoder(his_embedded)
        print(f"his_encoded shape: {his_encoded.shape}")
        
        his_encoded = his_encoded.view(batch_size, self.hparams.history_size, -1)
        print(f"his_encoded reshaped: {his_encoded.shape}")
        
        user_vector = self.userencoder(his_encoded)
        print(f"user_vector shape: {user_vector.shape}")
        # Scoring mode
        if pred_input_title_one is not None:
            news_one = self.word_embedding(pred_input_title_one.view(-1, self.hparams.title_size))
            news_vector_one = self.newsencoder(news_one)
            pred_one = torch.matmul(news_vector_one, user_vector.unsqueeze(-1))
            return torch.sigmoid(pred_one.squeeze(-1))
            
        # Training mode
        pred_embedded = self.word_embedding(pred_input_title.view(-1, self.hparams.title_size))
        pred_vectors = self.newsencoder(pred_embedded)
        pred_vectors = pred_vectors.view(batch_size, -1, pred_vectors.size(-1))
        
        scores = torch.matmul(pred_vectors, user_vector.unsqueeze(-1)).squeeze(-1)
        return torch.sigmoid(scores, dim=-1)

class KerasLikeModel:
    def __init__(self, model):
        self.model = model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.optimizer = None
        self.criterion = None
        
    def compile(self, optimizer='adam', loss='categorical_crossentropy'):
        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            
        if loss == 'categorical_crossentropy':
            self.criterion = nn.CrossEntropyLoss()
        elif loss == 'binary_crossentropy':
            self.criterion = nn.BCELoss()
            
    def fit(self, train_dataloader, validation_data=None, epochs=1, callbacks=None):
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch_idx, (inputs, labels) in enumerate(train_dataloader):
                history, candidates = inputs
                history = torch.from_numpy(history).to(self.device)
                candidates = torch.from_numpy(candidates).to(self.device)
                labels = torch.from_numpy(labels).float().to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(history, candidates)
                loss = self.criterion(outputs, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                
            if callbacks:
                for callback in callbacks:
                    callback.on_epoch_end(epoch, {'val_loss': total_loss/len(train_dataloader)})

class NRMSWrapper:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        
    def fit(self, train_dataloader, validation_data=None, epochs=1, callbacks=None, writer=None):
        if writer is None:
            writer = SummaryWriter('runs/nrms')
            
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch_idx, (inputs, labels) in enumerate(train_dataloader):
                history, candidates = inputs
                history = torch.from_numpy(history).to(self.device)
                candidates = torch.from_numpy(candidates).to(self.device)
                labels = torch.from_numpy(labels).float().to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(history, candidates)
                loss = self.criterion(outputs, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                
                if batch_idx % 100 == 0:
                    print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
                    
            if validation_data is not None:
                val_loss = self._validate(validation_data)
                writer.add_scalar('Loss/val', val_loss, epoch)
                
                if callbacks:
                    for callback in callbacks:
                        callback.on_epoch_end(epoch, {'val_loss': val_loss})
                        
        writer.close()
        return self
    
    def _validate(self, validation_data):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, labels in validation_data:
                history, candidates = inputs
                history = torch.from_numpy(history).to(self.device)
                candidates = torch.from_numpy(candidates).to(self.device)
                labels = torch.from_numpy(labels).float().to(self.device)
                
                outputs = self.model(history, candidates)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
        return total_loss / len(validation_data)
    
    def predict(self, dataloader):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for inputs, _ in dataloader:
                history, candidates = inputs
                history = torch.from_numpy(history).to(self.device)
                candidates = torch.from_numpy(candidates).to(self.device)
                outputs = self.model(history, candidates)
                predictions.append(outputs.cpu().numpy())
        return np.concatenate(predictions)
        
    def save_weights(self, filepath):
        torch.save(self.model.state_dict(), filepath)
        
    def load_weights(self, filepath):
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))