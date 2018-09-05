import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class matchLSTM(nn.Module):
    def __init__(self, args, word_embedding):
        super(matchLSTM, self).__init__()
        self.args = args
        self.word_embedding = word_embedding
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # embedding layer with pretrained glove vector
        self.embeddings = nn.Embedding(len(self.word_embedding), args.embed_dim, padding_idx=0)
        self.init_weights()

        self.lstm_premise = nn.LSTM(args.embed_dim, args.hidden_dim, batch_first=True)
        self.lstm_hypo = nn.LSTM(args.embed_dim, args.hidden_dim, batch_first=True)
        self.match_lstm = nn.LSTMCell(args.hidden_dim * 2, args.hidden_dim)

        self.w_e = nn.Parameter(torch.Tensor(args.hidden_dim))
        nn.init.uniform_(self.w_e)

        self.w_s = nn.Linear(args.hidden_dim, args.hidden_dim, bias=False)
        self.w_t = nn.Linear(args.hidden_dim, args.hidden_dim, bias=False)
        self.w_m = nn.Linear(args.hidden_dim, args.hidden_dim, bias=False)
        # --> equation (6)
        self.fc = nn.Linear(args.hidden_dim, args.output_dim)
        
        self.dropout = nn.Dropout(p=args.dropout)
    
    def init_weights(self):
        self.embeddings.weight.data = torch.Tensor(self.word_embedding)
        self.embeddings.weight.requires_grad = False

    def forward(self, premise, premise_len, hypothesis, hypothesis_len):
        # premise LSTM
#        print 'premise size', premise.size()
        premise = premise.to(self.device)
        prem_max_len = premise.size(1) # max len

         # sort tensors by length
        premise_len, p_idx = premise_len.sort(0, descending=True)
        premise = premise[p_idx]
        
         # embed sequences
        premise_embed = self.embeddings(premise) # (N, max_len, embed_dim)

         # pack them up
        packed_premise = pack_padded_sequence(premise_embed, premise_len, batch_first=True)

         # to LSTM
        h_s, (_, _) = self.lstm_premise(packed_premise)

         # unpack the output
        h_s, _ = pad_packed_sequence(h_s, batch_first=True)

        # hypothesis LSTM
#        print 'hypothesis size', hypothesis.size()
        hypothesis = hypothesis.to(self.device)
        hypo_max_len = hypothesis.size(1)

        hypo_len, h_idx = hypothesis_len.sort(0, descending=True)
        hypo_embed = self.embeddings(hypothesis)
        packed_hypo = pack_padded_sequence(hypo_embed, hypo_len, batch_first=True)
        h_t, (_,_) = self.lstm_hypo(packed_hypo)
        h_t, _ = pad_packed_sequence(h_t, batch_first=True)
#        print h_t.size() # (N, len, embed_dim)
        
        # k-th hidden state
#        h_t[:,k,:]
 
        # match LSTM
        batch_size = premise_embed.size(0)

        # matchLSTM initialize
        h_m_k = torch.zeros((batch_size, self.args.hidden_dim), device=self.device)
        c_m_k = torch.zeros((batch_size, self.args.hidden_dim), device=self.device)
        
        # last hidden states
        last_hidden = torch.zeros((batch_size, self.args.hidden_dim), device=self.device)

        for k in range(hypo_max_len):
            h_t_k = h_t[:,k,:]       
    
            e_kj = torch.zeros((prem_max_len,batch_size), device=self.device)
            w_e_ = self.w_e.expand(batch_size, self.args.hidden_dim)
            w_e_ = w_e_.unsqueeze(1)
            
            for j in range(prem_max_len):
                # equation (6)
                # s_t_m size should be (N, hidden_dim, 1)
                s_t_m = torch.tanh(self.w_s(h_s[:,j,:]) + self.w_t(h_t_k) + self.w_m(h_m_k))
                s_t_m = s_t_m.unsqueeze(2)

                e = torch.bmm(w_e_, s_t_m) # (N, 1, 1)
                e_kj[j] = e.squeeze() # (N)
            
            # equation (3)
            alpha_kj = F.softmax(e_kj, dim=0) # (l, N)

            # equation (2)
            a_k = torch.bmm(alpha_kj.t().unsqueeze(1), h_s).squeeze() # (N, hidden_dim)
            
            # equation (7)
            m_k = torch.cat((a_k, h_t_k), 1) # (N, hidden_dim * 2)

            # equation (8)
            h_m_k, c_m_k = self.match_lstm(m_k, (h_m_k, c_m_k)) # (N, hidden_dim)

            # last hidden states
            for idx, h_len in enumerate(hypothesis_len):
                if (k+1) == h_len:
                    last_hidden[idx] = h_m_k[idx] # (N, hidden_dim)
        
        out = self.fc(last_hidden) # (N, output_dim)
        
        return out

