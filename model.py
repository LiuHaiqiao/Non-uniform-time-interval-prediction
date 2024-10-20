import torch.nn as nn
from Embed import DataEmbedding,TemporalEmbedding

class Model(nn.Module):
    def __init__(self, emb_dim, n_heads, e_layers, d_layers, dropout, c_in=200):
        super(Model, self).__init__()
        self.embedding = DataEmbedding(c_in=c_in, d_model=emb_dim, dropout=dropout)
        self.de_embedding = TemporalEmbedding(d_model=emb_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=n_heads, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model=emb_dim, nhead=n_heads, dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=d_layers)
        self.projection = nn.Linear(emb_dim, c_in)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        history_input = self.embedding(x_enc, x_mark_enc) # Batch_size, seq_len, emb_dim
        history_input = history_input.permute(1,0,2) # seq_len, batch_size, emb_dim
        dec_input = self.de_embedding(x_mark_dec) # batch_size, 1, emb_dim
        dec_input = dec_input.permute(1,0,2) # 1, batch_size, emb_dim
        encoder_output = self.encoder(history_input) # seq_len, batch_size, emb_dim
        decoder_ouput = self.decoder(dec_input, encoder_output) # 1, batch_size, emb_dim
        output = self.projection(decoder_ouput.squeeze(0)) # batch_size, c_in
        output = output.unsqueeze(1)
        return output  # [batch_size, 1 , c_in]