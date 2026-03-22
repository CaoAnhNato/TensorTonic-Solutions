import numpy as np

class BertEmbeddings:
    """
    BERT Embeddings = Token + Position + Segment
    """
    
    def __init__(self, vocab_size: int, max_position: int, hidden_size: int):
        self.hidden_size = hidden_size
        
        # Token embeddings
        self.token_embeddings = np.random.randn(vocab_size, hidden_size) * 0.02
        
        # Position embeddings (learned, not sinusoidal)
        self.position_embeddings = np.random.randn(max_position, hidden_size) * 0.02
        
        # Segment embeddings (just 2 segments: A and B)
        self.segment_embeddings = np.random.randn(2, hidden_size) * 0.02
    
    def forward(self, token_ids: np.ndarray, segment_ids: np.ndarray) -> np.ndarray:
        """
        Compute BERT embeddings.
        """
        seq_len = token_ids.shape[1]

        # Compute token_emb
        E_token = self.token_embeddings[token_ids]

        # Compute posi_emb
        position = np.arange(seq_len)
        E_posit = self.position_embeddings[position]

        # Compute seg_emb
        E_segment = self.segment_embeddings[segment_ids]

        output_emb = E_token + E_posit + E_segment
      
        return output_emb