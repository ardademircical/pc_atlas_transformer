class AtlasModelRBConfig():
        
    def __init__(
            self,
            vocab_size=50265,
            embed_dim=256,
            num_layers=6,
            num_heads=8,
            forward_expansion = 4,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            layer_norm_eps=1e-12,
            pad_token_id=0,
            classifier_dropout=None,
            **kwargs,
        ):
            super().__init__(pad_token_id=pad_token_id, **kwargs)

            self.vocab_size = vocab_size
            self.embed_dim = embed_dim
            self.num_layers = num_layers
            self.num_heads = num_heads
            self.hidden_act = hidden_act
            self.forward_expansion = forward_expansion
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.layer_norm_eps = layer_norm_eps
            self.classifier_dropout = classifier_dropout