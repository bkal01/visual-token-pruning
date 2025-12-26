from transformers import Qwen3VLForConditionalGeneration

class PrunedQwen3VL(Qwen3VLForConditionalGeneration):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.captured_attentions = []
        self.register_hooks()

    def register_hooks(self):
        """
        hook into Qwen3VLTextAttention and capture the attention weights during prefill
        """
        def hook(module, input, output):
            attn_output, attn_weights = output
            if attn_weights.shape[2] > 1:
                self.captured_attentions.append(attn_weights.cpu())
            return output

        for idx, layer in enumerate(self.model.language_model.layers):
            layer.self_attn.register_forward_hook(hook)