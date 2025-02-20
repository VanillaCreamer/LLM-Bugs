# LLM-Bugs

## Training Bugs

### RuntimeError: probability tensor contains either inf, nan or element < 0

```python
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList, InfNanRemoveLogitsProcessor, MinLengthLogitsProcessor

logits_processor = LogitsProcessorList()
logits_processor.append(MinLengthLogitsProcessor(15, eos_token_id=2, device='cuda'))
logits_processor.append(InfNanRemoveLogitsProcessor())

...

model.generate(..,logits_processor=logits_processor,...)
```



## Customized LoRA Bugs

### After 4-bit quantization, only linear code adapted by customized Lora can receive self-defined parameters (e.g., style, etc.).
