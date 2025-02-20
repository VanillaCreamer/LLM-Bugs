# LLM-Bugs

## Training Bugs

### RuntimeError: probability tensor contains either inf, nan or element < 0

```python
# https://github.com/THUDM/ChatGLM-6B/issues/31#issuecomment-1987262130
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList, InfNanRemoveLogitsProcessor, MinLengthLogitsProcessor

logits_processor = LogitsProcessorList()
logits_processor.append(MinLengthLogitsProcessor(15, eos_token_id=2, device='cuda'))
logits_processor.append(InfNanRemoveLogitsProcessor())

...

model.generate(..,logits_processor=logits_processor,...)
```

请注意这个方法不能根治出现nan的问题，只是把nan值换成了可处理的值，nan对模型训练产生的影响依然存在，继续训下去大概率会崩溃。

经过排查后发现是模型输出了`"\n\n"`这样的字符串，导致计算reward时出现了nan值，进而导致训练出现崩溃，因此取消了上述processor，设置了min_new_tokens=20

还是会出现同样的bug，debug后发现是某一条输出中出现了重复256次的`"\n\n\n...\n"`这样的值，导致reward函数计算结果中出现`nan`，可能需要加大repetition_penalty，或者将这样的输出的reward手动设置为0（已实现）

目前发现可能是量化存在的问题，如果不量化或者8bit量化会在别的地方出现类似的bug，目前的做法是将reward手动设置为0，并取消量化，同时参考已有论文调整超参数保证输出质量，目前能正常训练，但依然存在reward为0（也就是输出为无意义的字符串）的情况。

其它有可能的原因：
（1）输入max length设置的太大了，模型接收了太多pad token；
（2）max_grad_norm设置太大了；
（3）需要先SFT再使用GRPO；

## Customized LoRA Bugs

### After 4-bit quantization, only linear code adapted by customized Lora can receive self-defined parameters (e.g., style, etc.).
