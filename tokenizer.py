from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers, processors, decoders
import regex
from datasets import dataset

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
tokenizer.decoder = decoders.ByteLevel()
tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
trainer = trainers.BpeTrainer(
    vocab_size=20000,
    min_frequency=2,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
tokenizer.train([text], trainer=trainer)
encoded = tokenizer.encode("I can feel the magic, can you?")