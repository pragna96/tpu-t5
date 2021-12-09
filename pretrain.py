from datasets import load_dataset
from tokenizers import trainers, Tokenizer, normalizers, ByteLevelBPETokenizer

# load dataset
dataset = load_dataset("csv", column_names=["input","target"], split="train", data_files="/home/pragnamandadi/pretrain.tsv",delimiter="\t")

# Instantiate tokenizer
tokenizer = ByteLevelBPETokenizer()

def batch_iterator(batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i: i + batch_size]["input"]

# Customized training
tokenizer.train_from_iterator(batch_iterator(), vocab_size=50257, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

# Save files to disk
tokenizer.save("./custom_t5/tokenizer.json")
from transformers import T5Config

config = T5Config.from_pretrained("google/t5-v1_1-base", vocab_size=tokenizer.get_vocab_size())
config.save_pretrained("./custom_t5")
