from tests.test_tokenizer import get_tokenizer_from_vocab_merges_path, VOCAB_PATH, MERGES_PATH
import numpy as np

tokenizer = get_tokenizer_from_vocab_merges_path(
    vocab_path=VOCAB_PATH,
    merges_path=MERGES_PATH,
    special_tokens=["<|endoftext|>"],
)

train_text = "/sda1/szl/cs336/TinyStories_train_small.txt"
val_text = "/sda1/szl/cs336/TinyStoriesV2-GPT4-valid.txt"
train_tokens_path = "/sda1/szl/cs336/ts_train.npy"
val_tokens_path = "/sda1/szl/cs336/ts_val.npy"


def tokenize_file(input_path: str, output_path: str, log_every: int = 10000):
    all_ids = []
    total_lines = 0
    total_tokens = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            ids = tokenizer.encode(line)
            all_ids.extend(ids)

            total_lines += 1
            total_tokens += len(ids)

            if total_lines % log_every == 0:
                print(
                    f"[{input_path}] lines={total_lines} tokens={total_tokens}",
                    flush=True,
                )

    arr = np.array(all_ids, dtype=np.int32)
    np.save(output_path, arr)
    print(f"saved to {output_path}, shape={arr.shape}", flush=True)


tokenize_file(train_text, train_tokens_path)
tokenize_file(val_text, val_tokens_path)
