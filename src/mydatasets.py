from torch import tensor as to_tensor
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class MyDataset(Dataset):
    def __init__(self, 
          seqs, 
          word_lengths,
          PAD_TOKEN,
          MAX_LEN_MY,
      ):
        self.seqs = seqs
        self.word_lengths = word_lengths
        self.PAD_TOKEN = PAD_TOKEN
        self.MAX_LEN_MY = MAX_LEN_MY
        assert (
            len(self.seqs) == len(self.word_lengths)
        ), "Length unmatched!"
    def __getitem__(self, idx: int):
        return (self.seqs[idx], 
                self.word_lengths[idx])
    def __len__(self):
        return len(self.seqs)
    def collator(self, batch):
        units, lengths = list(zip(*batch))
        units_padded = pad_sequence(units, 
            batch_first=True, 
            padding_value=self.PAD_TOKEN)
        units_padded = units_padded[:, :self.MAX_LEN_MY, ...]  # truncation, also for speech
        attention_mask = (units_padded != self.PAD_TOKEN)  # FIXME: why list?
        lengths = to_tensor(lengths).long()
        return {
            "input_ids": units_padded,
            "attention_mask": attention_mask,
            "word_lengths": lengths,  # need shorten of trunc?
        }
