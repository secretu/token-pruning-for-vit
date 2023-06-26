from models.modeling_bert import CoFiBertForSequenceClassification
import torch
from torchprofile import profile_macs
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm

if __name__ == '__main__':
    model = CoFiBertForSequenceClassification.from_pretrained("bert-base-uncased")
    seq_len = 128
    input_ids = torch.randint(0, 100, (1, seq_len))
    attention_mask = torch.ones_like(input_ids)
    token_type_ids = torch.zeros_like(input_ids)
    macs = profile_macs(model, (input_ids, attention_mask, token_type_ids))
    print(macs)

    # macs_list = []
    # seq_len_list = list(range(1, 4096 + 1024, 64))
    # for seq_len in tqdm(seq_len_list):
    #     hidden_states = torch.randn(1, seq_len, 768)
    #     attention_mask = torch.ones(1, seq_len)
    #     macs = profile_macs(model.bert.encoder.layer[0], (hidden_states, attention_mask))

    #     # input_ids = torch.randint(0, 100, (1, seq_len))
    #     # attention_mask = torch.ones_like(input_ids)
    #     # token_type_ids = torch.zeros_like(input_ids)
    #     # macs = profile_macs(model, (input_ids, attention_mask, token_type_ids))
    #     macs_list.append(macs)
    
    # plt.plot(seq_len_list, macs_list)
    # plt.xlabel('Sequence Length')
    # plt.ylabel('MACs')
    # plt.grid()
    # plt.savefig('macs.png')
