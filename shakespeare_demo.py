#%%
import glob
import yaml
import torch as t
from torch.utils.data import Dataset
import gradio as gr
from typing import Optional, Union
import requests
import re
import sampling
import transformer_replication
#%%
device = 'cuda' if t.cuda.is_available() else 'cpu'
#%%
class WordsDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        text = self.texts[idx]
        sample = (text, label)
        return sample

#%%
def tokenize(text):
    return re.split(r"\b", text)

def _remove_duplicates(text, string=" "):
    if string + string in text:
        text = text.replace(string + string, string)
        return _remove_duplicates(text, string)
    return text

def remove_duplicates(text):
    text = _remove_duplicates(text, ' ')
    text = _remove_duplicates(text, '\n')
    return text

# %%
class WordData():
    def __init__(self, text, start, end):
        self.complete_text = remove_duplicates(text)
        if start is not None and end is not None:
            self.complete_text = self.get_excerpt(start, end)
        self.complete_tokens = tokenize(self.complete_text)
        self.vocab = sorted(set(self.complete_tokens))
        self.token_to_id = dict(zip(self.vocab, list(range(len(self.vocab)))))
        self.id_to_token = dict(zip(list(range(len(self.vocab))), self.vocab))
        self.model_max_length = None

    @staticmethod
    def from_link(link, start=None, end=None):
        return WordData(requests.get(link).content.decode('utf-8'), start, end)
    
    @staticmethod
    def from_file(filename, start=None, end=None):
        with open(filename, encoding='utf-8') as f:
            text = f.read()
        return WordData(text, start, end)

    def get_excerpt(self, start="THE SONNETS", end="THE END", text=None):
        if text is None:
            text = self.complete_text
        assert start in text, f'get_excerpt: cannot find {start} in text'
        l_stripped = text.split(start, maxsplit=1)[1]
        assert end in l_stripped, f'get_excerpt: cannot find {end} in text'
        r_stripped = l_stripped.split(end, maxsplit=1)[0]
        return r_stripped

    def generate_autoregressive_dataset(self, sequence_length, text=None):
        self.model_max_length = sequence_length
        if text is None:
            text = self.complete_text
        token_ids = self.encode(text, return_tensors="pt")
        inputs = [token_ids[i:i + sequence_length] for i in range(len(token_ids) - sequence_length)]
        labels = [token_ids[i + 1:i + 1 + sequence_length] for i in range(len(token_ids) - sequence_length)]
        return WordsDataset(inputs, labels)

    def encode(self, initial_text: str, return_tensors: Optional[str] = None) -> Union[list, t.Tensor]:
        '''
        Tokenizes initial_text, then returns the token ids.

        Return type is list by default, but if return_tensors="pt" then it is returned as a tensor.
        '''
        tokens = tokenize(initial_text)
        token_ids = [self.token_to_id[t] for t in tokens]
        if return_tensors == "pt":
            return t.tensor(token_ids, device=device)
        return token_ids

    def decode(self, list_of_ids: Union[t.Tensor, list]) -> str:
        '''
        Converts ids to a list of tokens, then joins them into a single string.
        '''
        tokens = [self.id_to_token[int(i)] for i in list_of_ids]
        return "".join(tokens)
#%%
#%%
shakespeare = WordData.from_file('100-0.txt', start="1\n", end='ALL’S WELL THAT ENDS WELL')
# shakespeare = WordData.from_link('https://www.gutenberg.org/files/100/100-h/100-h.htm', start="1\n", end='ALL’S WELL THAT ENDS WELL')
print('Vocab size: ', len(shakespeare.vocab))
#%%
#%%
with open('config.yaml', 'r') as f:
    yaml_cfg = yaml.safe_load(f)
#%%
with open('model_state_dict.pt') as f:
    state_dict = t.load(
        'model_state_dict.pt'
    )
#%%
base_config = transformer_replication.TransformerConfig(
    num_layers=yaml_cfg['num_layers']['value'],
    num_heads=yaml_cfg['num_heads']['value'],
    vocab_size=len(shakespeare.vocab),
    hidden_size=yaml_cfg['hidden_size']['value'],
    max_seq_len=yaml_cfg['max_seq_len']['value'],
    dropout=yaml_cfg['dropout']['value'],
)
shakespeare.model_max_length = yaml_cfg['max_seq_len']['value']
model = transformer_replication.DecoderOnlyTransformer(base_config)

model.load_state_dict(state_dict)

#%%
def generate(
        text: str, max_tokens: int, temperature: float, 
        top_k: int,
) -> str:
    return sampling.sample_tokens(
        model, 
        shakespeare, 
        text, 
        max_tokens_generated=max_tokens, 
        temperature=temperature, 
        top_k=top_k,
    )

#%%
def safe_generate(
        text: str, max_tokens: int = 300, temperature: float = 1.0, 
        top_k: int = 20,
    ) -> str:
    try:
        return generate(
            text, max_tokens=max_tokens, temperature=temperature, top_k=top_k
        )
    except KeyError as e:
        return f"I'm sorry, {str(e)} is not in Shakespeare's vocabulary"
#%%
examples = [
    ["I sang a beautiful song"],
    ["To be free is to"],
    ["How I love thee"],
]
#%%
print(safe_generate(examples[-1]))
#%%

demo = gr.Interface(
    fn=safe_generate,
    inputs=[
        gr.components.Textbox(lines=5, label="Input Text"),
        gr.components.Slider(
            label='max tokens generated', minimum=1, maximum=1000, 
            value=300, step=1,
        ),
        gr.components.Slider(
            label='temperature', minimum=0, maximum=2, value=1, step=0.1,
        ),
        gr.components.Slider(
            label='top_k', minimum=1, maximum=100, value=10, step=1,
        ),
    ],
    outputs=gr.components.Textbox(label="Generated Text"),
    examples=examples
)
#%%
demo.launch()
# %%
'''
FIXME:
* cut everything after sonnet break
* deploy to heroku
* link from github home
'''
