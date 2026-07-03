# yummygpt
A GPT-style Decoder only language model
The model uses tiktoken's 'gpt-2' tokenizer and '<|endoftext|>' as its eot_token. 

##Training on the Wizard of Oz
The first test was with a Wizard of Oz text file with no eot tokens, i used 
```
        "d_model": 256,
        "sequence_length": 256,
        "n_heads": 4,
        "n_blocks": 2
```
and generated incoherent and grammatically incorrect sentences, but words are seen, including "Dorothy", "Kansas", "Zeb", and "The Prince":
