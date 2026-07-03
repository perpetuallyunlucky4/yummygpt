# yummygpt
A GPT-style Decoder only language model\
The model uses tiktoken's `gpt-2` tokenizer and `'<|endoftext|>'` as its eot_token and AdamW as its optimizer.
The code also allows for GPU acceleration(untested).

## Training on the Wizard of Oz
The first test was with a Wizard of Oz text file with no endoftext tokens, I used 
```
"d_model": 256,
"sequence_length": 256,
"n_heads": 4,
"n_blocks": 2
```
and generated incoherent and grammatically incorrect sentences, but words are seen, including "Dorothy", "The Tin Woodman", "Zeb", and "The Prince":

```
sun--our not make the palace," said Dorothy, "if you untied him, he
she will bearers of the Sorcerer.
to be the Wizard that the Wizard returned the sorceries you are able to
without happy, he knows.
```
```
At altogether; so They are a young girl could open about it. And you can't do not seem to eat the Emerald

"They are from the only you the Tin Woodman and bunting, and passed.it."
```
```
"That's true," said Zeb.
In the balloon, with a light into a cleverly through the air.
```
```
He will
"And we do if you must be planted at once come to go we belong there," the Prince.

center the earth," explained the girl. "We wouldn't defeated us yet, for we been
the people."
```

I am overall very happy with the model's performance, with a final loss of 2 - 2.5 after only 10000 epochs.

## Training on the TinyStories dataset
To download the dataset and create a text file, run load_TinyStories.py. The code downloads the first 12000 stories of the "train" dataset, appends eos tokens to the end of every story and saves it to TinyStories.txt





