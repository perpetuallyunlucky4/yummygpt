# yummygpt
A GPT-style Decoder only language model\
The code also allows for GPU to be used.

### Training on the Wizard of Oz
The first test was with a Wizard of Oz text file with no endoftext tokens, I used 
```
"d_model": 256,
"context_length": 256,
"n_heads": 4,
"n_blocks": 2
```
and generated incoherent and grammatically incorrect sentences, but words are seen, including "Dorothy", "The Tin Woodman", "Zeb", and "The Prince":

<details>
    <summary>Wizard of Oz</summary>
sun--our not make the palace," said Dorothy, "if you untied him, he
she will bearers of the Sorcerer.
to be the Wizard that the Wizard returned the sorceries you are able to
without happy, he knows.
</details>
<details>
      <summary>Wizard of Oz</summary>
At altogether; so They are a young girl could open about it. And you can't do not seem to eat the Emerald

"They are from the only you the Tin Woodman and bunting, and passed.it."
</details>
<details>
      <summary>Wizard of Oz</summary>
"That's true," said Zeb.
In the balloon, with a light into a cleverly through the air.
</details>
<details>
      <summary>Wizard of Oz</summary>
He will
"And we do if you must be planted at once come to go we belong there," the Prince.

center the earth," explained the girl. "We wouldn't defeated us yet, for we been
the people."
</details>

I am overall very happy with the model's performance, with a final loss of 2 - 2.5 after only 10000 epochs.

### Training on the TinyStories dataset
To download the dataset and create a text file, run load_TinyStories.py. The code downloads the first 12000 stories of the "train" dataset, appends eos tokens to the end of every story and saves it to TinyStories.txt

While experimenting with the dataset, I added model state_dict saving and config.json to save the model state and configuration after a KeyboardInterrupt. I also added weight tying from the input embedding `wte` and the final layer `fl`
`self.wte.weight = self.fl.weight`

I trained multiple models until the loss plateaued, all with 4 heads and 2, 4, 6 and 10 decoder blocks.

prompt: 
> Once, there was a shark named Ben. Ben loved to eat

<details>
  <summary>2 blocks</summary>
Once, there was a shark named Ben. Ben loved to eat beef. They would eat carrots and lettuce. One day, Ben and his friends went to the sea. They were playing and having fun.

When they got home, Ben saw a big bear on the stove. He wanted to show Ben his friends. He asked them to come inside and play with him. Ben said yes and they all went to the kitchen.

Ben said, "Thank you, Ben! You are a good friend!"

Ben and his friends ate the dessert. They were happy and played together. They all had a lot of fun. They had a lot of fun together.<|endoftext|>
</details>
<details>
<summary>4 blocks</summary>
Once, there was a shark named Ben. Ben loved to eat fish. One day, Ben saw a big shell in the park. He wanted to help the shell.

Ben said to the shark, "Don't worry, Ben. Let's go to the park and we can find a way to find a way to get there."

Ben said, "OK, let's go to the park."

They went and saw a big hill. They shouted, "Mom, I am the sun!" Ben said, "But there is a big, white cloud. The cloud is deep and scary. It will be dangerous."

Ben was scared, but he thought the cloud were dangerous. He did not like the rain. He reached for the bird. The cloud saw Ben and got his trophy. He was happy.

Ben was not scared anymore. He wanted to see the cloud. He said, "This cloud is a good spot. It is big and hard."

His mom said, "No, Ben. It's not safe. You have to be careful. You have to listen to me. You are not good. You are in trouble. You can't eat the sunset. You are not yours. You are just friends. You have to be careful. You are not right. You are not like you. You are not yours. You are not ignorant. You should listen to the cloud. You have to be careful and listen to me. You have to be careful and respectful. You are not nice to drop the clouds. You are not sorry. You are not the sun. You are in the sky. You don't have to go back to the cloud. You should not touch the clouds. But you should not touch the cloud. And don't touch the clouds."

Ben was not angry. He did not want to play with the rain, but he was not happy. He was just curious. He wanted to have a shelter. He wanted to see if he could go up and see what is on the rain.

Ben walked towards the cloud and look at the sky. It is not a sun, or a rainbow. It has a rainbow colors. It is a tree with a rainbow. It can make it grow bigger and bigger and bigger. It makes a rainbow. Ben and Lily are sad.

They hoped the rain would not go. They hoped the rain would go on the coats. They hoped the sky would come back. They hoped the rain would
</details>
<details>
  <summary>6 blocks</summary>
Once, there was a shark named Ben. Ben loved to eat fish. But one day, it started to rain. Ben knew if he was not outside, he said yes.

Ben took the fish home and set a picnic. He mixed yummy food with a big bowl of food. But as he drove, he noticed that some of the water was dirty. He wanted to eat some of the fish, but he was not happy.

Ben said, "Mum, this is not nice. You should eat all day. It's not good to eat our food. Put them back to the fish."

Mum smiled and said, "Yes, Ben. We will be back soon. We cannot buy another fish for you. We can be friends."

Ben and Ben agreed and went back to the sea. They made a big splashes with the sea. Ben was happy. Now, Ben and Lily shared their fish with the fish. They were all friends and had a lot of fun together.<|endoftext|>
</details>
<details>
  <summary>10 blocks</summary>
Once, there was a shark named Ben. Ben loved to eat beef. One day, he went to the market with his friend, a little girl named Sue. Sue had a big jar of oats. "Look, Tommy!" she said. "Do you want more oats?"

Tom smiled and said, "Yes, please!" He went up to the store to get some more cheese. Sue wanted one of the oats, but it was too big. They both decided to leave the store.

As they were looking around, they saw many small apples sitting on a big table. They were sad, but they could not find a fun place. They went home and found a big, juicy worm. Tom and Sue were happy that they could have fun with the big, red ball. They continued to play and find new things in the garden.<|endoftext|>
</details>

For the 10 block model, I increased the context length and d_model to 512. While the stories are all terrible, you can see that the storyline is much more consistent in the last model than in the 2 block model.

### Pre-Training, the Fineweb-edu dataset, and a conversational model
To load the dataset, I had to tweak the data loading process. Due to the model's large size, the text file is too big to be loaded at once, so I tokeize in chunks and write it into a .bin file full of tokens, and load it using np.memmap.

Run load_fineweb.py to load the binary. load_TinyStories now also uses the same pipeline for consistency.

Next, I needed to add conversational finetuning to the model, so load_ultrachat.py and load_axiom.py will load two datasets: ultrachat is from huggingface, and AXIOM is a test dataset I created myself. The formats are different, so I had to use different loaders. Also, the dataloaders for finetuning is different. The entire pipeline is changed to include the ends of conversations so that DataLoader2 only samples within conversations in training. Use DataLoader2 for finetuning -- read the comments in the code. load_axiom.py and load_ultrachat.py produces three files: inputs, targets, and indexes. Targets are inputs shifted by one, and indexes tell DataLoader2 where to sample within the dataset. Numpy memmaps are used for everything now because of the much larger datasets. More information on the data pipeline in appendix A.

### Final models
#### fineweb1024_1216_v1, 2, 3, 4 -- pretraining
architecture:
```
"d_model": 1024,
"context_length": 256,
"n_heads": 16,
"n_blocks": 12
```

Some generations from the v4 model:
```
rep penalty: 1.1
temperature: 0.8
top p: 0.8
```

<details>
    <summary>
        The capital of Germany is
    </summary>
vernacular German.
Occupation of the city has also been extremely slow going, especially in rural areas like Ewe-ben-Gurion in France. People from all over the country celebrate the anniversary with their own coats and accessories, and with costumes. In Ewe-ben-Gurion, however, there is no official celebration always, which means foreigners are actually enjoying a special weekend on their way to work. The municipality also raises some concerns about food safety as they have more than five hundred health professionals who work in the public premises. To ensure proper food safety, the city of Luxemburg is carefully investigated, and it was decided by the government on how to keep food safe during the holiday season.
In this regard, as in the above image, the town of Lind and the surrounding hills undergo a series of events right along its street. After the event, it is decided that a ��county council�� will be created in order to ensure uniformity of the four-month FAF holiday season.
Useful activities in the City of Lind and the surrounding mountains
On a daily basis, the mayor receives regular updates on his plans for the year, and residents are reminded that the month day for the special holiday season is June 11th. Since the city of Lind was not created until 1849, the mayor has asked the city people to send him a copy of the city map and to submit the city maps so that he can live on the same date as the next major state capitol.
After the Local Minister has released actual plans for the year, residents and guests enter the city and check details of each other before their arrival at the city. As well as Type 2 Diabetes and Diet Plan of the city, these require additional time outside to travel to the carbon neutral building site, to complete their meals, and to get enough exercise to help them maintain weight for the entire length of time. Thus, the Mayor of Lind and the town of Bern will send up proof of concept for the project to take place.
Local citizens celebrate the very first day of the first week of May. Each Wednesday they visit the albino site at the other end of the park, which includes the music hall. There are also most mornings where the mayor visits the plantings before beginning the project. They plan on spending two weeks in the tree planting, drawing plants from their garden in April and May.
As one of the city��s main attractions,
</details>

<details>
    <summary>
        Baking a cake requires many steps: 
    </summary>
__________________________________________________________________________. _______________________________________Freezing creates a bond between two parties.
. The first two items can be expressed as following (i) __________________. The second item is a simple object with a definite path between them, and (ii) __________________________.
The third item can be expressed as following (i) __________________________.
. The third item is a simple object with an definite path between them, and (iii) ______________________________________. The fifth item is a simple object having a definite path between them. The last item is a complicated object containing a descriptive path between them. The last item is the subject.
. The fourth item is an acronym of the subject. It is an abbreviation of the subject. It denotes a name of the object.
Different promotional channels are used to represent different types of content and products (such as multimedia or graphic). For example, advertising gives us different sorts of content, such as pictures and videos without text, music, movies or other media that we listen to instead of watching television. Advertising allows us to identify images and messages, which in turn helps to make our services more appealing.
. The fourth item is an acronym of the subject. This is always pronounced differently depending on the context in which the information was presented.
The content should have both positive and negative connotations, as it is often seen as an expression of emotionality. However, this sentence usually means ��useful�� or ��impossible�� in its context. So, if you are asked to describe something in terms of how you perceive yourself, you may sometimes respond only to a negative connotation. If you think about how you react in the first place, you might need to explain it to someone else. Negative connotations could include ��as good as you feel good,�� ��conviction is worthwhile,�� or ��detachment is necessary.��
However, when speaking about a kind of content, there is usually more than one type of content. For example, a noun is an expression of emotions. A noun might have different meaning depending on the context in which it is presented. It may also have several meanings.
A verb most commonly refers to a verb or word that describes a fact (e.g., the action, action, etc.) or a concept (e.g., ��the people you��re talking to are friends.��
</details>

<details>
    <summary>
        Winston Smith is the main protagonist of George Orwell's book 1984. 
    </summary>
    A short video interview with Harold Gumman (one also available here) looks at "George Orwell" as a fictional character who is portrayed in various ways. His understanding of how he writes a simple, unscripted letter to his wife, and how Orwell uses a variety of media on their work can help unravel this mystery.
'The Conservative Party and the Anti-Fascism Movement'
In his book, The Conservative Party and the Anti-Fascism Movement, Stephen Burke describes the movement's vision for the British public after World War I to help counter any claims that it was opposed to any ideas about British oppression. He argues that the Party was successful in bringing British power back into being.
In an interview with Richard Nixon on May 30, 1992, he quotes and explains how he thinks the party could be right along with the government and the people.
There are four main categories of views. Firstly, the party is the very opposite of the party itself as opposed to the party itself. This is because there is only one way to organise a party. Secondly, the party is a party designed to prolong the period of time between the foundations of the country and the state. Thirdly, the party is the opposite since it is merely a political party. Fourthly, the party does not have such support from both sides - something which was never seen before. Thirdly, the party is too dependent on the government to properly govern its affairs. Lastly, the party has the right to organize and promote welfare issues in socialist countries.
However, the party itself does not have this support from the government and the governments. However, the party has the right to make certain laws respecting the government. These include:
- The law that the government provides that this government should act accordingly;
- If the government's policy is against those policies, then it will exclude them from the regime; and
- If the government fails to address those policies, then it will violate anti-fascism principles, while losing the majority of its support.
Why the Party has lost all traction
John Lennon's book describes how it has lost a significant amount of traction over the last two decades. It includes the party name, James Buchanan, most notably Harry Reid.
"While Sam Durand' Communism is widely criticised now, it has comeback against many thought leaders today," Kennedy said at the 2016 audience event. "As Bill Kennedy said in the 1960s, 'A
</details>

As seen, the model can make fairly good text, but the topic and focus is not there yet. Finetuning is supposed to help with this.
#### ultrachat1024_1216_v41, 42 -- finetuning

#### axiom models -- further finetuning/alingment

### Improvements for the next iteration
#### Rotatory Positional Encodings
I've already started working, and I will definitely add it next time

#### SwiGLU
Honestly don't understand the logic behind this, but adding complexity to the model should really help

#### RMSNorm
Another thing I don't understand, but it seems cool and something else that can be easily and immediately implemented

#### MOE (Mixture of Experts)
The coolest one on the list, I will add it once I like the performance of the model on its own and when I get access to better computers probably

## Appendix A -- The AXIOM dataset format
The format I used for AXIOM is simple:
```
USER
user text here

ASSISTANT
assistant text here

USER
continued comversation

ENDTEXT
```

## Appendix B -- Data pipeline and training steps for conversational model

### Pretraining on fineweb
The data is loaded from huggingface into a .bin file in tokenized form (`load_fineweb.py`). DataLoader then takes in a numpy memmap of the binary and slices context_len tokens, and generates input and target batches from there (shift inputs by one to get targets). Model is then trained on the data (`train_model.py`) and saved.

### Finetuning on Axiom and UltraChat
Data is loaded from huggingface again. This time, it is loaded into three separate files: inputs, targets, and index (`load_axiom.py`, `load_ultrachat.py`). The files both take in different formats of data, and split the data into conversations. The index of each conversation in the binary as well as the length of the tokenized conversation are both saved into the index file. The inputs and targets are generated into separate files. DataLoader2 Then takes in all three files and samples WITHIN the conversations, which is different from the fineweb process. Dataloader2 then pads the conversations if the entire conversation is too short (pad is <system>). DataLoader2 then produces the same input and target batches and trains.

