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

While experimenting with the dataset, I added model state_dict saving and config.json to save the model state and configuration after a KeyboardInterrupt. I also added weight tying from the input embedding `wte` and the final layer `fl`
`self.wte.weight = self.fl.weight`

I trained multiple models until the loss plateaued, all with 4 heads and 2, 4, and 6 decoder blocks.

prompt: 
> Once, there was a shark named Ben. Ben loved to eat

2 blocks:
```
Once, there was a shark named Ben. Ben loved to eat beef. They would eat carrots and lettuce. One day, Ben and his friends went to the sea. They were playing and having fun.

When they got home, Ben saw a big bear on the stove. He wanted to show Ben his friends. He asked them to come inside and play with him. Ben said yes and they all went to the kitchen.

Ben said, "Thank you, Ben! You are a good friend!"

Ben and his friends ate the dessert. They were happy and played together. They all had a lot of fun. They had a lot of fun together.<|endoftext|>
```

4 blocks:
```
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
```
6 blocks: 
```
Once, there was a shark named Ben. Ben loved to eat fish. But one day, it started to rain. Ben knew if he was not outside, he said yes.

Ben took the fish home and set a picnic. He mixed yummy food with a big bowl of food. But as he drove, he noticed that some of the water was dirty. He wanted to eat some of the fish, but he was not happy.

Ben said, "Mum, this is not nice. You should eat all day. It's not good to eat our food. Put them back to the fish."

Mum smiled and said, "Yes, Ben. We will be back soon. We cannot buy another fish for you. We can be friends."

Ben and Ben agreed and went back to the sea. They made a big splashes with the sea. Ben was happy. Now, Ben and Lily shared their fish with the fish. They were all friends and had a lot of fun together.<|endoftext|>

```
The 2 block model produces the simplest and shortest stories, but the topic drifts very quickly. The 2 block model also has little understanding of the meaning of words as it shows the shark eating beef, carrots, and lettuce. Ther stories also show litte memory of events and story structure.

The 4 block model I find the most disappointing. The model still does not really understand well the meaning of the story and the structure of sentences. While all the text it produces is grammatically correct, the stories produced are essentially picked at random from a bag of related sentences. The model also did not complete the story within the given 500 tokens, showing how much the model's stories drift topics, never finding "closure".

The 6 block model is also mostly grammatically correct, but there is no visible improvement. However, it shows a more complex story structure without drifting topics too much, which is nice to see in the bigger models.

Overall compared to the Wizard of Oz, the model performs much better, but I am still dissapointed. Next, I will be running bigger models on GPUs on Kaggle.

