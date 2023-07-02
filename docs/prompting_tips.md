# Prompting Tips

Similar to large language models, vision-language models benefit from thoughtful and specific prompting. Here are a few tips to help you get what you're looking for:

- **Be aware of all meanings of a word:** If you ask for are looking for a breed of cat and request "a sphinx", you're likely to also get images of the Egyptian statue. Instead, qualify what you're looking for, e.g. "a sphinx, a breed of cat".

- **Ask for the type of media you want:** You'll notice that if you type "a dog", we automatically transform the prompt to say "a photo of a dog". This is because if you only request "a dog" you may get cartoons, clip art, etc. However, if you actually do want those things, change the prompt to ask for them specifically.

- **Be aware of biases:** When you ask CLIP for "a person smiling", nearly all of the results are women smiling. If you use this for a classifier, your model may simply classify all women as "a person smiling". To mitigate this, you may try other prompts (for example "a human smiling" could work better), or you could create two classes (one for "a woman smiling" and one for "a man smiling") and then deal with the combining logic in your app. Future versions of Zeroshot will support multiple prompts in one super-class.

- **Include negative classes:** By default, Zeroshot include an "other" class with a variety of images to provide negative examples. However, if this isn't specific enough, try including a different class of things you *don't* want to see. For example, if you specifically want to classify "a Fedex truck", include another class for "a truck" so that the model understands you want a specific type of vehicle.

- **Don't get too complicated:** The CLIP model isn't powerful enough at language to understand extremely specific prompts. It also sometimes does badly with negatives, such as "not a photo of a dog". Future versions of zeroshot will support more powerful search models, but for now don't try and get too fancy.
