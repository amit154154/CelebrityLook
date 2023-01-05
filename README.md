# CelebrtyLook

App that uses GAN inversion and MobileStyleNet to transform your face into that of a celebrity, specific text, a image of a friend, on the edge.

## Implemntation

using MobileStyleNet that designed to be efficient and fast, making it well-suited for use on mobile devices that trained with distillation of StyleGan2.

We want to have GAN inversion and manipulation on the device(phone) itself, I implemented the SOTA paper
["Bridging CLIP and StyleGAN through Latent Alignment for Image Editing"](https://arxiv.org/abs/2210.04506) (2022)

the training process looking like this:

![Demonstration of the training process](assets/mapper_training.png)

when G is the MobileStyleNet and $E_I$ distillation of openclip Large model to EffiectFormer Large and $E_T$ is openclip text encoder(dont need for the app).

## Experiments


### GAN Inversion

GAN inversion inference is very simple:  $G(f(E_I(X)))$  when f is the simple layers that are not frozen in the training.

examples: 

<div style="display: flex;">
	<img src="assets/inversion images/0.jpg" width="200" height="200"> 
	<img src="assets/inversion images/0_inversion.jpg" width="200">
</div>


<div style="display: flex;">
	<img src="assets/inversion images/2.jpg" width="200" height="200"> 
	<img src="assets/inversion images/2_inversion.jpg" width="200">
</div>

this method can do gan inversion but it is not even close to the SOTA methods to do gan inversion, it is saves some elements from the input image but it is lose almost completely the identity of the face.
### Generate faces using text
Because we use the clip image encoder to train the mapper, we can use the text encoder to create delta W+ from the mean image to image close to a given text.
look in the training process image and the paper to see how exactly it works.

texts used:
1. blonde woman with sun glasses.
2. very happy woman with black hair and glasses.
3. man with a hat and nice beard.
4. man with brown beard.

<div style="display: flex;">
	<img src="assets/generate images/blonde_woman_with_glasses.jpg" width="200" height="200"> 
	<img src="assets/generate images/black_hair_woman.jpg" width="200">
	<img src="assets/generate images/man with a hat and nice beard.jpg" width="200">
	<img src="assets/generate images/man_with_brown_beard.jpg" width="200">
</div>

### Image text manuipulation 

Because we use the clip image encoder to train the mapper, we can use the text encoder to create delta W+ from an input W+ to given text.
(the inference is in the paper)
texts used:
1. asian with pink wig.
2. blond woman with sunglasses.
3. fat man with beard.
4. man with glasses.

<div style="display: flex;">
	<img src="assets/examples/asian_with_pink_wig.gif" width="200" height="200">  	<img src="assets/examples/blond_woman_with_sunglasses.gif" width="200">
	<img src="assets/examples/fat_man_with_beard.gif" width="200">
	<img src="assets/examples/man_with_glasses.gif" width="200">
</div>

experiment that don't show in the paper is to change head position to the image using different texts.
texts used:

1. Donald Trump looking left.
2. Donald Trump looking left.
3. man looking up.

<div style="display: flex;">
	<img src="assets/head movment examples/Donald_Trump_looking_left.gif" width="200" height="200">  	<img src="assets/head movment examples/Donald_Trump_looking_left_2.gif" width="200">
	<img src="assets/head movment examples/man_looking_up.gif" width="200">
</div>

As we can see the model understand very good simple elements such as glasses, sex etc but with more complex text such as looking like celebrite its harder for it(also we as humans who see celebrties many times can see very small errors in the image), but it understands really well also head position and directions.


### Problem with text manipulation

As we saw this method can create delta W+ from an input W+ to given text. but, its hard to know how much of the delta we should add to create an image that looks like the
given text and the input W+ image representation.

The problem is that every W+ latent would need different constant C to multiply the delta and add to the latent for the best result.

All the examples below used const C between 0 and 0.8(Such that every frame C getting bigger).
example texts used:

1. man with blue eyes and a hat.
2. man with blue eyes and a hat.
3. man with glasses looking up.

<div style="display: flex;">
	<img src="assets/examples problem/man with blue eyes and a hat_2.gif" width="200" height="200">  	<img src="assets/examples problem/man with blue eyes and a hat.gif" width="200">
	<img src="assets/examples problem/man with glasses looking up.gif" width="200">
</div>

As we can see the optimal C can be change even with the same text but different latent to start with.

to solve this problem I created new method to find the optimal C for specific W+ latent and text encoding using a simple projection layer, and this is training process:

![Demonstration of the training process](assets/projection_layer_training.png)

the training process looks complicated but simply put we use clip loss to train the projection layer to find optimal C for W+ latent and text encoding.

clip loss:

$L_c = mse(C,C_I(X,X'))$ 

when $C_I$ is function to find clip similarity between to images and C is hyper-parameter when I used for the training equal to 0.7, for the best results for balance between the similarity to the input image and similarity for the text input.




