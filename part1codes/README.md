# The followings describe what happens here.

## 1: Training Dataset Used for the Model

For our training dataset, we have road images from three countries (see Fig.1), which are Japan, India and Czech. For data in Japan, there are 9053 images from 2018 collections and 13133 images from 2019 collections. For data in India and Czech, there are nearly 9892 images and 3595 images. We can also see that for different types of cracks, there are different number of data. In addition, we can also see the types of road damages in Fig.2. We can see that there are many different damage types such as damages about constriction joint part, white line blur, and equal interval. For training dataset, we will have GT labels for each image data (see Fig.3). There are two GT labels that we have, the first one is the position of the crack region boundary box, the second one is the type of cracks in images.


<p float="left">
<img src="https://github.com/WallaceSUI/Detroit-City-Road-Assessment/blob/main/part1codes/reportdata/fig1.png" width = "33%" /><img src="https://github.com/WallaceSUI/Detroit-City-Road-Assessment/blob/main/part1codes/reportdata/fig2.png" width = "33%" /><img src="https://github.com/WallaceSUI/Detroit-City-Road-Assessment/blob/main/part1codes/reportdata/fig3.png" width = "33%" />
</p>


## 2: Testing Dataset Used for the Model

Since we need to apply our model directly into real-world dataset from Detroit city, we conduct data mining on the whole map from the streets of Detroit city and extract images from each street (see Fig.4). Since the map is very huge, we first extract 1.5% of the entire dataset, which has 45826 images. We can see that each image comes from a combination of multiple views and they contain not only street information, but also information about the cars, the air, and trees. Based on these conditions, it is necessary to conduct data prepossessing on the raw dataset to let our model have a better performance.

<center class="half">
<img src="https://github.com/WallaceSUI/Detroit-City-Road-Assessment/blob/main/part1codes/reportdata/fig4.png" width = "33%" />
</center>

## 3: Image Data Preprocessing

In the original image data, what we can see is not from a single view. It seems like a combination of different views (see Fig.5). So the first thing that we need to do is to transform the view in each image. We use NFOV view transformation for this part of work. After the NFOV view transformation (see Fig.5), we can see that the new data only has one view inside it and the image seems like more normal now.

Then we can see that for these image data, there are still some problems here. For example, we can see the trees and cars in the street, but we cannot consider any cracks on these objects. Based on these conditions, it is better that we can have a mask to cover the region outside the roads, which can make our model focus on only road region. After the manipulation, we can see some examples in Fig.6. We can find that after we add masks into the original images, output images can only have the clear region for the street and all the things outside the street are masked.

<p float="left">
<img src="https://github.com/WallaceSUI/Detroit-City-Road-Assessment/blob/main/part1codes/reportdata/fig5.png" width = "33%" /><img src="https://github.com/WallaceSUI/Detroit-City-Road-Assessment/blob/main/part1codes/reportdata/fig6.png" width = "33%" />
</p>

## 4: Model Design and Results
There are many different frameworks and model structures that we can use for this part. We first tried different combination of the framework and model structure. Then we find that combining SSD with mobilenet is a good choice here. The key idea in the SSD framework is the one-shot mechanism (see Fig.7). Compared with other methods, one-shot mechanism can make SSD framework to be very fast on any condition, Besides this, SSD also has two other novel designs. The first is using different multi-scale of feature maps during the training, and the second is using different multi-scale of prior boxes for detection. These designs can help the model detect not only big objects but also small objects in the image.

The mobilenet is another design which considers using the depthwise separable convolution (see Fig.7). Different from the normal convolutional layer, mobilenet can apply different kernels on different channels. This can help to reduce a lot of time for training but remain the similar performance.

After combining these two structures together (see Fig.8, our model can directly predict the crack regions in our testing dataset. We can see that for each road image, our model can predict the crack’s bounding box in the image with its crack type, confidence score and the position information of the bounding box.

<p float="left">
<img src="https://github.com/WallaceSUI/Detroit-City-Road-Assessment/blob/main/part1codes/reportdata/fig7.png" width = "33%" /><img src="https://github.com/WallaceSUI/Detroit-City-Road-Assessment/blob/main/part1codes/reportdata/fig8.png" width = "33%" /><img src="https://github.com/WallaceSUI/Detroit-City-Road-Assessment/blob/main/part1codes/reportdata/fig9.png" width = "33%" />
</p>
