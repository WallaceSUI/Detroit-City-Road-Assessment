# The followings describe what happens here.

## 1: Training Dataset Used for the Model

For our training dataset, we have crack shapes dataset under a very close view (see Fig.10). The first rows are the examples of training dataset. They have a very close view into each crack shape. And the second row is GT labels for each images. We can see that for each image in the first row, we will have a very clear GT data in the second row. This can guarantee that our model can have a good understanding about the crack’s shapes.

<p float="left">
<img src="https://github.com/WallaceSUI/Detroit-City-Road-Assessment/blob/main/part2codes/reportdata/fig10.png" width = "33%" />
</p>


## 2: Testing Dataset Used for the Model

For our testing dataset, we hope to capture the accurate crack shapes in the images. Therefore, we directly use the bounding box results from our last detection and use them as our input data (see Fig.11). We can see that in each image, there is a different type of crack shape. We hope our model can find these crack shapes so that we can have a good estimation about the size of cracks.

<p float="left">
<img src="https://github.com/WallaceSUI/Detroit-City-Road-Assessment/blob/main/part2codes/reportdata/fig11.png" width = "33%" />
</p>

## 3: Model Design and Results
For this part, we use the latest state-of-the-art method, which is [15]. In this paper, the whole framework is in Fig.12. The first layer is a bottom-up structure, which means that each convolutional layer will extract different deep features from it. After the first layer, there is a feature pyramid. The feature pyramid receives the deep features from the last layer and combine them together to extract some potential connections between each feature. Then after the side network and hierarchical boosting, we combine features to output the final prediction result.

Based on this model, we input our bounding boxes results from the last detection and check results (see Fig.13). We can see that our model cannot predict the accurate crack shapes in images. These prediction results are far away from the real crack shapes in images.

<p float="left">
<img src="https://github.com/WallaceSUI/Detroit-City-Road-Assessment/blob/main/part2codes/reportdata/fig12.png" width = "33%" /><img src="https://github.com/WallaceSUI/Detroit-City-Road-Assessment/blob/main/part2codes/reportdata/fig13.png" width = "33%" />
</p>

## 4: Analyze reasons for the problem
Based on the problem we met in results of testing dataset, we try to analyze what happens here during the testing. We compare the training data with the testing data in Fig.14. We can see that for the training dataset, all the crack shapes are concave, which means that there is only one type of crack in the dataset. However, if we look at the examples in testing set, there are many different types of cracks in the image. Some cracks are raised, some are not cracks but damage on the road like white lines. This is why our model cannot handle our testing situation even if we use the latest best model for crack shapes detection.

<p float="left">
<img src="https://github.com/WallaceSUI/Detroit-City-Road-Assessment/blob/main/part2codes/reportdata/fig14.png" width = "33%" />
</p>
