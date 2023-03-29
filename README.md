# Maalexi-Internship-Visual-Inspection-of-Onions

![image](https://user-images.githubusercontent.com/70502367/228473220-65e51743-811c-4bd9-87b3-50860d9f608b.png)


```
In collaboration with
```

![image](https://user-images.githubusercontent.com/70502367/228473536-75aee696-750c-487a-81fe-6f17427d6189.png)


```
Internship Report on
```


## Create Tech Prototype

### cargo inspections of Agro-commodities at ports

```
Presented by
```
###### VISHWESHWAR PARAMESHWAR BHAT

###### AYUSHMAN HAZARIKA

###### ROHAN BHAGWAT

###### T. AISHWARYA

#### Along with Internal Mentor

#### Dr. Prashant P Patavardhan

## Problem Statement

“We would like to reduce the cost and time required to do cargo inspections of Agro-

commodities at ports - perishables like onions, and mangoes, and semi-perishables like rice

and wheat flour. Currently, these are done manually by experts who charge tens of thousands

of rupees a day and are too expensive for smaller exporters.

We would like to see a small-scale prototype of an automated or semi-automated solution.

Check for quality of product, accuracy of labelling or both.”

We would want to start by first taking an individual item, and the item that we have considered

is **onion**. We aim at creating an automated solution to solve the problem mentioned above and

to check the quality of the onions.

## Introduction

Considering the present trends in the cargo inspection industry, it is observed that lots of man

power is needed to do the inspection. This not only wastes the time but also results in

inaccuracy, and charges a lot of money. There are various companies which provide cargo

inspection services, and they charge very high fees. To eliminate this, we have designed an

automated system, which can we used to determine the quality of goods without any hassle and

manpower.

Automation improves the nation's quality, production, and economic growth in agriculture

science. The selection of fruits and vegetables has an impact on the export market and quality

assessment. The most important sensory quality of fruits and vegetables is their look, which

influences their market value as well as consumer preference and selection.

Images are the most fundamental approach in the physical classification of food products and

the agricultural business for representing conception for the human brain. Visual quantification

of the elements affecting fruits and vegetables is possible, but it is time-consuming, expensive,

and vulnerable to subjective judgement and physical influence. These inspections and the

"best-if-used-before date" help establish market values. The quality assessment was carried out

by skilled human investigators using their senses of touch and sight.

This approach is highly erratic, capricious, and rarely yields consistent results across

investigators. Machine vision systems are ideally suited for traditional analysis and quality





assurance in this sort of environment since it is a continuous task to analyse fruits and

vegetables for various aspect criteria. Computer vision systems and image processing are a

rapidly expanding study field in agriculture.

Computer vision-based pattern recognition and image processing are established techniques

for safety and quality analysis of many agricultural applications, and information science is a

field that is expanding quickly.

## Methodology

### Dataset

1\. A dataset of onions was considered for training the model for the project which was

created from scratch. The created dataset for the model to train is made using images

taken from a light box under accurate conditions.

2\. For creating a perfect dataset with minimal noise and disturbance we use the light box

it is a box with translucent sides and white backdrop]. The box is intended to be empty

so that we can place the onions inside the box which helps to take a photo and get a

result that has professional-quality lighting, with no shadows disrupting the plain, even

background.

3\. For the model to be trained properly we need both types of onions [good and bad]. We

place the onions in different angles to get a proper dataset which can be used in real-

time applications.

4\. Similarly, we take photos for both the types of onions. The dataset created has around

800 images of onion as input for training the system. The proposed system is

implemented using Python and Jupyter notebook and the data-set of images is taken in

.jpg format.

### Model Pre-processing and Training Phase

1\. In the training phase of the model, after creating the dataset for the model to train, the

dataset images are pre-processed which is needed for eliminating the unwanted

information from the image. Using image processing techniques, higher accuracy can

be obtained from the given images.





2\. In the pre-processing step, the images are first normalised through normalisation for

reducing the data loss of the images, then gray scaling of image to reduce the

complexities related to computational requirements, ( grayscale means that the value

of each pixel represents only the intensity information of the light).

3\. Such images typically display only the darkest black to the brightest white. We use

structuring element which is a binary image (consisting of 1’s and 0’s) that is used to

probe an image for finding for finding the region of interest.

4\. The pattern of 1’s and 0’s specifies the shape of structuring element. Then the images

are introduced to data augmentation for diversifying and increasing the dataset size

when the classifier is trained. After the pre-processing, the **features like colour, shape,**

**size, texture, etc are extracted from the images**. The extracted features are used to

train the classifier.

5\. The proposed system uses Convolutional Neural Networks (CNN), The model

architecture as the consist of **5 convolution layers**, A convolutional layer is the main

building block of a CNN.

6\. It contains a set of filters (or kernels), parameters of which are to be learned throughout

the training. The size of the **filters is usually smaller than the actual image, a drop**

**out layer with 50% drop out ratio** a [the dropout layer is a mask that nullifies the

contribution of some neurons towards the next layer and leaves unmodified all others]

, and **3 dense layers** [Dense Layer is simple layer of neurons in which each neuron

receives input from all the neurons of previous layer] and a **deep learning algorithm**,

which consists of a class of neural networks as a classifier for image recognition by a

specialised way of **processing on the grid of pixels.**

7\. This process trains the system for **classifying and grading**. For the training phase, the

images of the dataset consist of fresh and stale classes of onion for which we must train

the model.


![image](https://user-images.githubusercontent.com/70502367/228474447-81762b70-69f0-4bde-b841-282103fb27de.png)





### Model Testing Phase

1\. In the testing phase of the model, a set of testing images are given as input for the

system. These images go through the **image-processing and feature extraction**

**processes**. Based on the system classification the given input onion is graded and the

result is displayed. The training and testing phases are shown in the above block

diagram.

2\. CNN network **has three additional hidden layers,** namely an input layer, an output

layer, and middle layers that are composed of multiple convolution layers as well as

fully connected layers.

3\. The CNN uses convolution for extracting the features and flattens the values and then

uses functions such as **’ELu’ [is a function that tend to converge cost to zero faster**

**and produce more accurate results]** for further reducing the matrices. The accuracy

of the model is calculated as the ratio between the number of correct predictions to the

total number of predictions.

4\. This model is then checked to see whether it is fixed for the given application and then

the model is used for testing the inputs. With this CNN architecture, the system converts

the given input image into an array of pixel values using the convolution and after a

certain number of times the convolution is done, then the data is flattened using ‘ELu’.





The model runs through the data many times, these are called **epochs**. As the number

of epochs increases, the more the model improves to a certain extent.

5\. The accuracy of the model was determined by checking the number of test cases that

were returned correctly when tested with a particular input. The developed model

returned the value of the onion as good or bad and gave **a testing accuracy of about**

**96-97%, Precision of 96-97% and Recall factor of 95-96%.**

## Materials used and Process of obtaining the Dataset

1\. Light Box – for making the data set

2\. Camera – 48 mega pixels

3\. Tripod Stand

4\. Different types of Onions – good, bad etc

### Software requirements**

1\. Python interpreter – Jupyter Notebook

## Steps followed in obtaining the data set

1\. Set up a well-lit place for the procurement of pictures

2\. Set up the light box

![image](https://user-images.githubusercontent.com/70502367/228474872-f42f50ec-76d8-425a-b85d-ca4af7049e16.png)


3\. Fix the tripod stand at 30 CMS from the light box with the camera exactly lying on the

parallel axis of the light box’s base.

![image](https://user-images.githubusercontent.com/70502367/228475142-b090ae5a-51ec-4043-8a27-250fd1ae2106.png)


4\. Place the onions one after the other and click pictures in different angles maintaining

the same distance and lighting conditions.


![image](https://user-images.githubusercontent.com/70502367/228475631-91d6fc90-b956-4680-8e0c-ccda6e9f3042.png)


5\. Add additional lights from the corners of the light box in order to minimize shadows.


![image](https://user-images.githubusercontent.com/70502367/228475715-a97bf73a-4fba-4c36-a3ac-662a5654fef5.png)



## Code

![image](https://user-images.githubusercontent.com/70502367/228476015-6452e046-6435-43a5-ba51-6fcb8f47fe5d.png)


![image](https://user-images.githubusercontent.com/70502367/228476095-5ee9cca0-a185-468a-b7bc-ff353e153f21.png)



![image](https://user-images.githubusercontent.com/70502367/228476153-acc82ce1-4c7f-4c48-abf2-50cbdca82e57.png)


![image](https://user-images.githubusercontent.com/70502367/228476232-7b929bb0-127b-4078-aa91-d1b255fa3117.png)


![image](https://user-images.githubusercontent.com/70502367/228476360-a738ed2b-831b-4f44-972a-a283792aee50.png)



![image](https://user-images.githubusercontent.com/70502367/228476398-423724db-70a1-4266-929d-961433da35fa.png)




![image](https://user-images.githubusercontent.com/70502367/228476423-139c9a4c-f922-4b1d-a4df-c4aa7eea9a24.png)



![image](https://user-images.githubusercontent.com/70502367/228476456-dd5a7056-9f8a-4d74-b44f-24ed8cd61d35.png)



![image](https://user-images.githubusercontent.com/70502367/228476497-419492d9-d954-4d1b-91c1-314f829510f9.png)



![image](https://user-images.githubusercontent.com/70502367/228476525-19669fb2-4a60-43c8-8fc8-34303cb9dbc7.png)



![image](https://user-images.githubusercontent.com/70502367/228476564-7db43618-7e7a-45c7-9b84-c072eef94fa0.png)


![image](https://user-images.githubusercontent.com/70502367/228476591-7afd24bd-7344-4e06-b949-75a7ded1aed7.png)

![image](https://user-images.githubusercontent.com/70502367/228476623-4d61ea23-f5f3-4f45-93f6-30b74aa88628.png)


![image](https://user-images.githubusercontent.com/70502367/228476663-0d0d1be5-ad14-4eda-8ae8-1df51d6eb300.png)

![image](https://user-images.githubusercontent.com/70502367/228476702-23946c37-a40e-4aac-a63c-58d3a43eaa9e.png)



## Results

1\. The model was built using several inbuilt libraries and functions, as shown in **Fig a.**

below followed by the overall **architecture of the model** and its various layers defined

as depicted in **Fig. b.**


![image](https://user-images.githubusercontent.com/70502367/228476855-ed523d9f-93ba-4146-bd8c-7ff7d62de9e5.png)


**Fig a. Overall model definition**


![image](https://user-images.githubusercontent.com/70502367/228476899-e7429131-1a76-46e9-9c2d-316d80d04663.png)

**Fig b. Model Architecture**

2\. The project is set up with first converting each image in the dataset into a grey scale

image for uniformity and unbiased classification and to fasten the learning process of





the model. Gray scale images are frequently used in machine learning to streamline the

input data and lower the processing demands of the model. **Fig c.** shows the **original**

**image of the onion** from the dataset.



![image](https://user-images.githubusercontent.com/70502367/228477003-64ccdcff-8a7b-4fb1-924c-11e232b27b95.png)


**Fig c. original image of the onion**

3\. **Grayscale images** require less memory and processing resources to alter because they

only have one channel instead of three. Yet, when colour is a crucial component of the

data being studied, adopting grayscale photos can also lead to an information loss. **Fig**

**d.** represents **the grey scaled image of the sample onion**.


![image](https://user-images.githubusercontent.com/70502367/228477070-c505354e-5826-4cab-9425-edf9ad67f97c.png)


**Fig d. Grey scaled image of the onion**

4\. By adjusting a threshold value based on local pixel intensity, adaptive thresholding is a

technique used in image processing and computer vision to segment an image into

foreground and background regions. **Adaptive thresholding** modifies the threshold





value for each pixel based on the intensity values of its neighbours, in contrast to global

thresholding, which sets the **same threshold value for the entire image**.


![image](https://user-images.githubusercontent.com/70502367/228477184-1104a9e2-6d47-4e15-a111-c5c4f2e8032c.png)

![image](https://user-images.githubusercontent.com/70502367/228477214-a11a7065-56e0-4260-8366-58dbebbb5780.png)

**Fig. e Image after adaptive thresholding using ostu’s method Fig. f Image after dilation**

5\. When the object(onion) of interest has irregular shapes or sizes or when the illumination

or contrast fluctuates across the image, this method proved to be helpful. The adaptive

thresholding algorithm was used to calculate the local threshold value using a sliding

window of a given size and then applied it to the associated pixel. This process is shown

in **Fig**. **e**.

6\. An image can be automatically segmented into foreground and background regions

using **Otsu's thresholding**. The technique is based on the idea of reducing the inter-

class variance, which **measures the distance between the two classes**, and the intra-

class variance of the **picture intensities,** which measures the dispersion of intensities

within each class (foreground and background).

7\. It is frequently used in applications including picture segmentation, object recognition,

and image analysis. Otsu's method is a simple and efficient methodology for **automatic**

**binary thresholding**. It is computationally effective and can handle photos with

various **lighting and contrast situations**.

8\. We have **used this process for automating the thresholding process in the model**,

and then **Dilation** is performed as shown in **Fig. f.** Dilation is a morphological method

used frequently in adaptive thresholding in image processing. With adaptive

thresholding, pixels are either thought of as **belonging to the object of interest** or as

belonging to the background, dividing the image into binary zones.





9\. Dilation was carried out **to expand the boundaries of the onion in the binary image**

by **adding pixels to the edges of the onion**. This was done by using a structuring

element, which is a small binary image that is used as a probe to compare against the

original image.

10\. It **helped in closing gaps in object borders and to soften the edges of onion**. The

accuracy of further processing processes like object recognition or segmentation can be

increased by adding pixels to the edges of the onions, making them more connected and

continuous.

11\. Once the processing is done for the sample, the model then proceeds onto **computing**

**the area of the onion in the image using the contours of the onion** as shown in **Fig.**

**g.**


![image](https://user-images.githubusercontent.com/70502367/228477386-25ec27cf-9132-4995-8fb6-04bd2bb141a2.png)



![image](https://user-images.githubusercontent.com/70502367/228477405-c6bf482e-7e85-4f39-ab78-a1beac2e445d.png)



![image](https://user-images.githubusercontent.com/70502367/228477436-8c1e63fa-4ed5-4e33-bcad-ae8cdacc870c.png)


**Fig g. Area of the onion computed after adaptive thresholding and dilation**

12\. The images are flipped in all directions and translated so that the orientation of the

onion does not yield false outputs and hence **maximizes accuracy of the model.** These

stages are shown in **figs h, i, j and k below.**


![image](https://user-images.githubusercontent.com/70502367/228477521-83f6e8e8-b350-45db-a85d-9e4336554599.png)


![image](https://user-images.githubusercontent.com/70502367/228478357-b4c3e98e-e92e-4461-aa23-d8ac271798b1.png)


**Fig h. Flipped horizontally Fig i. Flipped Vertically**

![image](https://user-images.githubusercontent.com/70502367/228478397-256f0f63-d4b4-42a7-80ea-7e037afb3304.png)


![image](https://user-images.githubusercontent.com/70502367/228478433-2dbe7bef-03e4-43ef-b663-3daa60594447.png)

**Fig j. and k After Translation**

13\. Translation in machine learning is the process of moving input data a specific amount

in one or more directions. Many different sorts of input data, including images, text,

and audio, can be translated.

14\. **Translation in image processing** entails moving an image by a predetermined number

of pixels either horizontally or vertically. This can be helpful for data augmentation,

where different versions of the same image are produced with slight differences to

expand the **training dataset and strengthen the generalisation capabilities** of the

model.

15\. The results yielded by the model **after 6 iterations are** depicted in **Fig. l** as Recall

being **96.522%,** Precision being **96.646%** and Accuracy being **96.288%.**


![image](https://user-images.githubusercontent.com/70502367/228478616-16627f73-ed79-46cb-94f4-9e07d04aaf08.png)



**Fig. l Results of the Model**





16\. Once the entire model is run to find the results, it is classified into 2 categories namely

**Fig. m** which depicts a **good onion** and **Fig. n** which depicts a **bad onion.**

![image](https://user-images.githubusercontent.com/70502367/228478736-f810ebdb-cf87-408b-b57e-be11a2d77065.png)


**Fig m. Test 1 image output as a good onion predicted by the model**


![image](https://user-images.githubusercontent.com/70502367/228478821-2340a393-bab5-4aad-b649-46359f45bbb8.png)


**Fig n. Test 2 image output as a bad onion predicted by the model**

17\. An **N x N matrix called a confusion matrix** is used to assess the effectiveness of a

classification model, where N is the total number of target classes. In the matrix, the

actual target values are contrasted with those that the machine learning model predicted.





18\. The fundamental terms listed below will enable us to identify the measurements we are

looking for:

• **True Positives (TP):** are when the projected value and the actual value are both

positive.

• **True negatives (TN):** are when the prediction and the actual value are both

negative.

• **False positives (FP):** occur when the forecast is correct but the actual result is

incorrect. known as the Type 1 mistake as well.

• **False negative (FN):** When the fact is positive but the prediction is negative, this

is known as a false negative (FN). Type 2 mistake is another name for it.

19\. The prediction of the above values for our model is presented below with values ranging

in the domain of the number of inputs for the model in **Fig. o.**

20\. The **highest and lowest predicted** values **after averaging over 6 iterations** in the

confusion matrix of the model are:

![image](https://user-images.githubusercontent.com/70502367/228479998-37af44eb-2183-4f38-937e-299d13431991.png)



![image](https://user-images.githubusercontent.com/70502367/228479074-72cfa181-4fb5-464d-99da-96b14e5ea784.png)



**Fig o. Confusion matrix values of the model**

### Performance of the proposed solution

1\. ***Accuracy:*** The accuracy of the model was evaluated by comparing its predictions with

the actual labels of the test data. The dataset used for the evaluation of the accuracy was

created manually. The results showed that the model achieved an accuracy of nearly

100% in detecting and classifying onions as good or bad. Although this accuracy is acceptable only because we operate over a small dataset, it can be tried and tested truly

only by increasing the size of the training dataset or by fine-tuning the model.

2\. ***Speed:*** The speed of the model was evaluated by measuring the time it took to process

an image and make a prediction. The results showed that the model was able to process

an image in less than a second, making it suitable for real-time applications.

3\. ***Cost Effectiveness:*** The cost effectiveness of the model was evaluated by comparing

the cost incurred in designing the light boxes with the cost of alternative methods for

capturing images. The results showed that the cost of designing the light boxes was

significantly lower compared to other methods, making the model cost-effective.

### Limitations of the proposed solution

A. Environmental and Physical Limitations:

1\. ***Lighting conditions:*** Poor lighting can impact the accuracy of the CNN model in

detecting and classifying an onion as good or bad.

2\. ***Camera quality:*** Poor camera quality can also impact the accuracy of the model as it

may not capture the necessary details required for proper classification.

3\. ***Background noise:*** The presence of other objects or clutter in the background can make

it more difficult for the model to accurately identify and classify the onion.

B. Model Limitations:

1\. ***Overfitting:*** Overfitting can occur if the model is too complex or if it is trained on too

few examples, leading to poor performance on new, unseen data.

2\. ***Lack of robustness:*** The model may not be robust to variations in the onion’s

appearance, such as changes in lighting conditions or camera angles.

3\. ***Bias in the data:*** If the data used to train the model is biased, the model may not be

able to accurately detect and classify an onion as good or bad.

4\. ***Inadequate pre-processing:*** Poor pre-processing techniques can impact the accuracy

of the model, such as incorrect cropping or resizing of the onion images.





## Validation

### The Generalisation Ability of the Model

The capacity of a machine learning model to perform well on novel, untried data is known as

generalisation ability. There are many approaches to assess a model's generalisation potential,

but some of the most popular ones which we used are as follows:

**1. Train-test Split:** Data was divided into a training set and a test set using a train-test

split. After the model was trained on the training set, its effectiveness was assessed on

the test set. This method gives a solid indication of the **model's generalizability**

because the test set incorporates data that the model **has not encountered during**

**training.**

2\. **Cross-validation:** It is a method for assessing a model's performance by splitting the

data into numerous subsets, or "folds." The model was trained on all folds except one,

after which its performance on the held-out fold was assessed and gave **about 90%**

**resemblance with the original accuracy.**

3\. **Out-of-sample testing:** In this technique, the model is trained using a portion of the

data, and its performance is assessed using a whole new dataset that wasn't utilised

during training. **This approach is a part of improvisation and as of now has not**

**given us accurate results.**

### Performance in Real-World Scenarios

The model when used at the cargo ports, must follow a sequence of steps in order to obtain

maximum accuracy. They are:

1\. Initially the image is to be taken from camera at well-lit conditions.

2\. Distance between the onion in the light box and the camera must be at a minimum

distance of 30 cm.

3\. Image should be resized to (228\*228\*3).

4\. The obtained images must be fed to the ML Model for classification.

5\. The web app can also be used for displaying the calculated results.





## Conclusion

Infections in plants are a significant risk to worldwide food supplies. Extreme diseases in

plants result in annual agricultural yield losses. To avoid these losses, we have designed a

model to predict the quality and size of onions by using Convolution Neural Networks

Algorithm. A new approach was discussed in this project to use deep learning methods to

spontaneously identify a bad onion from an image. The model established was able to

differentiate between a good and bad onion which can be diagnosed visually.

The complete process was defined from the selection of images used for validation and training.

We summarised the results and concluded that deep learning detection, segmentation, and

classification achieves the highest precision. A deep CNN is accomplished to identify onions

with a classification accuracy of 96 - 97% for batch size 32 using our dataset of onion crops.

Besides, the performance can be improved by using a large dataset. More advanced feature

extraction techniques based on deep learning will be developed.

The proposed method can be used in real-world situation by making a web application. It can

also be implemented by using a single camera, which will be connected to the computer and it

will take pictures and tell us about the quality of the onion or any other commodity which we

need. Initially image is taken from camera image should be resized to (228\*228\*3) distance

between the onion and cam is min 30 cm. The onion must be inside the white box will be

difficult to classify the model and it can classify the flipped onion as well.

Overall, while an automated inspection system has the potential to provide significant benefits,

it is important to carefully evaluate the feasibility and practicality of such a system and to

consider any potential limitations or challenges that may arise.

It is also important to consider any regulatory or safety requirements that may be applicable to

the cargo inspection industry, such as requirements for third-party validation or certification of

inspection systems.





## Future scope

1\. The project’s future scope is to focus on **locating the position of the damaged area of**

**the onions along with generating the conclusions** about the type of damage or

rottenness and area of damage.

2\. Another future aim is to **develop a cargo port friendly prototype** that will be built of

a designated photo box with **cameras at all fixed positions**.

3\. Increasing the accuracy of the model by **implementing the out-of-sample testing**

**method** to make sure that the model performs with the same accuracy for unseen data.

4\. Enhancing the **user-interface of the existing web application** and display more

readable and precise outputs.

5\. The box will have lights of a particular type and the onion pictures of the sample will

be captured from time to time. These pictures will be fed as inputs to the **AI Model**

**developed with improved functionalities** to meet the above requirements.

6\. Since the data set, we considered is comparatively of a smaller size, the focus on

**increasing the size of the data sets and trying to maintain the same accuracy** for

larger inputs as well can be a prospective future scope.

7\. Improvising the existing model to **obtain a vast range of outputs and increased data**

**sets** pave our path to make our model efficient with each improvement.

8\. The existing model can also be used for other Agro commodities which has a definite

shape, so that it can be used to get the area. Like potatoes, and other such vegetables

and fruits can be used by using a different dataset of whichever commodity we want.

We must make the dataset by clicking the pictures of the commodity in the light box

and train the model in the same way we did for the onions.





## Acknowledgement 

The successful completion of this project would be incomplete without the mention of the

people who made it possible and whose constant guidance crowned my effort with success.

We would like to extend our gratitude to **Mr. Rohit Majhi,Co-founder, Maalexi** for

facilitating us to present this project and for supporting us throughout with everything.

We thank **Mr. Chalam Plachikkat, Co-CEO, RaiseToPi** India Private Limited, for his

constant support and encouragement.

We would like to thank our **Project Guide, Mr. Ajay** for always guiding us with his

experience.

Finally, we extend our heart-felt gratitude to our internal **Guide Dr. Prashant P Patavardhan**

family for their encouragement and support without which we would not have come so far.

Moreover, we thank all our friends for their invaluable support and cooperation.





## Project Details

The drive link contains the data set, code and all the materials used in the making of this project.

https://drive.google.com/drive/folders/1TyRWBaVdkuw\_zf7SY987KqdKN6LccOwr


