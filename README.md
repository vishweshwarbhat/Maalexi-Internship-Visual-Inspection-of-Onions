# Maalexi-Internship-Visual-Inspection-of-Onions

![image](https://user-images.githubusercontent.com/70502367/228473220-65e51743-811c-4bd9-87b3-50860d9f608b.png)


```
In collaboration with
```

![image](https://user-images.githubusercontent.com/70502367/228473536-75aee696-750c-487a-81fe-6f17427d6189.png)

```
Internship on
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

##### [Detailed report](https://github.com/vishweshwarbhat/Maalexi-Internship-Visual-Inspection-of-Onions/blob/main/Internship%20Report.pdf)

## [Problem Statement](https://github.com/vishweshwarbhat/Maalexi-Internship-Visual-Inspection-of-Onions/blob/main/Internship%20Report.pdf)
“We would like to reduce the cost and time required to do cargo inspections of Agro-
commodities at ports - perishables like onions, and mangoes, and semi-perishables like rice
and wheat flour. Currently, these are done manually by experts who charge tens of thousands
of rupees a day and are too expensive for smaller exporters.

We would like to see a small-scale prototype of an automated or semi-automated solution.
Check for quality of product, accuracy of labelling or both.”

We would want to start by first taking an individual item, and the item that we have considered
is onion. We aim at creating an automated solution to solve the problem mentioned above and
to check the quality of the onions.

[more details...](https://github.com/vishweshwarbhat/Maalexi-Internship-Visual-Inspection-of-Onions/blob/main/Internship%20Report.pdf)
 
## OVERVIEW

Considering the present trends in the cargo inspection industry, it is observed that lots of
manpower is needed to do the inspection. This not only wastes the time but also results in
inaccuracy and charges a lot of money. There are various companies which provide cargo
inspection services, and they charge very high fees. To eliminate this, we have designed an
automated system, which can we used to determine the quality of goods without any hassle and
manpower. Automation improves the nation's quality, production, and economic growth in
agriculture science. The selection of fruits and vegetables has an impact on the export market
and quality assessment. Visual quantification of the elements affecting fruits and vegetables is
possible, but it is time-consuming, expensive, and vulnerable to subjective judgement and
physical influence. These inspections and the "best-if-used-before date" help establish market
values. The quality assessment was carried out by skilled human investigators using their
senses of touch and sight.

This approach is highly erratic, capricious, and rarely yields consistent results across
investigators. Machine vision systems are ideally suited for traditional analysis and quality
assurance in this sort of environment since it is a continuous task to analyse fruits and
vegetables for various aspect criteria. Computer vision systems and image processing are a
rapidly expanding study field in agriculture.

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
need. Initially image is taken from camera image should be resized to (228*228*3) distance
between the onion and cam is min 30 cm. The onion must be inside the white box will be
difficult to classify the model and it can classify the flipped onion as well.Overall, while an
automated inspection system has the potential to provide significant benefits, it is important to
carefully evaluate the feasibility and practicality of such a system and to consider any potential
limitations or challenges that may arise.
It is also important to consider any regulatory or safety requirements that may be applicable to
the cargo inspection industry, such as requirements for third-party validation or certification of
inspection systems.

## STEPS TO RUN THE CODE

1. At first, download the anaconda ide.
2. Now make a folder inside the system for the project.

3. Put all the files and data to be used in the same folder.

4. Download all the required libraries in the anaconda using pip and the anaconda command
prompt and then download all the data and dataset in the folder.
5. Open the .ipynb file in the jupyter notebook.
6. Run all the cells to train and pre-process the data and get trained model.
7. Open the app.py specify the trained model directory.
8. Open the anaconda prompt and run app .py
9. Using the command stream lit run app.py, while doing it should be noted that command
prompt should be pointing to the project folder.
10. Upload the image and get the results.



This is a offline tool, your data stays locally and is not send to any server!
Feedback & Bug Reports
