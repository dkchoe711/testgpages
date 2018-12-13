---
title: Conclusion and Future Work
notebook: future_work
nav_include: 5
---



# Results Summary
Below is a summary of the mutiple models tested on the sensing shirt data of a single subject. The data set used in these models had two additional features added, the 1st and 2nd derivatives of the angles with respect to time to retain the time dependence and continuous nature of arm position over time. As can be seen from the summary tables in the **Sensing Shirt Models** section, the addition of the engineered features resulted in an improvement in accuracy. 

Interestingly, the models that performed the best on the three angles were all distinct, as can be seen below. 

##  Results for Abduction/Adduction (ab)

The model that performed the best at predicting abduction/adduction was the Random Forest, with a max_feature split hyper parameter of 0.5, and a tree depth of 10. 

[![](results1.png)](results1.png)


##  Results for Horizontal Flexion/Extension (hf)
The model that performed the best at predicting horizontal flexion/extension was the Boosting ensemble method using AdaBoost, with a max_feature split hyper parameter of 0.5, and a tree depth of 12 and 100 estimators.  
[![](results2.png)](results2.png)


##  Results for Internal Rotation (ir)
The model that performed the best in prediction of internal rotation angles was the ANN, with 3 hidden layers with 100 nodes. 
[![](results3.png)](results3.png)



Below are graphs of each method, showing the predictions of the models versus the ground truth provided by the motion capture data for both the train and test sets. The shaded area in the top row plots are the mean standard deviation, which is a measure of precision. Additionally, the RMSE and MAE are plotted below, which are an indication of the model accuracy. Judging from the plots below, it appears that the models for Abduction and Horizontal Flexion tend to over-estimate the angles at the lower end and over-estimate at the upper end. 


## Abduction/Adduction - Random Forest
[![](ab.png)](ab.png)

## Horizontal Flexion/Extension - Boosting (AdaBoost)
[![](hf.png)](hf.png)

## Internal Rotation - Artificial Neural Network
[![](ir.png)](ir.png)




# Conclusion

The initial motivation of this project was to create a sensing shirt that is capable of estimating the wearer's shoulder position. This goal aligns with our method of training and testing on a single subject, because the intended use case is to have tbe same user wear the sensing shirt - by training the model on a single subject (and not switching between multiple), using a single person's data seems to be a reasonable approach.

However one thing to continue doing in developing a predictive model would be to include more additional features. Even across all the different models that were teested, the best ones had a modest improvement over the remaining models. Therefore, moving forward hopefiully by adding more features, it might be possible to improve model performance even more. 
