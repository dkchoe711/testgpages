---
title: Literature Review and Previous Work
notebook: previous_work
nav_include: 4
---


1. EMG-based continuous and simultaneous estimation of arm kinematics in able-bodied individuals and stroke survivors.

Liu, Jie, et al. "EMG-Based Continuous and Simultaneous Estimation of Arm Kinematics in Able-Bodied Individuals and Stroke Survivors." Frontiers in neuroscience 11 (2017): 480.

The aim for this study is to develop a model for decoding multi-joint dynamic arm movements based on multi-channel surface EMG signals. More specifically, the authors collected data in 10 stroke subjects and 10 able-bodied subjects to estimate angles of shoulder horizontal adduction, elbow flexion, and wrist flexion from EMG signals from six muscles are recorded. 

A non-linear autoregressive exogenous (NARX) model, a type of recurrent neural network, was developed in this paper. The paper reported that the model accounted variance (VAF) > 98% for all three joints in all subjects, with no specific R^2 values specified. 

Due to the lack of rigor in this paper, we did not proceed with NARX. However, recurrent neural network, also suggested by the TA, sounded promising as our data was in time series. 

2. Hierarchical recurrent neural network for skeleton based action recognition.

Du, Yong, Wei Wang, and Liang Wang. "Hierarchical recurrent neural network for skeleton based action recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.

In this paper, an end-to-end hierarchical recurrent neural network (RNN) was proposed for skeleton based action classification. The authors tested their algorithm on the HDM05 and MSR Actioin3D datasets and compared with five other deep RNN architectures. With an accuracy of 98.22%, this model beat the other ones and achieved state-of-the-art performance.

This paper further gives us confidence in solving the problem with RNN. More specially, we looked at long short-term memory (LSTM) RNN, based on the recommendation from TA.

3. Keras documentation
https://keras.io/layers/recurrent/
This document explains the detail of executing RNN and LSTM in Keras environment. 


4. Understanding LSTM Networks
http://colah.github.io/posts/2015-08-Understanding-LSTMs/?fbclid=IwAR1OGRNofSXhHG3rfcsjwrGnzSF5SBcZGH_yT5gEkQHrCMudSAOo1EWrxLQ
This website helps us conceptually understand LSTM networks. 
