---
title: Shoulder Angle Estimation with Soft Sensing Shirt
notebook: index
---


## AC209A Data Science Final Project
**Fall 2018**<br/>
**Group # 50:** Dabin Choe, Yichu Jin, Evelyn Park

**Overview:**

Musculoskeletal disorders (MSDs) are the most frequent cause of nonfatal occupational injuries and illnesses. Right after the hand, the shoulder is the second most frequently injured upper extremity and when injured, demands the most time away from work compared to all other body parts. There is epidemiologic evidence of shoulder MSDs being caused by repetitive work and non-neutral postures. Specifically, repetitive or prolonged overhead work, defined as working with a hand above the head, has been shown to be a significant risk factor of shoulder MSDs. Therefore, there is a need for continuing monitoring of upper arm postures and shoulder movement in industrial workplaces to assess postural hazards of shoulder MSDs.


**Motivation:**
The Harvard Biodesign Lab designed and built a soft sensing shirt that is capable of monitoring the user’s shoulder positions in a conformal, transparent, and comfortable manner. The sensing shirt contains six soft strain sensors around the user’s shoulder joint. The sensors measure the amounts of stretch experienced by the shirt, which can be used to estimate the user’s shoulder position. 

<img src="notebooks\img\sensing_shirt.png" width="400">


**Problem Statement:**

Is there a way to predict the position of the shoulder using only the sensor readings from the six embedded sensors in the shirt? 

1. To predict three shoulder motion angles (adduction/abduction, horizontal flexion/extension, and internal/external rotation) using the sensor outputs.
2. To evaluate the model’s accuracy and precision using data from different human subjects.




