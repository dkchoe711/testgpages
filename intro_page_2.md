---
title: Data Description and Initial EDA
notebook: data_description_EDA
nav_include: 2
---



## Contents
{:.no_toc}
*  
{: toc}
 

### 1) Experimental Protocol Details
The data used in this project was collected during tests session in the Wyss Motion Capture Lab, where 6 human subjects were asked to wear the soft sensing shirt and perform a series of arm motions. Each subject performed 3 trials of arm movements. Each trial contains 3 sequential conditions: 

* Pre-defined movements
* Composite movements 
* Random movements 

Pre-defined movements include shoulder abduction, flexion, extension, horizontal flexion, and horizontal extension; all from minimum to maximum range of motion. 
Composite movements include drawing a figure “∞” with the arm at different heights.
Random movements were trials where the subject was told to move the arm in a random manner for the duration of the trial.
The total number of observations (n) for these subjects are: 80137, 102941, 91690, 118276, 100600, and 108239.



### 2) Collected Data
Recorded data from each session includes timestamps, embedded sensor raw outputs (total of 6 sensors in the shirt), and the motion captured (MOCAP) shoulder angles (total of 3 degrees of freedom: shoulder adduction, horizontal flexion, and internal rotation). The MOCAP data is used as the angle ground truth. Therefore, the dataset has 7 initial predictors and 3 outcome variables.

#### 2a) Description of Raw Data? 

### 3) Data Cleaning
One thing that needed to be considered when first looking at the data is the fact in the variability present in ranges of values presented by the MOCAP data for the three angles: Abduction (ab), Horizontal Flexion (hf) and Internal Rotation (ir). This corresponds to the range of motion across all subjects in the three directions. Although the ranges are relatively consistent across the 6 subjects for hf and ab, the ir ranges fluctuate considerably. The variability here could be due to true inter-subject variability (linked to posture), as well as the lack of standardization of the definition of 0&deg; rotation in ir during the experimental data collection. 

**PUT FIGURES HERE**

Since the MOCAP is used as the ground truth for the model predictions, it was decided to develop a model using the data of a single subject to avoid any complications arising from using an inconsistent ground truth for ir across multiple subjects. This aligns with the goal of creating a model that would have a personalized calibration period for each individual who would use it to train the model, and then have the model predict for each subject. Later on, it may be worth extending the model to be more general, but the use case for the scope of this project to be a subject-individualized model. 


### 4) Dataset Selection
Once the decision to pursue a single-subject model, a single subject was selected from the 6 total subjects that were tested in the protocol. The data from the three separate trials, each containing the three conditions of pre-defined, composite, and random movements, were pooled together. Additionally, since the sampling frequency of the MOCAP system is very high (200 Hz), the dataset was downsampled to allow for faster computation time. 

### 5) Feature Selection and Engineering
For the initial data exploration, the timestamp data was dropped in order to remove absolute time-dependency from the model. This was done to mitigate the risk of the model being biased towards a certain outcome based on what time the motion was recorded since the protocol involved following a sequence of moments, which will not always be the case in later trials. 

However, since the arm motions are continuous, including previous sensor values may be useful in the models, especially in RNN and LSTM models, to predict future positions of the arm. For this purpose, the 1st and 2nd derivatives of the position of the arm were calculated and included as features. Additionally, for linear regression, polynomial features of order 2 were included as well. 



# 2. EDA

Initial EDA was done on the pooled data of a single subject - the 3 trials collected were concatenated to create a larger dataset. The initial approach was to look at the raw output of each of the 6 sensors embedded in the shirt during motion in the 3 directions (ab, hf, if), as well as to look at the reverse - the consistency with which data points with the arm in the same position have similar sensor values.

**Load Data**



```python
df = pd.read_csv('../data/M1_t1A.txt', header=None, names=['time','s1','s2','s3','s4','s5','s6','hf','ab','ir'])
df.head()
```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>s1</th>
      <th>s2</th>
      <th>s3</th>
      <th>s4</th>
      <th>s5</th>
      <th>s6</th>
      <th>hf</th>
      <th>ab</th>
      <th>ir</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>205.33</td>
      <td>176.98</td>
      <td>225.10</td>
      <td>208.81</td>
      <td>152.47</td>
      <td>164.64</td>
      <td>-12.149</td>
      <td>20.124</td>
      <td>-41.307</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.008333</td>
      <td>205.32</td>
      <td>177.00</td>
      <td>225.06</td>
      <td>208.67</td>
      <td>152.41</td>
      <td>164.62</td>
      <td>-11.648</td>
      <td>20.087</td>
      <td>-40.909</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.016667</td>
      <td>205.31</td>
      <td>177.02</td>
      <td>225.02</td>
      <td>208.53</td>
      <td>152.36</td>
      <td>164.60</td>
      <td>-11.142</td>
      <td>20.047</td>
      <td>-40.482</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.021000</td>
      <td>205.30</td>
      <td>177.03</td>
      <td>225.00</td>
      <td>208.45</td>
      <td>152.34</td>
      <td>164.59</td>
      <td>-10.891</td>
      <td>20.025</td>
      <td>-40.246</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.025000</td>
      <td>205.30</td>
      <td>177.04</td>
      <td>224.98</td>
      <td>208.38</td>
      <td>152.31</td>
      <td>164.59</td>
      <td>-10.659</td>
      <td>20.004</td>
      <td>-40.028</td>
    </tr>
  </tbody>
</table>
</div>





```python
fig, ax = plt.subplots(n_sens, 1, figsize=(20,20))

for k in range(0,n_sens):
    ax[k].plot(df.time,df[sens[k]])
    ax[k].set_ylabel(sens[k])
    
ax[-1].set_xlabel('Time');
```



![png](1a_sensing_shirt_yichu_eda_with_plots_SECTION_files/1a_sensing_shirt_yichu_eda_with_plots_SECTION_7_0.png)


**Comments**: 
From the plot above it is visible that there is a distinct difference shown in certain sensors during movement in each of the three directions - this leads us to believe that using some subset of the signals output by the embedded sensors will be a viable way of predicting a corresponding arm angle. For clarification, the plot above was made prior to the feature removal of the timestamp for visualization purposes. 



```python
# pooled data 3D plot

cm = plt.get_cmap('viridis')
fig = plt.figure(figsize=(30,20))

for k in range(6):
    ax = fig.add_subplot(2,3,k+1,projection='3d')
#     ax.view_init(elev=-90,azim=0)

    sdat = df_pooled[sens[k]].values
    sdat_norm = (sdat-np.min(sdat))/(np.max(sdat)-np.min(sdat))
    ax.scatter(df_pooled['hf'],df_pooled['ab'],df_pooled['ir'],alpha=0.2,c=cm(sdat_norm))

    ax.set_xlabel('hf')
    ax.set_ylabel('ab')
    ax.set_zlabel('ir')
    ax.set_title(sens[k])

plt.show()
```



![png](1a_sensing_shirt_yichu_eda_with_plots_SECTION_files/1a_sensing_shirt_yichu_eda_with_plots_SECTION_9_0.png)


**Comments**: 
The figure shows how sensor readings vary in the 3D angle space of the shoulder joint. For each plot, the
axes are the ground truth shoulder angles as measured by motion capture (hf, ab, ir), while the sensor
value (normalized over the range) is represented by the point color. From this we can identify general
trends of how different sensors behave in different regions of the angle space. For example, from the
overall direction of the color gradients we can see that different sensors do better at picking up motion
along a specific axis than others. Sensors 1, 2, 4, 6 [s1, s2, s4, s6] are more sensitive to abduction,  s4, s5 are more sensitive to internal rotation, and s5 and s6 are more sensitive to motion in horizontal flexion. 



```python
names = ['Asa','Ci','Con','M1','M2','Siv']
tests = ['A','B','C']
dfs = [[],[],[],[],[],[]]

for ind, name in enumerate(names):
    for number in range(1,4):
        for test in tests:
            file_dir = 'data/' + name + '_t' + str(number) + test + '.txt'
            dfs[ind].append(pd.read_csv(file_dir, header=None, names=['time','s1','s2','s3','s4','s5','s6','hf','ab','ir']).drop('time', axis=1))
    dfs[ind] = pd.concat(dfs[ind], ignore_index=True)

dfs_raw = dfs.copy
    
asa_df = dfs[0]
ci_df = dfs[1]
con_df = dfs[2]
m1_df = dfs[3]
m2_df = dfs[4]
siv_df = dfs[5]
names = ['asa_df','ci_df','con_df','m1_df','m2_df','siv_df']
```




```python
def box_compare(var_name):
    plot_df = pd.concat([df[var_name] for df in dfs], axis=1)
    plot_df.columns = ['Sub1','Sub2','Sub3', 'Sub4', 'Sub5', 'Sub6']
    ax = sns.boxplot(data=plot_df)       
    ax.set_title('Ranges of '+ var_name, fontsize=20)
    ax.set_ylabel(var_name, fontsize=20)
    ax.tick_params(labelsize=20)
    plt.show();
        
```




```python

for var_name in asa_df.columns:
    box_compare(var_name)
```



![png](1b_sensing_shirt_yichu_eda_with_plots_SECTION_files/1b_sensing_shirt_yichu_eda_with_plots_SECTION_6_0.png)



![png](1b_sensing_shirt_yichu_eda_with_plots_SECTION_files/1b_sensing_shirt_yichu_eda_with_plots_SECTION_6_1.png)



![png](1b_sensing_shirt_yichu_eda_with_plots_SECTION_files/1b_sensing_shirt_yichu_eda_with_plots_SECTION_6_2.png)



![png](1b_sensing_shirt_yichu_eda_with_plots_SECTION_files/1b_sensing_shirt_yichu_eda_with_plots_SECTION_6_3.png)



![png](1b_sensing_shirt_yichu_eda_with_plots_SECTION_files/1b_sensing_shirt_yichu_eda_with_plots_SECTION_6_4.png)



![png](1b_sensing_shirt_yichu_eda_with_plots_SECTION_files/1b_sensing_shirt_yichu_eda_with_plots_SECTION_6_5.png)



![png](1b_sensing_shirt_yichu_eda_with_plots_SECTION_files/1b_sensing_shirt_yichu_eda_with_plots_SECTION_6_6.png)



![png](1b_sensing_shirt_yichu_eda_with_plots_SECTION_files/1b_sensing_shirt_yichu_eda_with_plots_SECTION_6_7.png)



![png](1b_sensing_shirt_yichu_eda_with_plots_SECTION_files/1b_sensing_shirt_yichu_eda_with_plots_SECTION_6_8.png)


**Comments**: 
As can be seen from plots above showing the values of the six embedded sensors for each of the six subjects, there are several outliers in the sensor outputs. The outliers on the lower ranges can be attributed to sensor
shorting due to shirt wrinkling during certain motions. The outliers on the higher ranges can be attributed
to occasional movement of the shirt and the sensor extreme positions during the random motion condition. 



```python
# Load data: M1

tests = ['A','B','C']
appended_data = []
count = 0

for number in range(1,4):
    for test in tests:
        file_dir = 'data/M1_t' + str(number) + test + '.txt'
        data = pd.read_csv(file_dir, header=None, names=['t','s1','s2','s3','s4','s5','s6','hf','ab','ir'])
        appended_data.append(data)
        if count > 0:
            appended_data[count].t = appended_data[count].t + max(appended_data[count-1].t)
        count += 1
        
df = pd.concat(appended_data, ignore_index=True)
```




```python
# Find observations with identical time stamp, 
sam_t_df = df[df['t'].duplicated(keep=False)]
sam_mocap_df = df[df[['hf','ab','ir']].duplicated(keep=False)]
sam_sen_df = df[df[['s1','s2','s3','s4','s5','s6']].duplicated(keep=False)]
```




```python
def func(df):
    return np.max(df) - np.min(df)
```




```python
sam_mocap_diff = []

for name, group in sam_mocap_df.groupby(['ab','ir','hf']):
    sam_mocap_diff.append([func(group.t), 
                           func(group.s1), func(group.s2), func(group.s3), 
                           func(group.s4), func(group.s5), func(group.s6)])
sam_mocap_diff = pd.DataFrame(sam_mocap_diff, columns=['time_diff','s1_diff','s2_diff',
                                                       's3_diff','s4_diff','s5_diff','s6_diff'])
```




```python
sam_mocap_max_diff = []
for col in sam_mocap_diff.columns:
    sam_mocap_max_diff.append(np.max(sam_mocap_diff[col]))
sam_mocap_max_diff = pd.DataFrame(sam_mocap_max_diff, 
                                  index=['max time_diff','max s1_diff','max s2_diff',
                                         'max s3_diff','max s4_diff','max s5_diff','max s6_diff']).T
sam_mocap_max_diff
```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>max time_diff</th>
      <th>max s1_diff</th>
      <th>max s2_diff</th>
      <th>max s3_diff</th>
      <th>max s4_diff</th>
      <th>max s5_diff</th>
      <th>max s6_diff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.01</td>
      <td>0.01</td>
      <td>0.0</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>0.01</td>
    </tr>
  </tbody>
</table>
</div>





```python
sam_sen_diff = []
count = 0
for name, group in sam_sen_df.groupby(['s1','s2','s3','s4','s5','s6']):
    count += 1
    if count == 871:
        display(group)
    sam_sen_diff.append([count, func(group.t), func(group.ab), func(group.hf), func(group.ir)])
sam_sen_diff = pd.DataFrame(sam_sen_diff, columns=['count','time_diff','ab_diff','hf_diff','ir_diff'])

```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>t</th>
      <th>s1</th>
      <th>s2</th>
      <th>s3</th>
      <th>s4</th>
      <th>s5</th>
      <th>s6</th>
      <th>hf</th>
      <th>ab</th>
      <th>ir</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>32248</th>
      <td>193.260</td>
      <td>211.93</td>
      <td>179.47</td>
      <td>229.24</td>
      <td>210.19</td>
      <td>154.54</td>
      <td>167.86</td>
      <td>-13.934</td>
      <td>17.636</td>
      <td>-40.091</td>
    </tr>
    <tr>
      <th>32249</th>
      <td>193.262</td>
      <td>211.93</td>
      <td>179.47</td>
      <td>229.24</td>
      <td>210.19</td>
      <td>154.54</td>
      <td>167.86</td>
      <td>-13.920</td>
      <td>17.638</td>
      <td>-40.081</td>
    </tr>
    <tr>
      <th>32250</th>
      <td>193.270</td>
      <td>211.93</td>
      <td>179.47</td>
      <td>229.24</td>
      <td>210.19</td>
      <td>154.54</td>
      <td>167.86</td>
      <td>-13.867</td>
      <td>17.644</td>
      <td>-40.046</td>
    </tr>
    <tr>
      <th>32251</th>
      <td>193.278</td>
      <td>211.93</td>
      <td>179.47</td>
      <td>229.24</td>
      <td>210.19</td>
      <td>154.54</td>
      <td>167.86</td>
      <td>-13.832</td>
      <td>17.648</td>
      <td>-40.026</td>
    </tr>
    <tr>
      <th>32252</th>
      <td>193.281</td>
      <td>211.93</td>
      <td>179.47</td>
      <td>229.24</td>
      <td>210.19</td>
      <td>154.54</td>
      <td>167.86</td>
      <td>-13.825</td>
      <td>17.649</td>
      <td>-40.024</td>
    </tr>
    <tr>
      <th>32253</th>
      <td>193.287</td>
      <td>211.93</td>
      <td>179.47</td>
      <td>229.24</td>
      <td>210.19</td>
      <td>154.54</td>
      <td>167.86</td>
      <td>-13.812</td>
      <td>17.652</td>
      <td>-40.018</td>
    </tr>
    <tr>
      <th>32254</th>
      <td>193.295</td>
      <td>211.93</td>
      <td>179.47</td>
      <td>229.24</td>
      <td>210.19</td>
      <td>154.54</td>
      <td>167.86</td>
      <td>-13.807</td>
      <td>17.655</td>
      <td>-40.022</td>
    </tr>
    <tr>
      <th>32255</th>
      <td>193.302</td>
      <td>211.93</td>
      <td>179.47</td>
      <td>229.24</td>
      <td>210.19</td>
      <td>154.54</td>
      <td>167.86</td>
      <td>-13.811</td>
      <td>17.660</td>
      <td>-40.031</td>
    </tr>
    <tr>
      <th>32256</th>
      <td>193.303</td>
      <td>211.93</td>
      <td>179.47</td>
      <td>229.24</td>
      <td>210.19</td>
      <td>154.54</td>
      <td>167.86</td>
      <td>-13.812</td>
      <td>17.661</td>
      <td>-40.033</td>
    </tr>
    <tr>
      <th>32257</th>
      <td>193.312</td>
      <td>211.93</td>
      <td>179.47</td>
      <td>229.24</td>
      <td>210.19</td>
      <td>154.54</td>
      <td>167.86</td>
      <td>-13.827</td>
      <td>17.668</td>
      <td>-40.052</td>
    </tr>
    <tr>
      <th>32258</th>
      <td>193.320</td>
      <td>211.93</td>
      <td>179.47</td>
      <td>229.24</td>
      <td>210.19</td>
      <td>154.54</td>
      <td>167.86</td>
      <td>-13.846</td>
      <td>17.677</td>
      <td>-40.078</td>
    </tr>
    <tr>
      <th>32259</th>
      <td>193.323</td>
      <td>211.93</td>
      <td>179.47</td>
      <td>229.24</td>
      <td>210.19</td>
      <td>154.54</td>
      <td>167.86</td>
      <td>-13.854</td>
      <td>17.681</td>
      <td>-40.089</td>
    </tr>
    <tr>
      <th>32260</th>
      <td>193.328</td>
      <td>211.93</td>
      <td>179.47</td>
      <td>229.24</td>
      <td>210.19</td>
      <td>154.54</td>
      <td>167.86</td>
      <td>-13.867</td>
      <td>17.687</td>
      <td>-40.107</td>
    </tr>
    <tr>
      <th>32261</th>
      <td>193.337</td>
      <td>211.93</td>
      <td>179.47</td>
      <td>229.24</td>
      <td>210.19</td>
      <td>154.54</td>
      <td>167.86</td>
      <td>-13.881</td>
      <td>17.701</td>
      <td>-40.132</td>
    </tr>
    <tr>
      <th>32262</th>
      <td>193.344</td>
      <td>211.93</td>
      <td>179.47</td>
      <td>229.24</td>
      <td>210.19</td>
      <td>154.54</td>
      <td>167.86</td>
      <td>-13.886</td>
      <td>17.715</td>
      <td>-40.150</td>
    </tr>
    <tr>
      <th>32263</th>
      <td>193.345</td>
      <td>211.93</td>
      <td>179.47</td>
      <td>229.24</td>
      <td>210.19</td>
      <td>154.54</td>
      <td>167.86</td>
      <td>-13.887</td>
      <td>17.717</td>
      <td>-40.152</td>
    </tr>
    <tr>
      <th>32264</th>
      <td>193.353</td>
      <td>211.93</td>
      <td>179.47</td>
      <td>229.24</td>
      <td>210.19</td>
      <td>154.54</td>
      <td>167.86</td>
      <td>-13.889</td>
      <td>17.734</td>
      <td>-40.169</td>
    </tr>
  </tbody>
</table>
</div>




```python
sam_sen_diff.iloc[sam_sen_diff.time_diff.idxmax]['count']
```





    871.0





```python
sam_sen_max_diff = []
for col in sam_sen_diff.columns:
    if col != 'count':
        sam_sen_max_diff.append(np.max(sam_sen_diff[col]))
sam_sen_max_diff = pd.DataFrame(sam_sen_max_diff, 
                                index=['max time_diff','max ab_diff',
                                       'max hf_diff','max ir_diff']).T
sam_sen_max_diff
```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>max time_diff</th>
      <th>max ab_diff</th>
      <th>max hf_diff</th>
      <th>max ir_diff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.093</td>
      <td>1.45</td>
      <td>3.5004</td>
      <td>3.489</td>
    </tr>
  </tbody>
</table>
</div>



**Comment**:
To further investigate on the amount of sensor signal drift,
we calculated maximum MOCAP angle differences for the observations with identical sensor outputs.
The max difference in abduction is 1.45 degrees, in horizontal flexion is 3.50 degrees, and in internal
rotation is 3.49 degrees. These values could be viewed as the limitations of the hardware and the
thresholds for algorithm performance.

