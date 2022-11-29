# Rework classification usingg XGboost
The following script is python-based ML algorithm.
The algorithms will make predictions whether a rework will be needed, based on different data parameters such as Tool name, product and Chuck No..
The Data in columns 'IDEAL_SETTING_XMAG' and 'IDEAL_SETTING_YSCALE' was imputed using a Linear Regression model.

## Data Distribution

Not well distributed - response ratio is biased
![REWORK_FLAG](https://user-images.githubusercontent.com/114855806/204291335-377c0181-93e1-4d18-83ea-5caedd39b56b.png)

![UNDERLAYER_OPERATION](https://user-images.githubusercontent.com/114855806/204291504-879bdb97-d282-4cfb-8462-84f0f030217c.png)
![OVERLAYER_OPERATION](https://user-images.githubusercontent.com/114855806/204291456-95011545-f721-4052-86fe-b6bb01d20cf6.png)

![UNDERLAYER_ENTITY](https://user-images.githubusercontent.com/114855806/204291368-e89499ba-d5ed-46a3-b5b8-3c06c515057a.png)
![OVERLAYER_ENTITY](https://user-images.githubusercontent.com/114855806/204291360-05d93da4-4bf6-486d-93cd-8f934a8b94f2.png)
![PRODUCT_FAMILY](https://user-images.githubusercontent.com/114855806/204291347-39b50874-38d7-4d00-86d7-2e4070e49a09.png)

![OL_START_CHK](https://user-images.githubusercontent.com/114855806/204291392-2096cc95-c54e-4bdc-b862-be44911200a6.png)
![HOTLOT_AT_OPER](https://user-images.githubusercontent.com/114855806/204291384-f4868243-82ef-43a5-82e9-8f278dee7d8b.png)

Reg performance (RWK group conclustions) 
![IDEAL_SETTING_XMAG](https://user-images.githubusercontent.com/114855806/204291403-7ed317d4-1770-42ea-96ac-f87e3d2f607f.png)
![IDEAL_SETTING_YSCALE](https://user-images.githubusercontent.com/114855806/204291413-2fb1faee-345e-4337-980b-8c1281dd083e.png)

![OUT_WAFER_QTY](https://user-images.githubusercontent.com/114855806/204291443-054c9f6b-ddfe-4393-8291-b747e7cf2f07.png)
![PERCENT_MEASURED_MIN](https://user-images.githubusercontent.com/114855806/196931501-f63970a0-cb9b-496e-bcb0-9257782569ed.png)

![LAYER_REWORK_COUNTER](https://user-images.githubusercontent.com/114855806/204291436-b0448bfa-6358-4dd2-80a2-b480f3a60f52.png)
![PERCENT_MEASURED_MIN](https://user-images.githubusercontent.com/114855806/204291463-1604179d-4a15-46e3-851a-65bf620ef8cd.png)
![REWORK_CATEGORY](https://user-images.githubusercontent.com/114855806/204291474-7ac30f1a-711b-4858-b350-b17ee2a9c046.png)
![Total_Open_Paths](https://user-images.githubusercontent.com/114855806/204291494-5e87ac4e-ce62-4cd9-be9f-21d77db8c352.png)
![segment](https://user-images.githubusercontent.com/114855806/204291513-c746702b-b0c7-4bf0-8a69-63e9a8648486.png)

## Data Correlation
PCA analysis is to come. 

