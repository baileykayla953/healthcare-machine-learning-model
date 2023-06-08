
# Machine-Learning-IMR Review with Medical Necessity Model
## Objective
Objective: To leverage machine learning to predict the possibility of medical services to be deemed medically necessary or not based on the presence of certain conditions.

Dataset: [Kaggle](https://www.kaggle.com/datasets/prasad22/ca-independent-medical-review)
![image](https://github.com/baileykayla953/Machine-Learning-/assets/118647940/67c6522b-045e-441c-bd37-61a28b9e6687)


# Database Setup
## SQL and CSV Export 
<img width="553" alt="image" src="https://github.com/baileykayla953/Machine-Learning-/assets/118647940/895e7094-4333-4ce7-8491-b6580ebea539">
<img width="497" alt="image" src="https://github.com/baileykayla953/Machine-Learning-/assets/118647940/e81036b6-434b-492f-bea7-87926560239a">

# Machine Learning Models


## "Type" Machine Learning Model 

### Installation/Dependencies
```bash
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report
```

## Roadmap

- Install Dependencies 

- Upload CSV file for kaggle dataset and create dataframe

- Find Unique Values 

- Find Value Counts for "type"
 
- Bin "Not Medically Necessary"

- Bin "Investigational/Experimental and Urgent Care" 

- Set "type" as target

- Drop Type 

- convert category data into numerical to create pd.get_dummies

- Split preprocessed data into our features and target arrays 

- Split the preprocessed data into a training and testing dataset 

- Create StandardScaler Instances 

- Fit StandardScaler 

- Scale the Data 



### Testing a deep neural network 

<img width="468" alt="image" src="https://github.com/baileykayla953/Machine-Learning-/assets/118647940/a3f75471-9105-467e-8d89-27b67903b801">
<img width="253" alt="image" src="https://github.com/baileykayla953/Machine-Learning-/assets/118647940/47ea3a4f-9269-470e-a00a-0c01c31a667d">



### Testing Logistic Regression 

<img width="398" alt="image" src="https://github.com/baileykayla953/Machine-Learning-/assets/118647940/ba7d1891-d3b4-465a-a467-7562b4bc992d">
<img width="407" alt="image" src="https://github.com/baileykayla953/Machine-Learning-/assets/118647940/b2447f2c-182c-4e8e-bfa2-878aa33e60e6">



### Testing Logistic Regression after Over Sampling the Data
<img width="401" alt="image" src="https://github.com/baileykayla953/Machine-Learning-/assets/118647940/e02ab965-c736-49ef-b5a8-7eebf9c13263">
<img width="369" alt="image" src="https://github.com/baileykayla953/Machine-Learning-/assets/118647940/c48444dc-c696-44cb-95da-9e0bd0973f97">






### Conclusion 
Creating this model to measure the type accuracy of why insurance claims are typically denied. We binned "Medical Necessity" as one bin due to the high number, and binned "Experimental/Investigational and Urgent together due to the subject matter of it not always immediately be not medically necessary as well as the numbers were much lower then the "Medical Necessity." In our first model we got 84% accuracy. We then tested Logistic Regression with a clarification report at 88% accuracy. We knew these numbers were possibly skewed due to data being imbalanced and decided to try a third model to ensure accuarcy was correct with over sampling data and got 87% accuarcy. 





## "Determination" Machine Learning Model 

### Installation/Dependencies
```bash
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report
```

## Roadmap

- Install Dependencies 

- Upload CSV file for kaggle dataset and create dataframe

- Find Unique Values 

- Find Value Counts for "determination"
 
- Change "Upheld Decision of Health Plan" as "No Additional Payment" and "Overturned Decision of Health Plan" as "Additional Payment 

- Bin "Investigational/Experimental and Urgent Care" 

- Set target as "determination" 

- Drop "determination"

- Convert Categories into numerical data and create pd.get_dummies

- Split preprocessed data into our features and target arrays 

- Split the preprocessed data into a training and testing dataset 

- Create StandardScaler Instances 

- Fit StandardScaler 

- Scale the Data 




<img width="675" alt="image" src="https://github.com/baileykayla953/Machine-Learning-/assets/118647940/daf71fc3-0b17-4787-b782-ddbe7fb9cb83">
<img width="366" alt="image" src="https://github.com/baileykayla953/Machine-Learning-/assets/118647940/09c926db-1347-492c-844c-5492fe999dc3">



<img width="588" alt="image" src="https://github.com/baileykayla953/Machine-Learning-/assets/118647940/3cb3a5f2-0985-45bf-a2ee-5cb6a16fe015">
<img width="503" alt="image" src="https://github.com/baileykayla953/Machine-Learning-/assets/118647940/48df59e0-526e-4e41-996f-0ef10a46d8a2">



<img width="606" alt="image" src="https://github.com/baileykayla953/Machine-Learning-/assets/118647940/d0c196c7-81d1-45af-9627-942a6e9c847c">
<img width="495" alt="image" src="https://github.com/baileykayla953/Machine-Learning-/assets/118647940/33fb93d5-20e1-4e63-bdc8-754e043ae5d6">





### Conclusion 
Creating this model to measure the determination accuracy of the decision of insurance claims being upheld or overturned. In our first model we got 72% accuracy. We then tested Logistic Regression with a clarification report at 70% accuracy. We did not have an imbalance in numbers, but due to the lower accuarcy score decided to try a third model to ensure accuarcy was correct with over sampling data and got 70% accuarcy. We came to the conclusion this process of IMR can be very subjective as it is decisions made by an Insurance Agent and is completed completed on case by case basis. With this said, we found that human error could be playing a factor in this low accuracy as well as this is typically a high percentage in the medical field. 




# Analysis 

### Objective: To leverage machine learning to predict the possibility of medical services to be deemed medically necessary or not based on the presence of certain conditions.

### Conclusion: We found that when hospitals labeled care as "Medical Necessity" it was typically viewed as medically Unnecessary by IMR Review and required additional payments. When care was marked as "Investigational/Experimental and Urgent" it was more likely to be viewed as "medically necessary" by IMR Review and required no additional payment. The data was overwhelmingly presenting "Medical Necessity" numbers to be much higher and requiring additional payment than "Investigational/Experimental and urgent" requiring no additional payment. 

### Presentation:[Analysis Presentation](https://view.officeapps.live.com/op/view.aspx?src=https%3A%2F%2Fraw.githubusercontent.com%2FPhil-Mart%2Fhealthcare-machine-learning-model%2Fmain%2FIMR%2520Analysis%2520and%2520Presentation%2FProject%25204_Medical%2520Necessity%2520Modeling_Draft%2520(1).pptx&wdOrigin=BROWSELINK)

Our orignal question for this dataset was asking "Why does it feel like Urgent Cares are popping up everywhere?" With this dataset, We can see Urgent Cares are more affordable than Hospitals. This data shows more cost effectiveness as we can see an overwhelming amount of data showing how often care is not considered a medical necessity and will require an additional payment. 
Why are Urgent Cares Popping Up Everywhere? 
[CNN Article](https://www.cnn.com/2023/01/28/business/urgent-care-centers-growth-health-care/index.html#:~:text=Urgent%20care%20has%20grown%20rapidly,increase%20from%202019%2C%20estimates%20IBISWorld.)



# Authors 
Phillip Martinez[github](https://github.com/Phil-Mart)

Kayla Bailey [github](https://github.com/baileykayla953/Machine-Learning-)

Dhawn Alexander [github](https://github.com/DhawnAlexander)

Jason Didier [github]() 




