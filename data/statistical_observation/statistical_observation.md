```python
import numpy as np
import pandas as pd
import sklearn as sk
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn import metrics
```


```python
from sklearn.tree import DecisionTreeClassifier
```


```python
traindata = pd.read_csv('data/train.csv')
testdata = pd.read_csv('data/test.csv')
```


```python
traindata.head()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
testdata.head()
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
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>892</td>
      <td>3</td>
      <td>Kelly, Mr. James</td>
      <td>male</td>
      <td>34.5</td>
      <td>0</td>
      <td>0</td>
      <td>330911</td>
      <td>7.8292</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <td>1</td>
      <td>893</td>
      <td>3</td>
      <td>Wilkes, Mrs. James (Ellen Needs)</td>
      <td>female</td>
      <td>47.0</td>
      <td>1</td>
      <td>0</td>
      <td>363272</td>
      <td>7.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <td>2</td>
      <td>894</td>
      <td>2</td>
      <td>Myles, Mr. Thomas Francis</td>
      <td>male</td>
      <td>62.0</td>
      <td>0</td>
      <td>0</td>
      <td>240276</td>
      <td>9.6875</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <td>3</td>
      <td>895</td>
      <td>3</td>
      <td>Wirz, Mr. Albert</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>315154</td>
      <td>8.6625</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <td>4</td>
      <td>896</td>
      <td>3</td>
      <td>Hirvonen, Mrs. Alexander (Helga E Lindqvist)</td>
      <td>female</td>
      <td>22.0</td>
      <td>1</td>
      <td>1</td>
      <td>3101298</td>
      <td>12.2875</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



## Dataset features description

<b>PassengerId</b> Unique Id of a passenger. 

<b>Survived</b> Survival of a person 0 = No, 1 = Yes 

<b>Pclass</b> Passenger class Ticket class 1 = 1st(Upper), 2 = 2nd(Middle), 3 = 3rd(Lower)

<b>Sex</b> is a gender of the person 

<b>Age</b> in years 

<b>SibSp</b> # of siblings(brother, sister, stepbrother, stepsister) / spouses(husband, wife) aboard the Titanic

<b>Parch</b> # of parents(mother, father) / children(daughter, son, stepdaughter, stepson) aboard the Titanic: Some children travelled only with a nanny, therefore parch=0 for them. 

<b>Ticket</b> Ticket number 

<b>Fare</b> Passenger fare 

<b>Cabin</b> Cabin number 

<b>Embarked</b> Port of Embarkation C = Cherbourg, Q = Queenstown, S = Southampton

# 1. Data preprocessing

Data preprocessing composed of 

* Converting data into single datatype format
* Treating missing data
* Correction of the data such as identifying outliers
* Feature transformation i.e creating a new feature by using one or more features. 
    
By following these steps, will enhance the predictive capability of the model. Moreover, some of the algorithms may not work if they encounter missing values in the data or if the input data is not in one data format.

<br>
Based on the data, we can notice that 

1. <b>Title</b> feature is a mix of numeric and alphanumeric data types. 

2. <b>Cabin</b> feature is alphanumeric.

3. <b>Name</b> feature may contain errors or typos as there are several ways used to describe a name including titles, round brackets, and quotes used for alternative or short names.



## 1.1 Identifying categorical features and numerical features in the dataset:

Grouping of similar samples into sets considered as categorical features. <b>Nominal, ordinal, ratio, or interval</b> based types of features come categorical features.


Numerical feature values change from sample to sample. <b>Discrete, continuous, or timeseries</b> based considered as numerical features.

Based on a specific feature type helps us to select the appropriate plots for visualization.

#### Categorical: Survived, Sex, and Embarked. Ordinal: Pclass.

#### Continous: Age, Fare. Discrete: SibSp, Parch.

# 2. Exploratory data analysis

In statistics, exploratory data analysis is an approach to analyzing data sets to summarize their main characteristics, often with visual methods. A statistical model can be used or not, but primarily EDA is for seeing what the data can tell us beyond the formal modeling or hypothesis testing task. 

Used statical methods for <I>Titanic dataset</I> are:
1. Mean: Average value of all data points
2. Median is the "middle" value of a sorted data points.
3. Mode: is the value that appears most often in data points
4. Standard deviation: value would tell you how much the data points deviate from the mean value of its data points.
5. Distribution plots: Helps us to understand the column values 
6. Regression: It helps to determine the relationship between one or more features to the dependent variable(target variable).

Let's apply one or more methods for summarizing each column's characteristics.


```python
traindata.info()
print('*********************************')
print('''From the above description Age, Cabin and Embarked columns have null values, these values have to be preprocessed.\n
Seven features are integer/floats.\n
Five features are strings (object).''')
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 12 columns):
    PassengerId    891 non-null int64
    Survived       891 non-null int64
    Pclass         891 non-null int64
    Name           891 non-null object
    Sex            891 non-null object
    Age            714 non-null float64
    SibSp          891 non-null int64
    Parch          891 non-null int64
    Ticket         891 non-null object
    Fare           891 non-null float64
    Cabin          204 non-null object
    Embarked       889 non-null object
    dtypes: float64(2), int64(5), object(5)
    memory usage: 83.7+ KB
    *********************************
    From the above description Age, Cabin and Embarked columns have null values, these values have to be preprocessed.
    
    Seven features are integer/floats.
    
    Five features are strings (object).
    


```python
testdata.info()
print('*********************************')
print('''From the above description Age, Cabin and Fare columns have null values, these values have to be preprocessed.\n
Six are integer/floats.\n
Five features are strings (object).''')
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 418 entries, 0 to 417
    Data columns (total 11 columns):
    PassengerId    418 non-null int64
    Pclass         418 non-null int64
    Name           418 non-null object
    Sex            418 non-null object
    Age            332 non-null float64
    SibSp          418 non-null int64
    Parch          418 non-null int64
    Ticket         418 non-null object
    Fare           417 non-null float64
    Cabin          91 non-null object
    Embarked       418 non-null object
    dtypes: float64(2), int64(4), object(5)
    memory usage: 36.0+ KB
    *********************************
    From the above description Age, Cabin and Fare columns have null values, these values have to be preprocessed.
    
    Six are integer/floats.
    
    Five features are strings (object).
    

<br>
Nevertheless, as per dataset description I can assume that <b>PassengerId</b>, <b>Cabin</b> and <b>Ticket</b> columns would not influence the the survial of a person. Hence, it would be better skip those and will also help us for easy analysis.  


```python
list_data = {'traindata':traindata, 'testdata':testdata}

for name, data in list(list_data.items()):
    print(f'{name.title()} shape before skipping : ', data.shape)
    data.drop(['PassengerId','Ticket','Cabin'],inplace=True, axis=1)
    print(f'{name.title()} shape after skipping data: ', data.shape)
    print('-'*50)
```

    Traindata shape before skipping :  (891, 12)
    Traindata shape after skipping data:  (891, 9)
    --------------------------------------------------
    Testdata shape before skipping :  (418, 11)
    Testdata shape after skipping data:  (418, 8)
    --------------------------------------------------
    

# 3. Statical analysis of each feature in the dataset

## 3.1 Survived feature

Since it is our target variable, it is important to know the distribution rate of the survived people in the data. If it has a  uniform distribution(i.e equal percentage) then we can build an unbiased predictive model. 


```python
ax = sns.countplot(x="Survived", data=traindata)
total = len(traindata['Survived'])
for p in ax.patches:
    print(p.get_height())
    percentage = '{:.1f}%'.format(100 * p.get_height()/total)
    x = p.get_x() + p.get_width()/2
    y = p.get_y() + p.get_height() + 1.5
    ax.annotate(percentage, (x, y))
plt.show()
```

    549
    342
    


![png](output_16_1.png)


The visualization shows that more than 20% of people have died compared to survived people. I assume that our predictive model may <b>estimate more sample under non-survived category(i.e biased)</b> especially for unseen data. Hence, it is a crucial step for choosing a model with a good performance based on any one of the following methods

1. Collecting more samples to maintain class balance, of course, which is not possible right now.
2. Choosing a correct metric to evaluate model such as 
    * Precision(A measure of a classifiers exactness)
    * Recall(A measure of a classifiers completeness)
    * Confusion Matrix(A breakdown of predictions into a table showing correct predictions)
    * ROC Curves (Like precision and recall, accuracy is divided into sensitivity and specificity and models can be chosen based on the balance thresholds of these values.)
3. over-sampling(Adding copies of instances from the under-represented class) or under-sampling (Delete instances from the over-represented class)

## 3.2 Pclass feature

#### How many people have survived or died according to passenger class?


```python
g = sns.catplot(x="Pclass", hue="Pclass", col="Survived",
                data=traindata, kind="count",
                height=4, aspect=.7);
```


![png](output_19_0.png)



```python
traindata[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# count_Pclass = pd.DataFrame(traindata.Pclass.value_counts()).to_dict()['Pclass']
# print(count_Pclass)
# survived_df["count_Pclass"] = survived_df["Pclass"].apply(lambda x: count_Pclass.get(x))
# survived_df['per']= survived_df['Survived']/survived_df['count_Pclass']
# survived_df
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
      <th>Pclass</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>0.629630</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>0.472826</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>0.242363</td>
    </tr>
  </tbody>
</table>
</div>



From the above percentage scores, the higher the passenger class(1 means high class and 3 means low class), the higher the survival rate. 

## 3.3 Name feature

Name column values are shown below. Generally, the names of people will not contribute any information to the survival rate. However, in the 19th century, people were given respect according to the title they have. Hence, by applying, the feature transformation will give their titles.


```python
traindata.Name
```




    0                                Braund, Mr. Owen Harris
    1      Cumings, Mrs. John Bradley (Florence Briggs Th...
    2                                 Heikkinen, Miss. Laina
    3           Futrelle, Mrs. Jacques Heath (Lily May Peel)
    4                               Allen, Mr. William Henry
                                 ...                        
    886                                Montvila, Rev. Juozas
    887                         Graham, Miss. Margaret Edith
    888             Johnston, Miss. Catherine Helen "Carrie"
    889                                Behr, Mr. Karl Howell
    890                                  Dooley, Mr. Patrick
    Name: Name, Length: 891, dtype: object




```python
# Applying feature transformation on Name column

for name, data in list(list_data.items()):
    data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.')
    data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr','Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    data['Title'] = data['Title'].replace('Mlle', 'Miss')
    data['Title'] = data['Title'].replace('Ms', 'Miss')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')
    data.drop(['Name'], axis=1, inplace=True)
traindata[['Title', 'Survived']].groupby(['Title'], as_index=False).mean().sort_values(by='Survived', ascending=False)
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
      <th>Title</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>3</td>
      <td>Mrs</td>
      <td>0.793651</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Miss</td>
      <td>0.702703</td>
    </tr>
    <tr>
      <td>0</td>
      <td>Master</td>
      <td>0.575000</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Rare</td>
      <td>0.347826</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Mr</td>
      <td>0.156673</td>
    </tr>
  </tbody>
</table>
</div>



It seems that women were saved during a titanic disaster more than the high status of male persons. 

## 3.4 Sex feature

From the Name feature, the female proportion survival rate is very high. It is around 74% 


```python
print(traindata[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
g = sns.catplot(x="Sex", hue="Sex", col="Survived",
                data=traindata, kind="count",
                height=4, aspect=.7);
```

          Sex  Survived
    0  female  0.742038
    1    male  0.188908
    


![png](output_27_1.png)


## 3.5 Age feature
<br>
Often people replace missing values with a mean value of their continuous range values. However, it is not an ideal scenario. For estimation of missing value can be done with the help of other correlated features i.e Pclass and Gender. <b>Median value</b> will be considered for replacing a null value for each subset. The sample was grouped according to Pclass and Gender which is shown below.


```python
train_age = traindata[['Age', 'Pclass','Sex']]
test_age = testdata[['Age', 'Pclass','Sex']]
df_age = pd.concat([train_age,test_age],ignore_index=True)

grid = sns.FacetGrid(df_age, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age',alpha=.5, bins=20)
grid.add_legend()
```




    <seaborn.axisgrid.FacetGrid at 0x16536a53608>




![png](output_29_1.png)


From the above image, we can see that each subset has different age distributions i.e mean, median and mode of an age.


```python
# checking null values and replacing them with a median value.

for name, data in list(list_data.items()):
    print(f'Age column null values in {name.title()}:',data['Age'].isnull().sum())
    for i in ['male', 'female']:
        for j in range(0, 3):
            subset_df = df_age[(df_age['Sex'] == i) & (df_age['Pclass'] == j+1)]['Age'].dropna()
            print(f'\nSubset from {name.title()} with Gender: {i},\tPclass: {j}:')
            print(subset_df.describe())
            print('Median value:', round(subset_df.median()))
            print('Mode value:', round(subset_df.mode()[0]))
            print('****************')
            data.loc[(data.Age.isnull()) & (data.Sex == i) & (data.Pclass == j+1),'Age'] = int(round(subset_df.median()))
    print(f'Age column null values after replacement in {name.title()}:',data['Age'].isnull().sum())
    data["Age"] = data["Age"].astype(int) 
    print('-'*20)
```

    Age column null values in Traindata: 177
    
    Subset from Traindata with Gender: male,	Pclass: 0:
    count    151.000000
    mean      41.029272
    std       14.578529
    min        0.920000
    25%       30.000000
    50%       42.000000
    75%       50.000000
    max       80.000000
    Name: Age, dtype: float64
    Median value: 42
    Mode value: 36.0
    ****************
    
    Subset from Traindata with Gender: male,	Pclass: 1:
    count    158.00000
    mean      30.81538
    std       13.97740
    min        0.67000
    25%       23.00000
    50%       29.50000
    75%       38.75000
    max       70.00000
    Name: Age, dtype: float64
    Median value: 30
    Mode value: 30.0
    ****************
    
    Subset from Traindata with Gender: male,	Pclass: 2:
    count    349.000000
    mean      25.962264
    std       11.682415
    min        0.330000
    25%       20.000000
    50%       25.000000
    75%       32.000000
    max       74.000000
    Name: Age, dtype: float64
    Median value: 25
    Mode value: 21.0
    ****************
    
    Subset from Traindata with Gender: female,	Pclass: 0:
    count    133.000000
    mean      37.037594
    std       14.272460
    min        2.000000
    25%       24.000000
    50%       36.000000
    75%       48.000000
    max       76.000000
    Name: Age, dtype: float64
    Median value: 36
    Mode value: 35.0
    ****************
    
    Subset from Traindata with Gender: female,	Pclass: 1:
    count    103.000000
    mean      27.499223
    std       12.911747
    min        0.920000
    25%       20.000000
    50%       28.000000
    75%       34.000000
    max       60.000000
    Name: Age, dtype: float64
    Median value: 28
    Mode value: 24.0
    ****************
    
    Subset from Traindata with Gender: female,	Pclass: 2:
    count    152.000000
    mean      22.185329
    std       12.205254
    min        0.170000
    25%       16.000000
    50%       22.000000
    75%       30.000000
    max       63.000000
    Name: Age, dtype: float64
    Median value: 22
    Mode value: 18.0
    ****************
    Age column null values after replacement in Traindata: 0
    --------------------
    Age column null values in Testdata: 86
    
    Subset from Testdata with Gender: male,	Pclass: 0:
    count    151.000000
    mean      41.029272
    std       14.578529
    min        0.920000
    25%       30.000000
    50%       42.000000
    75%       50.000000
    max       80.000000
    Name: Age, dtype: float64
    Median value: 42
    Mode value: 36.0
    ****************
    
    Subset from Testdata with Gender: male,	Pclass: 1:
    count    158.00000
    mean      30.81538
    std       13.97740
    min        0.67000
    25%       23.00000
    50%       29.50000
    75%       38.75000
    max       70.00000
    Name: Age, dtype: float64
    Median value: 30
    Mode value: 30.0
    ****************
    
    Subset from Testdata with Gender: male,	Pclass: 2:
    count    349.000000
    mean      25.962264
    std       11.682415
    min        0.330000
    25%       20.000000
    50%       25.000000
    75%       32.000000
    max       74.000000
    Name: Age, dtype: float64
    Median value: 25
    Mode value: 21.0
    ****************
    
    Subset from Testdata with Gender: female,	Pclass: 0:
    count    133.000000
    mean      37.037594
    std       14.272460
    min        2.000000
    25%       24.000000
    50%       36.000000
    75%       48.000000
    max       76.000000
    Name: Age, dtype: float64
    Median value: 36
    Mode value: 35.0
    ****************
    
    Subset from Testdata with Gender: female,	Pclass: 1:
    count    103.000000
    mean      27.499223
    std       12.911747
    min        0.920000
    25%       20.000000
    50%       28.000000
    75%       34.000000
    max       60.000000
    Name: Age, dtype: float64
    Median value: 28
    Mode value: 24.0
    ****************
    
    Subset from Testdata with Gender: female,	Pclass: 2:
    count    152.000000
    mean      22.185329
    std       12.205254
    min        0.170000
    25%       16.000000
    50%       22.000000
    75%       30.000000
    max       63.000000
    Name: Age, dtype: float64
    Median value: 22
    Mode value: 18.0
    ****************
    Age column null values after replacement in Testdata: 0
    --------------------
    

We have seen a method for replacing missing values as well as a visual relationship among <b>Age, Pclass and Gender</b>. At subset level of the data has followed almost <b>left-skewed distibution</b>. In another word, its mode value lies left to the mean value and most of the cases mean, median values are almost close to each other. 


```python
import scipy.stats as st
def get_best_distribution(data):
    dist_names = ["norm", "exponweib", "weibull_max", "weibull_min", "pareto", "genextreme"]
    dist_results = []
    params = {}
    for dist_name in dist_names:
        dist = getattr(st, dist_name)
        param = dist.fit(data)
        params[dist_name] = param
        # Applying the Kolmogorov-Smirnov test
        D, p = st.kstest(data, dist_name, args=param)
#         print("p value for "+dist_name+" = "+str(p))
        dist_results.append((dist_name, p))
#     print(dist_results)
#     print(params)
    # select the best fitted distribution
    best_dist, best_p = (max(dist_results, key=lambda item: item[1]))
    # store the name of the best fit and its p value

    print("Best fitting distribution: "+str(best_dist))
#     print("Best p value: "+ str(best_p))
#     print("Parameters for the best fit: "+ str(params[best_dist]))

    return best_dist, best_p, params[best_dist]
```


```python
print(get_best_distribution(traindata.dropna()[traindata['Survived']==0]['Age']))
print(get_best_distribution(traindata.dropna()[traindata['Survived']==1]['Age']))

from scipy.stats import genextreme,norm, exponweib
fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(12,6))
sns.distplot(traindata.dropna()[traindata['Survived']==0]['Age'], fit=exponweib, kde=False,ax=ax1,axlabel='Age wised Titanic non survivers')
sns.distplot(traindata.dropna()[traindata['Survived']==1]['Age'], fit=norm, kde=False,color='g', ax=ax2,axlabel='Age wised Titanic survivers')

```

    Best fitting distribution: genextreme
    ('genextreme', 1.4292621709318743e-07, (0.10210141422952809, 24.442709475020692, 11.175663347485344))
    Best fitting distribution: norm
    ('norm', 0.036310183888932604, (28.055882352941175, 13.954120015175585))
    




    <matplotlib.axes._subplots.AxesSubplot at 0x16536d79f48>




![png](output_34_2.png)


### Using density plots one can answer, which age group of people has survived at a high rate and also who had died at a high rate during the disaster.

* These probability values are also helpful for replacing missing data in the data with respect to survivers vs non-survivors.
* From the above density plot, we can observe that most of the people who were survived during a titanic disaster are in the range between 20 to 35 years old.

### Is it possible to determine the relationship between Age vs Fare features?


```python
traindata.dropna().plot.scatter(x='Age', y='Fare')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x16536f91b08>




![png](output_37_1.png)


Since most of the ticket fares are uniformly distributed for each age. It is hard to determine the relationship.

## 3.6 SibSp & Parch features


```python
traindata[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
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
      <th>SibSp</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>0.535885</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2</td>
      <td>0.464286</td>
    </tr>
    <tr>
      <td>0</td>
      <td>0</td>
      <td>0.345395</td>
    </tr>
    <tr>
      <td>3</td>
      <td>3</td>
      <td>0.250000</td>
    </tr>
    <tr>
      <td>4</td>
      <td>4</td>
      <td>0.166667</td>
    </tr>
    <tr>
      <td>5</td>
      <td>5</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>6</td>
      <td>8</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
traindata[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
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
      <th>Parch</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>3</td>
      <td>3</td>
      <td>0.600000</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>0.550847</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <td>0</td>
      <td>0</td>
      <td>0.343658</td>
    </tr>
    <tr>
      <td>5</td>
      <td>5</td>
      <td>0.200000</td>
    </tr>
    <tr>
      <td>4</td>
      <td>4</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>6</td>
      <td>6</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



However, SibSp and Parch features have describes a higher rate of survival for people who are with parents on the ship. More family members contribute to the decrease in survival rate.

Let's create a feature namely whether a person is alone or not by using Parch and SibSp features.  


```python
for name, data in list(list_data.items()):
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    data['IsAlone'] = 0
    data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1
#     data.drop(['Parch', 'SibSp', 'FamilySize'],inplace=True, axis=1) # drop unnecessary columns
traindata[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean().sort_values(by='Survived', ascending=False)
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
      <th>IsAlone</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0</td>
      <td>0.505650</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>0.303538</td>
    </tr>
  </tbody>
</table>
</div>



## 3.8 Fare feature


```python
fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(15,5))
sns.distplot(traindata.dropna()[traindata['Survived']==0]['Fare'],ax=ax1,axlabel='Fare Vs Titanic non survivers')
sns.distplot(traindata.dropna()[traindata['Survived']==1]['Fare'],color='g', ax=ax2,axlabel='Fare Vs Titanic survivers')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x16536e8f088>




![png](output_45_1.png)



```python
for name, data in list(list_data.items()):
#     data['Fare'].fillna(data['Fare'].dropna().median(), inplace=True)
    data['bin_Fare'] = pd.qcut(data.Fare, 4)
testdata['Fare'].fillna(testdata['Fare'].dropna().median(), inplace=True)
traindata[['bin_Fare', 'Survived']].groupby(['bin_Fare'], as_index=False).mean().sort_values(by='Survived', ascending=False)
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
      <th>bin_Fare</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>3</td>
      <td>(31.0, 512.329]</td>
      <td>0.581081</td>
    </tr>
    <tr>
      <td>2</td>
      <td>(14.454, 31.0]</td>
      <td>0.454955</td>
    </tr>
    <tr>
      <td>1</td>
      <td>(7.91, 14.454]</td>
      <td>0.303571</td>
    </tr>
    <tr>
      <td>0</td>
      <td>(-0.001, 7.91]</td>
      <td>0.197309</td>
    </tr>
  </tbody>
</table>
</div>




```python
for name, data in list(list_data.items()):
    data['Fare'] = pd.qcut(data.Fare, 4, labels=[0,1,2,3]).astype(int)
    data.drop(['bin_Fare'],inplace=True, axis=1)
```

People who bought tickets within range of (31.0, 512.329]$,  likey to be survived because those tickets belong higher than 3rd class passenger tickets.

## 3.9 Embarked feature

Embarked feature takes S, Q, C values based on port of embarkation.

C = Cherbourg, 

Q = Queenstown,

S = Southampton


```python
# In train data 2 missing values exist, hence these can be replaced by the mode values of the Embarked column.
freq_Embarked = traindata.Embarked.dropna().mode()[0]
traindata['Embarked'] = traindata['Embarked'].fillna(freq_Embarked)
traindata[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
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
      <th>Embarked</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>C</td>
      <td>0.553571</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Q</td>
      <td>0.389610</td>
    </tr>
    <tr>
      <td>2</td>
      <td>S</td>
      <td>0.339009</td>
    </tr>
  </tbody>
</table>
</div>



### Still, there are categorical features in the dataset. These columns will be converted into ordinal values.


```python
titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for name, data in list(list_data.items()):
    data['Sex'] = data['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
    data['Age'] = pd.cut(data.Age, bins=5, labels=[0,1,2,3,4]).astype(int)
    data['Title'] = data['Title'].map(titles).astype(int)
    data['Embarked'] = data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    data['Age_Class']= data['Age']* data['Pclass']
    print(data.info())
    print('-'*60)
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 12 columns):
    Survived      891 non-null int64
    Pclass        891 non-null int64
    Sex           891 non-null int32
    Age           891 non-null int32
    SibSp         891 non-null int64
    Parch         891 non-null int64
    Fare          891 non-null int32
    Embarked      891 non-null int32
    Title         891 non-null int32
    FamilySize    891 non-null int64
    IsAlone       891 non-null int64
    Age_Class     891 non-null int64
    dtypes: int32(5), int64(7)
    memory usage: 66.3 KB
    None
    ------------------------------------------------------------
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 418 entries, 0 to 417
    Data columns (total 11 columns):
    Pclass        418 non-null int64
    Sex           418 non-null int32
    Age           418 non-null int32
    SibSp         418 non-null int64
    Parch         418 non-null int64
    Fare          418 non-null int32
    Embarked      418 non-null int32
    Title         418 non-null int32
    FamilySize    418 non-null int64
    IsAlone       418 non-null int64
    Age_Class     418 non-null int64
    dtypes: int32(5), int64(6)
    memory usage: 27.9 KB
    None
    ------------------------------------------------------------
    


```python
# traindata['Embarked'] = pd.Categorical(traindata['Embarked'])
# dfDummies = pd.get_dummies(traindata['Embarked'], prefix = 'category')
# traindata.drop(['Embarked'],inplace=True, axis=1)
# traindata = pd.concat([traindata, dfDummies],axis=1)
```


```python
traindata.head()
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
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>Title</th>
      <th>FamilySize</th>
      <th>IsAlone</th>
      <th>Age_Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>



### Applying oversampling method on the data to avoid class imbalance.


```python
X_train = traindata.drop("Survived", axis=1)
Y_train = traindata["Survived"]
X_test  = testdata.copy()
X_train.shape, Y_train.shape, X_test.shape
```




    ((891, 11), (891,), (418, 11))




```python
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
oversample = SMOTE()
X_resampled, y_resampled = oversample.fit_resample(X_train, Y_train)
X1,X2,y1,y2 = train_test_split(X_resampled,y_resampled, test_size=0.2)
pd.DataFrame(y_resampled)[0].value_counts()
X1.shape, X2.shape, y1.shape, y2.shape
```




    ((878, 11), (220, 11), (878,), (220,))



# 4. Predicitve modeling

Sofar, we have seen statical analysis of the data. It has been preprocessed and analysed. Now, it is time to train a model. For classification purpose there are several machine learning algotihms are avaialble. Out of them I would to choose <b>Decision Tree </b> 

AUC (Area Under Curve) is used as a evaluation metric, since the target value is binary. Hence it’s a binary classification problem. AUC is a good way for evaluation for this type of problems.


#### Regularization:

It discourages learning a more complex or flexible model, so as to avoid the risk of overfitting. One of the ways of avoiding overfitting is to use cross validation, that helps in estimating the error over test set, and in deciding what parameters work best for your model.

## 4.1 Decision Tree

Decision Tree solves the problem of machine learning by transforming the data into a tree representation. Each internal node of the tree representation denotes an attribute and each leaf node denotes a class label. Decision tree algorithm can be used to solve both regression and classification problems.

#### Advantages:

* Compared to other algorithms decision trees requires less effort for data preparation during pre-processing.
* A decision tree does not require normalization of data.
* A decision tree does not require scaling of data as well.
* Missing values in the data also does NOT affect the process of building a decision tree to any considerable extent.
* A Decision trees model is very intuitive and easy to explain to technical teams as well as stakeholders.

#### Disadvantage:

* A small change in the data can cause a large change in the structure of the decision tree causing instability.
* Decision Tree algorithm is inadequate for applying regression and predicting continuous values.

#### Pruning

Growing the tree beyond a certain level of complexity leads to overfitting. In our data, age doesn’t have any impact on the target variable. Growing the tree beyond Gender is not going to add any value. Need to cut it at Gender. This process of trimming trees is called Pruning.

Avoid overfitting by changing the parameters:

* <b>max_depth:</b> How deep the tree can be. The deeper the tree, the more splits it has and it captures more information about the data. However, an increase in depth may good for training but it will fail to generalize the unseen data.

* <b>min_samples_leaf:</b> the Minimum number of samples required to split at leaf node, increase in percentage may the model cannot learn enough about the data. 

* <b>max_features:</b> max_features represents the number of features to consider when looking for the best split. An increase in number features for the best split may cause overfitting. 

* <b>max_leaf_nodes:</b> Reduce the number of leaf nodes




```python
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_resampled, y_resampled)
# Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X1, y1) * 100, 2)
print('Model accuracy: ',acc_decision_tree)
```

    Model accuracy:  88.72
    


```python
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
y_pred = decision_tree.predict(X1)
confusion_matrix(y_pred=y_pred, y_true=y1)
```




    array([[407,  30],
           [ 69, 372]], dtype=int64)




```python
y_probas = decision_tree.predict_proba(X1)[::,1]
# print(y_probas)
fpr, tpr, _ = metrics.roc_curve(y1,  y_probas)
auc = metrics.roc_auc_score(y1, y_probas)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.title('ROC plot with oversampling')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
```


![png](output_62_0.png)



```python
decision_tree_norm = DecisionTreeClassifier()
decision_tree_norm.fit(X_train, Y_train)
# Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree_norm.score(X_train, Y_train) * 100, 2)
print('Model accuracy: ',acc_decision_tree)


y_probas = decision_tree_norm.predict_proba(X_train)[::,1]
# print(y_probas)
fpr, tpr, _ = metrics.roc_curve(Y_train,  y_probas)
auc = metrics.roc_auc_score(Y_train, y_probas)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.title('ROC plot without oversampling')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
```

    Model accuracy:  88.89
    


![png](output_63_1.png)


#### From the above ROC curve, model is able predict each category with a certainity of 95%. Overall accuracy of the model is 88%

## Check whether it is a bias or variance problem

If a model has high bias, it will fail to capture the complexity of the data. This would be like using a linear model for data that has a quadratic form. In short, high bias means the model underfits the data. In contrast, high variance means a model is too complex like using a 100 degree polynomial for that quadratic data. In this situation, overfitting of the data. 


```python
from mlxtend.evaluate import bias_variance_decomp
avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(decision_tree, X1, y1,X2,y2, random_seed=123)
print('Average expected loss: %.3f' % avg_expected_loss)
print('Average bias: %.3f' % avg_bias)
print('Average variance: %.3f' % avg_var)
```

    Average expected loss: 0.179
    Average bias: 0.173
    Average variance: 0.067
    

From the above result, I can conclude the trained model an under fitted due to high bias value whereas its variance values it low. Nevertheless, the is good enough to capture a relatively complex relationship as well as good enough to generalize unseen data. Below shown cross-validation results. The model has performed very well on test data with very less standard deviation


```python
from sklearn.model_selection import cross_validate
cv_results = cross_validate(decision_tree_norm, X_train, Y_train, cv=5, scoring='accuracy')
cv_results['test_score']
```




    array([0.78212291, 0.78651685, 0.79775281, 0.78089888, 0.84269663])



### The following code is will be used for parameter tuning, however, the model works very well for default values.


```python
import itertools
import datetime
max_depths  = np.linspace(1, 5, 5, endpoint=True)
min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)
max_features = [i for i in range(2, X_train.shape[1])][:5]
max_leaf_nodes = np.linspace(2, 5, 4, endpoint=True)

parameters = list(itertools.product(max_depths,min_samples_leafs,max_features, max_leaf_nodes))
```


```python
def plot(data, title_, x_name):
    plt.plot(data)
    plt.title(title_)
    plt.xlabel(x_name)
    plt.show()
    plt.savefig(f'{title_}.png')
```


```python
def train(model, train_data, train_target, file_name = None, params= None):
    models_to_build = len(params)
    print(f'***********Total models to build are: {models_to_build}**************')
    scores = []
    best_score = 0
    for idx, parameter in enumerate(params):
        max_depth, features, min_samples, max_leaf_node = parameter
        model.max_depth= max_depth
        model.max_features=features
        model.min_samples_leaf=min_samples
        model.max_leaf_nodes=int(max_leaf_node)
        start_time = datetime.datetime.now()
        model.fit(train_data, train_target)
        end_time = datetime.datetime.now()
#         predictions = model.predict(train_data)
        acc = round(decision_tree.score(X_train, Y_train) * 100, 2)        
        scores.append(acc)
        text = f'Model:[{idx+1}/{models_to_build}]\tAcc Score: {acc}\t Computing time: {end_time-start_time}'
        print(text)
#         file_name = f'result/{file_name}.txt'
#         with open(file_name, 'w') as f:
#             f.write(text)
#             f.write('\n')
#         file_name = file_name.split('.')[0] + '.sav'
#         if best_score == 0:
#             best_score = mse_score
#             print('Best score achieved at: ',text)
#             pickle.dump(reg_model, open(file_name, 'wb'))
#         elif acc > best_score:
#             best_score = acc
#             print('Best score updated to: ',text)
#             pickle.dump(model, open(file_name, 'wb'))
#             print('Model saved at ', parameter)
#     plot(data=scores, title_= 'Train model performance', x_name='Acc Score')
    return model


train(model=decision_tree,train_data=X_train, train_target=Y_train, params=parameters)
```

    ***********Total models to build are: 500**************
    Model:[1/500]	Acc Score: 62.07	 Computing time: 0:00:00.001993
    Model:[2/500]	Acc Score: 61.73	 Computing time: 0:00:00.000994
    Model:[3/500]	Acc Score: 63.41	 Computing time: 0:00:00.000996
    Model:[4/500]	Acc Score: 62.07	 Computing time: 0:00:00.001000
    Model:[5/500]	Acc Score: 62.18	 Computing time: 0:00:00.002974
    Model:[6/500]	Acc Score: 66.67	 Computing time: 0:00:00.002024
    Model:[7/500]	Acc Score: 78.68	 Computing time: 0:00:00.001999
    Model:[8/500]	Acc Score: 66.67	 Computing time: 0:00:00.001030
    Model:[9/500]	Acc Score: 62.07	 Computing time: 0:00:00.000997
    Model:[10/500]	Acc Score: 62.74	 Computing time: 0:00:00.000997
    Model:[11/500]	Acc Score: 78.23	 Computing time: 0:00:00.001002
    Model:[12/500]	Acc Score: 63.41	 Computing time: 0:00:00.000994
    Model:[13/500]	Acc Score: 61.62	 Computing time: 0:00:00.001998
    Model:[14/500]	Acc Score: 61.73	 Computing time: 0:00:00
    Model:[15/500]	Acc Score: 63.41	 Computing time: 0:00:00.000993
    Model:[16/500]	Acc Score: 66.78	 Computing time: 0:00:00.000964
    Model:[17/500]	Acc Score: 62.07	 Computing time: 0:00:00.000966
    Model:[18/500]	Acc Score: 66.67	 Computing time: 0:00:00.001004
    Model:[19/500]	Acc Score: 62.74	 Computing time: 0:00:00.001001
    Model:[20/500]	Acc Score: 78.23	 Computing time: 0:00:00.000972
    Model:[21/500]	Acc Score: 78.23	 Computing time: 0:00:00.000996
    Model:[22/500]	Acc Score: 78.68	 Computing time: 0:00:00.000998
    Model:[23/500]	Acc Score: 78.23	 Computing time: 0:00:00.000975
    Model:[24/500]	Acc Score: 78.23	 Computing time: 0:00:00.000970
    Model:[25/500]	Acc Score: 66.78	 Computing time: 0:00:00.000997
    Model:[26/500]	Acc Score: 78.23	 Computing time: 0:00:00.000995
    Model:[27/500]	Acc Score: 78.68	 Computing time: 0:00:00.000970
    Model:[28/500]	Acc Score: 62.18	 Computing time: 0:00:00
    Model:[29/500]	Acc Score: 66.78	 Computing time: 0:00:00.000971
    Model:[30/500]	Acc Score: 63.41	 Computing time: 0:00:00.000973
    Model:[31/500]	Acc Score: 63.41	 Computing time: 0:00:00.000974
    Model:[32/500]	Acc Score: 66.67	 Computing time: 0:00:00.000970
    Model:[33/500]	Acc Score: 62.74	 Computing time: 0:00:00.000998
    Model:[34/500]	Acc Score: 62.07	 Computing time: 0:00:00.000999
    Model:[35/500]	Acc Score: 63.41	 Computing time: 0:00:00.000972
    Model:[36/500]	Acc Score: 66.67	 Computing time: 0:00:00
    Model:[37/500]	Acc Score: 66.78	 Computing time: 0:00:00.000993
    Model:[38/500]	Acc Score: 78.23	 Computing time: 0:00:00.000975
    Model:[39/500]	Acc Score: 66.78	 Computing time: 0:00:00.000973
    Model:[40/500]	Acc Score: 62.07	 Computing time: 0:00:00.001021
    Model:[41/500]	Acc Score: 78.68	 Computing time: 0:00:00.000971
    Model:[42/500]	Acc Score: 78.68	 Computing time: 0:00:00.000972
    Model:[43/500]	Acc Score: 78.23	 Computing time: 0:00:00.000997
    Model:[44/500]	Acc Score: 78.23	 Computing time: 0:00:00.000995
    Model:[45/500]	Acc Score: 61.73	 Computing time: 0:00:00.000973
    Model:[46/500]	Acc Score: 78.23	 Computing time: 0:00:00
    Model:[47/500]	Acc Score: 66.67	 Computing time: 0:00:00.001022
    Model:[48/500]	Acc Score: 78.23	 Computing time: 0:00:00.000998
    Model:[49/500]	Acc Score: 78.68	 Computing time: 0:00:00.001013
    Model:[50/500]	Acc Score: 63.41	 Computing time: 0:00:00.000970
    Model:[51/500]	Acc Score: 66.78	 Computing time: 0:00:00.000972
    Model:[52/500]	Acc Score: 66.78	 Computing time: 0:00:00.000996
    Model:[53/500]	Acc Score: 66.78	 Computing time: 0:00:00.000999
    Model:[54/500]	Acc Score: 78.23	 Computing time: 0:00:00.000974
    Model:[55/500]	Acc Score: 78.23	 Computing time: 0:00:00.000996
    Model:[56/500]	Acc Score: 78.23	 Computing time: 0:00:00.001002
    Model:[57/500]	Acc Score: 78.68	 Computing time: 0:00:00.000997
    Model:[58/500]	Acc Score: 66.67	 Computing time: 0:00:00.000997
    Model:[59/500]	Acc Score: 66.67	 Computing time: 0:00:00.000995
    Model:[60/500]	Acc Score: 63.41	 Computing time: 0:00:00.000998
    Model:[61/500]	Acc Score: 66.78	 Computing time: 0:00:00.000999
    Model:[62/500]	Acc Score: 78.23	 Computing time: 0:00:00
    Model:[63/500]	Acc Score: 66.67	 Computing time: 0:00:00.000998
    Model:[64/500]	Acc Score: 66.78	 Computing time: 0:00:00.000995
    Model:[65/500]	Acc Score: 78.23	 Computing time: 0:00:00.000975
    Model:[66/500]	Acc Score: 66.78	 Computing time: 0:00:00.000974
    Model:[67/500]	Acc Score: 78.68	 Computing time: 0:00:00.000995
    Model:[68/500]	Acc Score: 78.23	 Computing time: 0:00:00.000972
    Model:[69/500]	Acc Score: 78.23	 Computing time: 0:00:00.001002
    Model:[70/500]	Acc Score: 78.68	 Computing time: 0:00:00.000976
    Model:[71/500]	Acc Score: 63.41	 Computing time: 0:00:00.000999
    Model:[72/500]	Acc Score: 78.68	 Computing time: 0:00:00.000997
    Model:[73/500]	Acc Score: 66.78	 Computing time: 0:00:00
    Model:[74/500]	Acc Score: 78.68	 Computing time: 0:00:00.001995
    Model:[75/500]	Acc Score: 78.23	 Computing time: 0:00:00.001997
    Model:[76/500]	Acc Score: 78.23	 Computing time: 0:00:00.001977
    Model:[77/500]	Acc Score: 66.78	 Computing time: 0:00:00.000998
    Model:[78/500]	Acc Score: 78.23	 Computing time: 0:00:00.000966
    Model:[79/500]	Acc Score: 78.68	 Computing time: 0:00:00.000975
    Model:[80/500]	Acc Score: 78.68	 Computing time: 0:00:00
    Model:[81/500]	Acc Score: 78.68	 Computing time: 0:00:00.001979
    Model:[82/500]	Acc Score: 78.23	 Computing time: 0:00:00.001021
    Model:[83/500]	Acc Score: 78.23	 Computing time: 0:00:00.000973
    Model:[84/500]	Acc Score: 78.68	 Computing time: 0:00:00.000975
    Model:[85/500]	Acc Score: 78.23	 Computing time: 0:00:00
    Model:[86/500]	Acc Score: 78.23	 Computing time: 0:00:00.001022
    Model:[87/500]	Acc Score: 78.23	 Computing time: 0:00:00.000974
    Model:[88/500]	Acc Score: 78.68	 Computing time: 0:00:00.000978
    Model:[89/500]	Acc Score: 78.23	 Computing time: 0:00:00.000975
    Model:[90/500]	Acc Score: 78.23	 Computing time: 0:00:00.001010
    Model:[91/500]	Acc Score: 78.23	 Computing time: 0:00:00.000996
    Model:[92/500]	Acc Score: 78.68	 Computing time: 0:00:00.000998
    Model:[93/500]	Acc Score: 78.23	 Computing time: 0:00:00.001000
    Model:[94/500]	Acc Score: 78.23	 Computing time: 0:00:00.000975
    Model:[95/500]	Acc Score: 78.23	 Computing time: 0:00:00.000972
    Model:[96/500]	Acc Score: 78.68	 Computing time: 0:00:00.000998
    Model:[97/500]	Acc Score: 78.68	 Computing time: 0:00:00.000997
    Model:[98/500]	Acc Score: 66.67	 Computing time: 0:00:00.000997
    Model:[99/500]	Acc Score: 78.23	 Computing time: 0:00:00.001021
    Model:[100/500]	Acc Score: 66.67	 Computing time: 0:00:00.000985
    Model:[101/500]	Acc Score: 78.68	 Computing time: 0:00:00.000972
    Model:[102/500]	Acc Score: 66.67	 Computing time: 0:00:00.000994
    Model:[103/500]	Acc Score: 64.2	 Computing time: 0:00:00.001149
    Model:[104/500]	Acc Score: 64.2	 Computing time: 0:00:00.000973
    Model:[105/500]	Acc Score: 78.68	 Computing time: 0:00:00.000998
    Model:[106/500]	Acc Score: 64.2	 Computing time: 0:00:00.000998
    Model:[107/500]	Acc Score: 68.24	 Computing time: 0:00:00.001993
    Model:[108/500]	Acc Score: 73.74	 Computing time: 0:00:00.001994
    Model:[109/500]	Acc Score: 62.07	 Computing time: 0:00:00.001022
    Model:[110/500]	Acc Score: 78.68	 Computing time: 0:00:00.000987
    Model:[111/500]	Acc Score: 69.81	 Computing time: 0:00:00.000974
    Model:[112/500]	Acc Score: 68.69	 Computing time: 0:00:00.000974
    Model:[113/500]	Acc Score: 61.62	 Computing time: 0:00:00.000997
    Model:[114/500]	Acc Score: 66.78	 Computing time: 0:00:00.000983
    Model:[115/500]	Acc Score: 63.08	 Computing time: 0:00:00.000976
    Model:[116/500]	Acc Score: 66.22	 Computing time: 0:00:00.001000
    Model:[117/500]	Acc Score: 63.41	 Computing time: 0:00:00.001943
    Model:[118/500]	Acc Score: 66.78	 Computing time: 0:00:00.000996
    Model:[119/500]	Acc Score: 63.86	 Computing time: 0:00:00.001021
    Model:[120/500]	Acc Score: 74.07	 Computing time: 0:00:00.000972
    Model:[121/500]	Acc Score: 62.07	 Computing time: 0:00:00.002005
    Model:[122/500]	Acc Score: 78.9	 Computing time: 0:00:00.000996
    Model:[123/500]	Acc Score: 79.24	 Computing time: 0:00:00.001000
    Model:[124/500]	Acc Score: 78.68	 Computing time: 0:00:00.000974
    Model:[125/500]	Acc Score: 78.68	 Computing time: 0:00:00.000972
    Model:[126/500]	Acc Score: 70.15	 Computing time: 0:00:00.000998
    Model:[127/500]	Acc Score: 73.96	 Computing time: 0:00:00.001024
    Model:[128/500]	Acc Score: 66.67	 Computing time: 0:00:00.000970
    Model:[129/500]	Acc Score: 66.67	 Computing time: 0:00:00
    Model:[130/500]	Acc Score: 70.37	 Computing time: 0:00:00.000998
    Model:[131/500]	Acc Score: 64.2	 Computing time: 0:00:00
    Model:[132/500]	Acc Score: 81.03	 Computing time: 0:00:00.000972
    Model:[133/500]	Acc Score: 78.68	 Computing time: 0:00:00.000998
    Model:[134/500]	Acc Score: 78.9	 Computing time: 0:00:00.000973
    Model:[135/500]	Acc Score: 78.9	 Computing time: 0:00:00.001023
    Model:[136/500]	Acc Score: 74.97	 Computing time: 0:00:00.000997
    Model:[137/500]	Acc Score: 78.68	 Computing time: 0:00:00.000974
    Model:[138/500]	Acc Score: 70.03	 Computing time: 0:00:00.000971
    Model:[139/500]	Acc Score: 78.68	 Computing time: 0:00:00.001000
    Model:[140/500]	Acc Score: 70.37	 Computing time: 0:00:00.000971
    Model:[141/500]	Acc Score: 66.67	 Computing time: 0:00:00.000968
    Model:[142/500]	Acc Score: 74.97	 Computing time: 0:00:00.000998
    Model:[143/500]	Acc Score: 78.68	 Computing time: 0:00:00.000998
    Model:[144/500]	Acc Score: 81.03	 Computing time: 0:00:00
    Model:[145/500]	Acc Score: 66.67	 Computing time: 0:00:00.000966
    Model:[146/500]	Acc Score: 78.68	 Computing time: 0:00:00.000970
    Model:[147/500]	Acc Score: 70.03	 Computing time: 0:00:00.000970
    Model:[148/500]	Acc Score: 78.68	 Computing time: 0:00:00.001996
    Model:[149/500]	Acc Score: 78.23	 Computing time: 0:00:00.000997
    Model:[150/500]	Acc Score: 78.34	 Computing time: 0:00:00.000997
    Model:[151/500]	Acc Score: 81.71	 Computing time: 0:00:00.000998
    Model:[152/500]	Acc Score: 78.68	 Computing time: 0:00:00
    Model:[153/500]	Acc Score: 63.41	 Computing time: 0:00:00.000998
    Model:[154/500]	Acc Score: 78.9	 Computing time: 0:00:00.000978
    Model:[155/500]	Acc Score: 78.68	 Computing time: 0:00:00.000998
    Model:[156/500]	Acc Score: 69.58	 Computing time: 0:00:00.000973
    Model:[157/500]	Acc Score: 66.78	 Computing time: 0:00:00.000978
    Model:[158/500]	Acc Score: 69.25	 Computing time: 0:00:00.000998
    Model:[159/500]	Acc Score: 78.23	 Computing time: 0:00:00.000974
    Model:[160/500]	Acc Score: 69.58	 Computing time: 0:00:00.000974
    Model:[161/500]	Acc Score: 78.68	 Computing time: 0:00:00.000973
    Model:[162/500]	Acc Score: 74.52	 Computing time: 0:00:00.001000
    Model:[163/500]	Acc Score: 78.56	 Computing time: 0:00:00.000994
    Model:[164/500]	Acc Score: 78.68	 Computing time: 0:00:00.000969
    Model:[165/500]	Acc Score: 66.78	 Computing time: 0:00:00.000972
    Model:[166/500]	Acc Score: 78.68	 Computing time: 0:00:00.000998
    Model:[167/500]	Acc Score: 78.68	 Computing time: 0:00:00.000996
    Model:[168/500]	Acc Score: 70.37	 Computing time: 0:00:00.001995
    Model:[169/500]	Acc Score: 78.68	 Computing time: 0:00:00.000998
    Model:[170/500]	Acc Score: 74.52	 Computing time: 0:00:00
    Model:[171/500]	Acc Score: 78.68	 Computing time: 0:00:00.000997
    Model:[172/500]	Acc Score: 78.68	 Computing time: 0:00:00.000965
    Model:[173/500]	Acc Score: 78.23	 Computing time: 0:00:00.000967
    Model:[174/500]	Acc Score: 66.78	 Computing time: 0:00:00.000967
    Model:[175/500]	Acc Score: 78.79	 Computing time: 0:00:00.001044
    Model:[176/500]	Acc Score: 81.71	 Computing time: 0:00:00.000995
    Model:[177/500]	Acc Score: 78.68	 Computing time: 0:00:00.000998
    Model:[178/500]	Acc Score: 74.97	 Computing time: 0:00:00.000972
    Model:[179/500]	Acc Score: 78.68	 Computing time: 0:00:00.000997
    Model:[180/500]	Acc Score: 78.9	 Computing time: 0:00:00.000974
    Model:[181/500]	Acc Score: 78.23	 Computing time: 0:00:00.000976
    Model:[182/500]	Acc Score: 81.71	 Computing time: 0:00:00.000997
    Model:[183/500]	Acc Score: 81.71	 Computing time: 0:00:00.001955
    Model:[184/500]	Acc Score: 81.71	 Computing time: 0:00:00
    Model:[185/500]	Acc Score: 78.68	 Computing time: 0:00:00.000970
    Model:[186/500]	Acc Score: 81.03	 Computing time: 0:00:00.000997
    Model:[187/500]	Acc Score: 81.71	 Computing time: 0:00:00.000974
    Model:[188/500]	Acc Score: 81.71	 Computing time: 0:00:00.000998
    Model:[189/500]	Acc Score: 78.23	 Computing time: 0:00:00.000977
    Model:[190/500]	Acc Score: 78.9	 Computing time: 0:00:00.000966
    Model:[191/500]	Acc Score: 80.36	 Computing time: 0:00:00.001994
    Model:[192/500]	Acc Score: 78.9	 Computing time: 0:00:00.000997
    Model:[193/500]	Acc Score: 78.23	 Computing time: 0:00:00.000998
    Model:[194/500]	Acc Score: 70.03	 Computing time: 0:00:00.000998
    Model:[195/500]	Acc Score: 81.71	 Computing time: 0:00:00.000995
    Model:[196/500]	Acc Score: 79.69	 Computing time: 0:00:00.000999
    Model:[197/500]	Acc Score: 62.07	 Computing time: 0:00:00.000998
    Model:[198/500]	Acc Score: 78.9	 Computing time: 0:00:00.000998
    Model:[199/500]	Acc Score: 78.68	 Computing time: 0:00:00.000998
    Model:[200/500]	Acc Score: 80.36	 Computing time: 0:00:00.000998
    Model:[201/500]	Acc Score: 62.07	 Computing time: 0:00:00.000996
    Model:[202/500]	Acc Score: 70.59	 Computing time: 0:00:00.000968
    Model:[203/500]	Acc Score: 68.91	 Computing time: 0:00:00.000996
    Model:[204/500]	Acc Score: 66.44	 Computing time: 0:00:00.001032
    Model:[205/500]	Acc Score: 66.78	 Computing time: 0:00:00.000971
    Model:[206/500]	Acc Score: 69.81	 Computing time: 0:00:00
    Model:[207/500]	Acc Score: 73.29	 Computing time: 0:00:00.000994
    Model:[208/500]	Acc Score: 78.79	 Computing time: 0:00:00.001022
    Model:[209/500]	Acc Score: 78.68	 Computing time: 0:00:00.000973
    Model:[210/500]	Acc Score: 78.68	 Computing time: 0:00:00.000974
    Model:[211/500]	Acc Score: 71.16	 Computing time: 0:00:00.000975
    Model:[212/500]	Acc Score: 68.01	 Computing time: 0:00:00.000997
    Model:[213/500]	Acc Score: 62.74	 Computing time: 0:00:00.000993
    Model:[214/500]	Acc Score: 62.07	 Computing time: 0:00:00.000973
    Model:[215/500]	Acc Score: 61.73	 Computing time: 0:00:00.000967
    Model:[216/500]	Acc Score: 76.21	 Computing time: 0:00:00.000981
    Model:[217/500]	Acc Score: 62.07	 Computing time: 0:00:00.000974
    Model:[218/500]	Acc Score: 69.25	 Computing time: 0:00:00.000973
    Model:[219/500]	Acc Score: 63.86	 Computing time: 0:00:00.000998
    Model:[220/500]	Acc Score: 74.41	 Computing time: 0:00:00.000997
    Model:[221/500]	Acc Score: 63.41	 Computing time: 0:00:00.000976
    Model:[222/500]	Acc Score: 81.03	 Computing time: 0:00:00
    Model:[223/500]	Acc Score: 81.48	 Computing time: 0:00:00.000997
    Model:[224/500]	Acc Score: 77.1	 Computing time: 0:00:00.000971
    Model:[225/500]	Acc Score: 78.68	 Computing time: 0:00:00.000975
    Model:[226/500]	Acc Score: 66.78	 Computing time: 0:00:00.000997
    Model:[227/500]	Acc Score: 63.52	 Computing time: 0:00:00.000970
    Model:[228/500]	Acc Score: 74.97	 Computing time: 0:00:00.000971
    Model:[229/500]	Acc Score: 62.07	 Computing time: 0:00:00.000997
    Model:[230/500]	Acc Score: 71.04	 Computing time: 0:00:00.000997
    Model:[231/500]	Acc Score: 78.23	 Computing time: 0:00:00.000998
    Model:[232/500]	Acc Score: 75.08	 Computing time: 0:00:00.000998
    Model:[233/500]	Acc Score: 61.73	 Computing time: 0:00:00.001029
    Model:[234/500]	Acc Score: 80.36	 Computing time: 0:00:00.000996
    Model:[235/500]	Acc Score: 67.9	 Computing time: 0:00:00.001011
    Model:[236/500]	Acc Score: 81.71	 Computing time: 0:00:00.001029
    Model:[237/500]	Acc Score: 66.78	 Computing time: 0:00:00.000972
    Model:[238/500]	Acc Score: 78.68	 Computing time: 0:00:00.000974
    Model:[239/500]	Acc Score: 74.86	 Computing time: 0:00:00.000997
    Model:[240/500]	Acc Score: 77.89	 Computing time: 0:00:00.001001
    Model:[241/500]	Acc Score: 66.67	 Computing time: 0:00:00.000974
    Model:[242/500]	Acc Score: 78.68	 Computing time: 0:00:00.000973
    Model:[243/500]	Acc Score: 80.36	 Computing time: 0:00:00.001996
    Model:[244/500]	Acc Score: 76.88	 Computing time: 0:00:00.000974
    Model:[245/500]	Acc Score: 78.68	 Computing time: 0:00:00
    Model:[246/500]	Acc Score: 66.78	 Computing time: 0:00:00.000997
    Model:[247/500]	Acc Score: 81.03	 Computing time: 0:00:00.001026
    Model:[248/500]	Acc Score: 79.24	 Computing time: 0:00:00.000964
    Model:[249/500]	Acc Score: 62.18	 Computing time: 0:00:00.000961
    Model:[250/500]	Acc Score: 66.78	 Computing time: 0:00:00.000997
    Model:[251/500]	Acc Score: 78.68	 Computing time: 0:00:00.000996
    Model:[252/500]	Acc Score: 80.36	 Computing time: 0:00:00.000998
    Model:[253/500]	Acc Score: 78.23	 Computing time: 0:00:00.000974
    Model:[254/500]	Acc Score: 74.97	 Computing time: 0:00:00.000970
    Model:[255/500]	Acc Score: 80.36	 Computing time: 0:00:00.000994
    Model:[256/500]	Acc Score: 79.8	 Computing time: 0:00:00.000998
    Model:[257/500]	Acc Score: 78.23	 Computing time: 0:00:00.000970
    Model:[258/500]	Acc Score: 69.25	 Computing time: 0:00:00.000995
    Model:[259/500]	Acc Score: 81.71	 Computing time: 0:00:00.000997
    Model:[260/500]	Acc Score: 78.34	 Computing time: 0:00:00.001995
    Model:[261/500]	Acc Score: 78.23	 Computing time: 0:00:00.000974
    Model:[262/500]	Acc Score: 81.71	 Computing time: 0:00:00.000998
    Model:[263/500]	Acc Score: 78.68	 Computing time: 0:00:00.000986
    Model:[264/500]	Acc Score: 80.47	 Computing time: 0:00:00.000975
    Model:[265/500]	Acc Score: 78.23	 Computing time: 0:00:00.000974
    Model:[266/500]	Acc Score: 69.25	 Computing time: 0:00:00.000997
    Model:[267/500]	Acc Score: 79.69	 Computing time: 0:00:00.000973
    Model:[268/500]	Acc Score: 81.03	 Computing time: 0:00:00.000975
    Model:[269/500]	Acc Score: 78.23	 Computing time: 0:00:00.001992
    Model:[270/500]	Acc Score: 66.78	 Computing time: 0:00:00.000974
    Model:[271/500]	Acc Score: 78.9	 Computing time: 0:00:00.000977
    Model:[272/500]	Acc Score: 80.36	 Computing time: 0:00:00.000968
    Model:[273/500]	Acc Score: 78.68	 Computing time: 0:00:00.001001
    Model:[274/500]	Acc Score: 78.9	 Computing time: 0:00:00.000969
    Model:[275/500]	Acc Score: 78.34	 Computing time: 0:00:00.000974
    Model:[276/500]	Acc Score: 80.02	 Computing time: 0:00:00.002036
    Model:[277/500]	Acc Score: 78.68	 Computing time: 0:00:00.000995
    Model:[278/500]	Acc Score: 78.9	 Computing time: 0:00:00.000995
    Model:[279/500]	Acc Score: 74.07	 Computing time: 0:00:00.000974
    Model:[280/500]	Acc Score: 81.14	 Computing time: 0:00:00.000971
    Model:[281/500]	Acc Score: 66.78	 Computing time: 0:00:00.000973
    Model:[282/500]	Acc Score: 78.9	 Computing time: 0:00:00.000997
    Model:[283/500]	Acc Score: 78.68	 Computing time: 0:00:00.000997
    Model:[284/500]	Acc Score: 83.16	 Computing time: 0:00:00.000997
    Model:[285/500]	Acc Score: 78.68	 Computing time: 0:00:00.000994
    Model:[286/500]	Acc Score: 74.97	 Computing time: 0:00:00.000973
    Model:[287/500]	Acc Score: 78.68	 Computing time: 0:00:00.000997
    Model:[288/500]	Acc Score: 78.56	 Computing time: 0:00:00.000997
    Model:[289/500]	Acc Score: 78.68	 Computing time: 0:00:00.001000
    Model:[290/500]	Acc Score: 80.36	 Computing time: 0:00:00.000973
    Model:[291/500]	Acc Score: 78.68	 Computing time: 0:00:00.000971
    Model:[292/500]	Acc Score: 80.02	 Computing time: 0:00:00.000975
    Model:[293/500]	Acc Score: 66.78	 Computing time: 0:00:00.001995
    Model:[294/500]	Acc Score: 78.68	 Computing time: 0:00:00.000996
    Model:[295/500]	Acc Score: 78.68	 Computing time: 0:00:00.000996
    Model:[296/500]	Acc Score: 81.03	 Computing time: 0:00:00.000975
    Model:[297/500]	Acc Score: 66.67	 Computing time: 0:00:00.000974
    Model:[298/500]	Acc Score: 78.9	 Computing time: 0:00:00.000996
    Model:[299/500]	Acc Score: 74.97	 Computing time: 0:00:00.000997
    Model:[300/500]	Acc Score: 80.7	 Computing time: 0:00:00.000985
    Model:[301/500]	Acc Score: 61.73	 Computing time: 0:00:00.000999
    Model:[302/500]	Acc Score: 67.23	 Computing time: 0:00:00.000998
    Model:[303/500]	Acc Score: 70.15	 Computing time: 0:00:00.001970
    Model:[304/500]	Acc Score: 65.99	 Computing time: 0:00:00.000970
    Model:[305/500]	Acc Score: 66.67	 Computing time: 0:00:00.000999
    Model:[306/500]	Acc Score: 74.86	 Computing time: 0:00:00.000974
    Model:[307/500]	Acc Score: 78.68	 Computing time: 0:00:00.000974
    Model:[308/500]	Acc Score: 72.5	 Computing time: 0:00:00.000996
    Model:[309/500]	Acc Score: 63.41	 Computing time: 0:00:00.000985
    Model:[310/500]	Acc Score: 65.99	 Computing time: 0:00:00.000998
    Model:[311/500]	Acc Score: 66.78	 Computing time: 0:00:00.000997
    Model:[312/500]	Acc Score: 77.67	 Computing time: 0:00:00.000998
    Model:[313/500]	Acc Score: 62.07	 Computing time: 0:00:00.000999
    Model:[314/500]	Acc Score: 65.66	 Computing time: 0:00:00.000588
    Model:[315/500]	Acc Score: 62.74	 Computing time: 0:00:00.001609
    Model:[316/500]	Acc Score: 78.68	 Computing time: 0:00:00.000923
    Model:[317/500]	Acc Score: 66.78	 Computing time: 0:00:00.001510
    Model:[318/500]	Acc Score: 64.2	 Computing time: 0:00:00.000995
    Model:[319/500]	Acc Score: 78.68	 Computing time: 0:00:00.000997
    Model:[320/500]	Acc Score: 73.63	 Computing time: 0:00:00.000997
    Model:[321/500]	Acc Score: 78.23	 Computing time: 0:00:00.000982
    Model:[322/500]	Acc Score: 67.34	 Computing time: 0:00:00.000972
    Model:[323/500]	Acc Score: 78.68	 Computing time: 0:00:00.001000
    Model:[324/500]	Acc Score: 78.68	 Computing time: 0:00:00.000996
    Model:[325/500]	Acc Score: 78.23	 Computing time: 0:00:00.000973
    Model:[326/500]	Acc Score: 69.81	 Computing time: 0:00:00.000972
    Model:[327/500]	Acc Score: 77.67	 Computing time: 0:00:00.000997
    Model:[328/500]	Acc Score: 76.88	 Computing time: 0:00:00.000997
    Model:[329/500]	Acc Score: 66.67	 Computing time: 0:00:00.001022
    Model:[330/500]	Acc Score: 70.03	 Computing time: 0:00:00.000973
    Model:[331/500]	Acc Score: 70.82	 Computing time: 0:00:00.000995
    Model:[332/500]	Acc Score: 74.07	 Computing time: 0:00:00.000974
    Model:[333/500]	Acc Score: 78.23	 Computing time: 0:00:00
    Model:[334/500]	Acc Score: 78.68	 Computing time: 0:00:00.000998
    Model:[335/500]	Acc Score: 62.96	 Computing time: 0:00:00.000975
    Model:[336/500]	Acc Score: 74.86	 Computing time: 0:00:00.000999
    Model:[337/500]	Acc Score: 66.78	 Computing time: 0:00:00.000997
    Model:[338/500]	Acc Score: 74.52	 Computing time: 0:00:00.000974
    Model:[339/500]	Acc Score: 70.26	 Computing time: 0:00:00
    Model:[340/500]	Acc Score: 79.91	 Computing time: 0:00:00.000997
    Model:[341/500]	Acc Score: 66.67	 Computing time: 0:00:00.000997
    Model:[342/500]	Acc Score: 78.34	 Computing time: 0:00:00.000974
    Model:[343/500]	Acc Score: 78.68	 Computing time: 0:00:00.001022
    Model:[344/500]	Acc Score: 80.13	 Computing time: 0:00:00.000999
    Model:[345/500]	Acc Score: 66.67	 Computing time: 0:00:00.000973
    Model:[346/500]	Acc Score: 69.81	 Computing time: 0:00:00.000996
    Model:[347/500]	Acc Score: 79.01	 Computing time: 0:00:00.001995
    Model:[348/500]	Acc Score: 80.47	 Computing time: 0:00:00.000997
    Model:[349/500]	Acc Score: 66.67	 Computing time: 0:00:00.000996
    Model:[350/500]	Acc Score: 74.97	 Computing time: 0:00:00.000997
    Model:[351/500]	Acc Score: 70.71	 Computing time: 0:00:00.001995
    Model:[352/500]	Acc Score: 82.49	 Computing time: 0:00:00.001995
    Model:[353/500]	Acc Score: 78.23	 Computing time: 0:00:00.001994
    Model:[354/500]	Acc Score: 78.9	 Computing time: 0:00:00.001995
    Model:[355/500]	Acc Score: 81.03	 Computing time: 0:00:00.000998
    Model:[356/500]	Acc Score: 72.95	 Computing time: 0:00:00.000998
    Model:[357/500]	Acc Score: 78.23	 Computing time: 0:00:00.002993
    Model:[358/500]	Acc Score: 70.59	 Computing time: 0:00:00.001989
    Model:[359/500]	Acc Score: 69.81	 Computing time: 0:00:00.001000
    Model:[360/500]	Acc Score: 82.49	 Computing time: 0:00:00.000999
    Model:[361/500]	Acc Score: 78.68	 Computing time: 0:00:00.000997
    Model:[362/500]	Acc Score: 74.86	 Computing time: 0:00:00.000997
    Model:[363/500]	Acc Score: 78.68	 Computing time: 0:00:00.001996
    Model:[364/500]	Acc Score: 82.15	 Computing time: 0:00:00.001993
    Model:[365/500]	Acc Score: 78.68	 Computing time: 0:00:00.000995
    Model:[366/500]	Acc Score: 78.68	 Computing time: 0:00:00.000999
    Model:[367/500]	Acc Score: 76.43	 Computing time: 0:00:00.001026
    Model:[368/500]	Acc Score: 80.36	 Computing time: 0:00:00.000967
    Model:[369/500]	Acc Score: 78.23	 Computing time: 0:00:00.000966
    Model:[370/500]	Acc Score: 69.81	 Computing time: 0:00:00.001938
    Model:[371/500]	Acc Score: 81.71	 Computing time: 0:00:00.000996
    Model:[372/500]	Acc Score: 80.36	 Computing time: 0:00:00.000965
    Model:[373/500]	Acc Score: 63.41	 Computing time: 0:00:00.000998
    Model:[374/500]	Acc Score: 81.71	 Computing time: 0:00:00.001029
    Model:[375/500]	Acc Score: 81.03	 Computing time: 0:00:00.000974
    Model:[376/500]	Acc Score: 79.24	 Computing time: 0:00:00.000974
    Model:[377/500]	Acc Score: 78.23	 Computing time: 0:00:00
    Model:[378/500]	Acc Score: 67.12	 Computing time: 0:00:00.000998
    Model:[379/500]	Acc Score: 78.34	 Computing time: 0:00:00.000998
    Model:[380/500]	Acc Score: 81.03	 Computing time: 0:00:00.001028
    Model:[381/500]	Acc Score: 78.23	 Computing time: 0:00:00.000974
    Model:[382/500]	Acc Score: 78.9	 Computing time: 0:00:00.000973
    Model:[383/500]	Acc Score: 81.71	 Computing time: 0:00:00.000971
    Model:[384/500]	Acc Score: 81.71	 Computing time: 0:00:00.000973
    Model:[385/500]	Acc Score: 78.68	 Computing time: 0:00:00.000999
    Model:[386/500]	Acc Score: 74.86	 Computing time: 0:00:00.001000
    Model:[387/500]	Acc Score: 80.36	 Computing time: 0:00:00.000995
    Model:[388/500]	Acc Score: 79.01	 Computing time: 0:00:00.001994
    Model:[389/500]	Acc Score: 78.68	 Computing time: 0:00:00.000997
    Model:[390/500]	Acc Score: 78.68	 Computing time: 0:00:00.001028
    Model:[391/500]	Acc Score: 72.62	 Computing time: 0:00:00.000996
    Model:[392/500]	Acc Score: 81.71	 Computing time: 0:00:00.000973
    Model:[393/500]	Acc Score: 78.68	 Computing time: 0:00:00.000974
    Model:[394/500]	Acc Score: 81.71	 Computing time: 0:00:00.000973
    Model:[395/500]	Acc Score: 81.71	 Computing time: 0:00:00.000995
    Model:[396/500]	Acc Score: 79.01	 Computing time: 0:00:00.000997
    Model:[397/500]	Acc Score: 78.68	 Computing time: 0:00:00.001034
    Model:[398/500]	Acc Score: 74.52	 Computing time: 0:00:00.000959
    Model:[399/500]	Acc Score: 81.71	 Computing time: 0:00:00.000998
    Model:[400/500]	Acc Score: 81.71	 Computing time: 0:00:00.000971
    Model:[401/500]	Acc Score: 62.18	 Computing time: 0:00:00.000960
    Model:[402/500]	Acc Score: 63.3	 Computing time: 0:00:00.001979
    Model:[403/500]	Acc Score: 62.74	 Computing time: 0:00:00.000973
    Model:[404/500]	Acc Score: 79.46	 Computing time: 0:00:00.000975
    Model:[405/500]	Acc Score: 62.07	 Computing time: 0:00:00.000974
    Model:[406/500]	Acc Score: 62.07	 Computing time: 0:00:00
    Model:[407/500]	Acc Score: 78.68	 Computing time: 0:00:00.000973
    Model:[408/500]	Acc Score: 65.43	 Computing time: 0:00:00.000996
    Model:[409/500]	Acc Score: 62.18	 Computing time: 0:00:00.000998
    Model:[410/500]	Acc Score: 73.96	 Computing time: 0:00:00
    Model:[411/500]	Acc Score: 69.02	 Computing time: 0:00:00.000997
    Model:[412/500]	Acc Score: 71.38	 Computing time: 0:00:00.001171
    Model:[413/500]	Acc Score: 63.41	 Computing time: 0:00:00.000997
    Model:[414/500]	Acc Score: 65.99	 Computing time: 0:00:00.000973
    Model:[415/500]	Acc Score: 64.2	 Computing time: 0:00:00.000972
    Model:[416/500]	Acc Score: 74.86	 Computing time: 0:00:00.000996
    Model:[417/500]	Acc Score: 62.18	 Computing time: 0:00:00.000997
    Model:[418/500]	Acc Score: 74.97	 Computing time: 0:00:00.000973
    Model:[419/500]	Acc Score: 70.03	 Computing time: 0:00:00
    Model:[420/500]	Acc Score: 80.47	 Computing time: 0:00:00.002017
    Model:[421/500]	Acc Score: 62.07	 Computing time: 0:00:00.001021
    Model:[422/500]	Acc Score: 70.37	 Computing time: 0:00:00
    Model:[423/500]	Acc Score: 78.68	 Computing time: 0:00:00.001021
    Model:[424/500]	Acc Score: 74.41	 Computing time: 0:00:00.000974
    Model:[425/500]	Acc Score: 62.07	 Computing time: 0:00:00.000974
    Model:[426/500]	Acc Score: 67.45	 Computing time: 0:00:00.000997
    Model:[427/500]	Acc Score: 81.03	 Computing time: 0:00:00.000992
    Model:[428/500]	Acc Score: 77.1	 Computing time: 0:00:00.000975
    Model:[429/500]	Acc Score: 66.78	 Computing time: 0:00:00
    Model:[430/500]	Acc Score: 69.58	 Computing time: 0:00:00.000997
    Model:[431/500]	Acc Score: 70.37	 Computing time: 0:00:00
    Model:[432/500]	Acc Score: 73.74	 Computing time: 0:00:00.000997
    Model:[433/500]	Acc Score: 62.74	 Computing time: 0:00:00.000971
    Model:[434/500]	Acc Score: 69.25	 Computing time: 0:00:00.000974
    Model:[435/500]	Acc Score: 76.43	 Computing time: 0:00:00.001019
    Model:[436/500]	Acc Score: 71.72	 Computing time: 0:00:00.000973
    Model:[437/500]	Acc Score: 78.23	 Computing time: 0:00:00.000973
    Model:[438/500]	Acc Score: 78.9	 Computing time: 0:00:00.001004
    Model:[439/500]	Acc Score: 76.99	 Computing time: 0:00:00.001020
    Model:[440/500]	Acc Score: 76.32	 Computing time: 0:00:00.000980
    Model:[441/500]	Acc Score: 66.67	 Computing time: 0:00:00.000963
    Model:[442/500]	Acc Score: 70.59	 Computing time: 0:00:00.000973
    Model:[443/500]	Acc Score: 78.68	 Computing time: 0:00:00.000998
    Model:[444/500]	Acc Score: 69.81	 Computing time: 0:00:00.000997
    Model:[445/500]	Acc Score: 78.23	 Computing time: 0:00:00.001020
    Model:[446/500]	Acc Score: 78.9	 Computing time: 0:00:00.000998
    Model:[447/500]	Acc Score: 80.02	 Computing time: 0:00:00.000996
    Model:[448/500]	Acc Score: 78.68	 Computing time: 0:00:00.001004
    Model:[449/500]	Acc Score: 66.67	 Computing time: 0:00:00.000965
    Model:[450/500]	Acc Score: 70.59	 Computing time: 0:00:00.001967
    Model:[451/500]	Acc Score: 68.24	 Computing time: 0:00:00.000997
    Model:[452/500]	Acc Score: 78.9	 Computing time: 0:00:00.000965
    Model:[453/500]	Acc Score: 66.67	 Computing time: 0:00:00.000967
    Model:[454/500]	Acc Score: 78.68	 Computing time: 0:00:00.000997
    Model:[455/500]	Acc Score: 70.59	 Computing time: 0:00:00.000988
    Model:[456/500]	Acc Score: 79.69	 Computing time: 0:00:00.000999
    Model:[457/500]	Acc Score: 63.41	 Computing time: 0:00:00.000974
    Model:[458/500]	Acc Score: 74.64	 Computing time: 0:00:00.000973
    Model:[459/500]	Acc Score: 81.37	 Computing time: 0:00:00.000974
    Model:[460/500]	Acc Score: 80.36	 Computing time: 0:00:00
    Model:[461/500]	Acc Score: 78.23	 Computing time: 0:00:00.000967
    Model:[462/500]	Acc Score: 80.36	 Computing time: 0:00:00.000997
    Model:[463/500]	Acc Score: 78.23	 Computing time: 0:00:00.001030
    Model:[464/500]	Acc Score: 79.8	 Computing time: 0:00:00.001020
    Model:[465/500]	Acc Score: 78.68	 Computing time: 0:00:00.000997
    Model:[466/500]	Acc Score: 78.23	 Computing time: 0:00:00.000512
    Model:[467/500]	Acc Score: 76.88	 Computing time: 0:00:00.000973
    Model:[468/500]	Acc Score: 81.03	 Computing time: 0:00:00.000984
    Model:[469/500]	Acc Score: 78.68	 Computing time: 0:00:00.001508
    Model:[470/500]	Acc Score: 80.36	 Computing time: 0:00:00.000993
    Model:[471/500]	Acc Score: 81.03	 Computing time: 0:00:00.000997
    Model:[472/500]	Acc Score: 78.9	 Computing time: 0:00:00.001995
    Model:[473/500]	Acc Score: 66.67	 Computing time: 0:00:00.000997
    Model:[474/500]	Acc Score: 81.71	 Computing time: 0:00:00.000997
    Model:[475/500]	Acc Score: 81.71	 Computing time: 0:00:00.001007
    Model:[476/500]	Acc Score: 79.12	 Computing time: 0:00:00.000996
    Model:[477/500]	Acc Score: 78.23	 Computing time: 0:00:00.000997
    Model:[478/500]	Acc Score: 81.71	 Computing time: 0:00:00.000998
    Model:[479/500]	Acc Score: 78.68	 Computing time: 0:00:00.000998
    Model:[480/500]	Acc Score: 81.59	 Computing time: 0:00:00.001995
    Model:[481/500]	Acc Score: 78.23	 Computing time: 0:00:00.001000
    Model:[482/500]	Acc Score: 78.68	 Computing time: 0:00:00.000981
    Model:[483/500]	Acc Score: 78.9	 Computing time: 0:00:00.000967
    Model:[484/500]	Acc Score: 80.36	 Computing time: 0:00:00.000973
    Model:[485/500]	Acc Score: 78.23	 Computing time: 0:00:00
    Model:[486/500]	Acc Score: 81.71	 Computing time: 0:00:00.001021
    Model:[487/500]	Acc Score: 78.68	 Computing time: 0:00:00.000999
    Model:[488/500]	Acc Score: 79.01	 Computing time: 0:00:00.000996
    Model:[489/500]	Acc Score: 78.68	 Computing time: 0:00:00.000998
    Model:[490/500]	Acc Score: 78.68	 Computing time: 0:00:00.000977
    Model:[491/500]	Acc Score: 78.68	 Computing time: 0:00:00.001000
    Model:[492/500]	Acc Score: 83.16	 Computing time: 0:00:00.000995
    Model:[493/500]	Acc Score: 78.68	 Computing time: 0:00:00.000997
    Model:[494/500]	Acc Score: 78.68	 Computing time: 0:00:00.000997
    Model:[495/500]	Acc Score: 78.9	 Computing time: 0:00:00.000997
    Model:[496/500]	Acc Score: 80.02	 Computing time: 0:00:00.001980
    Model:[497/500]	Acc Score: 78.23	 Computing time: 0:00:00.000973
    Model:[498/500]	Acc Score: 80.36	 Computing time: 0:00:00.000973
    Model:[499/500]	Acc Score: 70.37	 Computing time: 0:00:00.000974
    Model:[500/500]	Acc Score: 81.37	 Computing time: 0:00:00.000997
    




    DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                           max_depth=5.0, max_features=0.5, max_leaf_nodes=5,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=6, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, presort='deprecated',
                           random_state=None, splitter='best')


