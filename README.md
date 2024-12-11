# Artificial Intelligence & Machine Learning

Machine learning is a subset of artificial intelligence (AI) that focues on the develoment of algorithms and statistical models that enable computers to learn and improve from expreience, without being explicitly programmed.
The core idea is to allow computers to learn from data and make predictions or decisions based on that learning.

## Key concepts in ML
1. **Data**: Data is the foundation of ML. It can come in various forms, such as text, images, audio or structured data like spreadsheets. The quality and quntity of data greatly infulence the performance of a machine learning model.
2. **Features**: Featuresaretheindividualmeasurablepropertiesorcharacteristicsof the data that are used as input for machine learning algorithms. For example, in an image recognition task, the pixel values of an image could be the features.
3. **Algorithms**: Machinelearningalgorithmsarethemathematicalmodelsor procedures used to learn patterns and relationships from data. These algorithms can be classified into different categories based on their learning style, such as supervised learning, unsupervised learning, semi-supervised learning, and reinforcement learning.
4. **Training**: Trainingamachinelearningmodelinvolvesfeedingitwithlabeleddata (in supervised learning) or unlabeled data (in unsupervised learning), allowing the model to learn the underlying patterns. During training, the model adjusts its parameters to minimize the difference between its predictions and the actual outcomes.
5. **Evaluation**: Aftertraining,theperformanceofamachinelearningmodelneedsto be evaluated using separate test data to assess how well it generalizes to new, unseen data. Common evaluation metrics include accuracy, precision, recall, F1-score, and area under the receiver operating characteristic curve (AUC-ROC), among others.
6. **Deployment**: Onceamachinelearningmodelhasbeentrainedandevaluated,it can be deployed to make predictions or decisions on new data in real-world applications. Deployment involves integrating the model into existing systems or platforms where it can be utilized.

Machine learning has numerous applications across various domains, including but not limited to:

* Image and speech recognition
* Natural language processing (NLP)
* Recommendation systems
* Predictive analytics
* Autonomous vehicles
* Healthcare (e.g., disease diagnosis)
* Finance (e.g., fraud detection)

Overall, machine learning has the potential to revolutionize industries by automating tasks, extracting valuable insights from data, and enabling intelligent decision-making systems.

## Understanding the Data
* **Data Description** - Understand the dataset, including the structure, varialbles, and types (numerical, categorial).
* **Domain Knowledge** - Leverage domain knowledge to understand teh context and relevance of the data.

## Exploratory Data Analysis (EDA)
Exploratory Data Analysis (EDA) is a process of describing the data by means of statistical and visualization techniques in order to bring important aspects of that data into focus for further analysis. This involves inspecting the dataset from many angles, describing & summarizing it without making any assumptions about its contents.
EDA is a significant step to take before diving into statistical modeling or machine learning, to ensure the data is really what it is claimed to be and that there are no obvious errors. It should be part of data science projects in every organization.

### EDA process: https://www.analyticsvidhya.com/blog/2021/08/exploratory-data-analysis-and-visualization-techniques-in-data-science/
* **Understand the Data** : Gather information about the data, such as the number of rows and columns, and the type of information each column contains. This includes understanding single variables and their distributions.

  - [X] Import necessary Python libraries
  ```python
  import pandas as pd
  import matplotlib.pyplot as plt
  import seaborn as sns
  ```
  
  - [X]  In our example we have used the [Car Prediction](https://github.com/tarunkhurana2015/artificial_intel-machine_learning/blob/main/EDA/CarPrice.csv) data set
  ```python
   df = pd.read_csv('CarPrice.csv')
  ```

  - [X]  Use the `df.head()` to understand the first few rows.
  ```python
   df.head()
  ```
  <img width="1257" alt="image" src="https://github.com/user-attachments/assets/6d05885c-35ab-43f2-b447-5e2e510230e6">

  - [X]  Use the `df.shape` to understand the size of the dataset
  ```python
   df.shape
  ```
  <img width="152" alt="image" src="https://github.com/user-attachments/assets/5030dd55-ca8e-4523-b68a-7e51f12dcd7d">

   - [X]  Use the `df.info()` to understand the data types
  ```python
   df.info()
  ```
  <img width="367" alt="image" src="https://github.com/user-attachments/assets/95757b76-05c7-45f9-bb03-247cbaacb76a">


* **Clean the Data**: Fix issues like missing or incorrect values. Preprocessing is essential to ensure the data is ready for analysis and predictive modeling.

  - [X]  Clean up the column `CarName` to extract the `CompanyName` and drop the `CarName` column
   ```python
    CompanyName = df['CarName'].apply(lambda x: x.split(' ')[0])
    df.insert(3,"CompanyName", CompanyName)
    df.drop(['CarName'], axis=1, inplace=True)
    df.head()
   ```
   ![image](https://github.com/user-attachments/assets/51c2fdf8-b682-4b65-b601-15cb7adac4c5)

  - [X]  Check for data uniqueness
  ```python
  df.CompanyName.unique()
  ```
   ![image](https://github.com/user-attachments/assets/f7f21988-2d93-494f-8465-88c6d23ff61e)

  - [X]  Clean up the wrong spellings/ repeated data
  ```python
   # make the case consistent
   df.CompanyName = df.CompanyName.str.lower()

   # handle the names mismatch
   def replace_name(a,b):
   df.CompanyName.replace(a,b,inplace=True)

   replace_name('maxda', 'mazda')
   replace_name('porcshce', 'porsche')
   replace_name('toyouta', 'toyota')
   replace_name('vokswagen', 'volkswagen')
   replace_name('vw', 'volkswagen')

   df.CompanyName.unique()
  ```
   ![image](https://github.com/user-attachments/assets/41758bae-145d-4eef-9512-37ec1e8fe3c4)

  - [X]  Check for any data duplications
  ```python
  df.loc[df.duplicated()]
  ```
   ![image](https://github.com/user-attachments/assets/5f148ded-40e6-44d0-9ea9-f057f07a5fe2)

* **General Statistical Analysis** - Compute mean, median, standard deviation, and other summary statistics.
  
  ```python
  df.describe()
  ```
  ![image](https://github.com/user-attachments/assets/2de456be-db37-4c60-a7c2-b6b4a7026c6a)

* **Data Visualization**
  * **Histogram and Box Plots** - For understanding the distribution of variables.
  * **Scatter Plots** - To visualize the relationship between predictor(s) and response variable.
  * **Pair Plots** - To understand relationships between multiple variables.
* **Correlation Matrix** - Identify the linear relationship between variables.
* **Feature Scaling/ Engineering** - Normalize or standardize numerical features. Create new features or transform existing ones to capture underlying patterns.
  ```python
  cars_lr = cars[['price', 'aspiration','carbody', 'drivewheel','wheelbase',
                  'curbweight', 'enginetype', 'cylindernumber', 'enginesize', 'boreratio','horsepower',
                   'fuelsystem', 'carlength','carwidth', 'price']]
   # define the map function to map the column name to the unique values in that column
  def dummies(x, df):
      temp = pd.get_dummies(df[x], drop_first=True)
      df = pd.concat([df, temp], axis=1)
      df.drop([x], axis=1, inplace=True)
      return df

  cars_lr = dummies('fuelsystem', cars_lr)
  cars_lr = dummies('aspiration',cars_lr)
  cars_lr = dummies('carbody',cars_lr)
  cars_lr = dummies('drivewheel',cars_lr)
  cars_lr = dummies('enginetype',cars_lr)
  cars_lr = dummies('cylindernumber',cars_lr)

  cars_lr.columns
  ```
  ![image](https://github.com/user-attachments/assets/a0b45d86-c383-4def-91dd-9a78ae1649fc)
  
* **Encoding Categorical Variables** - Use techniques like one-hot encoding or label encoding.
  ```python
   # One-hot encode categorical variables if needed
  X = pd.get_dummies(X, columns=['fueltype'], drop_first=True)
  ```

## Model Training/ Fitting

  - [ ] Model Selection/ Training/ fitting
      - [ ] Decide the algorithm
          - [ ] Linear Regression
              - [ ] Simple Linear Regression
              - [ ] Multiple Linear Regression
              - [ ] Polynomial Pregression
          - [ ] Regularization (Ridge, Lasso, Elastic Net)

  ![image](https://github.com/user-attachments/assets/d8998278-339f-4560-b717-fee0ee211f2a)

  **Under Fitting** - The trained model is not working well on the training data and can't generalize to new data.
  **Over Fitting** - the trained model is working very well on the training data and can't generalize well to the new data.
  

