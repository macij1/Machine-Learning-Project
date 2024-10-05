import streamlit as st

st.write(
    """Introduction 

Many students have trouble deciding how hard a class can be due to many vary factors. In our project, we wish to simplify this by looking at grade distributions for classes, professor rating, and class size to determine the difficulty of a class. Our main dataset we will use is a grade distribution of all Georgia Tech classes [2]. This website can filter by college, course number, professor, and semester and relays the average GPA of the class during that time. We will also use professor rating websites to determine how the professor ratings can affect the GPA [3]. Finally, we will also factor in class sizes for determining difficulty [1]. 

 

Problem Definition 

A common dilemma college students face is determining how difficult a class can be, especially without using external resources, which is important for Georgia Tech students as they very often care about their grade in these classes as well as how much effort is required. We aim to simplify this process with an ML approach through classifying class difficulty into ranges from easy to very hard with the previously mentioned factors of grade distribution, professor rating, and class size. 

 

Methods 

Data Acquisition 

We aim to automate the data acquisition process since the grade distribution data cannot be downloaded directly as well as professor rating and class size. We will consider using a web-scraping tool such as BeautifulSoup in our data pipeline, and if successful, we will apply this to other websites such as RateMyProfessor. 

 

Preprocessing 

One-Hot Encoding: 
We will utilize Scikit-learn's OneHotEncoder to convert categorical data, such as professor names, department names, or different semesters, into binary values. 

Standard Scaling: 
We will apply Scikit-learn's StandardScaler to remove the mean and scale the features to unit variance, which is crucial for algorithms sensitive to feature scaling, such as SVM or K-means. 

Handling Missing Values: 
We will use Scikit-learn's SimpleImputer to define strategies for imputing missing values, such as replacing them with zero, mean, median, or the most frequent value. 

Other advanced preprocessing methods we have considered include non-linear transformations and normalization. 

 

Machine Learning Methods 

Random Forest: 
We will employ Scikit-learn's RandomForestClassifier or RandomForestRegressor to predict grade distributions based on features like professor, department, semester, or class size and to analyze feature importance. 

Support Vector Machines (SVM): 
We will utilize Scikit-learn's SVC (Support Vector Classifier) to classify classes based on expected difficulty levels (e.g., “Easy,” “Medium,” “Hard,” and “Very Hard”) and compare these classifications with simpler metrics. 

K-Means: 
We will use Scikit-learn's KMeans to explore clustering in our database and identify natural groupings among classes, such as clusters with consistently high or low grade distributions. 

 

Potential Methods for Further Exploration: 

Principal Component Analysis (PCA) 

Gaussian Mixture Models (GMM) 

Gradient Boosting Machines (GBM) 

 

Results/Discussion 

Since we are mainly doing classification, we will want to use good metrics for classification. Our main scoring method will be the F1 score as it gives a good balance of judging precision and accuracy of our ML methods. This score will easily, but not entirely, tell us how well our methods are performing. The F1 score ranges from 1 to 0 with 1 being perfect, so we are looking for >0.8 from our methods. We will also use the recall score of our ML methods to determine how well they correctly identify a class’s difficulty. Finally, even though the accuracy score usually does not tell the full story, we will use it as well to tell how often our classifiers fail to correctly identify. 

 

Word Count (not including anything this line and below): 589 

 

Citations 

https://www.aasa.org/resources/resource/small-classes-big-possibilities 

[1] “Small classes, big possibilities,” Default, https://www.aasa.org/resources/resource/small-classes-big-possibilities [accessed Oct. 4, 2024]. 

 

https://lite.gatech.edu/lite_script/dashboards/grade_distribution.html 

[2] Grade distribution, https://lite.gatech.edu/lite_script/dashboards/grade_distribution.html [accessed Oct. 4, 2024]. 

 

https://www.ratemyprofessors.com/ 

[3] “Find and rate your professor or school,” Rate My Professors, https://www.ratemyprofessors.com/ [accessed Oct. 4, 2024]. 

 
Gantt Chart:
https://gtvault.sharepoint.com/:x:/s/CS4641ProjectGroup120/EZY21_YXBgZOlRWj8EyASzAB2wrWbY-PQEUM1Bkezptn8Q?e=pntQ71


Contributions: 

John Andrade 

Introduction, problem definition, citations, Gantt chart 

Juan Macias-Romero 

GitHub repository, Methods, Streamlit 

Mattias Anderson 

Proposal review, Video 

Alexandre Abchee 

Methods, Data Acquisition  

Hayk Arsenyan 

Results/Discussion, Methods 

 """
)
