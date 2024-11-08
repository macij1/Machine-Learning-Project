import streamlit as st
import ml_120 as utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import streamlit as st

# Title of the app
st.title("Group 120: Machine Project Overview")

# Introduction Section
st.header("Introduction")
st.write("""
Many students struggle with determining how hard a class can be due to many varying factors. In our project, we aim to simplify this by looking at grade distributions for classes, professor ratings, and class sizes to determine the difficulty of a class.

Our main dataset is the grade distribution of all Georgia Tech classes [2]. This website can filter by college, course number, professor, and semester, and provides the average GPA of the class during that time. We will also use professor rating websites to see how professor ratings affect GPA [3]. Finally, we will factor in class sizes for determining difficulty [1].
""")

# Problem Definition Section
st.header("Problem Definition")
st.write("""
A common dilemma for college students is determining how difficult a class can be, especially without external resources. This is crucial for Georgia Tech students, as they often care about their grade and the amount of effort required in these classes. We aim to simplify this process with an ML approach, classifying class difficulty into ranges from "easy" to "very hard" using grade distribution, professor rating, and class size as features.
""")

# Methods Section
st.header("Methods")

# Data Acquisition Subheader
st.subheader("Data Acquisition")
st.write("""
We plan to automate data acquisition, as grade distribution data is not directly downloadable, and professor ratings and class sizes need to be gathered as well. We will consider using web scraping tools like BeautifulSoup in our data pipeline, and if successful, we will apply this to other websites like RateMyProfessor.
""")

# Preprocessing Subheader
st.subheader("Preprocessing")

st.write("""
- **One-Hot Encoding**: We will use Scikit-learn's `OneHotEncoder` to convert categorical data (e.g., professor names, department names, semesters) into binary values.
- **Standard Scaling**: We will apply Scikit-learn's `StandardScaler` to remove the mean and scale features to unit variance. This is crucial for algorithms sensitive to feature scaling, such as SVM or K-means.
- **Handling Missing Values**: We will use Scikit-learn's `SimpleImputer` to define strategies for imputing missing values (e.g., replacing them with zero, mean, median, or the most frequent value).

We will also explore advanced preprocessing methods like non-linear transformations and normalization.
""")

# Machine Learning Methods Subheader
st.subheader("Machine Learning Methods")
st.write("""
- **Random Forest**: We will use Scikit-learn's `RandomForestClassifier` or `RandomForestRegressor` to predict grade distributions based on features like professor, department, semester, and class size, and to analyze feature importance.
- **Support Vector Machines (SVM)**: We will use Scikit-learn's `SVC` (Support Vector Classifier) to classify classes into difficulty levels (e.g., “Easy,” “Medium,” “Hard,” “Very Hard”) and compare these classifications with simpler metrics.
- **K-Means**: We will use Scikit-learn's `KMeans` to explore clustering in our database and identify natural groupings among classes, such as clusters with consistently high or low grade distributions.
""")

# Potential Methods for Further Exploration Subheader
st.subheader("Potential Methods for Further Exploration")
st.write("""
- **Principal Component Analysis (PCA)**
- **Gaussian Mixture Models (GMM)**
- **Gradient Boosting Machines (GBM)**
""")

# Results and Discussion Section
st.header("Results/Discussion")
st.write("""
Since we are primarily doing classification, our main evaluation metric will be the **F1 score**, as it provides a good balance of precision and recall. We are aiming for an F1 score greater than 0.8. We will also use the **recall score** to determine how well our model correctly identifies class difficulty. Finally, although the **accuracy score** doesn't tell the full story, we will include it to measure how often our classifiers fail to correctly identify class difficulty.
""")

# Citations Section
st.header("Citations")
st.write("""
1. [Small Classes, Big Possibilities](https://www.aasa.org/resources/resource/small-classes-big-possibilities) [Accessed Oct. 4, 2024]
2. [Grade Distribution - Georgia Tech](https://lite.gatech.edu/lite_script/dashboards/grade_distribution.html) [Accessed Oct. 4, 2024]
3. [Rate My Professors](https://www.ratemyprofessors.com/) [Accessed Oct. 4, 2024]
""")

# Gantt Chart Section
st.header("Gantt Chart")
st.write("You can view our Gantt chart [here](https://gtvault.sharepoint.com/:x:/s/CS4641ProjectGroup120/EZY21_YXBgZOlRWj8EyASzAB2wrWbY-PQEUM1Bkezptn8Q?e=pntQ71).")

# Contributions Section
st.header("Contributions")
st.write("""
- **John Andrade**: Introduction, problem definition, citations, Gantt chart
- **Juan Macias-Romero**: GitHub repository, Methods, Streamlit
- **Mattias Anderson**: Proposal review, Video
- **Alexandre Abchee**: Methods, Data Acquisition
- **Hayk Arsenyan**: Results/Discussion, Methods
""")



X_train, y_train = utils.preprocess_data('data/Grade_Distribution_Data.xlsx')
Xv, Xv_scaled, labels = utils.two_d_Kmeans(X_train, y_train)
u_labels = np.unique(labels)

# Visualization
Xv['Cluster'] = labels
Xv_scaled['Cluster'] = labels

# plt.title('Raw data')
# plt.scatter(Xv['AverageGrade'], 
#             Xv['Number'],
#             alpha=0.5,
#             s=20,
#             rasterized=True)
# plt.legend()
# plt.show()

# # Plot the data with clusters in the original scale
# plt.title('Scaled data')
# for cluster in Xv_scaled['Cluster'].unique():
#     cluster_data = Xv_scaled[Xv_scaled['Cluster'] == cluster]
#     plt.scatter(cluster_data['AverageGrade'], cluster_data['Number'], label=f'Cluster {cluster}')  
# plt.legend()
# plt.show()

# # Plot the data with clusters in the original scale
# plt.title('Original data')
# for cluster in Xv['Cluster'].unique():
#     cluster_data = Xv[Xv['Cluster'] == cluster]
#     plt.scatter(cluster_data['AverageGrade'], 
#                cluster_data['Number'],
#                label=f'Cluster {cluster}',
#                alpha=0.5,
#                s=20,
#                rasterized=True)
# plt.legend()
# plt.show()


# Raw data plot
st.title('Raw data')
scatter_data = pd.DataFrame({
    'AverageGrade': Xv['AverageGrade'],
    'Number': Xv['Number']
})
st.scatter_chart(
    data=scatter_data,
    x='AverageGrade',
    y='Number'
)

# Scaled data plot with clusters
st.title('Scaled data')
st.scatter_chart(
    data=Xv_scaled,
    x='AverageGrade',
    y='Number',
    color='Cluster'
)

# Original data plot with clusters
st.title('Original data')
st.scatter_chart(
    data=Xv,
    x='AverageGrade',
    y='Number',
    color='Cluster'
)