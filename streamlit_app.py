import streamlit as st
import src.ml_120 as ml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from PIL import Image

# Load the image
image = Image.open("src/image.png")

# Title of the app
st.title("Group 120: Machine Project Final Report")

# Introduction Section
st.header("Introduction")
st.write(
    """
    Determining the difficulty level of a course is often challenging for students, as multiple factors contribute to how demanding a class may be. In this project, we aim to simplify this process by analyzing three primary data points: grade distributions, professor ratings, and class sizes at Georgia Tech. Our objective is to classify classes into difficulty tiers, such as “Easy,” “Medium,” and “Hard,” using machine learning techniques. By making these insights available, we hope to provide students with an accessible resource for better course planning.
    """
)

# Problem Definition Section
st.header("Problem Definition")
st.write(
    """
    Our approach addresses a common dilemma for students: assessing the effort required for various courses. Instead of relying solely on subjective online reviews or anecdotal feedback, our model will use actual performance data. This structured classification can serve as an informed guide to help students anticipate course rigor and workload.
    """
)

# Data Collection & Preprocessing Section
st.header("Data Collection & Preprocessing")
st.write(
    """
    Our initial data collection efforts were ambitious, but challenges arose. We were unable to obtain professor ratings from external sources or gather class sizes efficiently without manually reviewing each data point. Despite these limitations, we successfully acquired Georgia Tech grade distribution data from the Office of Institutional Research & Planning. This dataset included various class features, but we selected only the average grade, standard deviations of the grades, instructor name, subject, and course number. Our main preprocessing was encoding our strings in integers for machine learning compatibility. However, this initial encoding did not include semantic meaning, such as associating harder subjects with higher numerical values. We also removed labs and recitations to exclude significant outliers and eliminated classes missing an average grade, as this indicated a 100% withdrawal rate. Four our truth values, we manually labeled courses as “very easy,” “easy,” “medium,” “hard,” and “very hard” based on course numbers and average GPA using a simple algorithm. Finally, we decided to try and normalize our data using a simple method, but while this did give slightly better results, we ultimately decided to use scalarization as it gave even better results than normalization for our models. 
    """
)

# Machine Learning Model Section
st.header("Machine Learning Model")
st.subheader("K-means actual implementation")
st.write(
    """
    After first preprocessing our data, we then used this preprocessed data as the training data for the K-means clustering, which was our first implemented model. We selected kmeans because it's a relatively simple and easy to use starter model for our data, and to act as a baseline for later comparisons. We used scikit-learn for the k-means clustering implementation, then used Streamlit for the visualization of the results. We decided to plot the grade distribution data variables such as course number and average grade as the independent variables and the true labels as the dependent variable for better visualized clusters.
    """
)

st.subheader("Support Vector Machine")
st.write(
    """
    We then moved on to adding super vector machine model to our code. We selected this model due to its ability to easily classify where the data would have a good amount of linear separability, which we believe the data has. This is due to its use of a hyperplane. We also experimented with other types of kernels for the SVM.
    """
)

st.subheader("Multilayer Perceptron")
st.write(
    """
    Next, we added an MLP mode mainly because we wanted to see how a neural network would work with our given data, but also because we believed that an MLP specifically would have the best classification for our model due to its ability to capture complex nature. While we believed that the data was mostly linear, we wanted to see if the MLP would have better classification because it saw some connections we, and other models, didn’t see.     
    """
)

st.subheader("Random Forest")
st.write(
    """
    Finally, we also implemented random forests as our last model. We chose this model due to its ability to counter overfitting as we feared that our previous models may only be useful for the training data that we gave it. We wanted to ensure that our code could be used outside of the limited data that we gave it.     
    """
)

# Results Section
st.header("Results")
st.write(
    """
    As seen in the visualization, our original labels do not match the output clusters. One reason for this could be that our K-means algorithm is actually finding a different pattern to determine the difficulty of the class that we aren’t noticing. Another reason is that the variables input into the algorithm are not good indicators of the difficulty of the class. We originally thought the K-means would find a connection between higher course number and a more difficult class, but our implementation actually saw a wide variety of difficulties among the higher courses. This means course number is most likely not a good indicator of the difficulty to input for a K-means implementation. We may also be trying to find correlations between variables from classes that are too dissimilar either in course number or class type. If we chose a dataset too variably different then we would need many more input variables to accurately determine good clustering for difficulty among classes.

    We also think that our encoding may have some problems as well. We have no semantics, such as encoding harder subjects with higher numbers, within our encoding. This could be acceptable in some ML methods, but K-Means is an unsupervised method that measures difference.
    """
)

# Next Steps Section
st.header("Next Steps")
st.write(
    """
    First, we want to improve our encoding methods to have more semantics, especially for kmeans. To do this, we would want professor ratings to help with our new encoding method to ensure lower rated professors are encoding with higher values. Next, we would want to introduce new preprocessing methods such as PCA to reduce the number of features. Finally, we will implement our other models, such as SVMs and random forests, for our analysis and comparison of the algorithms.
    """
)

# Contributions Table
st.header("Contributions Table")
st.write(
    """
    - **Hayk Arsenyan**: Report Review, Methods Review
    - **Alexandre Abchee**: Report Writing, Methods Review, Report Review
    - **John Andrade**: Preprocessing, Methods Review, Data Collection, Report Review
    - **Mattias Anderson**: Methods Implementation, Preprocessing, Results, Report Review, Video
    - **Juan Macias-Romero**: Methods Implementation, Streamlit, Data Collection, Report Review
    """
)


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


# IMAGE

X_train, y_train= ml.preprocess_data('data/Grade_Distribution_Data.xlsx')
X, Xv_scaled, labels, fm_score, silhouette  = ml.Kmeans(X_train, y_train)
u_labels = np.unique(labels)

# Display the image
# st.image(image, caption="Problems with link", use_column_width=True)

# Visualization
X['Cluster'] = labels
Xv_scaled['Cluster'] = labels

# plt.title('Raw data')
# plt.scatter(X['AverageGrade'], 
#             X['Number'],
#             alpha=0.5,
#             s=20,
#             rasterized=True)
# plt.legend()
# plt.show()

# # # Plot the data with clusters in the original scale
# # plt.title('Scaled data')
# # for cluster in Xv_scaled['Cluster'].unique():
# #     cluster_data = Xv_scaled[Xv_scaled['Cluster'] == cluster]
# #     plt.scatter(cluster_data['AverageGrade'], cluster_data['Number'], label=f'Cluster {cluster}')  
# # plt.legend()
# # plt.show()

# # # Plot the data with clusters in the original scale
# # plt.title('Original data')
# # for cluster in Xv['Cluster'].unique():
# #     cluster_data = Xv[Xv['Cluster'] == cluster]
# #     plt.scatter(cluster_data['AverageGrade'], 
# #                cluster_data['Number'],
# #                label=f'Cluster {cluster}',
# #                alpha=0.5,
# #                s=20,
# #                rasterized=True)
# # plt.legend()
# # plt.show()


# Raw data plot
st.title('Raw data')
scatter_data = pd.DataFrame({
    'Course Number': X['Number'],
    'True label': y_train
})
st.scatter_chart(
    data=scatter_data,
    x='Course Number',
    y='True label'
)

# Scaled data plot with clusters
st.title('Scaled data')
st.scatter_chart(
    data=Xv_scaled,
    x='AverageGrade',
    y='Number',
    color='Cluster'
)

# Scaled data plot with clusters
st.title('Original data')
st.scatter_chart(
    data=X,
    x='AverageGrade',
    y='Number',
    color='Cluster'
)

# Original data plot with clusters: Course Number
st.title('Course Number')
scatter_data = pd.DataFrame({
    'Course Number': X['Number'],
    'True label': y_train,
    'Cluster': X['Cluster']
})
st.scatter_chart(
    data=scatter_data,
    x='Course Number',
    y='True label',
    color='Cluster'
)

# Original data plot with clusters: Average_Grade
st.title('Average Grade')
scatter_data = pd.DataFrame({
    'Average_Grade': X['AverageGrade'],
    'True label': y_train,
    'Cluster': X['Cluster']
})
st.scatter_chart(
    data=scatter_data,
    x='Average_Grade',
    y='True label',
    color='Cluster'
)

# Original data plot with clusters: Average_Grade
st.title('Instructor')
scatter_data = pd.DataFrame({
    'Instructor': X['Instructor_bin'],
    'True label': y_train,
    'Cluster': X['Cluster']
})
st.scatter_chart(
    data=scatter_data,
    x='Instructor',
    y='True label',
    color='Cluster'
)

# Original data plot with clusters: Subject
st.title('Subject')
scatter_data = pd.DataFrame({
    'Subject': X['Subject'],
    'True label': y_train,
    'Cluster': X['Cluster']
})
st.scatter_chart(
    data=scatter_data,
    x='Subject',
    y='True label',
    color='Cluster'
)

st.header("Clustering Evaluation Metrics")
st.write(f"**Fowlkes-Mallows score:** {fm_score}")
st.write(f"**Silhouette score:** {silhouette}")
st.write(f"*Please read our report for more information*")


