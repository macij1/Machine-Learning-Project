import streamlit as st
import src.ml_120 as ml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from PIL import Image

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
st.subheader("Overall")
st.write(
    """
    All things considered, our models performed well, except for k-means. The classification was usually correct, and our F1 scores were very high for our other models. In the future, our models would work much better, especially K-means, if we were able to obtain more data relevant to the course difficulty that do not have too much correlation to the strongest feature, average GPA, such as course schedules and the ratings of the professor. 
    """
)

st.subheader("K-Means ")
st.write(
    """
    K-means under performance could have been a result of a few issues. The first is that the k-means algorithm might have identified patterns in the data that do not match our assumptions about course difficulty. It also could have been that the input variables may not be reliable indicators of difficulty, as evidenced by the wide variation in difficulty among higher course numbers. These problems could have been reduced by better preprocessing, but due to the performance of other models, we are not sure. In the future, we could run other models to find feature importance and limit the k-means to only those features to have a better chance of good clustering in the results.     
    """
)

st.subheader("Suport Vector Machine ")
st.write(
    """
    SVM performed extremely well for our purposes, and this is most likely due to how we created our truth labels. They allowed for very easy linear separations of all the clusters, which is where SVMs excel. We used what we found to be the two most important features to define the hyperplane, which were the average GPA and the course number. Looking at the results, the SVM was able to clearly define clusters and was best defined by the linear kernel. Compared to the other models, the SVM clearly defined groupings when comparing the course number and the average GPA. This makes sense because the course number and average GPA were some of the strongest features, which is necessary for strong SVM results.     
    """
)

st.subheader("Multilayer Perceptron ")
st.write(
    """
    The neural net performed well for the limited feature pool given. For our implementation, we fine-tuned our hyperparameters to try to our weighted F1-score and accuracy which was 0.97, which is very good. As seen in the loss plot, the neural network trained off the provided data very well to provide an accurate model using all features we could obtain. Compared to the other models, the multilayer perceptron had a very strong accuracy while being able to use all of the given features.      
    """
)

st.subheader("Random Forests ")
st.write(
    """
    As seen in the visualizations, our labels do not fully align with the output cluster. A reason could be that our Random Forest model is identifying different patterns that aren’t related to our assumptions. Another reason is that some variables are not strong indicators is that the course number and instructors were not as influential as we initially thought. 

    We originally thought the model would find a clear connection between higher course numbers and increased difficulty. The implementation revealed a different variety of difficulties among higher numbered classes. This meant the course number was not a good indicator of difficulty. As expected, our most influential feature was the average GPA with the standard deviation of the class grade coming in second. When comparing the random forests results with the other results it is shown that random forests best indicated the strengths of each feature more than the other models.     
    """
)

# Next Steps Section
st.header("Next Steps")
st.write(
    """
    In the future, we would improve our encoding methods by enhancing encoding semantics to align with expected difficulty measures, such as associating lower professor ratings with higher values. We would also implement additional models to explore and implement other models, such as SVM and Random Forest, to compare their performance against K-Means and determine the best algorithm for classifying course difficulty. We also would like to expand the number of features we could obtain. We would develop methods to automatically gather class size and professor ratings data for richer input features. 

    We would keep our models the same as they worked well for what we wanted from each, except for maybe k-means. However, we would want to tweak parameters of each model a little more, especially for k-means, if we used it again, and random forest.      
    """
)


# Contributions Table
st.header("Contributions Table")
st.write(
    """
    - **Hayk Arsenyan**: Feature Engineering, Data Collection, Methods Review, Report Review
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


# Data Acquisition and preprocessing
X, Y, data = ml.preprocess_data('data/Grade_Distribution_Data.xlsx')
X_train, X_test, Y_train, Y_test = ml.split(data) 
X_train_scaled = ml.scale(X_train)
X_test_scaled = ml.scale(X_test)
X_scaled = ml.scale(X) # without splitting, for KMeans

# KMeans
labels, fm_score, silhouette  = ml.Kmeans(X, Y)
u_labels = np.unique(labels)

# Visualization
X['Cluster'] = labels
X_scaled['Cluster'] = labels


# Raw data plot
st.title('Raw data')
scatter_data = pd.DataFrame({
    'Course Number': X['Number'],
    'True label': Y_train
})
st.scatter_chart(
    data=scatter_data,
    x='Course Number',
    y='True label'
)

# Scaled data plot with clusters
st.title('Scaled data')
st.scatter_chart(
    data=X_scaled,
    x='AverageGrade',
    y='Number',
    color='Cluster'
)

# # Scaled data plot with clusters
# st.title('Original data')
# st.scatter_chart(
#     data=X,
#     x='AverageGrade',
#     y='Number',
#     color='Cluster'
# )

# # Original data plot with clusters: Course Number
# st.title('Course Number')
# scatter_data = pd.DataFrame({
#     'Course Number': X['Number'],
#     'True label': Y_train,
#     'Cluster': X['Cluster']
# })
# st.scatter_chart(
#     data=scatter_data,
#     x='Course Number',
#     y='True label',
#     color='Cluster'
# )

# # Original data plot with clusters: Average_Grade
# st.title('Average Grade')
# scatter_data = pd.DataFrame({
#     'Average_Grade': X['AverageGrade'],
#     'True label': Y_train,
#     'Cluster': X['Cluster']
# })
# st.scatter_chart(
#     data=scatter_data,
#     x='Average_Grade',
#     y='True label',
#     color='Cluster'
# )

# # Original data plot with clusters: Subject
# st.title('Subject')
# scatter_data = pd.DataFrame({
#     'Subject': X['Subject'],
#     'True label': Y_train,
#     'Cluster': X['Cluster']
# })
# st.scatter_chart(
#     data=scatter_data,
#     x='Subject',
#     y='True label',
#     color='Cluster'
# )

st.write("### Clustering Evaluation Metrics")
st.write(f"**Fowlkes-Mallows score:** {fm_score}")
st.write(f"**Silhouette score:** {silhouette}")

# Random Forest
st.write("# Random Forest")
ml.RF(X_train_scaled, X_test_scaled, Y_train, Y_test)

# SVM
ml.SVM(X_train_scaled, X_test_scaled, Y_train, Y_test)

# Neural Network
ml.NN(X_train_scaled, X_test_scaled, Y_train, Y_test)


