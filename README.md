CUSTOMER SEGMENTATION BASED ON K MEANS CLUSTERING

This project implements the K-Means clustring algorithm in a data set that contains CustomerID	Gender	Age	Annual Income (k$)	Spending Score (1-100) features
K-means is  an unsupervised machine learning technique used for data exploration and pattern recognition.


Project Goals
*Apply k means clustering to cluster the customers in order to employ better marketing strategies
*Visually represent the identified clusters using appropriate data visualisation techniques
*Visually represent how the feature variables are related to each other

Dataset Characteristics
1)CustomerID	Gender	Age	Annual Income (k$)	Spending Score (1-100) is the feature set i will be using for the k means model

KMEANS
K-Means clustering is a widely used unsupervised machine learning algorithm for grouping similar data points together. It works by iteratively minimizing the distance between data points and their assigned cluster representatives, called centroids.
With k means clustering you need to push the number of clusters you want, this is achieved by finding the most suitable number of clusters with tne lowest inertis.
from sklearn.cluster import KMeans
cs = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init = 'k-means++', max_iter=300, n_init=10,random_state=0 )
    kmeans.fit(scaled_features)

    #init = 'k-means' initalizes the number of clusters at the very begining of clustering
    #n_init = 10 number of times the kmeans algorithm will be run with different seeds

    cs.append(kmeans.inertia_)

plt.plot(range(1, 11), cs, marker = 'o')
plt.title("Elbow Method")
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

![Screenshot (28)](https://github.com/user-attachments/assets/472ffcf6-bda2-44d0-9a80-7605a2146836)


After we determined the best number of clusters using the elbow method, we move on to normalise the data otherwise known as scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaled_feature = scaler.fit_transform(data[['Annual Income (k$)','Spending Score (1-100)']])
scaled_features = pd.DataFrame(scaled_feature)
scaled_features.head()

The next step is seting up a new model an dintroducing a new column to the data set so that we can plot with it as the number of clusters
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters= 5, random_state=  42)
kmeans.fit(scaled_features)
#Adding cluster labels to the original dataset
data['clusters'] = kmeans.labels_
#Adding a new column to the data

The final step is visualising the clusters
plt.figure(figsize= (10,6))
sns.scatterplot(x='Annual Income (k$)', y= 'Spending Score (1-100)', data = data , hue= 'clusters', palette='colorblind')
#Hue allows you to colour a scatter point based on a categorical variable
plt.xlabel('Annual_salary')
plt.ylabel('Spending score')
plt.title("Customer segments based on annual income and spending Score")
plt.show()

![Screenshot (29)](https://github.com/user-attachments/assets/d923a73d-ede0-4ab1-9746-3e81cad9f0bd)


SQL QUERIES
DESCRIPTION QUESTIONS:Querying the data to get valuable insights
![image](https://github.com/user-attachments/assets/b0d38c57-a08b-4751-9d23-4e19f4d04495)


