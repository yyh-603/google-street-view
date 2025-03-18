import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.preprocessing import image
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns

# ordered by city location, counter-clockwise
ORDERED_CITY = ["Keelung City", "New Taipei City", "Taipei City", "Taoyuan City", "Hsinchu County", "Hsinchu City", "Miaoli County", "Taichung City", "Changhua County", "Nantou County", "Yunlin County", "Chiayi County", "Chiayi City", "Tainan City", "Kaohsiung City", "Pingtung County", "Taitung County", "Hualien County", "Yilan County"]

if __name__ == '__main__':

    # read CSV file
    df = pd.read_csv('output.csv')

    # map city name to index
    city_to_idx = {key: value for value, key in enumerate(ORDERED_CITY)}

    # read images and labels
    X, Y = [], []
    for index, row in df.iterrows():
        img = image.load_img('./data/' + row['picture_name'], target_size=(144, 144))
        img = image.img_to_array(img)
        img = img.flatten()
        X.append(img)
        Y.append(city_to_idx[row['city']])

    # use PCA to reduce dimension
    pca = PCA(n_components=1200)
    X = pca.fit_transform(X)

    # split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=114514)

    # train K-means model
    kmeans = KMeans(n_clusters=19, random_state=114514)
    kmeans.fit(X_train)

    # predict and evaluate
    train_pred = kmeans.predict(X_train)

    # find the most frequent label in each cluster
    # count[label][truth]: for each cluster, count the number of each ground truth
    count = dict()
    for pred, truth in zip(train_pred, y_train): # pred is the cluster number, truth is the ground truth
        if pred not in count:
            count[pred] = dict()
        if truth not in count[pred]:
            count[pred][truth] = 0
        count[pred][truth] += 1

    # find the most frequent label in each cluster
    # for each cluster, find the ground truth with the highest count
    labels_mapping = dict()
    for cluster_id, d in count.items():
        mx = 0
        for truth, cnt in d.items():
            if cnt > mx:
                mx = cnt
                labels_mapping[cluster_id] = truth
    
    # predict and evaluate
    y_pred = [labels_mapping[x] for x in kmeans.predict(X_test)]
    print(accuracy_score(y_test, y_pred))

    # plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(15, 15))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=ORDERED_CITY, yticklabels=ORDERED_CITY)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')