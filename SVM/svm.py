from sklearn import svm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform
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
    pca = PCA(n_components=2000)
    X = pca.fit_transform(X)

    # split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=114514)

    # configure SVM
    clf = svm.SVC(cache_size=2000)
    param_dist = {
        'C': loguniform(1, 10),
        'gamma': loguniform(1e-9, 1e-8),
    }

    # use RandomizedSearchCV to find the best hyperparameters
    rand_search = RandomizedSearchCV(
        clf, 
        param_distributions=param_dist, 
        cv=5, 
        n_iter=20, 
        verbose=1, 
        n_jobs=-1, 
        scoring='f1_micro')
    
    rand_search.fit(X_train, y_train)
    print(f'Best params: {rand_search.best_params_}')
    print(f'Best scores on training data: {rand_search.best_score_}')
    print(f'Score on test data: {rand_search.score(X_test, y_test)}')

    # plot the scatter plot of hyperparameters and accuracy
    results = rand_search.cv_results_
    mean_test_scores = results['mean_test_score']
    params = results['params']

    C_values = [param['C'] for param in params]
    gamma_values = [param['gamma'] for param in params]

    fig, ax = plt.subplots()
    scatter = ax.scatter(C_values, gamma_values, c=mean_test_scores, cmap='viridis', edgecolor='k')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('C')
    ax.set_ylabel('gamma')
    ax.set_title('Random Search Accuracy')
    cbar = plt.colorbar(scatter)
    cbar.set_label('Mean Accuracy')
    plt.savefig('rand_search.png')

    # plot confusion matrix
    y_pred = rand_search.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(15, 15))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=ORDERED_CITY, yticklabels=ORDERED_CITY)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')