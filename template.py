#PLEASE WRITE THE GITHUB URL BELOW!
#

import sys

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def load_dataset(dataset_path):
    """load csv file given path"""
    df = pd.read_csv(dataset_path)
    return df

def dataset_stat(dataset_df):
    """statistic of dataset"""
    # check dataset validity
    try:
        assert 'target' in dataset_df.columns
    except:
        raise Exception('`target` column not found in csv.' )
    
    # sta
    num_features = len(dataset_df.columns) - 1
    num_0 = dataset_df[dataset_df['target']==0].shape[0]
    num_1 = dataset_df[dataset_df['target']==1].shape[0]
    return num_features, num_0, num_1
    
def split_dataset(dataset_df, testset_size):
    """split dataset accorading to testset_size"""
    # get feature columns names
    feat_columns = list(dataset_df.columns)
    feat_columns.remove('target')
    
    # train test split
    # use `stratify` to deal with unbalanced labels
    x_train, x_test, y_train, y_test = train_test_split(
        dataset_df[feat_columns], dataset_df['target'], test_size=testset_size,
        stratify=dataset_df['target'], random_state=42)
    
    # convert to numpy array
    x_train, x_test = x_train.values, x_test.values
    y_train, y_test = y_train.values, y_test.values
    
    return x_train, x_test, y_train, y_test

def decision_tree_train_test(x_train, x_test, y_train, y_test):
    """train and test decision tree"""
    # train model
    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    
    # cal metrics
    pred_test = model.predict(x_test)
    acc = accuracy_score(y_test, pred_test)
    precision = precision_score(y_test, pred_test)
    recall = recall_score(y_test, pred_test)

    return acc, precision, recall
 
def random_forest_train_test(x_train, x_test, y_train, y_test):
    """train and test random forest"""
    # train model
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    
    # cal metrics
    pred_test = model.predict(x_test)
    acc = accuracy_score(y_test, pred_test)
    precision = precision_score(y_test, pred_test)
    recall = recall_score(y_test, pred_test)

    return acc, precision, recall

def svm_train_test(x_train, x_test, y_train, y_test):
    """train and test svm"""
    # standard scaler
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    
    # train model
    model = SVC()
    model.fit(x_train, y_train)
    
    # cal metrics
    x_test = scaler.transform(x_test) # transform with fitted scaler
    pred_test = model.predict(x_test)
    acc = accuracy_score(y_test, pred_test)
    precision = precision_score(y_test, pred_test)
    recall = recall_score(y_test, pred_test)

    return acc, precision, recall

def print_performances(acc, prec, recall):
    """print performance of model"""
    print ("Accuracy: ", acc)
    print ("Precision: ", prec)
    print ("Recall: ", recall)

if __name__ == '__main__':
	#Do not modify the main script!
	data_path = sys.argv[1]
	data_df = load_dataset(data_path)

	n_feats, n_class0, n_class1 = dataset_stat(data_df)
	print ("Number of features: ", n_feats)
	print ("Number of class 0 data entries: ", n_class0)
	print ("Number of class 1 data entries: ", n_class1)
 
	print ("\nSplitting the dataset with the test size of ", float(sys.argv[2]))
	x_train, x_test, y_train, y_test = split_dataset(data_df, float(sys.argv[2]))
 
	acc, prec, recall = decision_tree_train_test(x_train, x_test, y_train, y_test)
	print ("\nDecision Tree Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = random_forest_train_test(x_train, x_test, y_train, y_test)
	print ("\nRandom Forest Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = svm_train_test(x_train, x_test, y_train, y_test)
	print ("\nSVM Performances")
	print_performances(acc, prec, recall)
