# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Handle table-like data and matrices
import numpy as np
import pandas as pd

# # Modelling Algorithms
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.svm import SVC, LinearSVC
# from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier

# # Modelling Helpers
# from sklearn.preprocessing import Imputer , Normalizer , scale
# from sklearn.cross_validation import train_test_split , StratifiedKFold
# from sklearn.feature_selection import RFECV

# Visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

student = pd.read_excel(
    'D:/000_VU_Study/P5/DataMIning/Machine_Learning_Data_Mining/assignment1/data/ODI-2019.xlsx')

student = student[['Have you taken a course on machine learning?',
                   'Have you taken a course on information retrieval?', 'Have you taken a course on databases?']]
student = student.replace({'yes': '1'})
student = student.replace({'no': '0'})
student = student.replace({'ja': '1'})
student = student.replace({'nee': '0'})
student = student.replace({'unknown': '0'})

student.to_excel('student.xlsx')

sns.heatmap(student)
plt.show(sns.heatmap(student))
