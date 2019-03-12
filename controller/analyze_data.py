from flask import Blueprint, render_template, session, abort, request, jsonify, redirect
import pandas as pd
from key import *


from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

app_analyze_data = Blueprint('app_analyze_data', __name__)

#############################################################################
# /analyze_data
# Analyze Data using ML techniques
#############################################################################
@app_analyze_data.route("/analyze_data", methods=['GET','POST'])
def analyze_data():
	if request.args.get("doi") and request.args.get("id"):
		global API_KEY
		print("Started...")
		
		doi = request.args.get("doi")
		id = request.args.get("id")
		dataset_name = request.args.get("name")
		
		data = pd.read_table("https://dataverse-test.irss.unc.edu/api/access/datafile/"+id+"/?persistentId=doi:"+doi+"&key="+API_KEY)
		print("Downloaded Data from "+doi)
		data_head = data.head(50).to_numpy()
		
		
		data_col_names = data.columns.values
		data_describe = data.describe().round(3).to_html(classes='table table-dark')
		
		datalength = len(data.columns)

		
		X = data.values[:,0:datalength-1]
		Y = data.values[:,datalength-1]
		
		###Validation Size
		validation_size = 0.20
		seed = 7
		scoring = 'accuracy'
		
		X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X,Y,test_size=validation_size, random_state=seed)
		
		models = []
		models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr'))) #logistic regression
		models.append(('LDA', LinearDiscriminantAnalysis())) #LDA
		models.append(('KNN', KNeighborsClassifier())) #KNearest Neighbor  5
		models.append(('CART', DecisionTreeClassifier())) #DecisionTree/CART
		models.append(('NB', GaussianNB())) #Naive Bayes
		mymodels = {'LR': LogisticRegression(solver='liblinear', multi_class='ovr'), 'LDA': LinearDiscriminantAnalysis(),
					'KNN': KNeighborsClassifier(), 'CART': DecisionTreeClassifier(), 'NB': GaussianNB()}
		#models.append(('SVM', SVC(gamma='auto')))
		# evaluate each model in turn
		results = []
		names = []
		means = []

		best_model_mean = -1
		best_model_name = ""
		for name, model in models:
			kfold = model_selection.KFold(n_splits=10, random_state=seed)
			cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
			results.append(cv_results)
			names.append(name)
			if best_model_mean < cv_results.mean():
				best_model_mean = cv_results.mean()
				best_model_name = name
			#msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
			#print(msg)
			
		mymodel = mymodels[best_model_name]
		mymodel.fit(X_train, Y_train)
		predictions = mymodel.predict(X_validation)
		mymodel_accuracy = accuracy_score(Y_validation, predictions)
		mymodel_confusion = confusion_matrix(Y_validation, predictions)
		mymodel_report = classification_report(Y_validation, predictions)
		
		
		return render_template("analyze_data.html", shape=data.shape,data_head=data_head, data_describe=data_describe,
								data_col_names = data_col_names, dataset_name = dataset_name, algo_names=names,
								algo_results = results, best_model_mean=best_model_mean, best_model_name=best_model_name,
								mymodel_accuracy = mymodel_accuracy, mymodel_confusion = mymodel_confusion,
								mymodel_report = mymodel_report)
	return "<h2>Cannot Analyze the data</h2>"