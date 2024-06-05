import pandas as pd
import numpy as np
from distEst_lib import MultivarContiDistributionEstimator
from sklearn import svm
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import mechanism_learning as ml

syn_data_dir = "E:/Happiness source/PhD/UoB/python_packages/mechanism_learning/test_data/"
testcase_dir = "frontdoor_discY_contZ_contX_discU/"
X_train = pd.read_csv(syn_data_dir + testcase_dir + "X_train.csv")
Y_train = pd.read_csv(syn_data_dir + testcase_dir + "Y_train.csv")
Z_train = pd.read_csv(syn_data_dir + testcase_dir + "Z_train.csv")
X_train = np.array(X_train)
Y_train = np.array(Y_train).reshape(-1,1)
Z_train = np.array(Z_train)

X_test = pd.read_csv(syn_data_dir + testcase_dir + "X_test.csv")
Y_test = pd.read_csv(syn_data_dir + testcase_dir + "Y_test.csv")
X_test = np.array(X_test)
Y_test = np.array(Y_test).reshape(-1,1)

joint_yz_data = np.concatenate((Y_train, Z_train), axis = 1)

n_bins = [0,20]
dist_estimator_yz = MultivarContiDistributionEstimator(data_fit=joint_yz_data, n_bins = n_bins)
pdf_yz, pyz = dist_estimator_yz.fit_histogram()
dist_estimator_y = MultivarContiDistributionEstimator(data_fit=Y_train, n_bins = [0])
pdf_y, py = dist_estimator_y.fit_histogram()

dist_map = {"Y,Z": lambda Y, Z: pdf_yz([Y,Z]),
            "Y',Z": lambda Y_prime, Z: pdf_yz([Y_prime,Z]),
            "Y": lambda Y: pdf_y(Y),
            "Y'": lambda Y_prime: pdf_y(Y_prime)}

# train the deconfounded svc
clf_mechanism = ml.mechanism_classifier(cause_data = {"Y": Y_train}, 
                                        mediator_data = {"Z": Z_train},
                                        effect_data = {"X": X_train}, 
                                        dist_map = dist_map, 
                                        ml_model = svm.SVC(kernel = 'linear', C=5), 
                                        rebalance = False, n_samples = None, cb_mode = "fast")

# train the confounded svc
clf_conf = svm.SVC(kernel = 'linear', C=5)
clf_conf.fit(X_train, Y_train.reshape(-1))

# Compare models' decision boundaries
weight = clf_mechanism.coef_[0]
bias = clf_mechanism.intercept_[0]
k = -weight[0] / weight[1]
b = -bias / weight[1]
x_ = np.linspace(-4, 4, 100)
decison_boundary_deconf = k * x_ + b

weight = clf_conf.coef_[0]
bias = clf_conf.intercept_[0]
k = -weight[0] / weight[1]
b = -bias / weight[1]
x_ = np.linspace(-4, 4, 100)
decison_boundary_conf = k * x_ + b

fig,ax=plt.subplots()
scatter = ax.scatter(x= X_test[:,0], y = X_test[:,1], c = Y_test, s = 10, cmap='coolwarm', alpha = 0.5)
handles_scatter, labels_scatter = scatter.legend_elements(prop="colors")

plt.xlim(-4,4)
plt.ylim(-6,6)
x_ = np.linspace(-4, 4, 100)

true_b = plt.plot([0, 0], [-6, 6], '-.k', linewidth=2, label="True boundary")
confounder = plt.plot([-6,6], [0,0], '-.y', linewidth = 2, label = "Confounder boundary")
clf_b_conf = plt.plot(x_, decison_boundary_conf, '-.r', linewidth = 2, label= 'Confounded SVM decision boundary')
clf_b_deconf = plt.plot(x_, decison_boundary_deconf, '-.g', linewidth = 2, label= 'De-confounded SVM decision boundary')

ax.legend(handles=handles_scatter+true_b+clf_b_conf+clf_b_deconf+confounder, labels=['-1','1','True Y boundary','Confounded SVM decision boundary', 'De-confounded SVM decision boundary', 'Confounder boundary'], loc='lower right')

plt.title('Decision boundary comparison')
plt.show()

## compare their performance on un-confounded test set
y_pred_deconf = clf_mechanism.predict(X_test)
print("Report of de-confonded model:")
print(classification_report(Y_test, y_pred_deconf))

y_pred_conf = clf_conf.predict(X_test)
print("Report of confonded model:")
print(classification_report(Y_test, y_pred_conf))
