import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import scipy.stats as stats
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices
%matplotlib inline
hrdf = pd.read_csv("D:/AdityaData/hr_comma_sep.csv")
hrdf.columns
hrdf.info()
 Data Pre-Processing
Missing Value Treatmen
# a. Separating out Categorical and Numerical Variables
cat_list = []
num_list = []

for variable in hrdf.columns:
    if hrdf[variable].dtype.name in ['object']:
        cat_list.append(variable)
    else:
        num_list.append(variable)

##print("Categorical Variables : ", cat_list, '\n')
##print("Numerical Variables : ", num_list)

for i in cat_list:
    a = hrdf[i].fillna('Missing')
    print("\nThe distinct categories of variable '{0}' is :\n ".format(i), a.value_counts(), '\n')

nmiss_df = hrdf.isnull().sum(axis=0)
nmiss_df.name = 'NMiss'
print("The conclusion is that none of the categorical or numerical variable requires missing values treatment \n")
pd.concat([hrdf.describe().T,nmiss_df], axis=1,join='inner')
bp = PdfPages('BoxPlots with Attrition Split.pdf')

for num_variable in num_list:
    fig,axes = plt.subplots(figsize=(10,4))
    sns.boxplot(x='left', y=num_variable, data = hrdf)
    sns.plt.title(str('Box Plot of ') + str(num_variable))
    bp.savefig(fig)
bp.close()

bp = PdfPages('BoxPlots with Total View.pdf')

for num_variable in num_list:
    fig,axes = plt.subplots(figsize=(10,4))
    sns.boxplot(y=num_variable, data = hrdf)
    sns.plt.title(str('Box Plot of ') + str(num_variable))
    bp.savefig(fig)
bp.close()
## Categorical Variable
chisq_df = pd.DataFrame()
for cat_variable in cat_list:
    cross_tab = pd.crosstab(hrdf[cat_variable],hrdf['left'], margins=False)
    stats.chi2_contingency(observed=cross_tab)[1]
    temp = pd.DataFrame([cat_variable,stats.chi2_contingency(observed=cross_tab)[0],stats.chi2_contingency(observed=cross_tab)[1] ]).T
    temp.columns = ['Variable', 'ChiSquare','P-Value']
    chisq_df = pd.concat([chisq_df, temp], axis=0, ignore_index=True)
    
print(chisq_df, '\n')

## Numerical Variable
tstats_df = pd.DataFrame()
for num_variable in num_list:
    tstats = stats.ttest_ind(hrdf[hrdf['left']==1][num_variable],hrdf[hrdf['left']==0][num_variable])
    temp = pd.DataFrame([num_variable, tstats[0], tstats[1]]).T
    temp.columns = ['Variable Name', 'T-Statistic', 'P-Value']
    tstats_df = pd.concat([tstats_df, temp], axis=0, ignore_index=True)

print(tstats_df)
bp = PdfPages('Density Plots with Attrition Split.pdf')

for num_variable in num_list:
    fig,axes = plt.subplots(figsize=(10,4))
    #sns.distplot(hrdf[num_variable], kde=False, color='g', hist=True)
    sns.distplot(hrdf[hrdf['left']==0][num_variable], label='Not Attrited', color='g', hist=True, norm_hist=False)
    sns.distplot(hrdf[hrdf['left']==1][num_variable], label='Attrited', color='r', hist=True, norm_hist=False)
    sns.plt.xlabel(str("X variable ") + str(num_variable) )
    sns.plt.ylabel('Density Function')
    sns.plt.title(str('Attrition Split Density Plot of ')+str(num_variable))
    sns.plt.legend()
    bp.savefig(fig)

bp.close()
Data Exploratory Analysis
- Variable Reduction using Somer's D values
num_features = ['satisfaction_level', 'time_spend_company', 'Work_accident', 
               'promotion_last_5years', 'monthly_hrs_extreme', 'last_evaluation_extreme', 'number_project_grp']
somersd_df = pd.DataFrame()
for num_variable in num_features:
    logreg = sm.logit(formula = str('left ~ ')+str(num_variable), data=hrdf)
    result = logreg.fit()
    summ = result.summary()
    y_score = pd.DataFrame(result.predict())
    y_score.columns = ['Score']
    somers_d = 2*metrics.roc_auc_score(hrdf['left'],y_score) - 1
    temp = pd.DataFrame([num_variable,somers_d]).T
    temp.columns = ['Variable Name', 'SomersD']
    somersd_df = pd.concat([somersd_df, temp], axis=0)

somersd_df
Model Build and Diagnostics
Train and Test split
X.drop('salary_medium', axis=1, inplace=True)
train_features = X.columns.difference(['left'])
train_X, test_X = train_test_split(X, test_size=0.3, random_state=42)
train_X.columns
 Model Build and Diagnostics
- Model build on the train_X sample
logreg = sm.logit(formula='left ~ ' + "+".join(train_features), data=train_X)
result = logreg.fit()
summ = result.summary()
summ
train_gini = 2*metrics.roc_auc_score(train_X['left'], result.predict()) - 1
print("The Gini Index for the model built on the Train Data is : ", train_gini)
train_predicted_prob = pd.DataFrame(result.predict(train_X))
train_predicted_prob.columns = ['prob']
train_actual = train_X['left']
# making a DataFrame with actual and prob columns
hr_train_predict = pd.concat([train_actual, train_predicted_prob], axis=1)
hr_train_predict.columns = ['actual','prob']
hr_train_predict.head()
## Intuition behind ROC curve - confusion matrix for each different cut-off shows trade off in sensitivity and specificity
roc_like_df = pd.DataFrame()
train_temp = hr_train_predict.copy()

for cut_off in np.linspace(0,1,50):
    train_temp['predicted'] = train_temp['prob'].apply(lambda x: 0 if x < cut_off else 1)
    train_temp['tp'] = train_temp.apply(lambda x: 1 if x['actual']==1 and x['predicted']==1 else 0, axis=1)
    train_temp['fp'] = train_temp.apply(lambda x: 1 if x['actual']==0 and x['predicted']==1 else 0, axis=1)
    train_temp['tn'] = train_temp.apply(lambda x: 1 if x['actual']==0 and x['predicted']==0 else 0, axis=1)
    train_temp['fn'] = train_temp.apply(lambda x: 1 if x['actual']==1 and x['predicted']==0 else 0, axis=1)
    sensitivity = train_temp['tp'].sum() / (train_temp['tp'].sum() + train_temp['fn'].sum())
    specificity = train_temp['tn'].sum() / (train_temp['tn'].sum() + train_temp['fp'].sum())
    roc_like_table = pd.DataFrame([cut_off, sensitivity, specificity]).T
    roc_like_table.columns = ['cutoff', 'sensitivity', 'specificity']
    roc_like_df = pd.concat([roc_like_df, roc_like_table], axis=0)

train_temp.sum()
plt.subplots(figsize=(10,4))
plt.scatter(roc_like_df['cutoff'], roc_like_df['sensitivity'], marker='*', label='Sensitivity')
plt.scatter(roc_like_df['cutoff'], roc_like_df['specificity'], marker='*', label='Specificity')
plt.scatter(roc_like_df['cutoff'], 1-roc_like_df['specificity'], marker='*', label='FPR')
plt.title('For each cutoff, pair of sensitivity and FPR is plotted for ROC')
plt.legend()
## Finding ideal cut-off for checking if this remains same in OOS validation
roc_like_df['total'] = roc_like_df['sensitivity'] + roc_like_df['specificity']
roc_like_df[roc_like_df['total']==roc_like_df['total'].max()]
