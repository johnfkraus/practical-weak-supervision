#!/usr/bin/env python
# coding: utf-8

# # Labeling Prime Numbers with Snorkel
# 
# Version 1
# 
# This notebook is adapted from Chapter 2 of [Practical Weak Supervision](https://learning.oreilly.com/library/view/practical-weak-supervision/9781492077053/).
# 
# 

# 
# 
# [Weak Supervision vs. Rule-Based Classifiers](https://snorkel.ai/weak-supervision/)
# 
# Weak supervision has some similarities—and some very important differences—to rule-based classifiers. The obvious similarity is that the inputs to each look like rules (i.e., simple functions that output labels or predictions). The important difference between them is that the rule-based classifier stops there—the rules are the classifier. Such systems are generally brittle because they do not generalize to other examples, even ones that are very similar to those that are labeled by one or more rules.
# 
# With weak supervision, on the other hand, the rules (or “labeling functions”) are used to create a training set for a machine-learning-based model. That model is postulated to be more powerful, utilize a much richer feature set, and take advantage of other ML techniques, such as transfer learning from foundation models.  The resulting model is supposed to be more robust than a corresponding rule-based classifier.
# 
# Each labeling function suggests training labels for multiple unlabeled data points, based on human-provided subject matter expertise. A label model, of which there are multiple types, aggregates those weak labels into one training label per data point to create a training set. The ML model is trained on that training set and learns to generalize beyond just those data points that were labeled by labeling functions.
# 

# 
# The steps applied below for demonstrating Snorkel data labeling are:
# 
# 1. Create a small data set and smaller validation set.  Each data point is an integer.
# 2. Use labeling functions to create a list of labels for each data point.  The labels for the small data set are represented in a label matrix (named "L" or "Lxxx").
# 3. Use a (generative) model to resolve the list of labels for each data point to a single label.
# 4. Evaluate the accuracy of the (generative) model by applying it to the validation set and comparing predicted labels to ground truth.  
# 5. If the generative model is satisfactory, use its predicted labels for the small data set to train a logistic regression model (or perhaps some other model you prefer).
# 6. Use the logistic regression model to label a large data set.
# 7. Assess the accuracy of the large data set labels by comparing the labels to ground truth.  Ground truth is readily determined in the case of prime numbers.
# 
# Scenario 1:
# three labeling functions
# 

# In[25]:


import pandas as pd
import numpy as np
import random
from snorkel.labeling import labeling_function
from snorkel.labeling import PandasLFApplier
from snorkel.labeling.model import LabelModel
from snorkel.labeling import LFAnalysis
from snorkel.labeling.model import MajorityLabelVoter

import logging
logging.basicConfig(level=logging.DEBUG)

import is_it_prime   # custom code for prime number operations
scenario = 1


# ### Create a small example dataset

# In[26]:


if scenario == 1:
    # in the book's example '37' appeared twice in the data set.  Why?
    data =    [5, 21, 1, 29, 32, 37, 10, 20, 10, 26, 2, 37, 34, 11, 22, 36, 12, 20, 31, 25]

df = pd.DataFrame(data, columns=['Number'])


# ### Creating a small validation set

# Ground truth array for the validation set. 
# 
# 22 -> not prime [0]<br>
# 11 -> prime [1]<br>
# etc...

# In[27]:


import is_it_prime  # custom code
#  the book example used [22, 11, 7, 2, 32], we use:
if scenario == 1:
    validation_data = [22, 11, 7, 2, 32, 101, 102]  # a list

validation_true_labels = is_it_prime.array_map(validation_data)  # an ndarray


# In[28]:


df_val = pd.DataFrame(validation_data, columns=['Number'])
df_tl = pd.DataFrame(validation_true_labels, columns=['true_labels'])


# ### Labeling functions

# In[29]:


ABSTAIN = -1
NON_PRIME = 0
PRIME = 1


# In[30]:


# if odd, abstain else non-prime.
@labeling_function()
def is_odd(record):
    if record["Number"]%2 == 1:
        return ABSTAIN
    else:
        return NON_PRIME


# In[31]:


@labeling_function()
def is_two(record):
    if record["Number"] == 2:
        return PRIME
    else:
        return ABSTAIN


# In[32]:


@labeling_function()
def is_known_prime(record):
    # known_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]  # original list from book code
    known_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 101]
    if record["Number"] in known_primes:
        return PRIME
    else:
        return ABSTAIN


# In[33]:


# if > 3 and not evenly divisible by 3 ABSTAIN else NON_PRIME; so this kind of combines two function, unlike the above rules, where we apply rules for odd numbers and 2 in separate functions.  Also, 3 is a known prime in the known primes labeling function above.  Is this a problem ("multicollinearity" or something)?. 
@labeling_function()
def gt3_ndb_3(record):
    if record["Number"]>3 and record["Number"]%3 == 0:
        return NON_PRIME
    else:
        return ABSTAIN


# ### Calculating: Polarity, Coverage, Overlaps and Conflicts for the labeling functions

# In[34]:


# Make a list of the labeling functions to be applied
if scenario == 1:
  lfs = [is_odd, is_two, is_known_prime]
# gt3_ndb_3]


# In[35]:


def apply_labeling_functions(lfs_, df_):
    """ Returns an ndarray with a column for each labeling function and a row for each data point
    Args:
        lfs_ (list): A list of labeling functions.
        df_ (Pandas DataFrame): A dataframe having one column labeled Number, a row for each data point, and an index starting with zero.
    """
    applier = PandasLFApplier(lfs=lfs_)
    if df_.shape[1] != 1:
        print("check shape? ", df_.shape)
    L_label_matrix = applier.apply(df=df_)
    return L_label_matrix

# generate a label matrix for training the generative model, with the matrix having shape (m=# of labeling functions, n=number of data points)
L_train = apply_labeling_functions(lfs, df)


# #### Applying the labeling functions to the dataset, for illustration of calculating the above metrics

# In[36]:


# show Number and columns of "noisy" labels.
df["is_odd"] = df.apply(is_odd, axis=1)
df["is_two"] = df.apply(is_two, axis=1)
df["is_known_prime"] = df.apply(is_known_prime, axis=1)
if gt3_ndb_3 in lfs:
    df["gt3_ndb_3"] = df.apply(gt3_ndb_3, axis=1)

df


# In[37]:


# LFAnalysis(L=L_train, lfs=lfs).lf_polarities()
# L_valid = apply_labeling_functions(lfs,  pd.DataFrame(validation_data, columns=['Number']))
LFAnalysis(L_train, lfs).lf_summary()


# Polarity (How many distinct labels do each labeling function emit, not counting abstentions?)
# 
# Coverage (what is the proportion of points to which each labeling function applies a label, as opposed to abstaining?)
# 
# Overlaps (proportion of data points labeled the same by two LFs)
# 
# Conflicts  (proportion of data points labeled differently by two LFs)
# 
# Calculating: Correct, Incorrect and Empirical Accuracy on the validation set

# In[38]:


# apply the labeling functions to the validation set; returns a label from each L.F. for each data point. 
L_valid = apply_labeling_functions(lfs,  pd.DataFrame(validation_data, columns=['Number']))
LFAnalysis(L_valid, lfs).lf_summary(validation_true_labels)


# #### Applying the labeling functions to the validation set, just for illustration of calculating the above metrics

# In[39]:


def illustrate_labeling(data, truth, lfs):
    df_val = pd.DataFrame(validation_data, columns=['Number'])
    df_val["is_odd"] = df_val.apply(is_odd, axis=1)
    df_val["is_two"] = df_val.apply(is_two, axis=1)
    df_val["is_known_prime"] = df_val.apply(is_known_prime, axis=1)
    if gt3_ndb_3 in lfs:
        df_val["is_known_prime"] = df_val.apply(is_known_prime, axis=1)
    df_val["ground_truth"] = validation_true_labels
    return df_val

df_val = illustrate_labeling(validation_data, validation_true_labels, lfs)
df_val



# ### Using MajorityLabelVoter to determine the label

# In[40]:


majority_model = MajorityLabelVoter()
preds_train_majority_label = majority_model.predict(L=L_train)
preds_valid_majority_label = majority_model.predict(L=L_valid)
df["maj_label_pred"] = preds_train_majority_label
df[df["Number"] == 2]


# In[41]:


# doesn't work:
print("predicted: ", preds_valid_majority_label, ", truth: ", validation_true_labels)
metrics2 = majority_model.score(L_valid, validation_true_labels, metrics=['accuracy'])
# metrics2
val_accuracy_maj_label_voter_model = (preds_valid_majority_label == validation_true_labels).mean()
print(val_accuracy_maj_label_voter_model)

def get_accuracy(label_preds_ndarray, true_labels_ndarray):
    return (preds_valid_majority_label == validation_true_labels).mean()

print(get_accuracy(preds_valid_majority_label, validation_true_labels))


# for reference: METRICS = {<br>
#      "accuracy": Metric(skmetrics.accuracy_score), <br>
#      "coverage": Metric(_coverage_score, ["preds"]), <br>
#      "precision": Metric(skmetrics.precision_score), <br>
#      "recall": Metric(skmetrics.recall_score), <br>
#      "f1": Metric(_f1_score, ["golds", "preds"]), <br>
#      "f1_micro": Metric(_f1_micro_score, ["golds", "preds"]), <br>
#      "f1_macro": Metric(_f1_macro_score, ["golds", "preds"]), <br>
#      "fbeta": Metric(skmetrics.fbeta_score), <br>
#      "matthews_corrcoef": Metric(skmetrics.matthews_corrcoef), <br>
#      "roc_auc": Metric(_roc_auc_score, ["golds", "probs"]),  
# 
# metrics_list = ['accuracy', 'coverage', 'precision', 'recall', 'f1', 'f1_micro', 'f1_macro', 'matthews_corrcoef', 'roc_auc']
# metrics = random_model.score(preds_valid_random, validation_true_labels, metrics_list)  
# accuracy_random_model = metrics["accuracy"]
# df_valid_metrics["accur_rand_mod"] = np.ndarray(int(accuracy_random_model))
# metrics
# 

# In[42]:


def get_metrics(model, L, truth):
    metrics_list = ['accuracy', 'coverage', 'precision', 'recall', 'f1', 'f1_micro', 'f1_macro', 'matthews_corrcoef', 'roc_auc']
    metrics = model.score(L, truth, metrics_list)
    return metrics

metrics = get_metrics(majority_model, L_valid, validation_true_labels)
metrics
#print(pd.DataFrame(metrics))

df11 = pd.DataFrame.from_dict(metrics, orient='index',
                       columns=['LabelModel'])
print(df11)
df11 = pd.concat([df11, df11], ignore_index=True)
df11


# In[43]:


df_val = pd.DataFrame(validation_data, columns=['Number'])
df_val["is_odd"] = df_val.apply(is_odd, axis=1)
df_val["is_two"] = df_val.apply(is_two, axis=1)
df_val["is_known_prime"] = df_val.apply(is_known_prime, axis=1)
if gt3_ndb_3 in lfs:
    df_val["gt3_ndb_3"] = df_val.apply(gt3_ndb_3, axis=1)
df_val["pred_maj_label"] = preds_valid_majority_label
df_val["ground_truth"] = validation_true_labels
df_val


# In[44]:


# np.round(majority_model.get_weights(), 2)
# 'MajorityLabelVoter' object has no attribute 'get_weights'


# ### Using LabelingModel to determine the label
# 
# Documentation: https://snorkel.readthedocs.io/en/v0.9.3/packages/_autosummary/labeling/snorkel.labeling.LabelModel.html
# 
# A model for learning the LF accuracies and combining their output labels.
# 
# This class learns a model of the labeling functions’ conditional probabilities of outputting the true (unobserved) label Y, P(lf | Y), and uses this learned model to re-weight and combine their output labels.
# 
# This class is based on the approach in [Training Complex Models with Multi-Task Weak Supervision](https://arxiv.org/abs/1810.02840), published in AAAI‘19. In this approach, we compute the inverse generalized covariance matrix of the junction tree of a given LF dependency graph, and perform a matrix completion-style approach with respect to these empirical statistics. The result is an estimate of the conditional LF probabilities, P(lf | Y), which are then set as the parameters of the label model used to re-weight and combine the labels output by the LFs.
# 
# Currently this class uses a conditionally independent label model, in which the LFs are assumed to be conditionally independent given Y.
# 
# 

# In[45]:


label_model = LabelModel(verbose=True)
# no y_train data!
label_model.fit(L_train=L_train, n_epochs=200, seed=100)
preds_train_label = label_model.predict(L=L_train)

L_valid = apply_labeling_functions(lfs, pd.DataFrame(validation_data, columns=['Number']))

preds_valid_label = label_model.predict(L=L_valid)

# L_valid = applier.apply(df_val)
# LFAnalysis(L_valid, lfs).lf_summary()
LFAnalysis(L_valid, lfs).lf_summary(validation_true_labels)


# In[46]:


preds_train_labelingModel = label_model.predict(L=L_train)
preds_valid_labelingModel = label_model.predict(L=L_valid)


# Examine the weights of the label_model for each classification source (labeling function).  Labeling functions that make more mistakes would be expected to have lower weights.

# In[47]:


df["preds_label_model"] = preds_train_labelingModel


# In[48]:


#df[["Number", "preds_train_random", "majorityClass_pred", "maj_label_pred", "preds_labelingModel"]]
df


# #### LabelModel with class balance
# 
#     r"""A model for learning the LF accuracies and combining their output labels.
# 
#     This class learns a model of the labeling functions' conditional probabilities
#     of outputting the true (unobserved) label `Y`, `P(\lf | Y)`, and uses this learned
#     model to re-weight and combine their output labels.
# 
#     This class is based on the approach in [Training Complex Models with Multi-Task
#     Weak Supervision](https://arxiv.org/abs/1810.02840), published in AAAI'19. In this
#     approach, we compute the inverse generalized covariance matrix of the junction tree
#     of a given LF dependency graph, and perform a matrix completion-style approach with
#     respect to these empirical statistics. The result is an estimate of the conditional
#     LF probabilities, `P(\lf | Y)`, which are then set as the parameters of the label
#     model used to re-weight and combine the labels output by the LFs.
# 
#     Currently this class uses a conditionally independent label model, in which the LFs
#     are assumed to be conditionally independent given `Y`.
# 
#     Examples
#     --------
#     >>> label_model = LabelModel()
#     >>> label_model = LabelModel(cardinality=3)
#     >>> label_model = LabelModel(cardinality=3, device='cpu')
#     >>> label_model = LabelModel(cardinality=3)
# 
#     Parameters
#     ----------
#     cardinality
#         Number of classes, by default 2
#     **kwargs
#         Arguments for changing config defaults
# 
#     Raises
#     ------
#     ValueError
#         If config device set to cuda but only cpu is available
# 
#     Attributes
#     ----------
#     cardinality
#         Number of classes, by default 2
#     config
#         Training configuration
#     seed
#         Random seed
#     """

# # Logging???

# In[53]:


# 7/20 prime numbers

label_model_wcb = LabelModel(verbose=True)
# logger = logging.getLogger()
# logger.setLevel(logging.INFO)
# no Y_train data!!
# label_model_wcb.fit(L_train=L_train, n_epochs=200, class_balance = [0.7, 0.3], seed=100, progress_bar=False, log_freq=1)

label_model_wcb.fit(L_train=L_train, n_epochs=200, class_balance = [0.7, 0.3], seed=100, progress_bar=False, log_freq=10)


# In[28]:


preds_train_label_wcb = label_model_wcb.predict(L=L_train)
preds_valid_label_wcb = label_model_wcb.predict(L=L_valid)
# L_valid = applier.apply(df_val)

L_valid = apply_labeling_functions(lfs, pd.DataFrame(validation_data, columns=['Number']))


# LFAnalysis(L_valid, lfs).lf_summary()
LFAnalysis(L_valid, lfs).lf_summary(validation_true_labels)


# In[29]:


# with class balance
# preds_train_labelingModel_wcb = label_model_wcb.predict(L=L_train)
df["preds_label_model_wcb"] = preds_train_label_wcb


# In[30]:


df[["Number", "preds_label_model", "preds_label_model_wcb"]]


# Looking at the weights of the label_model

# In[31]:


np.round(label_model_wcb.get_weights(), 2)


# In[32]:


accuracy_labeling_model_wcbal = (preds_valid_label_wcb == validation_true_labels).mean()
print(accuracy_labeling_model_wcbal)

# get_metrics(label_model_wcb, preds_valid_label_wcb, validation_true_labels)


# And the actual conditional probability values placed in a matrix with dimensions [number of labeling function, number of labels + 1 (for abstain), number of classes], rounded are as follows:
# 
# 
# def get_conditional_probs(self) -> np.ndarray:
# 
#     r"""Return the estimated conditional probabilities table.
# 
#     Return the estimated conditional probabilites table cprobs, where cprobs is an
#     (m, k+1, k)-dim np.ndarray with:
# 
#         cprobs[i, j, k] = P(\lf_i = j-1 | Y = k)
# 
#     where m is the number of LFs, k is the cardinality, and cprobs includes the
#     conditional abstain probabilities P(\lf_i = -1 | Y = y).
# 
#     Returns
#     -------
#     np.ndarray
#         An [m, k + 1, k] np.ndarray conditional probabilities table.
#     """
# ```
# array([
# labeling function 1  (m=0)
#        [[
# (j = 0)        P( lf=1 | Y = 0 ) = 0.114, P( lf=1 | Y=1 ) = 0.98 ],
#         [0.876, 0.01 ],
#         [0.01 , 0.01 ]],
# labeling function 2
#        [[0.114, 0.98 ],
#         [0.876, 0.01 ],
#         [0.01 , 0.01 ]],
# 
#        [[0.91 , 0.923],
#         [0.01 , 0.01 ],
#         [0.08 , 0.067]],
# 
#        [[0.869, 0.58 ],
#         [0.01 , 0.01 ],
#         [0.121, 0.41 ]]])
# ```
# 
# Looking at the conditional probabilities 

# In[33]:


np.round(label_model.get_conditional_probs(), 3)


# ### Use the model to generate a larger set of labeled data

# In[34]:


# new_data_to_be_labeled = range(50, 150)
# df_new_data = pd.DataFrame(new_data_to_be_labeled, columns=['Number'])
# L_train = applier.apply(df=df_new_data)
LFAnalysis(L=L_train, lfs=lfs).lf_summary()


# ### Logistic Regression
# 
# Now that we have a model (using Snorkel LabelModel with class balance) let's label some data.
# 
# Sources include ML Bookcamp, Snorkel docs

# Filtering out unlabeled data points
# 
# As we saw earlier, some of the data points in our train set received no labels from any of our LFs. These data points convey no supervision signal and tend to hurt performance, so we filter them out before training using a built-in utility.
# 

# In[35]:


# from snorkel.labeling import filter_unlabeled_dataframe

# df_train = pd.DataFrame(data, columns=['Number'])
# df_train_filtered, probs_train_filtered = filter_unlabeled_dataframe(
#     X=df_train, y=preds_train_label_wcb, L=L_train
# )


# The output of the Snorkel LabelModel is a set of labels which can be used with most popular libraries for performing supervised learning, such as TensorFlow, Keras, PyTorch, Scikit-Learn, Ludwig, and XGBoost. In the Snorkel spam tutorial we use the well-known library Scikit-Learn. Note that typically, Snorkel is used (and really shines!) with much more complex, training data-hungry models, but we will use Logistic Regression here for simplicity of exposition.  Source: https://www.snorkel.org/use-cases/01-spam-tutorial
# 
# the LabelModel outputs probabilistic (float) labels. If the classifier we are training accepts target labels as floats, we can train on these labels directly (see describe the properties of this type of “noise-aware” loss in our NeurIPS 2016 paper).
# 
# If we want to use a library or model that doesn’t accept probabilistic labels (such as Scikit-Learn), we can instead replace each label distribution with the label of the class that has the maximum probability. This can easily be done using the probs_to_preds helper method. We do note, however, that this transformation is lossy, as we no longer have values for our confidence in each label.
#  

# In[36]:


# from snorkel.utils import probs_to_preds
# preds_train_filtered = probs_to_preds(probs=probs_train_filtered)


# In[37]:


from sklearn.linear_model import LogisticRegression


# In[38]:


logistic_regr_model = LogisticRegression(solver='liblinear', random_state=1)
logistic_regr_model.fit(L_train, preds_train_label_wcb)  # ) y_train)


# In[39]:


validation_pred_probs = logistic_regr_model.predict_proba(L_valid)
validation_pred_probs


# In[40]:


df_val = pd.DataFrame(validation_data, columns=['Number'])
df_val["is_odd"] = df_val.apply(is_odd, axis=1)
# df_val["is_even"] = df_val.apply(is_even, axis=1)
df_val["is_two"] = df_val.apply(is_two, axis=1)
df_val["is_known_prime"] = df_val.apply(is_known_prime, axis=1)
df_val["gt3_ndb_3"] = df_val.apply(gt3_ndb_3, axis=1)
df_val["pred_majority"] = preds_valid_majority_label
df_val["ground_truth"] = validation_true_labels
df_val["log_reg_p(1)"] = validation_pred_probs[:,1]
gt_one_half = lambda x: (x > 0.5)
df_val["log_reg_pred"] = np.multiply(gt_one_half(df_val["log_reg_p(1)"]),1 )
df_val


# In[41]:


accuracy_val_log_regr = (df_val["log_reg_pred"] == df_val["ground_truth"]).mean()
print(accuracy_val_log_regr) 


# ### Use Logistic Regression model to label new, larger data set

# In[42]:


#new_data_to_be_labeled = range(50, 150)
# df_primes = is_it_prime.make_primes_df(200)
# df_new_data_to_be_labeled_by_regr_model = df_primes.loc[50:150, ["Number", "ground_truth"]].reset_index(drop=True)
# df_gr_truth_new_data = df_primes.loc[50:150, ["ground_truth"]].reset_index(drop=True)
# df_new_data = pd.DataFrame(new_data_to_be_labeled_by_regr_model, columns=['Number'])
# df2 = df_new_data_to_be_labeled_by_regr_model["Number"].reset_index(drop=True)
# applier2 = PandasLFApplier(lfs=lfs)
nums, labels = is_it_prime.make_list_of_num_and_labels(0, 200)
df_nums = pd.DataFrame(nums, columns=["Number"])
# L = applier2.apply(pd.DataFrame(nums))

L = apply_labeling_functions(lfs, df_nums)


# LFAnalysis(L, lfs).lf_summary()
# L = applier.apply(df=df2)
LFAnalysis(L=L, lfs=lfs).lf_summary()


# In[43]:


# get probabilistic labels
new_data_pred_probs = logistic_regr_model.predict_proba(L)
# convert probabilistic labels to zero or one.
gt_one_half = lambda x: (x > 0.5)

df_nums["log_reg_pred"] = np.multiply(gt_one_half(new_data_pred_probs[:, 1]),1 )

df_nums["ground_truth"] = labels
acc = (df_nums["log_reg_pred"] == df_nums["ground_truth"] ).mean()
acc


# In[86]:


#(df_nums["log_reg_pred"] != df_nums["ground_truth"])
print("Error df shape == ",df_nums.query("log_reg_pred != ground_truth" ).shape)  # 35/200 are errors
print("Some erroneous predictions: ")
df_nums.query("log_reg_pred != ground_truth" ).tail()  # 35/200 are errors


# Experiment 1
# 
# Parameters:
# 
# Labeling functions: 3, is_odd, is_two, is_known_prime.
# 
# Data set: 20 integers
# 
# Validation: 7 integers
# 
# 
# 
# Using a logistic regression model with three labeling functions, the Snorkel labeling model with class balance using three labeling functions labeled 20 data points (no ground truth/gold labels). The labeling achieved a 100% accuracy against a validation set of seven data points.
# 
# A logistic regression model trained using the data set of 20 numbers and the labels from the Snorkel labeling model was then used to label 200 data points (integers 0-200, which includes the original training and validation data points).  Accuracy of labeling by log regr of the 200 data points was 82.5% (based on ground truth).
# 
