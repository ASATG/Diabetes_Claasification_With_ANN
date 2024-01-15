#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# In[2]:


df = pd.read_csv("loan_approval_dataset.csv")


# In[3]:


df


# In[4]:


df.rename(columns=dict(zip(df.columns,df.columns.str.strip())),inplace=True)


# In[5]:


df['education']=df['education'].str.strip()
df['education']=df['education'].map({'Graduate':1,'Not Graduate':0})


# In[6]:


df['self_employed']=df['self_employed'].str.strip()
df['self_employed']=df['self_employed'].map({'Yes':1,'No':0})


# In[7]:


df['loan_status']=df['loan_status'].str.strip()
df['loan_status']=df['loan_status'].map({'Approved':1,'Rejected':0})


# In[8]:


Y = (df["loan_status"]).to_numpy().reshape(-1, 1)
X = (df.drop(["loan_status"],axis = 1)).to_numpy()
X_train,X_test,Y_train,Y_test = X[0:3500],X[3500:],Y[0:3500],Y[3500:]


# In[9]:


print(Y.shape)
print(X.shape)
print(Y_train.shape)
print(X_train.shape)
print(Y_test.shape)
print(X_test.shape)


# In[10]:


print(df.info())


# In[11]:


print(df.describe())


# In[12]:


df.isna().sum()


# In[13]:


df.fillna(df.mean())


# In[14]:


plt.figure(figsize=(6, 4))
sns.countplot(x='loan_status', data=df)
plt.title('Distribution of Outcome')
plt.show()


# In[15]:


sns.pairplot(df, hue='loan_status', diag_kind='kde')
plt.suptitle('Pairplot of Features by Outcome', y=1.02)
plt.show()


# In[15]:


import numpy as np

def standardize_data(X):
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    standardized_X = (X - means) / stds
    return standardized_X, means, stds

X_train, means, stds = standardize_data(X_train)
X_test = (X_test - means) / stds


# In[16]:


class Layer_Dense:
    def __init__(self,n_inputs, n_neurons):
        np.random.seed(0)
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.inputs= None
        
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        
    def backward(self,dvalues):
        self.dweights = np.dot(self.inputs.T,dvalues)
        self.dbiases = np.sum(dvalues, axis = 0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)


# In[17]:


class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0,inputs)
    def backward(self,dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


# In[18]:


class Activation_Sigmoid:
    def forward(self, inputs):
        self.inputs = inputs
        clipped_inputs = np.clip(inputs, -500, 500)  # Clip values to avoid overflow/underflow
        self.output = 1 / (1 + np.exp(-clipped_inputs))

    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output


# In[19]:


class Loss_BinaryCrossentropy:
    def __init__(self, epsilon=1e-15):
        self.epsilon = epsilon
        self.dinputs = None
    def forward(self, y_pred, y_true):
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        sample_losses = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return np.mean(sample_losses, axis=-1)
    def backward(self, dvalues, y_true):
        outputs = len(dvalues[0])
        self.dinputs = (-(y_true / dvalues - (1 - y_true) / (1 - dvalues)))
        self.dinputs = self.dinputs / len(dvalues)
        


# In[20]:


def calculate_accuracy(y_true, y_pred):
    y_pred_binary = (y_pred > 0.5).astype(int)
    correct_predictions = np.sum(y_true == y_pred_binary)
    total_samples = len(y_true)
    accuracy = correct_predictions / total_samples
    return accuracy


# In[21]:


layer1 = Layer_Dense(12,64)
activation1 = Activation_ReLU()

layer2 = Layer_Dense(64,32)
activation2 = Activation_ReLU()

layer3 = Layer_Dense(32,1)
activation3 = Activation_Sigmoid()

loss_function = Loss_BinaryCrossentropy()

learning_rate = 0.05
epochs = 5000

for epoch in range(epochs):
    layer1.forward(X_train)
    activation1.forward(layer1.output)
    
    layer2.forward(activation1.output)
    activation2.forward(layer2.output)
    
    layer3.forward(activation2.output)
    activation3.forward(layer3.output)
    
    loss = loss_function.forward(activation3.output,Y_train)
    accuracy = calculate_accuracy(Y_train,activation3.output)
    
    if np.isnan(loss).any() or np.isinf(loss).any():
        print(f'NaN or Inf values encountered in loss. Stopping training.')
        break
    
    loss_function.backward(activation3.output,Y_train)
    activation3.backward(loss_function.dinputs)
    layer3.backward(activation3.dinputs)
    activation2.backward(layer3.dinputs)
    layer2.backward(activation2.dinputs)
    activation1.backward(layer2.dinputs)
    layer1.backward(activation1.dinputs)
    
    layer1.weights += -learning_rate * layer1.dweights
    layer1.biases += -learning_rate * layer1.dbiases

    layer2.weights += -learning_rate * layer2.dweights
    layer2.biases += -learning_rate * layer2.dbiases

    layer3.weights += -learning_rate * layer3.dweights
    layer3.biases += -learning_rate * layer3.dbiases

    if epoch % 100 == 0:
        print(f'Iteration {epoch}, Loss: {np.mean(loss)}, Accuracy: {accuracy*100}')
    
def output_test(inputs,output_true):
    layer1.forward(inputs)
    activation1.forward(layer1.output)
    
    layer2.forward(activation1.output)
    activation2.forward(layer2.output)
    
    layer3.forward(activation2.output)
    activation3.forward(layer3.output)
    y_pred = activation3.output
    y_pred = (y_pred > 0.5).astype(int)
    count = 0
    for i in range(len(output_true)):
        if y_pred[i] == output_true[i]:
            count+=1
            
    print('Test accuracy:', (count/len(output_true))*100)
        
output_test(X_test,Y_test)


# In[22]:


numerical_columns = ['no_of_dependents', 'income_annum', 'loan_amount', 'loan_term', 'cibil_score', 'residential_assets_value', 'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value']

plt.figure(figsize=(12, 8))
sns.boxplot(data=df[numerical_columns])
plt.title('Boxplots of Numerical Variables')
plt.show()


# In[23]:


df.columns


# In[24]:


Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

outlier_threshold = 1.2

outliers = ((df < (Q1 - outlier_threshold * IQR)) | (df > (Q3 + outlier_threshold * IQR))).any(axis=1)
df_no_outliers = df[~outliers]

print("Rows with major outliers removed:")
print(df[outliers])

print("\nCleaned dataset:")
print(df_no_outliers)


# In[25]:


Y = (df_no_outliers["loan_status"]).to_numpy().reshape(-1, 1)
X = (df_no_outliers.drop(["loan_status"],axis = 1)).to_numpy()
X_train,X_test,Y_train,Y_test = X[0:3500],X[3500:],Y[0:3500],Y[3500:]


# In[26]:


print(Y.shape)
print(X.shape)
print(Y_train.shape)
print(X_train.shape)
print(Y_test.shape)
print(X_test.shape)


# In[27]:


import numpy as np

def standardize_data(X):
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    standardized_X = (X - means) / stds
    return standardized_X, means, stds

X_train, means, stds = standardize_data(X_train)
X_test = (X_test - means) / stds


# In[28]:


layer1 = Layer_Dense(12,64)
activation1 = Activation_ReLU()

layer2 = Layer_Dense(64,32)
activation2 = Activation_ReLU()

layer3 = Layer_Dense(32,1)
activation3 = Activation_Sigmoid()

loss_function = Loss_BinaryCrossentropy()

learning_rate = 0.05
epochs = 5000

for epoch in range(epochs):
    layer1.forward(X_train)
    activation1.forward(layer1.output)
    
    layer2.forward(activation1.output)
    activation2.forward(layer2.output)
    
    layer3.forward(activation2.output)
    activation3.forward(layer3.output)
    
    loss = loss_function.forward(activation3.output,Y_train)
    accuracy = calculate_accuracy(Y_train,activation3.output)
    
    if np.isnan(loss).any() or np.isinf(loss).any():
        print(f'NaN or Inf values encountered in loss. Stopping training.')
        break
    
    loss_function.backward(activation3.output,Y_train)
    activation3.backward(loss_function.dinputs)
    layer3.backward(activation3.dinputs)
    activation2.backward(layer3.dinputs)
    layer2.backward(activation2.dinputs)
    activation1.backward(layer2.dinputs)
    layer1.backward(activation1.dinputs)
    
    layer1.weights += -learning_rate * layer1.dweights
    layer1.biases += -learning_rate * layer1.dbiases

    layer2.weights += -learning_rate * layer2.dweights
    layer2.biases += -learning_rate * layer2.dbiases

    layer3.weights += -learning_rate * layer3.dweights
    layer3.biases += -learning_rate * layer3.dbiases

    if epoch % 100 == 0:
        print(f'Iteration {epoch}, Loss: {np.mean(loss)}, Accuracy: {accuracy*100}')
    
def output_test(inputs,output_true):
    layer1.forward(inputs)
    activation1.forward(layer1.output)
    
    layer2.forward(activation1.output)
    activation2.forward(layer2.output)
    
    layer3.forward(activation2.output)
    activation3.forward(layer3.output)
    y_pred = activation3.output
    y_pred = (y_pred > 0.5).astype(int)
    count = 0
    for i in range(len(output_true)):
        if y_pred[i] == output_true[i]:
            count+=1
            
    print('Test accuracy:', (count/len(output_true))*100)
        
output_test(X_test,Y_test)


# In[ ]:




