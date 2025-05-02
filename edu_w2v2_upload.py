# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 12:58:43 2024
Code to find meaningful actions in action sequences.
Word2vec, Doc2vec,and NN+data augmentation (if necessary)
@author: Minyoung  Yun ENSAM PARIS FRANCE
"""

import pandas as pd
import random
import string
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import nltk
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import numpy as np
from sklearn.cluster import DBSCAN
from adjustText import adjust_text
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from scipy.spatial import KDTree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import f1_score


def data_augmentation(reduced_docvect, scoreL0, scale0):
    a_doclist = [reduced_docvect]

    variation_percentage = 0.01  # 1%
     
    for ns in range(scale0):
        # Generate random variations within the range of -1% to +1%
        random_variations = np.random.uniform(-variation_percentage, variation_percentage, size=reduced_docvect.shape)

        # Apply the random variations to the original data
        varied_data = reduced_docvect * (1 + random_variations)
    
        a_doclist.append(varied_data)
    
    a_red_docvect = np.vstack([array for array in a_doclist])
    a_scoreL0 = scoreL0 * (scale0+1)
    
    return a_red_docvect, a_scoreL0

def ann00(reduced_docvect, scoreL0, case):
    if case ==0: #scroe 3 = 1, the rest = 0       
        indices = [index for index, value in enumerate(scoreL0) if value in [1,2]] #select data with only full/zero score
        scoreL00 = [0 if index in indices else value for index, value in enumerate(scoreL0)]
        scoreL00 = [1 if value ==3 else value for index, value in enumerate(scoreL00)]
        
        X= reduced_docvect
        y= np.array(scoreL00)
        nan_indices = [index for index, value in enumerate(y) if np.isnan(value)]
    else: #only score 3 and 0
        indices = [index for index, value in enumerate(scoreL0) if value in [0,3]]
        scoreL00 = [scoreL0[ind] for ind in np.array(indices)]
        scoreL00 = [1 if value ==3 else value for index, value in enumerate(scoreL00)]
        
        X= reduced_docvect[np.array(indices)]
        y= np.array(scoreL00)
        nan_indices = [index for index, value in enumerate(y) if np.isnan(value)]
          
    # Remove the element at the specified index
    X = np.delete(X, nan_indices, axis=0)
    y = np.delete(y, nan_indices)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert labels to one-hot encoded vectors
    y_train_encoded = to_categorical(y_train,num_classes=2)
    y_test_encoded = to_categorical(y_test,num_classes=2)

    # Build a simple neural network model
    model = keras.Sequential([
        layers.Input(shape=(2,)),
        layers.Dense(8, activation='relu'),
        layers.Dense(2, activation='softmax')  # 3 output classes
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) #loss='categorical_crossentropy'

    # Train the model
    model.fit(X_train, y_train_encoded, epochs=500, batch_size=32, validation_data=(X_test, y_test_encoded))

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test_encoded)
    y_pred = model.predict(X_test)
    binary_predictions = np.argmax(y_pred, axis=1)
    binary_predictions = (binary_predictions > 0.5).astype(int)
    print(f"Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}")
    return binary_predictions, y_test

def RF(reduced_docvert, scoreL0, case):
    if case ==0: #scroe 3 = 1, the rest = 0       
        indices = [index for index, value in enumerate(scoreL0) if value in [1,2]] #select data with only full/zero score
        scoreL00 = [0 if index in indices else value for index, value in enumerate(scoreL0)]
        scoreL00 = [1 if value ==3 else value for index, value in enumerate(scoreL00)]
        
        X= reduced_docvect
        y= np.array(scoreL00)
        nan_indices = [index for index, value in enumerate(y) if np.isnan(value)]
    else: #only score 3 and 0
        indices = [index for index, value in enumerate(scoreL0) if value in [0,3]]
        scoreL00 = [scoreL0[ind] for ind in np.array(indices)]
        scoreL00 = [1 if value ==3 else value for index, value in enumerate(scoreL00)]
        
        X= reduced_docvect[np.array(indices)]
        y= np.array(scoreL00)
        nan_indices = [index for index, value in enumerate(y) if np.isnan(value)]
    
    # Remove the element at the specified index
    X = np.delete(X, nan_indices, axis=0)
    y = np.delete(y, nan_indices)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert labels to one-hot encoded vectors
    y_train_encoded = to_categorical(y_train,num_classes=2)
    y_test_encoded = to_categorical(y_test,num_classes=2)

    # Initialize and train the Random Forest classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train_encoded)
    
    # Make predictions
    y_pred = rf_model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    
    return y_pred, accuracy
    

def sent_doc(filtered_df, word_nicknames):
    unique_sq = filtered_df['SEQID'].unique()
    
    nn_list = []
    for sq0 in unique_sq:
        filt_sq = filtered_df[filtered_df['SEQID'] == sq0]
        act0 = filt_sq["event_type"].tolist()
        
        nn_list0 = [word_nicknames[tt] for tt in act0]
        nn_list.append(nn_list0)
    return nn_list

def word2vec(result_df):
    # Get the unique values of a specific column, for example, 'column_name'
    #unique_values = result_df['event_type'].unique()
    unique_values = result_df['SEQID'].unique()
    uni_list = unique_values.tolist()

    #create nicknames and find the nickname list
    word_nicknames, nickname_list = create_nickname2(uni_list)
    nn_list = sent_doc(result_df, word_nicknames)
    #model = word2vec(nn_list)
    # Flatten the list of lists into a single list
    #flattened_list = [item for sublist in nn_list for item in sublist]
    # Find unique values using set
    #unique_nn = list(set(flattened_list))
    
    # Train the Word2Vec model
    model = Word2Vec(nn_list, vector_size=100, window=5, min_count=1, sg=1)
    
    return model

def var_ftr(df_info0, var0, limit0):
    if limit0 == 4:
        df_i_idx = df_info0['SEQID'].values
    else:
        df_i_idx0 = df_info0[df_info0[var0] == limit0]['SEQID']
        df_i_idx = df_i_idx0.values
    return df_i_idx

def word2vec1(result_df,df_i_idx):
    # Get the unique values of a specific column, for example, 'column_name'
    idx_arry0= result_df['SEQID'].values
    ovp_vals = np.intersect1d(df_i_idx, idx_arry0)
    seq_acn0 = [list(result_df[result_df['SEQID']== ovs]['event_type']) for ovs in ovp_vals]
    
    # Train the Word2Vec model
    model = Word2Vec(seq_acn0, vector_size=100, window=5, min_count=1, sg=1)    #skip gram
    return model, seq_acn0

def access_word2vec1(model,seq_acn0, n_components, PT0):
    # Flatten the list of lists
    flattened_list = [item for sublist in seq_acn0 for item in sublist]
    
    # Get the unique values using set()
    unique_words = list(set(flattened_list))
    
    #access word embeddings
    # Get the embedding vector for a specific word
    word_embedding = [model.wv[n0] for n0 in unique_words]
    #print(word_embedding)
    
    word_embeddings_reduced0 = PCA0(word_embedding, n_components)
    word_embeddings_reduced1 = t_SNE0(np.array(word_embedding), n_components)
    
    if PT0 == 0: word_embeddings_reduced = word_embeddings_reduced0 
    else: word_embeddings_reduced = word_embeddings_reduced1
    
    return unique_words, word_embeddings_reduced

def access_word2vec(model):
    #access word embeddings
    # Get the embedding vector for a specific word
    rev_wn0 = create_reverse_dict(word_nicknames)
    word_embedding = [model.wv[n0] for n0 in unique_nn]
    #print(word_embedding)

    word_embeddings_reduced = PCA0(word_embedding)
    return

def pre_doc2vec(result_df, max_seqid):
    unique_seqid = result_df['SEQID'].unique()
    
    tagged_data = []
    sent_list = []
    for isq in unique_seqid:
        if isq < max_seqid:
            acn_list0= result_df[result_df['SEQID'] == isq]['event_type'].tolist()
            tagged_data.append(TaggedDocument(words=acn_list0, tags=[str(isq)]))
            sent_list.append(acn_list0)        
    return tagged_data, sent_list, unique_seqid

def pre_doc2vec_edt(result_df, max_seqid, lookup_list, case):   
    unique_seqid = result_df['SEQID'].unique()
    unique_crop = unique_seqid[unique_seqid < max_seqid]
    #scoreL_idx = [[df_info0[df_info0['SEQID'] == ii].index.tolist()[0]+1,df_info0[df_info0['SEQID'] == ii]['U01a000S'].values[0]] for ii in unique_seqid0 if ii < max_seqid]
    #unique_seqid = [value[0] for value in scoreL_idx if value[1] in [0.0,3.0]]

    tagged_data = []
    sent_list = []
    for isq in unique_crop:
         acn_list0= result_df[result_df['SEQID'] == isq]['event_type'].tolist()
         #replacement_a = 'cluster1'; replacement_b = 'cluster2'
         # New list with replaced elements
         #new_list = [x if x in lookup_list else replacement_value for x in acn_list0]
         #new_list = [replacement_a if x in lookup_list1 else replacement_b if x in lookup_list2 else x for x in acn_list0]
         
         if case ==0:
             # Create a new list containing only the elements present in the lookup_list
             new_list = [x for x in acn_list0 if x not in lookup_list]    
         else:
             new_list = [x for x in acn_list0 if x in lookup_list]    
         
         tagged_data.append(TaggedDocument(words=new_list, tags=[str(isq)]))
         sent_list.append(new_list) 
         
    # Find all indices of sublists with the target length
    indices_with_target_length = [index for index, sublist in enumerate(sent_list) if len(sublist) == 4]

    return tagged_data, unique_crop, unique_seqid


def doc2vec0(tagged_data, unique_seqid, max_seqid, df_info0, n_components, PT0):
    # Initialize and train the Doc2Vec model
    model = Doc2Vec(vector_size=100, window=5, min_count=1, workers=4, epochs=100) #PV-DBOW BY DEFAULT
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    
    # Get the document vectors
    document_vectors = [model.docvecs[str(idx)] for idx in unique_seqid if idx < max_seqid]
    reduced_docvect0 = PCA0(document_vectors, n_components)
    reduced_docvect1 = t_SNE0(np.array(document_vectors), n_components)
    
    #filtered score values
    scoreL0 = [df_info0[df_info0['SEQID'] == ii]['U01a000S'].values[0] for ii in unique_seqid if ii < max_seqid]
    
    if PT0 == 0: reduced_docvect = reduced_docvect0 
    else: reduced_docvect = reduced_docvect1
    
    return reduced_docvect, scoreL0
    unique_crop = unique_seqid[unique_seqid < max_seqid]
    time_info0 = [result_df[result_df['SEQID']==iuc]['timestamp'].tolist() for iuc in unique_crop]
    time_info1 = [ii[1:3] for ii in time_info0] #time to the first action, and time for which the first action was carried out
    time_info2 = np.array(time_info1)
    
    ind0 = np.array([index for index, value in enumerate(scoreL0) if value in [0]]) #sel
    ind3 = np.array([index for index, value in enumerate(scoreL0) if value in [3]]) #sel
    t_avg0 = np.mean(time_info2[ind0,0])
    t_avg3 = np.mean(time_info2[ind3,0])
    
    time_info2_0 = [result_df[result_df['SEQID']==iuc]['timestamp'].tolist()[1:3] for iuc in unique_crop2[np.array(mnr_vlu)]]
    time_info2_1 = np.array(time_info2_0)
    t_avgq = np.mean(time_info2_1[:,0]) #the points of interest 
    
    plt.hist(time_info2[ind0,0])
    plt.xlim(0,200000)
    plt.show()
    
    return

# DIMENSINOALITY REDUCTION METHOD FOR VISUALISATION
def PCA0(word_embeddings_array, n_components):
    # Specify the number of dimensions for the reduced embeddings
    #n_components = 3  
    # Initialize the PCA object
    pca = PCA(n_components)   
    # Apply PCA to reduce the dimensionality of the word embeddings
    word_embeddings_reduced = pca.fit_transform(word_embeddings_array)
    
    return word_embeddings_reduced

# DIMENSINOALITY REDUCTION METHOD FOR VISUALISATION
def t_SNE0(data_points, n_components):
    # Create t-SNE model
    tsne_model = TSNE(n_components, random_state=42)
    
    # Apply t-SNE to transform data into 2D space
    embedded_data = tsne_model.fit_transform(data_points)
    return embedded_data

def visualize(word_embeddings_reduced, unique_words):
    # Visualize the reduced embeddings in 2D
    %matplotlib qt 
    #plt.figure(figsize=(20, 12))
    #plt.scatter(word_embeddings_reduced[:, 0], word_embeddings_reduced[:, 1])
        
    x = word_embeddings_reduced[:,0]
    y = word_embeddings_reduced[:,1]
    
    fig, ax = plt.subplots()
    ax.plot(x, y, 'bo')
    texts = [plt.text(x[i], y[i], ix) for i,ix in enumerate(unique_words)]
    adjust_text(texts,arrowprops=dict(arrowstyle='->', color='red'), expand_text=(1.05, 1.05))
        
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('2D Visualization of Word Embeddings')
    plt.grid(True)
    plt.show()
    
    ################################## 
    ##adjust text##
    #x = word_embeddings_reduced[:,0]
    #y = word_embeddings_reduced[:,1]
    #plt.figure(figsize=(20, 12))
    #plt.scatter(x, y)
    
    # Annotate data points with labels
    #texts = []; labels = unique_words
    #for i, txt in enumerate(labels):
    #    texts.append(plt.text(x[i], y[i], txt, ha='center', va='center'))
    
    # Adjust the position of labels to avoid overlap
    #adjust_text(texts)
    
    # Show the plot
    #plt.xlabel('X-axis')
    #plt.ylabel('Y-axis')
    #plt.title('Scatter Plot with Non-Overlapping Labels')
    #plt.show()
    return texts


def vis_doc2vec(reduced_docvect0,scoreL0):
    #2D
    #visualize a scatter plot with color-coded points
    x_values = reduced_docvect[:,0]
    y_values = reduced_docvect[:,1]
    color_values = scoreL0
    
    # Create a scatter plot with color-coded points
    plt.scatter(x_values, y_values, c=color_values, cmap='viridis', s=30, alpha=0.8)
    
    # Add a colorbar to show the color scale
    cbar = plt.colorbar()
    cbar.set_label('Color Scale')
    
    # Set labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('2D Scatter Plot with Color Coding')
    
    # Show the plot
    plt.show()
    
    #3D
    #to allow pop-up window
    %matplotlib qt    
    indices = [index for index, value in enumerate(scoreL0) if value in [0.0,3.0]]
    x_values = reduced_docvect[indices,0]
    y_values = reduced_docvect[indices,1]
    z_values = reduced_docvect[indices,2]
    color_values = np.array(scoreL0)[indices]
    
    #if all
    x_values = reduced_docvect1[:,0]
    y_values = reduced_docvect1[:,1]
    z_values = reduced_docvect1[:,2]      
    color_values = scoreL0
    
    
    # Create a 3D scatter plot with color-coded points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x_values, y_values, z_values, c=color_values, cmap='viridis', s=15, alpha=0.8)
    
    # Add a colorbar to show the color scale
    cbar = plt.colorbar(scatter)
    cbar.set_label('Color Scale')
    
    # Set labels and title
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    plt.title('3D Scatter Plot with Color Coding')
    
    # Show the plot
    plt.show()

def dbscan0(word_embeddings_reduced, unique_words, nnk):
    data0 = word_embeddings_reduced
    # Create DBSCAN model
    dbscan = DBSCAN(eps=0.4, min_samples=5)

    # Fit the model to the data and obtain cluster labels 
    #data0= reduced_docvect0 #reduced_docvect0 #document_vectors
    cluster_labels = dbscan.fit_predict(data0)
    #kmeans
    #kmeans_model = KMeans(n_clusters=nnk, random_state=42)
    #cluster_labelsk = kmeans_model.fit_predict(data0)
    
    # Print the cluster labels (-1 indicates noise points)
    #print("Cluster labels:", cluster_labels)

    # Plot the data points with cluster assignments
    plt.scatter(data0[:, 0], data0[:, 1], c=cluster_labels, cmap='viridis')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')   
    
    x = word_embeddings_reduced[:,0]
    y = word_embeddings_reduced[:,1]
    texts = [plt.text(x[i], y[i], ix) for i,ix in enumerate(unique_words)]
    return texts

def adj_text(texts):    
    adjust_text(texts, expand_text=(1.0005, 1.0005))
   
def clst_sil(data0, cluster_labels):
    # Calculate the Silhouette score
    silhouette_avg = silhouette_score(data0, cluster_labels)  
    print("Silhouette Score:", silhouette_avg)
    
    return silhouette_avg


t_file = "BE_logdata.txt"
info0 = 'raw_data/prgbelp1.csv'
# Use the read_csv() function with delimiter='\t' to read the data from the text file into a pandas DataFrame
df = pd.read_csv(t_file, delimiter='\t')
df_info0 = pd.read_csv(info0)
max_seqid = df_info0['SEQID'].max()

# Filter the DataFrame based on a specific column value
#filtered_df = df[(df['booklet_id'] == 'L13') & (df['item_id'] == 1)]
result_df = df[df['booklet_id'].str.contains('PS1') & (df['item_id'] == 1)] #problem solving only

#doc2vec
tagged_data, unique_crop, unique_seqid = pre_doc2vec(result_df, max_seqid)
reduced_docvect, scoreL0 = doc2vec0(tagged_data, unique_seqid, max_seqid, df_info0, n_components=2, PT0=1) #PT0=0 PDA, PTO=1 tSNE

#word2vec
df_i_idx = var_ftr(df_info0, var0 = 'U01a000S', limit0 = 0)  
model, seq_acn0 = word2vec1(result_df,df_i_idx)
unique_words, word_embeddings_reduced = access_word2vec1(model,seq_acn0, n_components=2, PT0=0) #PT0=0 PDA, PTO=1 tSNE
texts = dbscan0(word_embeddings_reduced, unique_words, nnk=3)
adj_text(texts)

#doc2vect + t_SNE + Siluette score
lookup_list = [['MAIL_DRAG', 'MAIL_DROP', 'MAIL_MOVED']] #['CONFIRMATION_OPENED', 'COFIRMATION_CLOSED', 'END'] ['MAIL_DRAG', 'MAIL_DROP', 'MAIL_MOVED'], ['START', 'MAIL_DRAG', 'MAIL_DROP', 'MAIL_VIEWED', 'MAIL_MOVED','FOLDER_VIEWED', 'NEXT_INQUIRY', 'NEXT_ITEM','CONFIRMATION_CLOSED','CONFIRMATION_OPENED','END']
lookup_list = [['START', 'MAIL_DRAG', 'MAIL_DROP', 'MAIL_VIEWED', 'MAIL_MOVED','FOLDER_VIEWED', 'NEXT_INQUIRY', 'NEXT_ITEM','CONFIRMATION_CLOSED','CONFIRMATION_OPENED','END', 'NEXT_BUTTON']]
#doc2vect with modified action lists
tagged_data, unique_crop, unique_seqid = pre_doc2vec_edt(result_df, max_seqid, lookup_list[0], case=1) # case0 not in the list, case1 in the list only. #(result_df, max_seqid, lookup_list)
reduced_docvect, scoreL0 = doc2vec0(tagged_data, unique_seqid, max_seqid, df_info0, n_components=2, PT0=1) #PT0=0 PDA, PTO=1 tSNE

#silhouette_avg
indices = [index for index, value in enumerate(scoreL0) if value in [0.0,3.0]] #select data with only full/zero score
scoreL1 = np.array(scoreL0)
silhouette_avg = clst_sil(reduced_docvect[np.array(indices)], scoreL1[np.array(indices)])
print(silhouette_avg)

# Assuming you have two arrays: y_true (true labels) and y_pred (predicted labels)
binary_predictions, y_test = ann00(reduced_docvect, scoreL0, case=1) #case=1 only sample 0,3, case=0 all samples (0,1,2,3) score1
f1_binary = f1_score(y_test, binary_predictions)
print("Binary F1 Score:", f1_binary)

#augmentation
a_red_docvect, a_scoreL0 = data_augmentation(reduced_docvect, scoreL0, scale0=10)
binary_predictions, y_test = ann00(a_red_docvect, a_scoreL0, case=0) #case=1 only sample 0,3, case=0 all samples (0,1,2,3)
f1_binary = f1_score(y_test, binary_predictions)
print("Binary F1 Score:", f1_binary)
