from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from ripser import ripser, Rips
import stablerank.srank as sr
import numpy as np
from tqdm import tqdm
import pickle

def sample_sizes(df,label_column = 'type'):
    #Create dictionary with sample sizes for each type
    types = set(df[label_column])
    sample_size = {}

    for t in types:
        sample_size[t] = len(df['type'].loc[df['type'] == t])

    return sample_size

def feature_pairing(df, n_pair = 2, correlation = True):
    """Group features with low correlation together in groups of size n_pair"""
    groups = []
    if correlation:
        corr = df.corr().abs()
        N=len(corr)
        for _ in tqdm(range(N//n_pair)):
            column = corr.columns[0]
            new_group = [column] #Add first remaining column to new group

            for i in range(n_pair-1):
                corr_current = corr.loc[new_group]
                #low_corr = corr.loc[column].idxmin()
                low_corr = corr.loc[new_group].sum().idxmin()
                new_group.append(low_corr)


                corr = corr.drop(columns = [low_corr])

            corr = corr.drop(new_group)
            corr = corr.drop(columns = [column])

            groups.append(new_group)
    else:
        N=len(df.columns)
        for i in tqdm(range(N//n_pair)):
            groups.append(list(df.columns[i*n_pair:i*n_pair+n_pair]))
                
    return groups

def data_to_pointcloud(df, feature_partition):
    """Turn each row of df into a point cloud by pairing up columns"""
    points_per_cloud = len(feature_partition)
    cloud_dimension = len(feature_partition[0])
    no_data_points = len(df)
    
    point_data = np.zeros([no_data_points,points_per_cloud,cloud_dimension])
    for i in range(points_per_cloud):
        point_data[:,i,:] = np.array( df.loc[:, feature_partition[i]] )
    return point_data

def get_TDA_data(point_clouds,max_dim = 1,dist_type = 'euclidean'):
    """Calculate persistent homology for each space in an array of point clouds"""
    N = len(point_clouds)
    all_data = []
    for i in tqdm(range(N)):
        pc = point_clouds[i,:,:]
        X = pc
        data = {}
        dist = squareform(pdist(pc, dist_type))

        # getting a distance object
        c_dist = sr.Distance(dist)

        # getting h0sr
        pcf = c_dist.get_h0sr(clustering_method = "complete", reduced = False)

        #getting bar codes
        bc = c_dist.get_bc(maxdim=max_dim)

        data = {"bar code": bc, "H0": pcf}
        #getting Hi for i>0
        for i in range(1,max_dim+1):

            data['H'+str(i)] = sr.bc_to_sr(bc, degree="H"+str(i))
        all_data.append(data)
    return all_data

def _single_row_weighted_KNN(train_features, train_labels, test_feature,k = 5, weights = [0.8,0.2]):
    """Finds k_list[i] nearest neighbours in train data of test data point
    whith distance: interleaving distance for homology group Hi for each i 
    (same data point can appear for Hi and Hj) and returns most frequently
    appearing labels for every point in test data"""
    label_set = set(train_labels)
    N_train = len(train_features)
    
    
    NN_per_type = {t:0 for t in label_set}
        
        
    dist = lambda x: np.sum([weights[i]*test_feature['H'+str(i)].interleaving_distance(x['H'+str(i)]) for i in range(len(weights))])
    dist_vec = np.vectorize(dist)
        
    #H_train_arr = np.array([train_features[j]['H'+str(i)] for j in range(N_train)])
    distances = dist_vec(train_features)
        
    k_smallest =  np.argpartition(distances, k)[1:k]
    for index in k_smallest:
        NN_per_type[train_labels[index]] +=1
    
    
    return max(NN_per_type,key = NN_per_type.get)

def weighted_KNN(train_features, train_labels, test_features,k = 5, weights = [0.8,0.2]):
    """Finds k_list[i] nearest neighbours in train data of points (point clouds) 
    in test data whith distance: interleaving distance for homology group Hi for 
    each i (same data point can appear for Hi and Hj) and returns most frequently
    appearing labels for every point in test data"""
    single_KNN = lambda x: _single_row_weighted_KNN(train_features, train_labels, x, k, weights)
    vec_KNN = np.vectorize(single_KNN)
    return vec_KNN(np.array(test_features))

def _prediction(labels,indexes):
    predicted_labels = np.array([labels[i] for i in indexes])
    count = {t:0 for t in set(predicted_labels)}
    for t in predicted_labels:
        count[t] += 1
    return max(count,key = count.get)

def exclude_one_KNN_accuracy(distance_matrix,labels, k=5, by_label = False):
    """Returns accuracy of KNN for data with given distance matrix where for each point in the data 
    the predicted label for this point is calculated using KNN for all other points in the data."""
    closest = np.argsort(distance_matrix)[:,1:k+1]
    N = len(distance_matrix)
    if by_label:
        type_set = set(labels)
        all_by_type = {t: 0 for t in type_set}
        correct_by_type = {t: 0 for t in type_set}
        correct = 0
        for i in range(N):
            true_label = labels[i]
            predicted_label = _prediction(labels,closest[i,:])
            all_by_type[true_label] +=1
            if predicted_label == true_label:
                correct_by_type[true_label] += 1
                correct += 1
        ratio_by_type = {t:correct_by_type[t]/all_by_type[t] for t in type_set}
        ratio_by_type['total'] = correct/N
        return ratio_by_type
        
        
        
    else:
        correct = 0
        for i in range(N):
            true_label = labels[i]
            predicted_label = _prediction(labels,closest[i,:])
            if predicted_label == true_label:
                correct += 1
        return correct/N

def exclude_one_KNN_type_distribution(distance_matrix,labels, k=5):
    """Returns accuracy of KNN for data with given distance matrix where for each point in the data 
    the predicted label for this point is calculated using KNN for all other points in the data."""
    closest = np.argsort(distance_matrix)[:,1:k+1]
    N = len(distance_matrix)
    
    type_set = set(labels)
    all_by_type = {t: 0 for t in type_set}
    predicted_by_type = {t1: {t2: 0 for t2 in type_set} for t1 in type_set}
    
    for i in range(N):
        true_label = labels[i]
        predicted_label = _prediction(labels,closest[i,:])
        all_by_type[true_label] +=1
        predicted_by_type[true_label][predicted_label] += 1
    ratio_by_type = {tt:{tp: predicted_by_type[tt][tp]/all_by_type[tt] for tp in type_set} for tt in type_set}
    return ratio_by_type

def dfunc_to_matrix(values, distance_function):
    """Turn a set of points and a distance function between these points into a distance matrix"""
    N = len(values)
    dmat = np.zeros([N,N])
    for i in range(N):
        for j in range(i,N):
            d = distance_function(values[i],values[j])
            dmat[i,j] = d
            dmat[j,i] = d
    return dmat

def make_PCA(df,N_components = 120, label_column = 'type'):
    """Return data frame with N_components first principal components of df"""
    #Make PCA 
    X = df.drop(columns=[label_column]).to_numpy()
    labels = df[label_column].to_numpy()
    scaler = StandardScaler()
    Xscaled = scaler.fit_transform(X)
    pca = PCA(n_components=N_components, svd_solver = 'full')
    principal_components = pca.fit_transform(Xscaled)

    #Normalize the principal components
    principal_components = (principal_components - principal_components.mean())/principal_components.std()

    #Column names:
    columns = ["PC" + str(i) for i in range(1,N_components +1)]

    #Data frame
    df_PCA = pd.DataFrame(principal_components,columns = columns)
    df_PCA[label_column] = labels
    
    return df_PCA

def avg_sr(TDA_data, labels, label, hom_index = 0):
    flag = True
    count = 0
    for i, data in enumerate(TDA_data):
        if labels[i] == label:
            count += 1
            if flag:
                H = data['H'+ str(hom_index)]
                flag = False
            else:
                H += data['H'+ str(hom_index)]
    return H/count


if __name__ == '__main__':
    DATASET = 'breast'
    df_read = pd.read_csv('data/'+DATASET+'cancer.csv').drop(columns = ['samples'])
    df_read = df_read.loc[df_read['type'] != 'cell_line']
    df_PCA = make_PCA(df_read, 120) #data frame with first 120 principal components of df_read
    
    sample_sizes = sample_sizes(df_PCA)
    labels = df_PCA['type'].to_numpy()
    df = df_PCA.drop(columns = ['type'])
    print("Sample sizes:", sample_sizes)
    
    partition = feature_pairing(df,2,True) #Create partition of data (groups of 2)
    point_clouds = data_to_pointcloud(df, partition) #Turn df into list of pointclouds by pairing elements in each row.
    
    #Compute TDA data for the generated point clouds (homology stable ranks up until dimension max_dim = 1)
    TDA_data = get_TDA_data(point_clouds,max_dim = 1,dist_type ='euclidean')
    
    weights = [1,0]
    #Distance function between points in TDA_data, 1d_0 + 5d_1 where d0/d1 is interlaving distance for H0/H1 stable ranks
    dfunc = lambda x,y: np.sum([weights[i]*x['H'+str(i)].interleaving_distance(y['H'+str(i)]) for i in range(len(weights))])
    
    dmat = dfunc_to_matrix(TDA_data, dfunc) #Distance matrix from distance function and data
    
    kNN = 6 #Number of nearest neigbours
    
    #For nicer formatting of output
    if DATASET == 'breast':
        label_order = ['basal',  'HER', 'luminal_A', 'luminal_B', 'normal']
        label_converter = {'basal': 'Basal',  'HER':'HER', 'luminal_A': 'Luminal A', 'luminal_B':'Luminal B', 'tumoral_non_BLC': 'Non BLC', 'normal': 'Normal'}
    else:
        label_order = ['glioblastoma','pilocytic_astrocytoma',  'ependymoma', 'medulloblastoma' , 'normal', ]
        label_converter = {'glioblastoma': 'Glioblastoma','pilocytic_astrocytoma':'Pilocytic Astrocytoma', 'ependymoma':'Ependymoma', 'medulloblastoma': 'Medulloblastoma', 'normal': 'Normal', 'Astrocytoma': 'Astrocytoma'}
    
    accuracy_dict = exclude_one_KNN_accuracy(dmat,labels, k=kNN, by_label = True)
    
    print('Weights:',weights)
    print(accuracy_dict)#Accuracy by label
    
    #Accuracy data formatted such that it can just be copy pasted into latex table 
    print('')
    print('==========================================================================================')
    print('To paste in latex table:')
    for label in label_order:
        print(round(accuracy_dict[label], 3),end = ' & ')
    print(round(accuracy_dict['total'], 3))
    print('==========================================================================================')
    
    prediction_percentages = exclude_one_KNN_type_distribution(dmat, labels, k=kNN)
    
    fig = plt.figure(figsize=(10,6))

    ax = fig.add_subplot(111)
    w = 0.1
    for i,t in enumerate(label_order[:]):
        data = prediction_percentages[t]
        names = [label_converter[tp] for tp in label_order[:]]
        values = [data[tp] for tp in label_order[:]]
        ax.bar(np.array(range(len(values))) + i*w, values, tick_label=names, width = w, label = label_converter[t])
        plt.xlabel('Predicted Value')
        plt.ylabel('Ratio')

    fig.legend()
    fig.show()
    plt.show()    