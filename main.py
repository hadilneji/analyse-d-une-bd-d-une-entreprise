import pandas
import sklearn
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
df = pandas.read_csv(r"D:\telechargement\git.csv")
#print(df)
bd=df
n=bd.shape[0] #nombre de lignes
p=bd.shape[1] #nombre de colonnes
#rechercher et afficher les types de départements dans la colonne sales
print(df['sales'].unique())
#remplacer sales =1 // support=2 // accounting=3// hr=4// technical=5 // management=6// IT=7// product_mng=8// marketing=9// RandD=10//
bd['sales']=bd['sales'].replace(['sales','support','accounting','hr','technical','management','IT','product_mng','marketing',
'RandD'],[1,2,3,4,5,6,7,8,9,10])
print(bd)
#rechercher et afficher les types de salaires dans la colonne salary
print(df['salary'].unique())
#remplacer high=3// medium=2// low=1
bd['salary']=bd['salary'].replace(['high','medium','low'],[3,2,1])
print (bd)
#moyenne
l=[]
l.append(bd.mean())
print(l)
#verifier l'existance des valeurs inconnues
print(bd.isnull().values.any())
sc=StandardScaler()
Z=sc.fit_transform(bd)
print(Z)
#Afficher la centrée réduite Z avec Pandas
print("la matrice centrée reduite ")
print((bd-bd.mean())/bd.std(ddof=0))
#Vérifiez les moyennes et les écarts types après la standardisation.
mcr=(bd-bd.mean())/bd.std(ddof=0);
print("Moyennes sur Z :",mcr.mean())
print("Ecartypes sur Z :", mcr.std())
#Afficher et analyser la matrice de corrélation.
#avec pandas
print(" \n matrice de correlation \n")
Corr=bd.corr()
print(Corr)
#Affichage graphique avancé de la matrice de corrélation#heatmap pour identifier visuellement les corrélations fortes
#librairie graphique
# print("\n \n premier affichage graphique ")
sns.heatmap(Corr,xticklabels=mcr.columns,yticklabels=mcr.columns,vmin=-
1,vmax=+1,center=0,cmap="RdBu",linewidths=0.5, annot=True)
sns.set()
#begin-didn't understood
# #make Lower triangular values Null
# # print(np.triu(np.ones(Corr.shape), k=1).astype(bool))
# #Corr.where(cond) remplace des valeurs dans Corr lorsque la condition est false
upper_corr_mat = Corr.where(np.triu(np.ones(Corr.shape), k=1).astype(bool))
print(upper_corr_mat)
# Convert to 1-D series and drop Null values
unique_corr_pairs = upper_corr_mat.unstack().dropna()
print(unique_corr_pairs)
#end-didn't understood
# Sort correlation pairs
sorted_mat = unique_corr_pairs.sort_values(ascending=False)
print("\n sorted matrice \n",sorted_mat,"\n")
#pairplot
# print("\n deuxieme affichage graphique")
sns.pairplot(mcr,corner=True,diag_kind='hist',vars=["sales",'salary'], )
sns.pairplot(mcr,corner=True,diag_kind='hist',vars=["average_montly_hours",
'number_project'], )
sns.set()
#Réaliser sur Z une ACP normée en utilisant la méthode pca du module
sklearn.decomposition
pca = PCA()
Y = pca.fit_transform(Z)
print(Y)
print("columns ")
columns = ['pca_%i' % i for i in range(10)]
Y_df=pandas.DataFrame(Y,columns=columns)
print(Y_df)
np.set_printoptions(suppress=True)
print(Y_df.corr())
print(np.corrcoef(Y_df,rowvar=False))
pca1=PCA()
Y1=pca1.fit_transform(Y)
print(pandas.DataFrame(Y1,columns=Y_df.columns))
val_propres= pca.explained_variance_
print(val_propres)
print("\n somme des valeurs propres \n")
#somme des valeurs propres = nbre des variables
print(np.sum(val_propres))
print(pca.explained_variance_)
# valeur corrigée
# print("\n valeur corrigée \n ")
# val_propres = (n -1) / n * pca.explained_variance_
# print(val_propres)
print("\n valeurs singulieres \n ")
print(pca.singular_values_ ** 2 / n)
#plt.bar(range(pca.n_components_), pca.explained_variance_ratio_)
plt.plot(range(pca.n_components_), val_propres )
plt.title("variance expliquée /CP ")
plt.xlabel('Composantes principales')
plt.ylabel('Valeur de variance expliquée')
plt.xticks(range(pca.n_components_))
plt.show()
plt.bar(range(pca.n_components_), pca.explained_variance_)
plt.title("Variance expliquée / CP")
plt.xlabel('Composantes principales')
plt.ylabel('Variance expliquée')
plt.xticks(range(pca.n_components_))
plt.show()
# proportion de variance expliquée
print("proportion de variance expliquée")
print(pca.explained_variance_ratio_ ) # Il n’est pas nécessaire d’effectuer une correction dans ce cas
exp_var_ratio = pca.explained_variance_ratio_
print(exp_var_ratio*100)
#Somme cumulative des valeurs propres
cum_sum_eigenvalues = np.cumsum(exp_var_ratio)
print(cum_sum_eigenvalues)
#Préparation du plot pour l'affichage
plt.plot(range(pca.n_components_),cum_sum_eigenvalues,'o-', linewidth=2,
color='blue')
plt.grid(which='both', linestyle='--')
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()
#positionnement des individus dans le premier plan
fig, axes = plt.subplots(figsize=(7,7))
axes.set_xlim(-5,5) #même limites en abscisse
axes.set_ylim(-5,5) #et en ordonnée
#placement des étiquettes des observations
for i in range(10):
 plt.annotate(bd.index[i],(Y_df.iloc[i,0],Y_df.iloc[i,1]))
#ajouter les axes
plt.plot([-5,5],[0,0],color='silver',linestyle='-',linewidth=1)
plt.plot([0,0],[-5,5],color='silver',linestyle='-',linewidth=1)
#affichage
plt.scatter(Y_df.pca_0, Y_df.pca_1, s=50)
plt.show()
#Calculer la matrice de corrélation des anciennes variables (Zj) et des nouvelles (Yk) du plan factoriel*.
Q=pca.components_
#𝑟(𝑍 𝑗, 𝑌 𝑘) = racine_carrée(𝜆𝑘)*𝑞𝑗k
p=pca.n_components_
corvar = np.zeros((p,p))
for k in range (7):
 corvar[:,k] = Q[k,:] * np.sqrt(val_propres[k]) #remplit par colonne
print(corvar)
CorrVariables=pandas.DataFrame(corvar,index=mcr.columns,columns=columns)
print(CorrVariables)
#on affiche pour les deux premiers axes
print(pandas.DataFrame({'id':mcr.columns,'PC0':corvar[:,0],'PC1':corvar[:,1
]}))
#Analyser la saturation des variables en projetant les variables (Zj) sur le cercle de corrélation C
#cercle des corrélations
fig, axes = plt.subplots(figsize=(4,4))
axes.set_xlim(-1,1)
axes.set_ylim(-1,1)
#affichage des étiquettes (noms des variables)
for j in range(mcr.shape[1]):
 plt.annotate(mcr.columns[j],(corvar[j,0],corvar[j,1]))
 plt.quiver(0, 0, corvar[j,0],corvar[j,1], angles = 'xy', scale_units =
'xy', scale = 1) # Tracé d'un vecteur
#plt.scatter(T.pca_0, T.pca_1, s=50, c=colormap[classe-1])
#ajouter un cercle
cercle = plt.Circle((0,0),1,color='blue',fill=False)
axes.add_artist(cercle)
#ajouter les axes
plt.plot([-1,1],[0,0],color='silver',linestyle='-',linewidth=1)
plt.plot([0,0],[-1,1],color='silver',linestyle='-',linewidth=1)
plt.show()
from sklearn.cluster import KMeans
#from sklearn import cluster.KMeans
kmeans=KMeans(n_clusters=3).fit(Y)
print(kmeans)
#Afficher les coordonnées de chaque centroïde et l’inertie associée.
centroids=kmeans.cluster_centers_
print(centroids)
plt.scatter(Y[:,0], Y[:,1], c= kmeans.labels_.astype(float), s=50,
alpha=0.5)
plt.scatter(centroids[:, 0],centroids[:, 1], c='red', s=50)
plt.show()
# Afficher les individus et leurs groupes.
import numpy as np
idk=np.argsort(kmeans.labels_)
df_kmeans=pandas.DataFrame(bd.index[idk],kmeans.labels_[idk])
print(df_kmeans)
#print(pandas.DataFrame(X.index[idk],kmeans.labels_[idk].count()))
print(kmeans.labels_)
print(kmeans.inertia_)
#distances aux centres de classes des observations
print(kmeans.transform(Y))
#Méthode de silhouette
from sklearn import metrics
l2=list()
for i in np.arange(2,14):
 y_pred=KMeans(n_clusters=i,init='k-means++',n_init=10).fit_predict(Y_df)
 l2.append(metrics.silhouette_score(Y_df, y_pred, metric='euclidean'))
 print('La silhouette index pour {0:d} classes est {1: 3f}'.format(i,metrics.silhouette_score(Y_df,y_pred,metric='euclidean')))
 import matplotlib.pyplot as plt

 plt.title("Silhouette")
 plt.xlabel("# of clusters")
 plt.plot(np.arange(2, 14, 1), l2)
 plt.show()
 from matplotlib import pyplot as plt
 from scipy.cluster.hierarchy import dendrogram, linkage

 Y_CAH = linkage(Y, method='ward', metric='euclidean')
 # method='ward : L'algorithme de liaison à utiliser
 # metric='euclidean' : La métrique de distance à utiliser.
 # Faire une matérialisation en 5 classes
 from scipy.cluster.hierarchy import fcluster

 plt.title('CAH avec matérialisation des classes')
 dendrogram(Y_CAH, labels=bd.index, orientation='top', color_threshold=15)
 plt.show()
 # Afficher les correspondances avec les groupes de la CAH
 groupes_cah = fcluster(Y_CAH, t=15, criterion='distance') - 1
 print(groupes_cah)
 import numpy as np
 idg = np.argsort(groupes_cah)
 print(pandas.DataFrame(bd.index[idg], groupes_cah[idg]))
 pandas.crosstab(groupes_cah, kmeans.labels_)

