import joblib
import pandas as pd 
import os 
from skimage.transform import resize 
from skimage.io import imread 
import numpy as np 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report



Categories=['Gliome','Méningiome','No_tumeur','Pituitaire']
flat_data=[] 
target_arr=[]
datadir='Tumeur_Cérébrale_dadaset/' 


for i in Categories: 
      
    print(f'chargement... catégorie :{i}') 
    path=os.path.join(datadir,i) 

    for img in os.listdir(path): 
        img_array=imread(os.path.join(path,img)) 
        img_resized=resize(img_array,(150,150,3))
        flat_data.append(img_resized.flatten()) 
        target_arr.append(Categories.index(i)) 

    print(f'catégorie chargée :{i} avec succès') 


flat_data=np.array(flat_data) 
target=np.array(target_arr)

dataframe=pd.DataFrame(flat_data)  
dataframe['Target']=target 
print(dataframe.shape)

input_data=dataframe.iloc[:,:-1] 
output_data=dataframe.iloc[:,-1]



x_train,x_test,y_train,y_test=train_test_split(input_data,output_data,test_size=0.20, random_state=77,  stratify=output_data) 



param_grid= {'n_neighbors': [2, 4 , 6 , 8 , 10 ], 
              'weights': ['uniform', 'distance'], 
              'metric': ['euclidean', 'manhattan']
              }

knn = KNeighborsClassifier() 
print('\ncréation un modèle en utilisant GridSearchCV.....')
model=GridSearchCV(knn,param_grid,verbose=2,cv=3) 


model.fit(x_train,y_train)

print("\n la meilleure note .....")
print(model.best_params_)
print("\n le meilleur paramètre .....")
print(model.best_score_)


print("\nTester le modèle à l'aide des données de test.....")
y_pred = model.predict(x_test) 
print(classification_report(y_test, y_pred, target_names=['Gliome','Méningiome','No_tumeur','Pituitaire']))
accuracy = accuracy_score(y_pred, y_test) 
print(f"\nLe modèle est précis à {accuracy*100}% ")




nom_fichier_modele = 'Tumeur_Cérébrale_KNN_TEST.joblib'
joblib.dump(model,nom_fichier_modele)
print(f"\nLe modèle a été sauvegardé avec succès dans {nom_fichier_modele}\n")


