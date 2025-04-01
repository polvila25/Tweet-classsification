from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score
#En aquest fitxer realitzarem el codi que necesitem per realitzar l'exercici 

#StratifiedKFold, manté la distribució de classes en cada partició, la qual cosa ajuda el model a ser més consistent i a tenir un millor rendiment en mitjana.

#funció per contruir un diccionari on key serà cada paraula del tweet 

#mirar valors nulls
def valors_nulls(data):
    percent_nan_columna = data.isna().mean(axis = 0) * 100
    return percent_nan_columna

#funció per eliminar les files amb valor null
def eliminar_valors_nulls(data):
    return data.dropna(axis=0)

def construir_diccionari(data):
    #anem a construir un diccionari buit amb els Positiu i Negatius i emmagatzemarà la frequencia de cada paraula 
    
    dicc = defaultdict(lambda: {"Positiu":0, "Negatiu":0})
    for _,row in data.iterrows():
        if row['sentimentLabel'] == 1:
            label = "Positiu"
        else:
            label = "Negatiu"

        #cada paraula del tweet construim el diccionari 
        for word in row["tweetText"].split():
            dicc[word][label] += 1

    return dicc


#funcio per classificar si una classe és positiva o negativa
#pos_train i neg_train -> aquestes dues probabilititats són a priori perquè ja les conneixem
def classificar(tweet, diccionari, pos_train, neg_train):
    tamany_dicc = len(diccionari)
    paraules = tweet.split()
    prob_pos = pos_train
    prob_neg = neg_train
    for paraula in paraules:
        if paraula in diccionari:
            suma_valors = sum(diccionari[paraula].values()) + tamany_dicc
            prob_pos *= diccionari[paraula]["Positiu"] / suma_valors
            prob_neg *= diccionari[paraula]["Negatiu"] / suma_valors
    if prob_pos > prob_neg:
        return 1
    else:
        return 0
    
# Clasificar tweets del conjunt de prova
def classificacio_test(test_data, dictionary, prior_pos, prior_neg):
    prediccions = []
    for tweet in test_data["tweetText"]:
        prediccio = classificar(tweet, dictionary, prior_pos, prior_neg)
        prediccions.append(prediccio)
    return prediccions

    

def validacio_creuada(n_splits, data):
    stKfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracy = []
    precision = []
    for train,test in stKfold.split(data, data['sentimentLabel']):
        
        train_data = data.iloc[train]
        test_data = data.iloc[test].copy() 
        print(f"Tamaño del conjunto de entrenamiento: {len(train_data)}")
        print(f"Tamaño del conjunto de prueba: {len(test_data)}")
        diccionari=construir_diccionari(train_data)
        print(f'Tamany diccionari {len(diccionari)}')
        #calculem la probabilitat de que sigui pos i neg
        prob_pos_apr = len(train_data[train_data['sentimentLabel'] == 1]) / len(train_data)
        prob_neg_apr = len(train_data[train_data['sentimentLabel'] == 0]) / len(train_data)
    
        test_data['prediccio'] = classificacio_test(test_data, diccionari, prob_pos_apr, prob_neg_apr)

        #evaluar el model 
        accurac = accuracy_score(test_data['sentimentLabel'], test_data['prediccio'])
        precisio = precision_score(test_data['sentimentLabel'], test_data['prediccio'])
        print(accurac)
        accuracy.append(accurac)
        precision.append(precisio)


    mean_accuracy = sum(accuracy) / n_splits
    mean_precision = sum(precision) / n_splits
    print(f"Accuracy mitja: {mean_accuracy}")
    print(f"Precision mitja: {mean_precision}")



    