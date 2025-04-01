#PARTB
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.model_selection import KFold
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score
from PartC import *
import matplotlib.pyplot as plt

#se li pasa el percentatge del train
def variar_train(data, percentatge):
    train_data, test_data = train_test_split(data, train_size=percentatge,stratify=data["sentimentLabel"], random_state=42)
    resultats = []
    #construim el diccionari
    
    diccionari = construir_diccionari(train_data)

    #calculem probabilitats a priori 
    prob_pos_apr = len(train_data[train_data['sentimentLabel'] == 1]) / len(train_data)
    prob_neg_apr = len(train_data[train_data['sentimentLabel'] == 0]) / len(train_data)

    prediccions = classificacio_test(test_data, diccionari, prob_pos_apr, prob_neg_apr)
    #evaluar el model 
    accuracy = accuracy_score(test_data['sentimentLabel'], prediccions)
    precisio = precision_score(test_data['sentimentLabel'], prediccions, zero_division=1)

    print(f'Amb percentatge {percentatge} tenim una accuracy de {accuracy}')

    resultats.append(accuracy)
    return resultats

def intervals_train(data, intervals):
    resultats = []
    for interval in intervals:
        resultat= variar_train(data, interval)
        resultats.append(resultat)
    

    # Crear la gráfica
    plt.figure(figsize=(10, 6))
    plt.plot(intervals, resultats, marker='o', linestyle='-', color='b')
    plt.title('Accuracy amb diferents percentatges del train')
  
    plt.grid(True)
    plt.show()
    


#anem a variar el nombre de paraules del diccionari i el tren igual 
#ESTRATEGIA 1

def construir_diccionari_personalitzat(data, max_paraules):
    #anem a construir un diccionari buit amb els Positiu i Negatius i emmagatzemarà la frequencia de cada paraula 
    
    dicc = defaultdict(lambda: {"Positiu": 0, "Negatiu": 0})
    for _, row in data.iterrows():
        if len(dicc) >= max_paraules:
            break  # Si el diccionario ya tiene el tamaño máximo, salimos del bucle principal
        label = "Positiu" if row['sentimentLabel'] == 1 else "Negatiu"
        for word in row["tweetText"].split():
            if len(dicc) < max_paraules or word in dicc:
                dicc[word][label] += 1
            if len(dicc) >= max_paraules:  # Controlar el tamaño dentro del bucle de palabras
                break

   
    return dicc


#anem a variar el nombre de paraules del diccionari i el tren igual 
#ESTRATEGIA 2
#
#def construir_diccionari_personalitzat(data, max_paraules):
#    #anem a construir un diccionari buit amb els Positiu i Negatius i emmagatzemarà la frequencia de cada paraula 
#    
#    dicc = defaultdict(lambda: {"Positiu":0, "Negatiu":0})
#    for _,row in data.iterrows():
#        if row['sentimentLabel'] == 1:
#            label = "Positiu"
#        else:
#            label = "Negatiu"
#
#        #cada paraula del tweet construim el diccionari 
#        for word in row["tweetText"].split():
#            dicc[word][label] += 1
#
#        # s'especifica el limti de paraules 
#    if max_paraules:
#        # Ordenar les paraules per més frequencia 
#        sorted_dicc = sorted(dicc.items(), key=lambda x: sum(x[1].values()), reverse=True)
#        dicc = dict(sorted_dicc[:max_paraules])
#
#    return dicc



def mida_diccionari(data, mida_dicc):
     
    train_data, test_data = train_test_split(data, train_size=0.8, stratify=data["sentimentLabel"], random_state=42)
    
    #construir diccionari personalitzat
    diccionari = construir_diccionari_personalitzat(train_data,mida_dicc)

    # Calcular las probabilidades a priori
    prob_pos = len(train_data[train_data["sentimentLabel"] == 1]) / len(train_data)
    prob_neg = len(train_data[train_data["sentimentLabel"] == 0]) / len(train_data)

    resultados = []

    prediccions = classificacio_test(test_data, diccionari, prob_pos, prob_neg)

    #metriques del model
    accuracy = accuracy_score(test_data["sentimentLabel"], prediccions)
    precision = precision_score(test_data["sentimentLabel"], prediccions)
    print(f'Per la mida del diccionari de {mida_dicc} paraules, té una accuracy de {accuracy} i precisió de {precision}')
       


    return mida_dicc, accuracy

    
    
def variar_mida(data, mides):
    resultats = []
    for mida in mides:
        resultat = mida_diccionari(data,mida)
        resultats.append(resultat)
    
    # Extraer tamaños de diccionario y accuracy
    mides_dicc = [resultat[0] for resultat in resultats]
    accuracies = [resultat[1] for resultat in resultats]
    
    # Crear el gráfico
    plt.figure(figsize=(10, 6))
    plt.plot(mides_dicc, accuracies, marker='o', label='Accuracy', color='blue', linestyle='-')

    # Configuración del gráfico
    plt.title('Efecto de la mida del diccionari sobre Accuracy i Precisión', fontsize=14)
    plt.xlabel('Mida del diccionari', fontsize=12)
    plt.ylabel('Valor', fontsize=12)
    plt.xscale('log')  # Escala logarítmica para el eje X
    plt.grid(alpha=0.5)
    plt.legend(fontsize=10)
    plt.show()




#mateixa mida, variar train 
#anem a probar per diferents mides de diccionari variant el train
def variar_train_mida(data, percentatge, mida_dicc):
    train_data, test_data = train_test_split(data, train_size=percentatge,stratify=data["sentimentLabel"], random_state=42)
    resultats = []

    #construim el diccionari per de manera personalitzada
    
    diccionari = construir_diccionari_personalitzat(train_data, mida_dicc)

    #calculem probabilitats a priori 
    prob_pos_apr = len(train_data[train_data['sentimentLabel'] == 1]) / len(train_data)
    prob_neg_apr = len(train_data[train_data['sentimentLabel'] == 0]) / len(train_data)

    prediccions = classificacio_test(test_data, diccionari, prob_pos_apr, prob_neg_apr)
    #evaluar el model 
    accuracy = accuracy_score(test_data['sentimentLabel'], prediccions)
    precisio = precision_score(test_data['sentimentLabel'], prediccions, zero_division=1)

    print(f'Amb percentatge {percentatge} tenim una accuracy de {accuracy} i precisió de {precisio}')

    resultats.append(accuracy)
    return resultats

def intervals_train_mida(data, intervals, mida_dicc):
    resultats = []
    print(f'Per mida del dicc de {mida_dicc}')
    for interval in intervals:
        resultat= variar_train_mida(data, interval, mida_dicc)
        resultats.append(resultat)
    

    # Crear la gráfica
    plt.figure(figsize=(7, 4))
    plt.plot(intervals, resultats, marker='o', linestyle='-', color='b')
    plt.title(f'Accuracy amb diferents percentatges del train i mida del dicc {mida_dicc}')
  
    plt.grid(True)
    # Guardar la gráfica en un archivo
    
    file_name = f"grafic_partB_midadicc_{mida_dicc}.png"
  
    plt.savefig(file_name, format='png', dpi=300, bbox_inches='tight')
    plt.close()


def dicc_variar_train(mides_dicc, mides_train, data):
    for mida_dicc in mides_dicc:
        intervals_train_mida(data, mides_train, mida_dicc)