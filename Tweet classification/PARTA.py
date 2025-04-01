#PART A
from sklearn.model_selection import train_test_split
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score
import matplotlib.pyplot as plt
from PARTB import construir_diccionari_personalitzat, classificacio_test



def classificar_laplace(tweet, diccionari, pos_train, neg_train, total_pos, total_neg, dicc_size):
    
    paraules = tweet.split()
    prob_pos = pos_train
    prob_neg = neg_train

    for paraula in paraules:
        if paraula in diccionari:
            freq_pos = diccionari[paraula]["Positiu"]
            freq_neg = diccionari[paraula]["Negatiu"]
        else:
            # Si no está, la paraula al dicc asumim que frecuencia inicial és 0
            freq_pos = 0
            freq_neg = 0

        # Apliquem Laplace Smoothing
        prob_pos *= (freq_pos + 1) / (total_pos + dicc_size)
        prob_neg *= (freq_neg + 1) / (total_neg + dicc_size)

    # Clasificar en función de las probabilidades
    return 1 if prob_pos > prob_neg else 0


# Clasificar tweets del conjunt de prrova
def classificacio_test_laplace(test_data, diccionari, prior_pos, prior_neg, total_pos, total_neg, dicc_size):
    prediccions = []
    for tweet in test_data["tweetText"]:
        prediccio = classificar_laplace(tweet, diccionari, prior_pos, prior_neg, total_pos, total_neg, dicc_size)
        prediccions.append(prediccio)
    return prediccions

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

     # Valors necesaris per aplicar Laplace Smoothing 
    total_pos = sum(diccionari[word]["Positiu"] for word in diccionari)
    total_neg = sum(diccionari[word]["Negatiu"] for word in diccionari)
    vocab_size = len(diccionari)

    prediccions = classificacio_test_laplace(test_data, diccionari, prob_pos_apr, prob_neg_apr, total_pos, total_neg, vocab_size)
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