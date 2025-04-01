import numpy as np
import pandas as pd
from PartC import *
from PARTB import intervals_train, variar_mida, dicc_variar_train
from PARTA import dicc_variar_train
#El main on s'executarà el codi 
def main():
    #llegim dataset 
    df = pd.read_csv("FinalStemmedSentimentAnalysisDataset.csv", delimiter=';')
    print(df.describe())
    print(df.dtypes)
    #mirem si conté valors nulls 
    print(valors_nulls(df))

    #hi han pocs valors nulls, els eliminem
    df = df.dropna(axis=0)

    print(df.shape)
    #anem analitzar l'atribut target, que serà determinar si un tweet sera bó o dolent 
    target = 'sentimentLabel'
    #df["tweetText"] = df["tweetText"].fillna("")
    print(valors_nulls(df))


    print(df[target].value_counts())
    X = df.drop(columns=[target])
    y = df['sentimentLabel']

    #PART C
   # num_splits = [3,5,10]
   # for split in num_splits:
   #     validacio_creuada(split, df)

    #PART B
    #cambiar valors del train
    valors_train = [0.5, 0.6,0.7,0.8,0.9]
    #intervals_train(df, valors_train)



    #cambiar valor del diccionari i train 80%
    #mides_dicc = [50,100,1000,5000,20000,60000,200000,500000, 650000]
    #variar_mida(df,mides_dicc)    
    mides_dicc = [100,1000,10000,20000,150000,300000,650000]

    #dicc_variar_train(mides_dicc,valors_train, df)

    #PART A
    dicc_variar_train(mides_dicc,valors_train, df)






if __name__ == '__main__':
    main()