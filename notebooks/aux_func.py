import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
from sklearn import metrics
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix, roc_curve
import pickle
from sklearn.metrics import ConfusionMatrixDisplay
import re
import matplotlib


### Función mortalidad (% / 100 Accidentes)
def mortalidad(columna, tipos, titulo, data):
    analysis = data.groupby([columna, "fatality"]).size()
    serie = list()
    for i in range(0, np.int8(len(analysis) / 2)):
        serie.append(analysis[:, 1].iloc[[i]].item() / (
                    analysis[:, 1].iloc[[i]].item() + (analysis[:, 0].iloc[[i]].item())) * 100)
    analysis = pd.Series(serie, index=range(0, np.int8(len(analysis) / 2)))
    plt.figure(figsize=(15, 5))
    plot = analysis.plot(kind="bar", title=titulo, color="#3A5683")
    plot.set_xticklabels(tipos, rotation=45)


### Función frecuencia absoluta
def frecuencia(columna, tipos, titulo, data):
    analysis = data.groupby(columna)["fatality"].count()
    plt.figure(figsize=(15, 5))
    plot = analysis.plot(kind="bar", title=titulo, color="#639A88")
    plot.set_xticklabels(tipos, rotation=45)


### Función mortalidad total
def mortalidadtotal(columna, tipos, titulo, data):
    analysis = data.groupby(columna)["fatality"].sum() / data["fatality"].sum() * 100
    plt.figure(figsize=(15, 5))
    plot = analysis.plot(kind="bar", title=titulo, color="#76B041")
    plot.set_xticklabels(tipos, rotation=45)


### Función Boxplot
def boxplot_fatality(var, data):
    f, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(x=var, y='fatality', data=data, orient='h')

### Función V de Cramer corregido
def cramers_corrected_stat(x,y):
    result=-1
    if len(x.value_counts())==1 :
        print("First variable is constant")
    elif len(y.value_counts())==1:
        print("Second variable is constant")
    else:
        conf_matrix=pd.crosstab(x, y)

        if conf_matrix.shape[0]==2:
            correct=False
        else:
            correct=True

        chi2 = ss.chi2_contingency(conf_matrix, correction=correct)[0]

        n = sum(conf_matrix.sum())
        phi2 = chi2/n
        r,k = conf_matrix.shape
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        result=np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))
    return round(result,6)


### Función Métricas del Modelo
def evaluate_model(ytest, ypred, ypred_proba = None):
    if ypred_proba is not None:
        print('ROC-AUC score of the model: {}'.format(roc_auc_score(ytest, ypred_proba[:, 1])))
    print('Accuracy of the model: {}\n'.format(accuracy_score(ytest, ypred)))
    print('Classification report: \n{}\n'.format(classification_report(ytest, ypred)))
    print('Confusion matrix: \n{}\n'.format(confusion_matrix(ytest, ypred)))

### Función para cargar el Modelo
def cargar_modelo(ruta):
    return pickle.load(open(ruta, 'rb'))


### Función agregadora de Métricas, Matriz de confusión, curva ROC y threshold óptimo
def model_analysis(modelo, xtest, ytest):
    matplotlib.rcParams['figure.figsize'] = (9, 9)
    ypred = modelo.predict(xtest)
    ypred_proba = modelo.predict_proba(xtest)
    # keep probabilities for the positive outcome only
    yhat = ypred_proba[:, 1]
    # calculate roc curves
    fpr, tpr, thresholds = roc_curve(ytest, yhat)
    # plot the roc curve for the model
    plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
    plt.plot(fpr, tpr, marker='.', label=re.findall('^[A-z]+', str(modelo)))
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    # show the plot
    plt.show()

    gmeans = np.sqrt(tpr * (1 - fpr))
    # locate the index of the largest g-mean
    ix = np.argmax(gmeans)

    # print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))

    # plot the roc curve for the model
    plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
    plt.plot(fpr, tpr, marker='.', label=re.findall('^[A-z]+', str(modelo)))
    plt.scatter(fpr[ix], tpr[ix], s=100, marker='o', color='black', label='Best')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    # show the plot
    plt.show()

    ypred_new_threshold = (ypred_proba[:, 1] > thresholds[ix]).astype(int)

    # Plot non-normalized confusion matrix
    titles_options = [("Confusion matrix, without normalization", None),
                      ("Normalized confusion matrix", 'true')]

    for title, normalize in titles_options:
        fig, ax = plt.subplots(figsize=(10, 10))
        disp = ConfusionMatrixDisplay.from_predictions(ytest, ypred_new_threshold,
                                                       cmap=plt.cm.Greens,
                                                       normalize=normalize,
                                                       ax=ax)
        ax.set_title(title)

    evaluate_model(ytest, ypred_new_threshold, ypred_proba)

