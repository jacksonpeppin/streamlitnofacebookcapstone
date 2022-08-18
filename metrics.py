import pandas as pd
import matplotlib.pyplot as plt

model_metrics = pd.read_pickle('./model_metrics/model_metrics.pkl')


def get_model_metrics():
    return model_metrics


def get_model_accuracy():
    acc_fig = plt.figure()
    plt.plot(model_metrics['accuracy'])
    plt.plot(model_metrics['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    return acc_fig


def get_model_loss():
    loss_fig = plt.figure()
    plt.plot(model_metrics['loss'])
    plt.plot(model_metrics['val_loss'])
    plt.title('model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    return loss_fig

