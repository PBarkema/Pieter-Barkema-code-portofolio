import tensorflow as tf
import numpy as np
import pandas as pd
from .base import Base
from data.evaluation import get_range_proba
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import more_itertools as mit
from sklearn.preprocessing import MinMaxScaler, Binarizer
from sklearn.utils import class_weight
from tensorflow import keras
import os
from keras import backend as K


def mcor(y_true, y_pred):
    # matthews_correlation
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())


def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


class DNN(Base):
    feature_num = 8
    time_step = 6
    input_size = feature_num * time_step
    output_size = 1
    lr = 0.0006
    epochs = 10

    def train(self, train, kpi_name):
        # label = np.array(train.loc[:, ['label']])
        # predict = np.random.randint(0, 2, len(label))
        # new_predict = get_range_proba(predict, label, 7)
        # print(f1_score(label, new_predict))
        # model = {}
        model_name = 'dnn_model_' + kpi_name + '.h5'
        if os.path.exists(model_name):
            model = keras.models.load_model(model_name, custom_objects={'precision': precision, 'recall': recall, 'f1': f1})

        else:
            model = keras.Sequential()
            model.add(keras.layers.Dense(128,
                                              kernel_regularizer=keras.regularizers.l1(0.000005),
                                              activation=tf.nn.relu, input_shape=(self.feature_num,)))
            model.add(keras.layers.Dropout(0.5))
            model.add(keras.layers.Dense(128,
                                              kernel_regularizer=keras.regularizers.l1(0.000005),
                                              activation=tf.nn.relu))
            model.add(keras.layers.Dropout(0.5))
            model.add(keras.layers.Dense(1, kernel_initializer=keras.initializers.truncated_normal(stddev=0.1),
                                              activation=tf.sigmoid))
            model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(lr=1e-3), metrics=[precision, recall, f1])
            print(model.summary())

            # whole_data_label = []
            # kpi_names = train['KPI ID'].values
            # kpi_names = np.unique(kpi_names)

            # for kpi_name in kpi_names:
            #    kpi_train = (train[train["KPI ID"] == kpi_name])
            kpi_train = train[train.columns.difference(['timestamp', 'KPI ID'])]
            kpi_label = np.array(kpi_train['label'])
            kpi_data = np.array(kpi_train[kpi_train.columns.difference(['label'])])
            kpi_label = kpi_label.reshape(kpi_label.shape[0], 1)
            kpi_data_label = np.concatenate((kpi_data, kpi_label), axis=1)
            kpi_data_label_negative = kpi_data_label[np.where(kpi_data_label[:, -1] == 0)]
            kpi_data_label_positive = kpi_data_label[np.where(kpi_data_label[:, -1] == 1)]
            refine_head_ratio = 0.5
            kpi_data_label_positive = kpi_data_label_positive[np.random.choice(kpi_data_label_positive.shape[0], size=int(len(kpi_data_label_negative) * (1-refine_head_ratio)), replace=True), :]
            splits = np.where(kpi_data_label[1:] != kpi_data_label[:-1])[0] + 1
            splits = splits[0::2]
            tmp = splits
            for i in range(1, 9):
                tmp = np.concatenate((tmp, splits + i))
            splits = tmp[tmp < len(kpi_data_label)]
            kpi_data_label_positive_refine = kpi_data_label[splits]
            kpi_data_label_positive_refine = kpi_data_label_positive_refine[np.where(kpi_data_label_positive_refine[:, -1] == 1)]
            print(len(kpi_data_label_positive_refine))
            kpi_data_label_positive_refine = kpi_data_label_positive_refine[np.random.choice(kpi_data_label_positive_refine.shape[0], size=int(len(kpi_data_label_negative) * refine_head_ratio), replace=True), :]
            kpi_data_label = np.concatenate((kpi_data_label_negative, kpi_data_label_positive, kpi_data_label_positive_refine))
            np.random.shuffle(kpi_data_label)

            whole_data = kpi_data_label[:, :-1]
            whole_label = kpi_data_label[:, -1]
            model.fit(x=whole_data, y=whole_label, batch_size=128, epochs=10, verbose=2)
            model.save(model_name)


    def test(self, test, kpi_name):
        model = {}
        model_name = 'dnn_model_' + kpi_name + '.h5'
        if os.path.exists(model_name):
            model = keras.models.load_model(model_name,
                                                 custom_objects={'precision': precision, 'recall': recall, 'f1': f1})
        answer = test.loc[:, ['KPI ID', 'timestamp']]
        answer['predict'] = pd.Series(np.zeros(len(answer)), index=answer.index)
        answer['predict_unbinary'] = pd.Series(np.zeros(len(answer)), index=answer.index)

        kpi_test = test[test.columns.difference(['timestamp', 'KPI ID', 'label'])]
        kpi_data = np.array(kpi_test)
        # kpi_data = np.concatenate([np.zeros([self.time_step - 1, self.feature_num]), kpi_data])
        # kpi_data = np.array([w for w in mit.windowed(kpi_data, n=self.time_step)])
        # kpi_data = kpi_data.reshape([kpi_data.shape[0], kpi_data.shape[1] * kpi_data.shape[2]])
        predicted = model.predict(kpi_data)
        predicted = np.reshape(predicted, (predicted.size, 1))
        # predicted = MinMaxScaler(feature_range=(0, 1)).fit_transform(predicted)
        answer.loc[answer['KPI ID'] == kpi_name, 'predict_unbinary'] = np.reshape(predicted, (predicted.size,))
        predicted = Binarizer(threshold=0.5).transform(predicted)
        answer.loc[answer['KPI ID'] == kpi_name, 'predict'] = np.reshape(predicted, (predicted.size,))
        return answer

    def train_and_test(self, train, test, kpi_name):
        self.train(train, kpi_name)
        return self.test(test, kpi_name)
