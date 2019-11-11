# -*- coding:utf-8 -*-
import pandas as pd
from preprocessing import Processor, grade_map
from model import TextCNN
import json
import codecs
from keras.callbacks import Callback
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from keras.callbacks import EarlyStopping,ModelCheckpoint
import logging
from config import config

LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, filename='./log/summary.log', format=LOG_FORMAT)


def train(conf):
    data_train = pd.read_csv(conf['train_file'])
    data_val = pd.read_csv(conf['val_file'])
    processor = Processor(conf)
    processor.init(conf['w2v_path'])
    train_x = processor.get_features(data_train)
    val_x = processor.get_features(data_val)
    labels = conf['labels']
    grade2idx, idx2grade = grade_map(data_train[labels[0]].tolist())
    with codecs.open('./data/grade_idx.map', 'w') as f:
        json.dump(grade2idx, f)

    for label in labels:
        train_y = processor.get_labels(data_train, label, grade2idx)
        val_y = processor.get_labels(data_val, label, grade2idx)
        model = TextCNN(conf['num_class'], conf['seq_len'],
                        processor.to_embedding(), conf['num_filters'],
                        conf['filter_sizes']).model
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        mtr = Metrics()
        model_checkpoint = ModelCheckpoint('./save_model/{}.krs.save_model'.format(label),
                                           monitor='val_loss', verbose=1,
                                           save_best_only=True, mode='min')
        early_stopping = EarlyStopping(monitor='val_loss', patience=3,
                                       verbose=1, mode='min')

        model_summary = model.summary()
        logging.info(str(model_summary))
        logging.info('start train for label : {}'.format(label))
        history = model.fit(x=train_x, y=train_y, batch_size=256,
                                     epochs=20, verbose=1,
                                     callbacks=[mtr, model_checkpoint,
                                                early_stopping],
                                     validation_data=(val_x, val_y),
                                     shuffle=True)
        logging.info('save_model train history for label : {}'.format(label))
        logging.info(str(history))
    logging.info('all labels model train finished')



class Metrics(Callback):
    def __init__(self, name='metrics'):
        self.name = name

    def on_train_begin(self, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        validation_input = self.validation_data[0]
        validation_label = self.validation_data[1]
        validation_predict = self.model.predict([validation_input])
        validation_pred_label = prob2label(validation_predict)
        validation_num_label = prob2label(validation_label)
        f1_support = precision_recall_fscore_support(validation_num_label,
                                                     validation_pred_label)
        print('validation result : ')
        print(f1_support)


def prob2label(arr):
    arr = np.array(arr)
    labels = np.argmax(arr, axis=1)
    return labels


if __name__ == '__main__':
    train(config)
