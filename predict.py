# -*- coding: utf-8 -*-
from preprocessing import Processor
from utils import read_json
import pandas as pd
from keras.models import load_model
import os
import gc
import numpy as np
from keras import backend as K
import tensorflow as tf


class Predict(object):
    def __init__(self, config):
        self.config = config
        self.labels = config['labels']
        self.processor = Processor(config)
        self.processor.init(self.config['w2v_path'])
        self.grade2idx = read_json('./data/grade_idx.map')
        self.idx2grade = {idx:grade for grade, idx in self.grade2idx.items()}

    def predict(self, comment):
        data = pd.DataFrame({self.config['feature_column']: [comment]})
        pred_map = self._predict(data)
        return pred_map.items()

    def _predict(self, data:pd.DataFrame):
        features = self.processor.get_features(data)
        pred_map = {}
        for label in self.labels:
            # K.clear_session()
            model = load_model(os.path.join('./save_model', label + '.krs.save_model'))
            prob_pred = model.predict(features)
            grade = self._prob2grade(prob_pred)
            print('{} grade : {}'.format(label,grade))
            pred_map[label] = grade
            K.clear_session()
            tf.reset_default_graph()
            # del model
            # gc.collect()
        gc.collect()
        return pred_map

    def _prob2grade(self, prob_pred):
        pred_ids = np.argmax(prob_pred, axis=1).flatten()
        grade = self.idx2grade[pred_ids[0]]
        return grade
