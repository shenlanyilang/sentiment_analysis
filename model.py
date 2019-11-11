# -*- coding:utf-8 -*-

from keras.layers import Embedding, Dense, Flatten, Input, SpatialDropout1D, Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPool1D
from keras.layers.merge import concatenate
from keras.models import Model


class TextCNN:

    def __init__(self, num_class,
                 sequence_length,
                 embedding_matrix,
                 num_filters,
                 filter_sizes):
        # input layer
        sequence_input = Input(shape=(sequence_length,), dtype='int32')
        # embedding layer
        embedding_layer = Embedding(embedding_matrix.shape[0],
                                    embedding_matrix.shape[1],
                                    weights = [embedding_matrix],
                                    input_length=sequence_length,
                                   trainable=True)
        embedded_sequences = embedding_layer(sequence_input)
        embed_dropout = SpatialDropout1D(rate=0.2)(embedded_sequences)
#         embed_dropout = embedded_sequences
        # create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for filter_size in filter_sizes:
            x = Conv1D(num_filters, filter_size, activation='relu')(embed_dropout)
            x = MaxPool1D(int(x.shape[1]))(x)
            pooled_outputs.append(x)
        merged = concatenate(pooled_outputs)
        x = Flatten()(merged)
        x = Dropout(rate=0.2)(x)
        outputs = Dense(num_class, activation='softmax')(x)
        self.model = Model(sequence_input, outputs)