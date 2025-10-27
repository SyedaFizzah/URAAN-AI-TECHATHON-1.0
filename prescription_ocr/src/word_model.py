"""CRNN-like model: CNN -> BiLSTM -> CTC
Fixed for CTC loss compatibility.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_word_model(alphabets, max_str_len, img_width=128, img_height=32):
    input_img = layers.Input(shape=(img_width, img_height, 1), name='image')

    # CNN layers
    x = layers.Conv2D(32, (3,3), padding='same', activation='relu')(input_img)
    x = layers.MaxPooling2D((2,2))(x)  # 64x16

    x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2,2))(x)  # 32x8

    x = layers.Conv2D(128, (3,3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((1,2))(x)  # 32x4

    x = layers.Conv2D(256, (3,3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((1,2))(x)  # 32x2

    # Reduce feature maps to 1 in height dimension
    x = layers.Conv2D(512, (2,2), activation='relu')(x)  # 31x1

    # Reshape for RNN: (timesteps, features)
    new_shape = (31, 512)
    x = layers.Reshape(new_shape)(x)

    # Bidirectional LSTM layers
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.2))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.2))(x)

    # Output layer
    y_pred = layers.Dense(len(alphabets) + 1, activation='softmax', name='output')(x)
    
    word_model = keras.models.Model(inputs=input_img, outputs=y_pred)

    # CTC model for training
    labels = layers.Input(name='gtruth_labels', shape=[max_str_len], dtype='int32')
    input_length = layers.Input(name='input_length', shape=[1], dtype='int64')
    label_length = layers.Input(name='label_length', shape=[1], dtype='int64')

    def ctc_lambda_func(args):
        y_pred, labels, input_length, label_length = args
        # For CTC: input_length should be <= number of timesteps in y_pred
        # We have 31 timesteps, so set input_length to 31
        return tf.keras.backend.ctc_batch_cost(
            labels, 
            y_pred, 
            input_length, 
            label_length
        )

    ctc_loss = layers.Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([
        y_pred, labels, input_length, label_length])

    word_model_CTC = keras.models.Model(
        inputs=[input_img, labels, input_length, label_length], 
        outputs=ctc_loss
    )
    
    return word_model, word_model_CTC


if __name__ == '__main__':
    with open('data/charList.txt','r',encoding='utf-8') as f:
        alphabets = f.read()
    model, _ = build_word_model(alphabets, max_str_len=32)
    model.summary()