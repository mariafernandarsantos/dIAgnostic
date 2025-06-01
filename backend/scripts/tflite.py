import tensorflow as tf

# Corrigido: carregando o modelo Keras (.h5)
model = tf.keras.models.load_model('./ml_models/pneumonia_detection_model.h5')

# Convertendo para TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Salvando o modelo convertido
with open('pneumonia_model.tflite', 'wb') as f:
    f.write(tflite_model)
