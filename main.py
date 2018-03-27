import numpy as np
import matplotlib.pyplot as plt
import json
import tensorflow as tf
from ModelClass import *
from Data import *

# load config
config = json.load(open("./config.json"))

# Load training data
training_data = Data(training_data_x=config["data"]["training_data_x"],
                     training_labels_y = config["data"]["training_labels_y"],
                        alphabet=config["data"]["alphabet"],
                        input_size=config["data"]["input_size"],
                        num_of_classes=config["data"]["num_of_classes"])
training_data.load_data()
training_inputs, training_labels = training_data.get_all_data()

# Load validation data
heldout_data = Data(training_data_x=config["data"]["validation_data_x"],
                     training_labels_y = config["data"]["validation_labels_y"],
                        alphabet=config["data"]["alphabet"],
                        input_size=config["data"]["input_size"],
                        num_of_classes=config["data"]["num_of_classes"])
heldout_data.load_data()
heldout_inputs, heldout_labels = heldout_data.get_all_data()

model = ModelClass(input_size=config["data"]["input_size"],
                   alphabet_size=config["data"]["alphabet_size"],
                   embedding_size=config[config["model"]]["embedding_size"],
                   conv_layers=config[config["model"]]["conv_layers"],
                   fully_connected_layers=config[config["model"]]["fully_connected_layers"],
                   num_of_classes=config["data"]["num_of_classes"],
                   threshold=config[config["model"]]["threshold"],
                   dropout_p=config[config["model"]]["dropout_p"],
                   optimizer=config[config["model"]]["optimizer"],
                   loss=config[config["model"]]["loss"])

model.train(training_inputs=training_inputs,
            training_labels=training_labels,
            validation_split=config["training"]["validation_split"],
            epochs=config["training"]["epochs"],
            batch_size=config["training"]["batch_size"],
            checkpoint_every=config["training"]["checkpoint_every"])

model.model.save(r'trained_model')

y=model.model.predict(x=heldout_inputs)

acc, acc_op = tf.metrics.accuracy(labels=tf.argmax(heldout_labels, 1), predictions=tf.argmax(y,1))
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.local_variables_initializer().run()
print(sess.run([acc, acc_op]))
print(sess.run([acc]))

plt.plot(model.model.history.history['loss'])
plt.plot(model.model.history.history['val_loss'])
plt.savefig('plot.png')