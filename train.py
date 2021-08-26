import os
from datetime import datetime
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from models.model import get_net
import tensorflow as tf
from training import load_data, VolumeDataGenerator

if __name__ == "__main__":

    base_path = os.path.abspath(__file__ + "/..")

    batch_size = 6
    epochs = 200

    training_path = base_path + "/data/training/training-set"
    validation_path = base_path + "/data/validation/validation-set"

    x_train, y_train = load_data(training_path)
    x_validation, y_validation = load_data(validation_path)

    print("Loaded Data")

    train_datagen = VolumeDataGenerator(
        min_max_normalization=True,
        scale_range=0.1,
        scale_constant_range=0.2
    )

    # Do not augment validation set
    val_datagen = VolumeDataGenerator(
        min_max_normalization=True
    )

    train_generator = train_datagen.flow(x_train, y_train, batch_size, shuffle=True)
    validation_generator = val_datagen.flow(x_validation, y_validation, batch_size, shuffle=True)

    now = datetime.now()
    logdir = base_path + "/data/tf-logs/" + now.strftime("%B-%d-%Y-%I:%M%p") + "/"

    tboard = TensorBoard(log_dir=logdir, histogram_freq=0, write_graph=True, write_images=False)

    current_checkpoint = ModelCheckpoint(filepath=base_path + '/data/model-weights/latest_model.hdf5', verbose=1)
    # period_checkpoint = ModelCheckpoint(base_path + '/data/model-weights/weights{epoch:03d}.hdf5', period=5)
    best_weight_checkpoint = ModelCheckpoint(filepath=base_path + '/data/model-weights/best_weights_checkpoint.hdf5',
                                             verbose=1, save_best_only=True)

    weights_path = base_path + "/data/model-weights/trailmap_model_original.hdf5"

    model = get_net()

    # This will do transfer learning and start the model off with our current best model.
    # Remove the model.load_weight line below if you want to train from scratch
    # We want to train from scratch due to differences in resolution.
    # model.load_weights(weights_path)
    
    model.fit(train_generator,
              epochs=200,
              validation_data=validation_generator,
              callbacks=[tboard, current_checkpoint, best_weight_checkpoint],
              verbose=1
              )

    model_name = 'model_' + now.strftime("%B-%d-%Y-%I:%M%p")
