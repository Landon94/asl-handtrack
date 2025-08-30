import tensorflow as tf
import os
from model import create_model
from data import create_datasets,create_test_dataset,download_data

BATCH_SIZE = 32
VALIDATION_SPLIT = 0.15
SEED = 42
IMAGE_SIZE = (180,180)

MODEL_PATH = '../asl_cnn.h5'

def main():

    train_data_path, test_data_path = download_data()

    train_ds, val_ds, class_names = create_datasets(train_data_path,IMAGE_SIZE,BATCH_SIZE,SEED,VALIDATION_SPLIT)
    test_ds = create_test_dataset(test_data_path,IMAGE_SIZE,BATCH_SIZE,class_names)

    if os.path.exists(MODEL_PATH):
        print(f"Loading existing model from {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH)
    else:
        model = create_model(IMAGE_SIZE + (3,), len(class_names))


    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"]
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='best_model.keras', save_best_only=True, monitor='val_loss')

    model.fit(
        train_ds,
        validation_data=val_ds,
        callbacks=[early_stopping,model_checkpoint],
        epochs=10
        )
    
    model.save(MODEL_PATH)


if __name__ == "__main__":
    main()
