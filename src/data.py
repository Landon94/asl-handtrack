import kagglehub
import os
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE

def download_data() -> (str,str):
    path = kagglehub.dataset_download("lexset/synthetic-asl-alphabet")

    train_data = os.path.join(path, 'Train_Alphabet')
    test_data = os.path.join(path, 'Test_Alphabet')

    return train_data,test_data


def create_datasets(directory: str, image_size: (int,int), batch_size: int=32, seed: int=42, val_split: int=0.15):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels='inferred',
        label_mode='int',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        seed=seed,
        validation_split=val_split,
        subset='training',
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels='inferred',
        label_mode='int',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        seed=seed,
        validation_split=val_split,
        subset='validation',
    )

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds,val_ds


def create_test_dataset(directory: str, image_size: tuple[int, int], batch_size: int = 32, class_names: list[str] | None = None):
    test_ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="int",
        class_names=class_names,   
        color_mode="rgb",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False,
    )
    return test_ds.cache().prefetch(tf.data.AUTOTUNE)
