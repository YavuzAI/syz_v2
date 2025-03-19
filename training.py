import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model import get_lr_scheduler, LearningRateLogger

def train_model(model, train_images, train_metadata, train_labels, val_images, val_metadata, val_labels, batch_size=16, epochs=10):
    """Train model with data augmentation."""
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1, 
        height_shift_range=0.1, 
        horizontal_flip=True)
    
    datagen.fit(train_images)

    lr_scheduler = get_lr_scheduler()
    lr_logger = LearningRateLogger()  # New callback    

    history = model.fit(   
    [train_images, train_metadata], train_labels,
    validation_data=([val_images, val_metadata], val_labels),
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[lr_scheduler, lr_logger]  # Include both LR scheduler and logger
    )

    return history