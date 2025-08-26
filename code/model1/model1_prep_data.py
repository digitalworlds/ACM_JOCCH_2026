import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from helpers.helper_functions import get_clips_np_v

num_frames = 32
batch_size = 1000

# Pass in videos in expected 2D array
# Each index corresponds to all videos of a label
def prep_data(training_videos, training_labels, validation_videos, validation_labels):
    keys = ['x_values', 'y_values', 'z_values', 'visibility_values']
    num_keys = len(keys)
    num_features = 33

    training_videos = [
        video[:, :, :num_keys].reshape(-1, num_features * num_keys)
        for video in training_videos
    ]
    validation_videos = [
        video[:, :, :num_keys].reshape(-1, num_features * num_keys)
        for video in validation_videos
    ]
    
    training_clips, training_labels = get_clips_np_v(num_frames, training_videos, training_labels)
    validation_clips, validation_labels = get_clips_np_v(num_frames, validation_videos, validation_labels)

    num_values = num_features * num_keys

    # Standardize
    scaler = StandardScaler()
    training_clips = training_clips.reshape(-1, num_values) # Flatten
    training_clips = scaler.fit_transform(training_clips)
    validation_clips = validation_clips.reshape(-1, num_values) # Flatten
    validation_clips = scaler.transform(validation_clips)
    training_clips = training_clips.reshape(-1, num_frames, num_values)
    validation_clips = validation_clips.reshape(-1, num_frames, num_values)

    indices = np.random.permutation(training_clips.shape[0])
    training_clips = training_clips[indices]
    training_labels = training_labels[indices]

    training_labels = training_labels.reshape(-1, 1)
    validation_labels = validation_labels.reshape(-1, 1)

    with tf.device('/CPU:0'):
        base_train_ds = tf.data.Dataset.from_tensor_slices(
            (training_clips.astype(np.float32),
            training_labels.astype(np.int32))
        )
        base_val_ds   = tf.data.Dataset.from_tensor_slices(
            (validation_clips.astype(np.float32),
            validation_labels.astype(np.int32))
        )
        
    train_dataset = (
        base_train_ds
        .shuffle(buffer_size=training_clips.shape[0])
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    validation_dataset = (
        base_val_ds
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_dataset, validation_dataset