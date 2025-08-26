import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from helpers.helper_functions import get_clips_np_v, convert_frames_to_angles

num_frames = 64
batch_size = 1000

def augment_clip(clip, label, 
                 noise_std=0.02, noise_p=0.5,
                 scale_range=(0.9,1.1), scale_p=0.5):
    # maybe add noise
    if tf.random.uniform(()) < noise_p:
        noise = tf.random.normal(tf.shape(clip), stddev=noise_std)
        clip = clip + noise
    # maybe scale
    if tf.random.uniform(()) < scale_p:
        scale = tf.random.uniform([], *scale_range)
        clip = clip * scale
    return clip, label

# Pass in videos in expected 2D array
# Each index corresponds to all videos of a label
def prep_data(training_videos, training_labels, validation_videos, validation_labels):
    keys = ['x_values', 'y_values', 'z_values', 'visibility_values']
    num_keys = len(keys)
    num_features = 33
    num_angles = 10

    training_videos = [
        np.concatenate((
            video[:, :, :num_keys].reshape(-1, num_features * num_keys), 
            convert_frames_to_angles(video[:, :, :2]),
            np.pad(np.diff(video[:, :, :3].reshape(-1, num_features * 3), axis=0), ((0,1), (0,0)), mode="edge"),
            np.pad(np.diff(convert_frames_to_angles(video[:, :, :2]), axis=0), ((0,1), (0,0)), mode="edge")),
            axis=-1
        )
        for video in training_videos
    ]
    validation_videos = [
        np.concatenate((
            video[:, :, :num_keys].reshape(-1, num_features * num_keys), 
            convert_frames_to_angles(video[:, :, :2]),
            np.pad(np.diff(video[:, :, :3].reshape(-1, num_features * 3), axis=0), ((0,1), (0,0)), mode="edge"),
            np.pad(np.diff(convert_frames_to_angles(video[:, :, :2]), axis=0), ((0,1), (0,0)), mode="edge")),
            axis=-1
        ) 
        for video in validation_videos
    ]
    
    training_clips, training_labels = get_clips_np_v(num_frames, training_videos, training_labels)
    validation_clips, validation_labels = get_clips_np_v(num_frames, validation_videos, validation_labels)

    num_values = num_features * len(keys) + num_features * 3 + num_angles * 2

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
        .map(augment_clip, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    validation_dataset = (
        base_val_ds
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_dataset, validation_dataset