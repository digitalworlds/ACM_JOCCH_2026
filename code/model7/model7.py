from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Activation, GlobalAveragePooling1D, AveragePooling1D, GlobalMaxPooling1D, MaxPooling1D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

def getModel(num_classes):
    num_frames = 80
    batch_size = 1000
    dropout_rate = 0.3
    keys = ['x_values', 'y_values', 'z_values', 'visibility_values']
    num_features = 33
    num_angles = 10

    input_shape = (num_frames, num_features * len(keys) + num_features * 3 + num_angles * 2)

    model = Sequential()

    # Block 1
    model.add(Conv1D(75, kernel_size=5, padding='same', activation=None, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate))
    
    # Block 2
    model.add(Conv1D(150, kernel_size=5, dilation_rate=2, padding='same', activation=None))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate))

    # Block 3
    model.add(Conv1D(300, kernel_size=5, dilation_rate=4, padding='same', activation=None))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate))

    # Global pooling and dense layers for classification
    model.add(GlobalAveragePooling1D())
    model.add(Dense(num_classes, activation='softmax'))     

    optimizer = Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Print the model summary to see the architecture
    model.summary()

    return model, num_frames, num_features, batch_size, keys
