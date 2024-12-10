from keras import layers, Model

def unet_shlw(input_size=(192, 256, 3), num_classes=1):
    inputs = layers.Input(shape=input_size)

    # Encoder
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    # Bottom layer
    c4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c4)

    # Decoder
    u5 = layers.UpSampling2D((2, 2))(c4)
    u5 = layers.Concatenate()([u5, c3])
    c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u5)
    c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c5)

    u6 = layers.UpSampling2D((2, 2))(c5)
    u6 = layers.Concatenate()([u6, c2])
    c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c6)

    u7 = layers.UpSampling2D((2, 2))(c6)
    u7 = layers.Concatenate()([u7, c1])
    c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c7)

    # Output
    outputs = layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(c7)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model


def unet_deep(input_size=(192, 256, 3), num_classes=1):
    inputs = layers.Input(shape=input_size)

    # Encoder
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    # Additional Encoder Layer
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)
    p5 = layers.MaxPooling2D((2, 2))(c5)

    # Bottom layer
    c6 = layers.Conv2D(2048, (3, 3), activation='relu', padding='same')(p5)
    c6 = layers.Conv2D(2048, (3, 3), activation='relu', padding='same')(c6)

    # Decoder
    u7 = layers.UpSampling2D((2, 2))(c6)
    u7 = layers.Concatenate()([u7, c5])
    c7 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c7)

    u8 = layers.UpSampling2D((2, 2))(c7)
    u8 = layers.Concatenate()([u8, c4])
    c8 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c8)

    u9 = layers.UpSampling2D((2, 2))(c8)
    u9 = layers.Concatenate()([u9, c3])
    c9 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c9)

    u10 = layers.UpSampling2D((2, 2))(c9)
    u10 = layers.Concatenate()([u10, c2])
    c10 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u10)
    c10 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c10)

    u11 = layers.UpSampling2D((2, 2))(c10)
    u11 = layers.Concatenate()([u11, c1])
    c11 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u11)
    c11 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c11)

    # Output
    outputs = layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(c11)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model
