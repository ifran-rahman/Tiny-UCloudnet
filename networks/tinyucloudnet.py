import tensorflow as tf
from tensorflow.keras.layers import Input, SeparableConv2D, MaxPooling2D, UpSampling2D, BatchNormalization, DepthwiseConv2D, Conv2D, Activation, GlobalAveragePooling2D, Reshape, multiply, concatenate, Dropout,Dense, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


def TinyUCloudNet(input_img):
    """
    Builds and returns an adaptable CloudSegNet model using SeparableConv2D.

    Parameters:
    input_shape (tuple): Shape of the input image (height, width, channels). Use None for dynamic sizes.

    Returns:
    cloudsegnet (Model): Compiled adaptable CloudSegNet model.
    """

    # Encoder
    x1 = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    p1 = MaxPooling2D((2, 2), padding='same')(x1)  # Downsample

    x2 = SeparableConv2D(8, (3, 3), activation='relu', padding='same')(p1)
    p2 = MaxPooling2D((2, 2), padding='same')(x2)  # Downsample

    x3 = SeparableConv2D(8, (3, 3), activation='relu', padding='same')(p2)
    encoded = MaxPooling2D((2, 2), padding='same')(x3)  # Bottleneck

    # Decoder
    x4 = SeparableConv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    u4 = UpSampling2D((2, 2))(x4)  # Upsample
    c4 = concatenate([u4, x3])  # Skip connection

    x5 = SeparableConv2D(8, (3, 3), activation='relu', padding='same')(c4)
    u5 = UpSampling2D((2, 2))(x5)  # Upsample
    c5 = concatenate([u5, x2])  # Skip connection

    x6 = SeparableConv2D(16, (3, 3), activation='relu', padding='same')(c5)
    u6 = UpSampling2D((2, 2))(x6)  # Upsample
    c6 = concatenate([u6, x1])  # Skip connection

    # Output layer
    output = Conv2D(1, (1, 1), activation='sigmoid')(c6)
    print('output shape', output.shape)

    return Model(input_img, output, name="DW_CloudSegUNet")