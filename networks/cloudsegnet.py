import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

def CloudSegNet(input_img):
    """
    Builds and returns the CloudSegNet model.

    Parameters:
    input_shape (tuple): Shape of the input image (height, width, channels).

    Returns:
    cloudsegnet (Model): Compiled CloudSegNet model.
    """

    # Encoder
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    print("Shape of encoded:", K.int_shape(encoded))

    # Decoder
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)

    # Use padding='same' in the final layer for adaptable output size
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    # Define the CloudSegNet model
    return Model(input_img, decoded)