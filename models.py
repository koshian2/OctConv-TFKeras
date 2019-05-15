from tensorflow.keras import layers
from oct_conv2d import OctConv2D
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

def _create_normal_residual_block(inputs, ch, N):
    # Conv with skip connections
    x = inputs
    for i in range(N):
        # adjust channels
        if i == 0:
            skip = layers.Conv2D(ch, 1)(x)
            skip = layers.BatchNormalization()(skip)
            skip = layers.Activation("relu")(skip)
        else:
            skip = x
        x = layers.Conv2D(ch, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(ch, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Add()([x, skip])
    return x

def _create_octconv_residual_block(inputs, ch, N, alpha):
    high, low = inputs
    # OctConv with skip connections
    for i in range(N):
        # adjust channels
        if i == 0:
            skip_high = layers.Conv2D(int(ch*(1-alpha)), 1)(high)
            skip_high = layers.BatchNormalization()(skip_high)
            skip_high = layers.Activation("relu")(skip_high)

            skip_low = layers.Conv2D(int(ch*alpha), 1)(low)
            skip_low = layers.BatchNormalization()(skip_low)
            skip_low = layers.Activation("relu")(skip_low)
        else:
            skip_high, skip_low = high, low

        high, low = OctConv2D(filters=ch, alpha=alpha)([high, low])
        high = layers.BatchNormalization()(high)
        high = layers.Activation("relu")(high)
        low = layers.BatchNormalization()(low)
        low = layers.Activation("relu")(low)

        high, low = OctConv2D(filters=ch, alpha=alpha)([high, low])
        high = layers.BatchNormalization()(high)
        high = layers.Activation("relu")(high)
        low = layers.BatchNormalization()(low)
        low = layers.Activation("relu")(low)

        high = layers.Add()([high, skip_high])
        low = layers.Add()([low, skip_low])
    return [high, low]


def _create_octconv_last_residual_block(inputs, ch, alpha):
    # Last layer for octconv resnets
    high, low = inputs

    # OctConv
    high, low = OctConv2D(filters=ch, alpha=alpha)([high, low])
    high = layers.BatchNormalization()(high)
    high = layers.Activation("relu")(high)
    low = layers.BatchNormalization()(low)
    low = layers.Activation("relu")(low)

    # Last conv layers = alpha_out = 0 : vanila Conv2D
    # high -> high
    high_to_high = layers.Conv2D(ch, 3, padding="same")(high)
    # low -> high
    low_to_high = layers.Conv2D(ch, 3, padding="same")(low)
    low_to_high = layers.Lambda(lambda x: 
                        K.repeat_elements(K.repeat_elements(x, 2, axis=1), 2, axis=2))(low_to_high)
    x = layers.Add()([high_to_high, low_to_high])
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x

def create_normal_wide_resnet(N=4, k=10):
    """
    Create vanilla conv Wide ResNet (N=4, k=10)
    """
    # input
    input = layers.Input((32,32,3))
    # 16 channels block
    x = layers.Conv2D(16, 3, padding="same")(input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    # 1st block
    x = _create_normal_residual_block(x, 16*k, N)
    # The original wide resnet is stride=2 conv for downsampling,
    # but replace them to average pooling because centers are shifted when octconv
    # 2nd block
    x = layers.AveragePooling2D(2)(x)
    x = _create_normal_residual_block(x, 32*k, N)
    # 3rd block
    x = layers.AveragePooling2D(2)(x)
    x = _create_normal_residual_block(x, 64*k, N)
    # FC
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(10, activation="softmax")(x)

    model = Model(input, x)
    return model

def create_octconv_wide_resnet(alpha, N=4, k=10):
    """
    Create OctConv Wide ResNet(N=4, k=10)
    """
    # Input
    input = layers.Input((32,32,3))
    # downsampling for lower
    low = layers.AveragePooling2D(2)(input)

    # 16 channels block
    high, low = OctConv2D(filters=16, alpha=alpha)([input, low])
    high = layers.BatchNormalization()(high)
    high = layers.Activation("relu")(high)
    low = layers.BatchNormalization()(low)
    low = layers.Activation("relu")(low)

    # 1st block
    high, low = _create_octconv_residual_block([high, low], 16*k, N, alpha)
    # 2nd block
    high = layers.AveragePooling2D(2)(high)
    low = layers.AveragePooling2D(2)(low)
    high, low = _create_octconv_residual_block([high, low], 32*k, N, alpha)
    # 3rd block
    high = layers.AveragePooling2D(2)(high)
    low = layers.AveragePooling2D(2)(low)
    high, low = _create_octconv_residual_block([high, low], 64*k, N-1, alpha)
    # 3rd block Last
    x = _create_octconv_last_residual_block([high, low], 64*k, alpha)
    # FC
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(10, activation="softmax")(x)

    model = Model(input, x)
    return model
