from tensorflow.keras import layers
from oct_conv2d import OctConv2D
from tensorflow.keras.models import Model

def _create_normal_residual_block(inputs, ch, N):
    # adujust channels
    x = layers.Conv2D(ch, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    # Conv with skip connections
    for i in range(N-1):
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
    # adjust channels
    high, low = OctConv2D(filters=ch, alpha=alpha)(inputs)
    high = layers.BatchNormalization()(high)
    high = layers.Activation("relu")(high)
    low = layers.BatchNormalization()(low)
    low = layers.Activation("relu")(low)
    # OctConv with skip connections
    for i in range(N-1):
        skip_high, skip_low = [high, low]

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

def create_normal_wide_resnet(N=4, k=10):
    """
    Create vanilla conv Wide ResNet (N=4, k=10)
    """
    # input
    input = layers.Input((32,32,3))

    # 1st block
    x = _create_normal_residual_block(input, 16*k, N)
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

    # 1st block
    high, low = _create_octconv_residual_block([input, low], 16*k, N, alpha)
    # 2nd block
    high = layers.AveragePooling2D(2)(high)
    low = layers.AveragePooling2D(2)(low)
    high, low = _create_octconv_residual_block([high, low], 32*k, N, alpha)
    # 3rd block
    high = layers.AveragePooling2D(2)(high)
    low = layers.AveragePooling2D(2)(low)
    high, low = _create_octconv_residual_block([high, low], 64*k, N, alpha)
    # concat
    high = layers.AveragePooling2D(2)(high)
    x = layers.Concatenate()([high, low])
    x = layers.Conv2D(64*k, 1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    # FC
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(10, activation="softmax")(x)

    model = Model(input, x)
    return model
