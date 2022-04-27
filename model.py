from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, BatchNormalization, Activation, Add, Conv2D, Conv2DTranspose, MaxPooling2D
from tensorflow.keras import backend as K

def ConvBNRes(model, filters, block=None):
    model = Conv2D(filters, kernel_size=(3,3), padding='same', kernel_initializer='glorot_normal')(model)
    model = BatchNormalization()(model)
    if block is not None:
        block = Conv2D(filters, kernel_size=(1,1), padding='same')(model)
        model = Add()([model, block])
    model = Activation('relu')(model)
    return model

def ResUnet2D(input_size, init_filters=16, n_classes=2):
    # Encoder 第一層
    inputs = Input(shape=input_size) # 512x512x1
    tmp = inputs
    enc1 = ConvBNRes(inputs, filters=init_filters) # 512x512x16
    enc1 = ConvBNRes(enc1, filters=init_filters, block=tmp) # 512x512x16
    
    # Encoder 第二層
    enc2 = MaxPooling2D(pool_size=(2,2))(enc1) # 256x256x16
    tmp = enc2
    enc2 = ConvBNRes(enc2, filters=init_filters*2) # 256x256x32
    enc2 = ConvBNRes(enc2, filters=init_filters*2, block=tmp) # 256x256x32
    
    # Encoder 第三層
    enc3 = MaxPooling2D(pool_size=(2,2))(enc2) # 128x128x32
    tmp = enc3
    enc3 = ConvBNRes(enc3, filters=init_filters*4) # 128x128x64
    enc3 = ConvBNRes(enc3, filters=init_filters*4, block=tmp) # 128x128x64
    
    # Encoder 第四層
    enc4 = MaxPooling2D(pool_size=(2,2))(enc3) # 64x64x64
    tmp = enc4
    enc4 = ConvBNRes(enc4, filters=init_filters*8) # 64x64x128
    enc4 = ConvBNRes(enc4, filters=init_filters*8, block=tmp) # 64x64x128
    
    # 最底層
    enc5 = MaxPooling2D(pool_size=(2,2))(enc4) # 32x32x128
    tmp = enc5
    enc5 = ConvBNRes(enc5, filters=init_filters*16) # 32x32x256
    enc5 = ConvBNRes(enc5, filters=init_filters*16, block=tmp) # 32x32x256
    
    # Decoder 第一層 (對應Encoder4)
    dec4 = Conv2DTranspose(init_filters*8, kernel_size=(3,3), padding='same', strides=2, kernel_initializer='he_normal')(enc5) # 64x64x128
    tmp = dec4
    dec4 = Concatenate(axis=-1)([dec4, enc4]) # 64x64x256
    dec4 = ConvBNRes(dec4, filters=init_filters*8) # 64x64x128
    dec4 = ConvBNRes(dec4, filters=init_filters*8, block=tmp) # 64x64x128
    
    # Decoder 第二層 (對應Encoder3)
    dec3 = Conv2DTranspose(init_filters*4, kernel_size=(3,3), padding='same', strides=2, kernel_initializer='he_normal')(dec4) # 128x128x64
    tmp = dec3
    dec3 = Concatenate(axis=-1)([dec3, enc3]) # 128x128x128
    dec3 = ConvBNRes(dec3, filters=init_filters*4) # 128x128x64
    dec3 = ConvBNRes(dec3, filters=init_filters*4, block=tmp) # 128x128x64
    
    # Decoder 第三層 (對應Encoder2)
    dec2 = Conv2DTranspose(init_filters*2, kernel_size=(3,3), padding='same', strides=2, kernel_initializer='he_normal')(dec3) # 256x256x32
    tmp = dec2
    dec2 = Concatenate(axis=-1)([dec2, enc2]) # 256x256x64
    dec2 = ConvBNRes(dec2, filters=init_filters*2) # 256x256x32
    dec2 = ConvBNRes(dec2, filters=init_filters*2, block=tmp) # 256x256x32
    
    # Decoder 第四層 (對應Encoder1)
    dec1 = Conv2DTranspose(init_filters, kernel_size=(3,3), padding='same', strides=2, kernel_initializer='he_normal')(dec2) # 512x512x16
    tmp = dec1
    dec1 = Concatenate(axis=-1)([dec1, enc1]) # 512x512x32
    dec1 = ConvBNRes(dec1, filters=init_filters) # 512x512x16
    dec1 = ConvBNRes(dec1, filters=init_filters, block=tmp) # 512x512x16
    
    # 輸入端(Output要分為幾個class)
    out = Conv2D(n_classes, kernel_size=(1,1), padding='same')(dec1) # 512x512x2
    out = Activation('softmax')(out) # 512x512x2
    
    model = Model(inputs, out)
    return model

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    dice = K.mean((2. * intersection + smooth)/(union + smooth))
    return dice

def dice_loss(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    dice = K.mean((2. * intersection + smooth)/(union + smooth))
    return K.categorical_crossentropy(y_true, y_pred) - K.log(dice)

def IoU(y_true, y_pred, smooth=1e-3):
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - K.sum(y_true * y_pred)
    IoU = (intersection + smooth) / (union + smooth)
    return IoU