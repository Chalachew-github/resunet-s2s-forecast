from tensorflow.keras.layers import Input, Add, Conv2D, BatchNormalization, Activation, MaxPooling2D, Conv2DTranspose, \
    Dropout, Cropping2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

import xarray as xr

xr.set_options(display_style='text')

import warnings

warnings.simplefilter("ignore")


# Define the ResNet block
def resnet_block(input_layer, filters, kernel_size=(3, 3), stride=(1, 1), lamda=0, dropout_rate=0, bn=True):
    x = Conv2D(filters, kernel_size, strides=stride, padding='same', kernel_regularizer=l2(lamda),
               bias_regularizer=l2(lamda))(input_layer)
    x = BatchNormalization()(x) if bn else x
    x = Activation('elu')(x)
    x = Conv2D(filters, kernel_size, padding='same', kernel_regularizer=l2(lamda), bias_regularizer=l2(lamda))(x)
    x = BatchNormalization()(x) if bn else x
    x = Activation('elu')(x)

    # Skip connection
    shortcut = Conv2D(filters, kernel_size=(1, 1), strides=stride, padding='same', kernel_regularizer=l2(lamda),
                      bias_regularizer=l2(lamda))(input_layer)
    shortcut = BatchNormalization()(shortcut) if bn else shortcut

    x = Add()([x, shortcut])
    return x


class UnetWithResBlocks:

    def __init__(self, v, train_patches, weighted_loss=False):

        self.train_patches = train_patches
        self.model_architecture = 'unet'
        self.weighted_loss = weighted_loss

        # params related to input/preproc.
        if self.train_patches == False:
            self.input_dims = 0
            self.output_dims = 0
        else:
            self.input_dims = 32
            self.output_dims = self.input_dims - 8
            self.patch_stride = 12
            self.patch_na = 4 / 8
        self.n_bins = 3
        self.region = 'global'  # 'europe'

        # params for model architecture
        self.filters = 2
        self.apool = True  # choose between average and max pooling, True = average
        self.n_blocks = 3  # 4  # 5
        self.bn = True  # batch normalization
        self.ct_kernel = (3, 3)  # (2, 2)
        self.ct_stride = (2, 2)  # (2, 2)

        # params related to model training
        self.optimizer_str = 'adam'
        self.call_back = True  # should early stopping be used?

        if v == 'tp':
            self.learn_rate = 0.001
            self.decay_rate = 0.005
            self.delayed_early_stop = True
        else:
            self.learn_rate = 1e-4
            self.decay_rate = 0
            self.delayed_early_stop = False

        if self.train_patches == True:
            self.bs = 32
            self.ep = 300
            self.patience = 10  # 3  # for callback
            self.start_epoch = 5  # 2  # epoch to start with early stopping
        else:  # global unet
            self.bs = 32
            self.ep = 300  # 20
            self.patience = 10  # for callback
            self.start_epoch = 5
            if self.call_back == False:
                self.ep = 30

    def build_model(self, dg_train_shape, dg_train_weight_target=None):
        inp_imgs = Input(shape=(dg_train_shape[1], dg_train_shape[2], dg_train_shape[3],))  # Input layer
        x = inp_imgs

        # Encoder / Contracting path
        encoder_blocks = []
        for i in range(self.n_blocks):
            x = resnet_block(x, self.filters * 2 ** i, lamda=0.01)  # Replace Conv layers with ResNet blocks
            encoder_blocks.append(x)
            x = MaxPooling2D((2, 2))(x)
        x = resnet_block(x, self.filters * 2 ** self.n_blocks, lamda=0.01)  # Bottleneck

        # Decoder / Expanding path
        for i in range(self.n_blocks - 1, -1, -1):
            x = Conv2DTranspose(self.filters * 2 ** i, (3, 3), strides=(2, 2), padding='same')(x)
            x = Concatenate()([x, encoder_blocks[i]])
            x = resnet_block(x, self.filters * 2 ** i, lamda=0.01)  # Replace Conv layers with ResNet blocks

        # Output layer
        out = Conv2D(3, (1, 1), activation='softmax')(x)

        # Crop to get rid of patch edges if training on patches
        if self.train_patches:
            out = Cropping2D(cropping=((4, 4), (4, 4)))(out)

        else:
            if self.region == 'europe':
                out = Cropping2D(cropping=((8, 8), (8, 8)))(out)
            if self.region == 'global':
                out = Cropping2D(cropping=((8, 8), (4, 3)))(out)

        if (self.train_patches == True) & (self.weighted_loss == True):
            weight_shape = dg_train_weight_target[0]
            weights = Input(shape=(weight_shape[1], weight_shape[2],))
            target_shape = dg_train_weight_target[1]
            target = Input(shape=(target_shape[1], target_shape[2], target_shape[3],))
            inputs = [inp_imgs]

            cnn = Model(inputs=[inputs] + [weights, target], outputs=out)

            cnn.target = target
            cnn.weight_mask = weights
            cnn.out = out
        else:
            cnn = Model(inputs=[inp_imgs], outputs=out)

        # cnn.summary()

        return cnn
