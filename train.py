import os
import argparse
import sys
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, History
from tensorflow.keras.optimizers import Adam
from model import ResUnet2D, dice_coef, dice_loss, IoU
from dataloader import Dataset
from sklearn.model_selection import train_test_split

def parse_args(args):
    parser = argparse.ArgumentParser(description='Unet3D training')
    parser.add_argument('--datalist', type = str, required = True, help = 'File path of data list.')
    parser.add_argument('--image-size', type = int, default = 512, help = 'Image sizes, default is 512.')
    parser.add_argument('--n-classes', type = int, required = True, help = 'Number of classes.')
    parser.add_argument('--output', type = str, help = 'Folder path to save the trained weights and the line charts of dice coefficient, loss and IoU.')
    parser.add_argument('--weight', type = str, default = None, help = 'File path of the pretrained weights(h5 file). Default is None.')
    parser.add_argument('--batch-size', type = int, default = 1, help = 'Batch size of the training. Default is 1.')
    parser.add_argument('--epochs', type = int, default = 50, help = 'Epoch of the training. Default is 50.')
    parser.add_argument('--init-f', type = int, default = 32, help = 'Number of the filter in the first encoder. Default is 32.')
    parser.add_argument('--init-lr', type = float, default = 1e-3, help = 'Set the learning rate of the model. Default is 1e-3.')
    return parser.parse_args(args)


def main(args = None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    
    os.makedirs(args.output, exist_ok=True)
    
    list_path = args.datalist
    if list_path.split('.')[-1] == 'csv':
        df = pd.read_csv(list_path)
    elif list_path.split('.')[-1] == 'xlsx':
        df = pd.read_excel(list_path, engine='openpyxl')
    
    # DataLoader
    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)
    train_gen = Dataset(train_df, args.batch_size)
    valid_gen = Dataset(valid_df, 1)
    
    # Parameters of training
    model = ResUnet2D((args.image_size, args.image_size, 1), args.init_f, args.n_classes)
    if args.weight:
        model.load_weights(args.weight) # , custom_objects={}
    model.summary()
    
    # Training
    history = History()
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-20, verbose=1)
    model_checkpoint = ModelCheckpoint(f'{args.out}/ResUnet2d-epoch{epoch:03d}-dice{val_dice_coef:.5f}.h5', verbose=1, save_best_only=True)
    
    model.compile(optimizer=Adam(learning_rate=args.init_lr), loss=dice_loss, metrics=[dice_coef, IoU])
    model.fit(train_gen, validation_data=valid_gen, batch_size = args.batch_size, epochs = args.epochs, callbacks=[history, model_checkpoint, reduce_lr], shuffle=True)
        
    # Save a dictionary into a pickle file.
    hist = {'dice_coef':history.history['dice_coef'], 'val_dice_coef':history.history['val_dice_coef'], 'loss':history.history['loss'], 'val_loss':history.history['val_loss'], 'IoU':history.history['IoU'], 'val_IoU':history.history['val_IoU']}
    out_df = pd.DataFrame(hist)
    out_df.to_csv(f'{args.output}/History.csv', index=True)
    
    # save plot(dice_coef, IoU, loss)
    plt.plot(history.history['dice_coef'])
    plt.plot(history.history['val_dice_coef'])
    plt.title('model dice_coef'), plt.ylabel('dice_coef'), plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(f'{args.output}/dice_coef.png', dpi=350, bbox_inches='tight')
    plt.close()
    
    plt.plot(history.history['IoU'])
    plt.plot(history.history['val_IoU'])
    plt.title('model IoU'), plt.ylabel('IoU'), plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(f'{args.output}/IoU.png', dpi=350, bbox_inches='tight')
    plt.close('all')
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss'), plt.ylabel('loss'), plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(f'{args.output}/loss.png', dpi=350, bbox_inches='tight')
    plt.close('all')

if __name__ == '__main__':
    main()