import argparse

# Performs predictions using a trained model.
# Predictions are performed the same method we used and are saved to a
# target directory. 

def get_args():
    parser = argparse.ArgumentParser(
        description='Train UNet Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '-c', '--target-tissue', metavar='C', type=str,
        help='Target tissue, either breast or dv', dest='target_tissue'
    )

    parser.add_argument(
        '-i', '--image', metavar='T', type=str,
        help='Directory of images', dest='image_dir'
    )
    parser.add_argument(
        '-m', '--input-mask', metavar='M', type=str,
        help='Directory of input masks', dest='input_mask_dir'
    )

    parser.add_argument(
        '-s', '--save-masks-dir', metavar='S', type=str,
        help='Directory to save masks', dest='save_masks_dir'
    )
    parser.add_argument(
        '-p', '--model-save-path', metavar='P', type=str,
        help='Path to saved model', dest='model_save_path'
    )

    return parser.parse_args()


if __name__ == '__main__':
    from dataset_3d import *
    from model_utils import pred_and_save_masks_3d_simple, pred_and_save_masks_3d_stacked
    from unet import UNet3D

    args = get_args()

    if args.target_tissue == 'breast':
        n_channels = 1
        n_classes = 1
    elif args.target_tissue == 'dv':
        n_channels = 2
        n_classes = 3
    else:
        print('Target tissue must either be breast or dv')
        raise

    unet = UNet3D(
        in_channels = n_channels, 
        out_classes = n_classes,
        num_encoding_blocks = 3,
        padding = True,
        normalization = 'batch'
    )

    if args.target_tissue == 'breast':

        input_dim = (144, 144, 96)

        transforms = tio.Compose([
            tio.Resize(input_dim)
        ])

        dataset = Dataset3DSimple(
            image_dir = args.image_dir,
            mask_dir = None,
            transforms = transforms,
            image_only = True
        )

        pred_and_save_masks_3d_simple(
            unet = unet,
            saved_model_path = args.model_save_path,
            dataset = dataset,
            n_classes = n_classes,
            n_channels = n_channels,
            save_masks_dir = args.save_masks_dir
        )

    else:

        z_input_dim = 96
        z_step_size = 96

        transforms = tio.Compose([
            tio.Resize((344, 344, z_input_dim))
        ])

        dataset = Dataset3DVerticalStack(
            image_dir = args.image_dir,
            mask_dir = None,
            additional_input_dir = args.input_mask_dir,
            z_input_dim = z_input_dim,
            z_step_size = z_step_size,
            transforms = transforms,
            one_hot_mask = True,
            image_only = True
        )

        pred_and_save_masks_3d_stacked(
            saved_model_path = args.model_save_path,
            dataset = dataset,
            unet = unet,
            n_classes = n_classes,
            n_channels = n_channels,
            save_masks_dir = args.save_masks_dir
        )