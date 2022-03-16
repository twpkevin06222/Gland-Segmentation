# batch generator by MIC-DKFZ: https://github.com/MIC-DKFZ/batchgenerators
import numpy as np
from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.spatial_transforms import MirrorTransform, SpatialTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, ContrastAugmentationTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.augmentations.crop_and_pad_augmentations import crop


def get_split_fold(data):
    """
    If the data set is already split according to folds with indices [0,1,2]
    where:
    train => 0
    testA => 1
    testB => 2
    @param data: csv file where the data sets are stored
    @return: dictionaries of train, testA, testB dictionary
    """
    # return indices of fold data
    train_idx = np.where(data['fold'] == 0)[0]
    testA_idx = np.where(data['fold'] == 1)[0]
    testB_idx = np.where(data['fold'] == 2)[0]

    # create dictionary for each data set
    train_ds = {'img_npy': [data['img_npy'].tolist()[i] for i in train_idx],
                'anno_npy': [data['anno_npy'].tolist()[i] for i in train_idx],
                'patient_id': [data['patient ID'].tolist()[i] for i in train_idx]}
    testA_ds = {'img_npy': [data['img_npy'].tolist()[i] for i in testA_idx],
                'anno_npy': [data['anno_npy'].tolist()[i] for i in testA_idx],
                'patient_id': [data['patient ID'].tolist()[i] for i in testA_idx]}
    testB_ds = {'img_npy': [data['img_npy'].tolist()[i] for i in testB_idx],
                'anno_npy': [data['anno_npy'].tolist()[i] for i in testB_idx],
                'patient_id': [data['patient ID'].tolist()[i] for i in testB_idx]}

    return {'train_ds':train_ds, 'testA_ds': testA_ds, 'testB_ds': testB_ds}


def get_train_transform(patch_size, prob=0.5):
    # We now create a list of transforms.
    # These are not necessarily the best transforms to use for BraTS, this is just
    # to showcase some things
    tr_transforms = []

    # the first thing we want to run is the SpatialTransform. It reduces the size of our data to patch_size and thus
    # also reduces the computational cost of all subsequent operations. All subsequent operations do not modify the
    # shape and do not transform spatially, so no border artifacts will be introduced
    # Here we use the new SpatialTransform_2 which uses a new way of parameterizing elastic_deform
    # We use all spatial transformations with a probability of 0.2 per sample. This means that 1 - (1 - 0.1) ** 3 = 27%
    # of samples will be augmented, the rest will just be cropped
    tr_transforms.append(
        SpatialTransform(
            patch_size,
            [i // 2 for i in patch_size],
            do_elastic_deform=True,
            alpha=(0., 300.),
            sigma=(20., 40.),
            do_rotation=True,
            angle_x=(-np.pi/15., np.pi/15.),
            angle_y=(-np.pi/15., np.pi/15.),
            angle_z=(0., 0.),
            do_scale=True,
            scale=(1/1.15, 1.15),
            random_crop=False,
            border_mode_data='constant',
            border_cval_data=0,
            order_data=3,
            p_el_per_sample=prob, p_rot_per_sample=prob, p_scale_per_sample=prob
        )
    )

    # now we mirror along the y-axis
    tr_transforms.append(MirrorTransform(axes=(1,)))

    # brightness transform
    tr_transforms.append(BrightnessMultiplicativeTransform((0.7, 1.5), per_channel=True, p_per_sample=prob))

    # Gaussian Noise
    tr_transforms.append(GaussianNoiseTransform(noise_variance=(0, 0.5), p_per_sample=prob))

    # blurring. Some BraTS cases have very blurry modalities. This can simulate more patients with this problem and
    # thus make the model more robust to it
    tr_transforms.append(GaussianBlurTransform(blur_sigma=(0.5, 2.0), different_sigma_per_channel=True,
                                               p_per_channel=prob, p_per_sample=prob))
    tr_transforms.append(ContrastAugmentationTransform(contrast_range=(0.75, 1.25), p_per_sample=prob))
    # now we compose these transforms together
    tr_transforms = Compose(tr_transforms)
    return tr_transforms


class DataLoader(DataLoader):
    def __init__(self, data, batch_size, patch_size, num_threads_in_multithreaded,
                 crop_status=False, crop_type="center",
                 seed_for_shuffle=1234, return_incomplete=False, shuffle=True,
                 infinite=True, margins=(0,0,0)):
        """
        data must be a list of patients as returned by get_list_of_patients (and split by get_split_deterministic)
        patch_size is the spatial size the returned batch will have
        """
        super().__init__(data, batch_size, num_threads_in_multithreaded, seed_for_shuffle, return_incomplete, shuffle,
                         infinite)
        # original patch size with [width, height]
        self.patch_size = patch_size
        self.n_channel = 3
        self.indices = list(range(len(data['img_npy'])))
        self.crop_status = crop_status
        self.crop_type = crop_type
        self.margins = margins

    @staticmethod
    def load_patient(img_path):
        img = np.load(img_path, mmap_mode="r")
        return img

    def generate_train_batch(self):
        # DataLoader has its own methods for selecting what patients to use next, see its Documentation
        idx = self.get_indices()
        gland_img = [self._data['img_npy'][i] for i in idx]
        img_seg = [self._data['anno_npy'][i] for i in idx]
        patient_id = [self._data['patient_id'][i] for i in idx]
        # initialize empty array for data and seg
        img = np.zeros((len(gland_img), self.n_channel, *self.patch_size), dtype=np.float32)
        seg = np.zeros((len(img_seg), self.n_channel, *self.patch_size), dtype=np.float32)
        # iterate over patients_for_batch and include them in the batch
        for i, (j,k) in enumerate(zip(gland_img, img_seg)):
            img_data = self.load_patient(j)
            seg_data = self.load_patient(k)
            # according to the documentation
            # the input image should use channel first as input
            # hence we use tensor manipulation to convert to channel first
            img_data = np.einsum('hwc->chw', img_data)
            seg_data = np.einsum('hwc->chw', seg_data)
            # now random crop to self.patch_size
            # crop expects the data to be (b, c, x, y, z) but patient_data is (c, x, y, z) so we need to add one
            # dummy dimension in order for it to work (@Todo, could be improved)
            if self.crop_status:
                img_data, seg_data = crop(img_data[None], seg=seg_data[None], crop_size=self.patch_size,
                                    margins=self.margins, crop_type=self.crop_type)
                img[i] = img_data[0]
                seg[i] = seg_data[0]
            else:
                img[i] = img_data
                seg[i] = seg_data
        return {'data': img, 'seg': seg, 'patient_id': patient_id}

