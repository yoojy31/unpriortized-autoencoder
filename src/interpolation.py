import scipy
import utils
import os
import random

# input : x_save -> (batch, ch, w, h)?
# ouput: None
# interpolate save

def interpolate_z(x_save, ae, save_img_dir, valid_dataset):
    # entirely random select? entirely fixed select? siquential?
    nums = [x for x in range(len(x_save))]
    # random.shuffle(nums)

    # 2개의 이미지 _ 55번, 66번
    temp = x_save[:2].clone()
    temp[0]=x_save[nums[55]].clone()
    temp[1]=x_save[nums[66]].clone()

    # 또 두개의 이미지 _ 21, 102번
    temp_two = x_save[2:4].clone()
    temp_two[0] = x_save[nums[21]].clone()
    temp_two[1] = x_save[nums[102]].clone()

    # two_image_latent_code = ae.encoder.forward(x_save[:2]) # maybe, (2,code_size,1,1)
    two_image_latent_code = ae.forward(temp, forward_type='encoder') # maybe, (2,code_size,1,1)
    two_image_latent_code_two = ae.forward(temp_two, forward_type='encoder') # maybe, (2,code_size,1,1)
    save_img_dir = save_img_dir+'/interpolate'
    utils.make_dir(save_img_dir)

    shape = two_image_latent_code.shape

    ################################################################################
    ######################## original image save ###################################
    origin_one = valid_dataset.post_process(temp[0])
    save_data_path = os.path.join(save_img_dir, 'aa1.000origin.png')
    scipy.misc.imsave(save_data_path, origin_one)

    origin_two = valid_dataset.post_process(temp[1])
    save_data_path = os.path.join(save_img_dir, 'a_0..0origin.png')
    scipy.misc.imsave(save_data_path, origin_two)

#     wall = x_save[0].clone()
#     wall[:,:,:] = 0
#     wall = valid_dataset.post_process(wall)
#     save_data_path = os.path.join(save_img_dir, 'a_b_wall.png')
#     scipy.misc.imsave(save_data_path, wall)
#     save_data_path = os.path.join(save_img_dir, 'a_b_wall2.png')
#     scipy.misc.imsave(save_data_path, wall)

    origin_one = valid_dataset.post_process(temp_two[0])
    save_data_path = os.path.join(save_img_dir, 'bb1.000origin.png')
    scipy.misc.imsave(save_data_path, origin_one)

    origin_two = valid_dataset.post_process(temp_two[1])
    save_data_path = os.path.join(save_img_dir, 'b_0..0origin.png')
    scipy.misc.imsave(save_data_path, origin_two)
    ################################################################################
    ################################################################################

    for i in range(11):                     # 0,1,2,3,4,5,6,7,8,9,10
        interpolated_code = (1. - i/10.0) * two_image_latent_code[0] + (i/10.0) * two_image_latent_code[1]  # maybe, (code_size/4,1,1)
        interpolated_code = interpolated_code.expand(1, shape[1], shape[2],shape[3])
        interpolated_image = ae.decoder.forward(interpolated_code)
        interpolated_image = interpolated_image.squeeze()
        interpolated_image = valid_dataset.post_process(interpolated_image)
        save_data_path = os.path.join(save_img_dir, 'a_%.2f.png'%((1.-i/10.0)))
        scipy.misc.imsave(save_data_path, interpolated_image)

        interpolated_code_two = (1. - i / 10.0) * two_image_latent_code_two[0] + (i / 10.0) * two_image_latent_code_two[1]  # maybe, (code_size/4,1,1)
        interpolated_code_two = interpolated_code_two.expand(1, shape[1], shape[2], shape[3])
        interpolated_image_two = ae.decoder.forward(interpolated_code_two)
        interpolated_image_two = interpolated_image_two.squeeze()
        interpolated_image_two = valid_dataset.post_process(interpolated_image_two)
        save_data_path = os.path.join(save_img_dir, 'b_%.2f.png' % ((1. - i / 10.0)))
        scipy.misc.imsave(save_data_path, interpolated_image_two)
