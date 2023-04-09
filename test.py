import os.path as osp
import glob
import cv2
import numpy as np
import torch
import imutils
import RRDBNet_arch as arch
import math
from skimage.metrics import structural_similarity as ssim

model_path = 'models/RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
device = torch.device('cpu')  # if you want to run on CPU, change 'cuda' -> cpu
# device = torch.device('cpu')

test_img_folder = 'LR/*'

model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

print('Model path {:s}. \nTesting...'.format(model_path))


def downsample(img_file, scale=0.4, plot=False):
    img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
    img_small = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation = cv2.INTER_NEAREST)

    if plot:
        cv2.imshow('Down sized', img_small)
    return img, img_small

def resizeNpImage(img, scale):
    img_resized = cv2.resize(img, (0,0), fx=scale, fy=scale,interpolation = cv2.INTER_NEAREST)
    return img_resized

def rotateImage(img, angle : 0) :
    rotated_image = imutils.rotate(img, angle)
    return rotated_image

def HR_to_LR(img) : 
    down_sized_image = resizeNpImage(img, 0.25)
    lr_image = resizeNpImage(down_sized_image, 4)
    return lr_image

def LR_to_HR(img):
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()

    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)
    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))

    return output

def add_padding(img, new_image_height, new_image_width):
    color = (0,0,0)
    old_image_height, old_image_width, channels = img.shape
    result = np.full((new_image_height,new_image_width,channels), color, dtype=np.uint8)

    x_center = (new_image_width - old_image_width) // 2
    y_center = (new_image_height - old_image_height) // 2

    result[y_center:y_center+old_image_height, 
       x_center:x_center+old_image_width] = img
    
    return result

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension


    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
    return err

def compare_ssim(imageA, imageB) :
    gray_scale_A = cv2.cvtColor(float_to_uint(imageA), cv2.COLOR_RGB2GRAY)
    gray_scale_B = cv2.cvtColor(float_to_uint(imageB), cv2.COLOR_RGB2GRAY)


    (score,_) = ssim(gray_scale_A, gray_scale_B, full = True)
    return score


def diagonal_length(length, breadth) :
    return math.ceil(math.sqrt(length*length + breadth*breadth))

def float_to_uint(img):
    Img = np.copy(img)
    Img *= 255
    Img = Img.astype(np.uint8)
    return Img


def normal_mode() : 
    idx = 0

    for path in glob.glob(test_img_folder):
        idx += 1
        base = osp.splitext(osp.basename(path))[0]
        print(idx, base)


        # read images
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        og = img

        longest_side = diagonal_length(img.shape[0],img.shape[1])

        img = add_padding(img, longest_side,longest_side )
        print(img.shape)

        ogoversize = resizeNpImage(og,4)

        # output = LR_to_HR(img)

        for i in range(0,8):
            angle = i * 45
            rotated_image = rotateImage(img,angle)
            rotated_image_same_dimensions = resizeNpImage(rotated_image,4)
            output = LR_to_HR(rotated_image)


            reduced_output = resizeNpImage(output,0.25)

        # cv2.imshow('HR',reduced_output)
        # cv2.imshow('LR',rotated_image)

            cv2.imshow('HR',output)
            cv2.imshow('LR',rotated_image_same_dimensions)

        # (score,diff) = ssim(cv2.cvtColor(float_to_uint(reduced_output),cv2.COLOR_RGB2GRAY),cv2.cvtColor(float_to_uint(rotated_image),cv2.COLOR_RGB2GRAY),full=True)
            score = compare_ssim(output, rotated_image_same_dimensions)
            print(score, " at : ", angle)

            cv2.waitKey(0)
        # test for only one image
        break


def blur_mode() :
    idx = 0
    for path in glob.glob(test_img_folder):
        idx+=1
        base = osp.splitext(osp.basename(path))[0]
        print(idx, base)

        # read image
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        
        blurred_image = cv2.GaussianBlur(img,(3,3),0)



        output1 = LR_to_HR(img)
        output2 = LR_to_HR(blurred_image)

        cv2.imshow('Gaussian Blur', output2)
        cv2.imshow('Original Image', output1)
        cv2.imshow('OG', resizeNpImage(img,4))
        cv2.waitKey(0)

        break


def post_processing_blur() : 
    idx = 0
    for path in glob.glob(test_img_folder):
        idx+=1
        base = osp.splitext(osp.basename(path))[0]
        print(idx,base)

        # read image
        img = cv2.imread(path, cv2.IMREAD_COLOR)

        output = LR_to_HR(img)

        blurred_output = cv2.GaussianBlur(output,(5,5),0)

        same_size_og = resizeNpImage(img,4)

        cv2.imshow('HR',output)
        cv2.imshow('Blurred HR', blurred_output)
        cv2.imshow('Input image', same_size_og)

        cv2.waitKey(0)

        ssim_blur = compare_ssim(blurred_output, same_size_og)
        ssim_true = compare_ssim(output, same_size_og)

        mse_blur = mse(blurred_output,same_size_og)
        mse_true = mse(output, same_size_og)

        print("True SSIM : ", ssim_true)
        print("True MSE : ", mse_true)

        print("Blurred SSIM : ", ssim_blur)
        print("Blurred MSE : ", mse_blur)

        cv2.waitKey(0)
        
        break




# Uncomment one function call

normal_mode()
# blur_mode()
# post_processing_blur()