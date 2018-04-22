import cv2, os
import numpy as np 
import matplotlib.image as mpimg

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66,200,3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

def load_image(img_dir,img):
	"""Loads images from a folder"""
	return mpimg.imread(os.path.join(img_dir,img.strip()))
    
def crop(img):
	"""
		Removing the sky from and the car front
	"""
	return img[60:-25,:,:]

def resize(img):
	"""
		Resize the image to input shape used by the model
	"""
	return cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)

def rgb2yuv(img):
	"""
		Convert image from RGB to yuv (Like the NVIDIA model)
	"""
	return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

def preprocess(img):
	"""
		Combine all pre process functions
	"""
	img = crop(img)
	img = resize(img)
	img = rgb2yuv(img)

	return img


def choose_image(image_dir,center,left,right,steering_angle):
	"""
		Choose a random image from center, left, or right and 
		adjust steering angle
	"""
	choice = np.random.choice(3)
	if choice == 0:
		return load_image(image_dir,left), steering_angle + 0.2
	elif choice == 1:
		return load_image(image_dir,right), steering_angle - 0.2
	return load_image(image_dir, center), steering_angle

def random_flip(img, steering_angle):
	"""
		Flip randomly the image left <-> right and adjust the steering angle
	"""
	if np.random.rand() < 0.5:
		img = cv2.flip(img, 1)
		steering_angle = -steering_angle
	return img, steering_angle

def random_translate(img, steering_angle, range_x, range_y):
	"""
		Shift randomly the image vertically and horizontaly
	"""
	trans_x = range_x * (np.random.rand() - 0.5)
	trans_y = range_y * (np.random.rand() - 0.5)
	steering_angle += trans_x * 0.002
	trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
	height, width = img.shape[:2]
	img = cv2.warpAffine(img, trans_m, (width, height))
	return img, steering_angle

def random_shadow(img):
	"""
		Generates and adds a random shadow
	"""
	x1, y1 = IMAGE_WIDTH * np.random.rand(), 0
	x2, y2 = IMAGE_WIDTH * np.random.rand(), IMAGE_HEIGHT
	xm, ym = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH] #gives all loca img
	#set 1 below the line and zero otherwise. (ym-y1)/(xm-x1) > (y2-y1)/(x2-x1)>0
	#x2 == x1 causes division by zero    
	#mask_cond = (ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) #> 0
	mask = np.zeros_like(img[:,:, 1])
	#mask[mask_cond] = 1
	#choose the side that should have shodow
	cond = mask == np.random.randint(2)
	s_ratio = np.random.uniform(low=0.2,high=0.5)
	#adjust Hue, Light and Saturation
	hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
	hls[:,:,1][cond] = hls[:,:,1][cond] * s_ratio
	return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)


def random_brightness(image):
    """
    Randomly adjust brightness of the image.
    """
    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:,:,2] =  hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def augment(data_dir, center, left, right, steering_angle, range_x=100, range_y=10):
    """
    Generate an augumented image and adjust steering angle.
    (The steering angle is associated with the center image)
    """
    image, steering_angle = choose_image(data_dir, center, left, right, steering_angle)
    image, steering_angle = random_flip(image, steering_angle)
    image, steering_angle = random_translate(image, steering_angle, range_x, range_y)
    image = random_shadow(image)
    image = random_brightness(image)
    return image, steering_angle

def batch_generator(image_dir, image_paths, steering_angles,batch_size, is_training): 
	"""
	Generate train image provide paths and steering_angles
	"""
	imgs = np.empty([batch_size,IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_CHANNELS])
	steers = np.empty(batch_size)
	while True:
		i = 0
		for index in np.random.permutation(image_paths.shape[0]):
			center,left,right = image_paths[index]
			steering_angle = steering_angles[index]
			#augmentation
			if is_training and np.random.rand() < 0.6:
				img, steering_angle = augment(image_dir, center, left, right, steering_angle)
			else:
				img = load_image(image_dir, center)
			#add image and steers to the batch
			imgs[i] = preprocess(img)
			steers[i] = steering_angle
			i += 1
			if i == batch_size:
				break
		yield imgs, steers	