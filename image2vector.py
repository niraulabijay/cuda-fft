import numpy as np
from PIL import Image

def get_image_2d_vector(image_path):
    # Open the image file
    img = Image.open(image_path).convert('L')
    # Convert the image to a numpy array
    img_array = np.array(img)

    np.savetxt("image2d.txt", img_array, fmt='%d')
    return img_array.tolist()


# Test the function
get_image_2d_vector('./sampleImage/fresno_state.jpg')