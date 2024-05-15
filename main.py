import numpy as np
from PIL import Image

def get_image_2d_vector(image_path):
    # Open the image file
    img = Image.open(image_path).convert('L')
    # Convert the image to a numpy array
    img_array = np.array(img)

    np.savetxt("image2d.txt", img_array, fmt='%f')
    return img_array.tolist()

# Test the function
# print(get_image_2d_vector('./sampleImage/squirrel.jpg'))

def create_image_from_2d_vector(file_path, output_image_path):
    # Load the 2D vector from the text file
    data = np.loadtxt(file_path, dtype=np.uint8, converters=float)
    # Create an image from the 2D vector
    img = Image.fromarray(data)
    # Save the image
    img.save(output_image_path)

# Loop from  0.000001 to 0.1 with step 10 and create image from 2d vector
for i in range(1, 7):
    create_image_from_2d_vector('./compressedTxt/compressed_image_'+str(i)+'.txt', './compressedImg/new_image_' + str(i) + '.jpg')


# create_image_from_2d_vector('image2dcuda.txt', 'new_squirrel.png')