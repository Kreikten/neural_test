import matplotlib.pyplot as plt

from neural_network import *
import numpy as np
import PIL
import datetime

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def concat_pics(pics:list):
    captcha = pics[0]

    for i in range(1, len(pics)):
        captcha = np.hstack([captcha, pics[i]])
     #print(captcha.shape)
    return captcha
def from_3_to_2(array):
    array_fixed = zeros((len(array), len(len(array))))
    for i in range(len(array)):
        for j in range (len(len(array))):
            array_fixed[i][j] = array[i][j][0]
    return array_fixed
def save_pic(picture):
    basename = "captcha"
    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    filename = basename + "_" + suffix + ".png"
    np_img = np.squeeze(picture, axis=2)
    data = PIL.Image.fromarray(np_img)


    data.save(filename)
    return filename

def generate_captcha(text: str):
    pic_list = []
    for i in text:
        generator = make_generator_model()
        discriminator = make_discriminator_model()
        noise = tf.random.normal([1, 100])
        generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
        # print(generator.summary())
        checkpoint_dir = ""
        checkpoint_prefix = os.path.join(checkpoint_dir, i, "ckpt-1")
        print(checkpoint_prefix)
        print(i)
        checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                         discriminator_optimizer=discriminator_optimizer,
                                         generator=generator,
                                         discriminator=discriminator)
        checkpoint.read(save_path=checkpoint_prefix)

        generated_image = generator(noise, training=False)
      #  plt.imshow(generated_image[0,:,:])
      #  plt.show()
        pic_list.append(generated_image[0])
    s = concat_pics(pic_list)
    #print(s.shape)
    fig = plt.figure()
    plt.imshow(s[:,:,0])
    plt.show()
    filename = save_pic(s)

    #data = PIL.Image.fromarray(s)
   # data = data.convert("L")

  #  data.save(filename)
    return filename

filename = generate_captcha("01x01z")
print(filename)
# generator = make_generator_model()
# discriminator = make_discriminator_model()
# noise = tf.random.normal([1, 100])
# generated_image = generator(noise, training=False)
# # Visualize the generated sample
# plt.imshow(generated_image[0, :, :, 0], cmap='gray')
# plt.show()
# cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
# generator_optimizer = tf.keras.optimizers.Adam(1e-4)
# discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
# #print(generator.summary())
# checkpoint_dir = ""
# checkpoint_prefix = os.path.join(checkpoint_dir, "z", "ckpt-15")
# print(checkpoint_prefix)
# checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
#                                  discriminator_optimizer=discriminator_optimizer,
#                                  generator=generator,
#                                  discriminator=discriminator)
# #checkpoint.read(checkpoint_prefix)
# #checkpoint.read(checkpoint_prefix)
# checkpoint.read(save_path=checkpoint_prefix)
# #print(generator.summary())
# noise = tf.random.normal([1, 100])
# generated_image = generator(noise, training=False)
# plt.imshow(generated_image[0, :, :, 0], cmap='gray')
# plt.show()


# import numpy as np
# def noisy(noise_typ,image):
#  #  print(image)
#    if noise_typ == "gauss":
#       row,col,ch= image.shape
#       mean = 0
#       var = 0.1
#       sigma = var**0.5
#       gauss = np.random.normal(mean,sigma,(row,col,ch))
#       gauss = gauss.reshape(row,col,ch)
#       noisy = image + gauss
#       return noisy
#    elif noise_typ == "s&p":
#       row,col,ch = image.shape
#       s_vs_p = 0.5
#       amount = 0.004
#       out = np.copy(image)
#       # Salt mode
#       num_salt = np.ceil(amount * image.size * s_vs_p)
#       coords = [np.random.randint(0, i - 1, int(num_salt))
#               for i in image.shape]
#       out[coords] = 1
#
#       # Pepper mode
#       num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
#       coords = [np.random.randint(0, i - 1, int(num_pepper))
#               for i in image.shape]
#       out[coords] = 0
#       return out
#    elif noise_typ == "poisson":
#       vals = len(np.unique(image))
#       vals = 2 ** np.ceil(np.log2(vals))
#       noisy = np.random.poisson(image * vals) / float(vals)
#       return noisy
#    elif noise_typ =="speckle":
#       row,col,ch = image.shape
#       gauss = np.random.randn(row,col,ch)
#       gauss = gauss.reshape(row,col,ch)
#       noisy = image + image * gauss
#       return noisy
#
# generated_image = noisy("gauss", generated_image[0])
# print(generated_image.shape)
# plt.imshow(generated_image[0, :, :, 0], cmap='gray')
# plt.show()