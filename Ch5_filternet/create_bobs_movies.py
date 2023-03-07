import os
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from PIL import Image

from allensdk.core.brain_observatory_cache import BrainObservatoryCache


def get_natural_scenes(output_dir='bob_images'):
    """Fetches the 118 Brain Obs natural scene images from the data, saves them in npy format"""
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    boc = BrainObservatoryCache(manifest_file='boc/manifest.json')
    data_set = boc.get_ophys_experiment_data(501498760)
    scenes = data_set.get_stimulus_template('natural_scenes')

    for i in range(0, len(scenes)):
        scene = scenes[i]
        base_name = os.path.join(output_dir, 'scene.{:03d}.gray_{}x{}'.format(i, scene.shape[0], scene.shape[1]))
        plt.imsave('{}.png'.format(base_name), scene, cmap='gray')
        # np.save('{}.npy'.format(base_name), scene)


def create_movie_natural_scenes(images, movie_path, image_dur=250.0,  gs_dur=500.0, res_row=120, res_col=240, fps=1000.0):
    frames_per_image = int((image_dur/1000.0)*fps)
    frames_per_gs = int(int((gs_dur/1000.0)*fps))
    n_frames = frames_per_gs + len(images)*frames_per_image + frames_per_gs

    output_mat = np.zeros((n_frames, res_row, res_col), dtype=np.float)
    c_frame = frames_per_gs
    for img_path in images:
        pic = Image.open(img_path).convert('L')
        pic = pic.resize((res_col, res_row))
        pic_data = np.asarray(pic)
        pic_data = pic_data.astype(dtype=np.float) * 2.0 / 255.0 - 1.0
        print(c_frame, pic_data)
        output_mat[c_frame:(c_frame + frames_per_image), :, :] = pic_data

        c_frame += frames_per_image

    np.save(movie_path, output_mat)

    # n_frames =
    # output_mat = np.zeros((n_frames, res_row, res_col), dtype=np.float)


def show_movie(movie_file, frames):
    movie_array = np.load(movie_file)
    # plt.figure()
    fig, ax = plt.subplots(1, len(frames), figsize=(20, 5*len(frames)))

    # print(movie_array[1000, :, :])

    for i, frame in enumerate(frames):
        ax[i].imshow(movie_array[frame, :, :], cmap='gray', vmin=-1.0, vmax=1.0)
        # ax[i].set_xticks([])
        ax[i].set_xticks([0])
        ax[i].set_xticklabels([frame])

        ax[i].set_yticks([])

    ax[0].set_xlabel('frame #', horizontalalignment='right')
    plt.subplots_adjust(wspace=0, hspace=0)

    plt.show()


def get_touchofevil_movies(output_dir='movies', res_row=120, res_col=240, fps=1000):
    frame_conv = int(np.floor(fps/30.0))
    def convert_movie(name, movie):
        t, x, y = movie.shape
        n_frames = frame_conv * t
        movie_updated = np.zeros((n_frames, res_row, res_col), dtype=np.uint8)
        c_frame = 0
        for frame in range(t):
            # Resize resolution
            img = Image.fromarray(movie[frame, :, :], mode='L')
            img = img.resize((res_col, res_row))
            img_data = np.asarray(img)

            # Upscale frame rate
            movie_updated[c_frame:(c_frame + frame_conv), :, :] = img_data
            c_frame += frame_conv

        np.save('{}/{}.{}ms.{}x{}.npy'.format(output_dir, name, c_frame, res_row, res_col), movie_updated)

    boc = BrainObservatoryCache(manifest_file='boc/manifest.json')
    data_set = boc.get_ophys_experiment_data(506248008)
    movie = data_set.get_stimulus_template('natural_movie_one')
    convert_movie('natural_movie_one', movie)

    movie = data_set.get_stimulus_template('natural_movie_two')
    convert_movie('natural_movie_two', movie)

    data_set = boc.get_ophys_experiment_data(649409874)
    movie = data_set.get_stimulus_template('natural_movie_three')
    convert_movie('natural_movie_three', movie)


if __name__ == '__main__':
    # images = glob('bob_images/scene*.png')
    # images = np.random.choice(images, size=8, replace=False)
    # create_movie_natural_scenes(images, movie_path='inputs/ns_movie.8images.npy')
    # show_movie(movie_file='inputs/ns_movie.8images.npy', frames=range(0, 3000, 250))

    get_touchofevil_movies()

