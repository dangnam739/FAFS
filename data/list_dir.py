import os
import json
import glob
import random
from shutil import move

ROOT_DIR = "/home/s/polyp_data/WLI"
SAVE_DIR = '/home/admin_mcn/namkd/rnd-domain-adaptation/data'

if __name__ == '__main__':

    data = []

    # for dirname in os.listdir(ROOT_DIR):
    img_dir = 'images'
    mask_dir = 'mask_images'

    for file in os.listdir(os.path.join(ROOT_DIR, img_dir)):
        img_path = os.path.join(img_dir, file)
        mask_name = file[:-5] + ".png"
        mask_path = os.path.join(mask_dir, mask_name)

        data.append({"image_name":img_path, "mask_name":mask_path, "has_target":True})
    # os.chdir("/home/admin_mcn/namkd/rnd-domain-adaptation")
    json_file = os.path.join(SAVE_DIR, "wli_train.json")
    with open(json_file, 'w') as outfile:
        json.dump(data, outfile)
    # move(json_file, SAVE_DIR)


    # data_dir = ['gtFine', 'leftImg8bit']
    # city = ['frankfurt', 'lindau', 'munster']
    # count = 0

    # for dirname in os.listdir(data_dir):
    #     val_dir = os.path.join(dirname, 'val')
    #     for city_name in city:
    #         city_dir = os.path.join(val_dir, city_name)

    #         for file in os.listdir(os.path.join(ROOT_DIR, city_dir)):
    #             if "label" in file:                        \
    #                 img_path = os.path.join(city_dir, file)
    #             label_path = os.path.join(label_dir, file)

    #             data.append({"image_name":img_path, "mask_name":label_path, "has_target":True})

    # with open('cityscapes_val.json', 'w') as outfile:
    #     json.dump(data, outfile)

    #split train-test dataset

    # TEST_DIR = "/home/s/polyp_data/BLI/test"

    # if not os.path.exists(TEST_DIR):
    #     os.makedirs(TEST_DIR)

    # file_path_type = [f"{ROOT_DIR}/images/*.jpeg"]
    # images = glob.glob(random.choice(file_path_type))
    # # print(images)
    # list_img_test = []
    # for i in range(2000):
    #     if len(list_img_test) >= 400:
    #         break

    #     random_image = random.choice(images)
    #     # print(random_image)
    #     if random_image not in list_img_test:
    #         list_img_test.append(random_image)

    # # with open('/home/s/polyp_data/BLI/test/tmp.txt', 'w') as outfile:
    # #     outfile.write(str(list_img_test))
    # for img_path in list_img_test:
    #     move(img_path, f'{TEST_DIR}/images')

    # for imgpath in os.listdir(f"{TEST_DIR}/images"):
    #     img_name = imgpath[:-5]

    #     move(f"{ROOT_DIR}/label_images/{img_name}.png", f"{TEST_DIR}/label_images")
    #     move(f"{ROOT_DIR}/mask_images/{img_name}.png", f"{TEST_DIR}/mask_images")
    #     move(f"{ROOT_DIR}/metadata/{img_name}.json", f"{TEST_DIR}/metadata")
