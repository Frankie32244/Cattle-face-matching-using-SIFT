import os

# 将/data/test文件夹下的图片 rename -> 001.jpg  002.jpg ... 
folder_path = "./data/test"  # 替换为实际包含文件的文件夹路径

# 获取文件夹下所有文件
file_list = os.listdir(folder_path)

if len(file_list) > 0:
    counter = 1
    for file_name in file_list:
        if file_name.lower().endswith(".jpg"):
            file_extension = os.path.splitext(file_name)[1]
            new_file_name = "{:03d}{}".format(counter, file_extension)
            old_file_path = os.path.join(folder_path, file_name)
            new_file_path = os.path.join(folder_path, new_file_name)
            os.rename(old_file_path, new_file_path)
            counter += 1
    print("File renaming completed.")
else:
    print("No files found in the folder.")
