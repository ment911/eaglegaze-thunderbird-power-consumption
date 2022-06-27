from thunder import Thunder
from decouple import config as envs #temp
from pathlib import Path #temp


if __name__ == '__main__':
    # comment for common
    # path_files = envs('STORAGE_PATH', cast=str) #temp
    # Path(path_files).mkdir(parents=True, exist_ok=True) #temp
    # print('Folders has been created')#temp
    Thunder().run()