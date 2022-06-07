from thunder import Thunder
from decouple import config as envs
from pathlib import Path


if __name__ == '__main__':
    # comment for common
    path_files = envs('STORAGE_PATH', cast=str)
    Path(path_files).mkdir(parents=True, exist_ok=True)
    print('Folders has been created')
    Thunder().run()