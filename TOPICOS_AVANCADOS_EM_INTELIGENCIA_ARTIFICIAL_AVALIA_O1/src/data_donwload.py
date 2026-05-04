import kagglehub 
import shutil 
import os
path = kagglehub.dataset_download (
"bluehorseshoe/uk-2016-road-safety-data"
)

print (path)
destino = os.getcwd ()
shutil. copytree(path, os-path.join(destino, "dataset"), dirs_exist_ok=True)