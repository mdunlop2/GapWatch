import os, sys
from pathlib import Path

# Add the git root directory to python path
sys.path.insert(0,os.getcwd())

# custom scripts
from common.data.labels.app.config_utils import JSONPropertiesFile


app_file_parent_path = Path(__file__).absolute().parent
CONFIG_LOC = os.path.join(app_file_parent_path, "config")
CONFIG_FILE_LOC = os.path.join(CONFIG_LOC, "config.json")
# Ensure the configuration file exists
try:
    # generate configuration file
    os.mkdir(CONFIG_LOC)
    print("Directory {} created.".format(CONFIG_LOC))
    # file = open(CONFIG_FILE_LOC, 'w')
    print("{} did not exist. \nIt will now be created.".format(CONFIG_LOC, CONFIG_FILE_LOC))
except:
    print("Directory {} already exists.".format(CONFIG_LOC))
    try:
        file = open(CONFIG_FILE_LOC, 'r')
        print("{} already exists.".format(CONFIG_FILE_LOC))
    except IOError:
        print("{} did not exist but the {} directory did. \nIt will now be created.".format(CONFIG_LOC, CONFIG_FILE_LOC))


file_path = CONFIG_FILE_LOC
default_properties = {} 
config_file = JSONPropertiesFile(file_path, default_properties)
config = config_file.get() 
print(config)
config["PROD"] = "k else"
config_file.set(config) #  save new config