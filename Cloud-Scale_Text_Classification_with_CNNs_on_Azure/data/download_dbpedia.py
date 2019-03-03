#Download data

import sys
import os
current_path = os.path.dirname(os.path.abspath(__file__))
root_path = current_path.rsplit('solutions')[0]
sys.path.insert(0,root_path)
from solutions.utils.python_utils import download_file

url = 'https://mxnetstorage.blob.core.windows.net/public/nlp/dbpedia_csv.tar.gz'
print("Downloading file %s" % url)
download_file(url)
