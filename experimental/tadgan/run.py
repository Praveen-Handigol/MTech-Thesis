import pandas as pd
import subprocess
import json
import time
import sys
"""
MIT License

Copyright (c) 2021 Arun

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


# importing module
import logging

# Create and configure logger
logging.basicConfig(filename="run.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')


# Creating an object
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# To perform all the operations (-one model per building)
prefix = "../../"
with open(prefix + 'config.json', 'r') as file:
    config = json.load(file)

# open and get all the unique buildings:
df = pd.read_csv(prefix + config["data"]["dataset_path"])
uni_b = df["building_id"].unique()
logger.info(f"unique builds : {uni_b}")
l = len(uni_b)
i = 1  # counter for builds


for b_id in df["building_id"].unique():
    logger.info(b_id)
    # Change configs
    config["data"]["only_building"] = int(b_id)

    with open(prefix + 'config.json', 'w') as file:
        json.dump(config, file)

    start_time = time.time()
    subprocess.run([sys.executable, "preprocess.py"])
    end_time = time.time()
    pre_time = end_time - start_time
    logger.info(f"Building {b_id} :: Time taken for preprocessing : {pre_time}")

    starting_time = time.time()
    subprocess.run([sys.executable, "train.py"])
    end_time = time.time()
    train_time = end_time - starting_time
    logger.info(f"Building {b_id} :: Time taken for training : {train_time}")

    start_time = time.time()
    subprocess.run([sys.executable, "test.py"])
    end_time = time.time()
    test_time = end_time - start_time
    logger.info(f"Building {b_id} :: Time taken for test : {test_time}")

    logger.info(f"Processed {i}/{l} buildings ...... Total time taken: {end_time - starting_time} secs")

    i = i + 1
