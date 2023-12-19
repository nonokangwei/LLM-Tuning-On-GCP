# Single, Mirrored and MultiWorker Distributed Training
import argparse
import os
import sys
import logging
import time
import json

logging.info('test logging')

cluster_spec_str = os.getenv("CLUSTER_SPEC", "")

master_ip_address = os.getenv("MASTER_ADDR", "")

master_port = os.getenv("MASTER_PORT", "")

if cluster_spec_str:
    logging.info(json.loads(cluster_spec_str))
    logging.info(cluster_spec_str)
    logging.info("Master address is: " + master_ip_address)
    logging.info("Master port is: " + master_port)
    
while True:
    time.sleep(5)
