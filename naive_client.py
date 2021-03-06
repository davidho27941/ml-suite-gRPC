from __future__ import print_function

import inference_server_pb2_grpc

import request_wrapper
import sys
import grpc
import numpy as np
import time

# gRPC server info

SERVER_ADDRESS = "18.236.140.120"
SERVER_PORT = 5000

# Number of dummy images to send
N_DUMMY_IMAGES = 1000
NUM_OF_CALL = 1000 

INPUT_NODE_NAME = "data"
OUTPUT_NODE_NAME = "fc1000/Reshape_output"

STACK = False
BATCH_SIZE = 16

IMAGE_LIST = "~/CK-TOOLS/dataset-imagenet-ilsvrc2012-aux/val.txt"
IMAGE_DIR = "~/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min"



def empty_image_generator(n):
    '''
    Generate empty images

    n: number of images
    '''
    for _ in range(n // BATCH_SIZE):
        if STACK:
            request = {INPUT_NODE_NAME: np.zeros((BATCH_SIZE, 224, 224), dtype=np.float32)}
        else:
            request = {INPUT_NODE_NAME: np.zeros((BATCH_SIZE, 3, 224, 224), dtype=np.float32)}
        request = request_wrapper.dictToProto(request)
        yield request


def dummy_client(n, print_interval=50):
    '''
    Start a dummy client

    n: number of images to send
    print_interval: print a number after this number of images is done
    '''
    print("Dummy client sending {n} images...".format(n=n))
    print("gRPC streaming disabled, batch size {batch}".format(batch=BATCH_SIZE))

    start_time = time.time()
    # Connect to server
    with grpc.insecure_channel('{address}:{port}'.format(address=SERVER_ADDRESS,
                                                         port=SERVER_PORT)) as channel:
        stub = inference_server_pb2_grpc.InferenceStub(channel)
        single_time = []
        speed = []
        # Make a call
        for i in range(NUM_OF_CALL):
            duration_start = time.time()
            responses = stub.Inference(empty_image_generator(BATCH_SIZE))
            responses = list(responses)
            duration_end = time.time() - duration_start
            latency = duration_end / BATCH_SIZE
            single_time.append( latency )
            speed.append( latency ** -1  )
            print("Request {0}/1000, {1} images finished in {2} seconds, speed: {3} images/s".format(i, BATCH_SIZE, BATCH_SIZE*latency, latency ** -1))
        with open('./log/ailab_remote/log_batch_{0}.txt'.format(BATCH_SIZE), 'w') as f:
            for i in range(len(single_time)):
                f.writelines("{0:.3f}, {1:.3f}\n".format(single_time[i], speed[i]))
    total_time = time.time() - start_time
    print("{n} images in {time} seconds ({speed} images/s)"
          .format(n=NUM_OF_CALL * BATCH_SIZE,
                  time=total_time,
                  speed=float(n) / total_time))
    

if __name__ == '__main__':
    
    dummy_client(N_DUMMY_IMAGES)
