import numpy as np
import matplotlib.pyplot as plt 
import os, glob

prefix = 'log'

def parse_data(mode):
    latency_dataset = np.zeros((16,1000))
    throughput_dataset = np.zeros((16,1000))
    for i in range(0,16):
        file_path = ""
        seq = ("log_batch_", str(i+1), ".txt")
        with open(os.path.join(prefix, mode, file_path.join(seq)), 'r', newline="") as f:
            lines = f.readlines()
            j = 0
            for line in lines:
                value = [float(s) for s in line.split(",")]
                #print(value[0], value[1])
                latency_dataset[i][j] = value[0] 
                throughput_dataset[i][j] = value[1]
                j += 1
            
    latency_mean = np.zeros(16)
    throughput_mean = np.zeros(16)

    for i in range(0,16):
        latency_mean[i] = latency_dataset[i].mean()
        throughput_mean[i] = throughput_dataset[i].mean()

    latency_err = np.zeros(16)
    throughput_err = np.zeros(16)

    for i in range(0,16): 
        sum_diff_latency, sum_diff_throughpuy = 0, 0
        for j in range(0,1000):
            sum_diff_latency += (latency_dataset[i][j] - latency_mean[i]) **2
            sum_diff_throughpuy += (throughput_dataset[i][j] - throughput_mean[i]) **2
        latency_err[i] = np.sqrt((sum_diff_latency)/1000)
        throughput_err[i] = np.sqrt((sum_diff_throughpuy)/1000)
    return latency_dataset, throughput_dataset, latency_mean, throughput_mean, latency_err, throughput_err

localhost_latency, localhost_throughput, localhost_latency_mean, localhost_throughput_mean, localhost_latency_err, localhost_throughput_err = parse_data("AWS_localhost")
ailab_latency, ailab_throughput, ailab_latency_mean, ailab_throughput_mean, ailab_latency_err, ailab_throughput_err = parse_data("ailab_remote")

x = []
for i in range(0,16):
    x.append(i+1)

plt.figure(figsize=(9,6))
plt.errorbar(x, localhost_latency_mean, yerr=localhost_latency_err, label='AWS F1 Localhost')
plt.errorbar(x, ailab_latency_mean, yerr=ailab_latency_err, label='Ailab to F1 instance.')
plt.xlabel('Batch size')
plt.ylabel('Latency (s)')
plt.legend(loc="upper right")
plt.title('Latency')
plt.savefig('latency_1000.png')
plt.show()

plt.figure(figsize=(9,6))
plt.errorbar(x, localhost_throughput_mean, yerr=localhost_throughput_err, label='Localhost')
plt.errorbar(x, ailab_throughput_mean, yerr=ailab_throughput_err, label='Ailab to F1 instance.')
plt.xlabel('Batch size')
plt.ylabel('Throughput (image/s)')
plt.legend(loc="upper right")
plt.title('Throughput')
plt.savefig("throughput_1000.png")
plt.show()

plt.figure(figsize=(9,6))
plt.errorbar(x, localhost_latency_mean, yerr=localhost_latency_err, label='AWS F1 Localhost')
#plt.errorbar(x, ailab_latency_mean, yerr=ailab_latency_err, label='Ailab to F1 instance.')
plt.xlabel('Batch size')
plt.ylabel('Latency (s)')
plt.legend(loc="upper right")
plt.title('Latency')
plt.savefig('latency_AWC_localhost_1000.png')
plt.show()

plt.figure(figsize=(9,6))
#plt.errorbar(x, localhost_latency_mean, yerr=localhost_latency_err, label='Localhost')
plt.errorbar(x, ailab_latency_mean, yerr=ailab_latency_err, label='Ailab to F1 instance.')
plt.xlabel('Batch size')
plt.ylabel('Latency (s)')
plt.legend(loc="upper right")
plt.title('Latency')
plt.savefig('latency_ailab_remote_1000.png')
plt.show()

plt.figure(figsize=(9,6))
plt.errorbar(x, localhost_throughput_mean, yerr=localhost_throughput_err, label='AWS F1 Localhost')
#plt.errorbar(x, ailab_throughput_mean, yerr=ailab_throughput_err, label='Ailab to F1 instance.')
plt.xlabel('Batch size')
plt.ylabel('Throughput (image/s)')
plt.legend(loc="upper right")
plt.title('Throughput')
plt.savefig("throughput_AWS_localhost_1000.png")
plt.show()

plt.figure(figsize=(9,6))
#plt.errorbar(x, localhost_throughput_mean, yerr=localhost_throughput_err, label='Localhost')
plt.errorbar(x, ailab_throughput_mean, yerr=ailab_throughput_err, label='Ailab to F1 instance.')
plt.xlabel('Batch size')
plt.ylabel('Throughput (image/s)')
plt.legend(loc="upper right")
plt.title('Throughput')
plt.savefig("throughput_ailab_remote_1000.png")
plt.show()