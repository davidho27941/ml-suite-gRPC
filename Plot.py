import numpy as np
import matplotlib.pyplot as plt 
import os 

latency_dataset = np.zeros((16,1000))
throughput_dataset = np.zeros((16,1000))

for i in range(0,16):
    log_path = ""
    seq = ("log_batch_", str(i+1), ".txt")
    with open(os.path.join('./log/AWS_localhost/', log_path.join(seq)), 'r', newline="") as f:
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

x = []
for i in range(0,16):
    x.append(i+1)

plt.figure(figsize=(9,6))
plt.errorbar(x, latency_mean, yerr=latency_err, label='Latency')
plt.xlabel('Batch size')
plt.ylabel('Latency')
plt.legend(loc="upper right")
plt.title('Latency')
plt.savefig('latency_AWS_localhost_1000.png')
plt.show()

plt.figure(figsize=(9,6))
plt.errorbar(x, throughput_mean, yerr=throughput_err, label='Throughput(image/s)')
plt.xlabel('Batch size')
plt.ylabel('Throughput')
plt.legend(loc="upper right")
plt.title('Throughput')
plt.savefig("throughput_AWS_localhost_1000.png")
plt.show()