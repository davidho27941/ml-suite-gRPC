import numpy as np
import matplotlib.pyplot as plt 
import os, glob

prefix = 'parsed_data'
compare_prefix = "../log"

N_EVENTS = 10000
N_TEST = 16
scaling = 10 ** 6
def parse_data(pre, mode):
    if ( mode == "ailab_remote"):
        latency_dataset = np.zeros((N_TEST,N_EVENTS))
        throughput_dataset = np.zeros((N_TEST,N_EVENTS))
        for i in range(0,N_TEST):
            file_path = ""
            seq = ("log_batch_", str(i+1), ".txt")
            with open(os.path.join(pre, mode, file_path.join(seq)), 'r', newline="") as f:
                lines = f.readlines()
                j = 0
                for line in lines:
                    value = [float(s) for s in line.split(",")]
                    #print(value[0], value[1])
                    latency_dataset[i][j] = value[0]  / ((i+1) )
                    throughput_dataset[i][j] = (value[0] / ((i+1) )) ** -1
                    j += 1
    else :
        latency_dataset = np.zeros((N_TEST,N_EVENTS))
        throughput_dataset = np.zeros((N_TEST,N_EVENTS))
        for i in range(0,N_TEST):
            file_path = ""
            seq = ("log_batch_", str(i+1), "_new.txt")
            with open(os.path.join(pre, file_path.join(seq)), 'r', newline="") as f:
                lines = f.readlines()
                j = 0
                for line in lines:
                    value = [float(s) for s in line.split(",")]
                    #print(value[0], value[1])
                    latency_dataset[i][j] = value[0]  / ((i+1) * scaling)
                    throughput_dataset[i][j] = (value[0] / ((i+1) * scaling)) ** -1
                    j += 1
            
    latency_mean = np.zeros(N_TEST)
    throughput_mean = np.zeros(N_TEST)

    for i in range(0,N_TEST):
        latency_mean[i] = latency_dataset[i].mean()
        throughput_mean[i] = throughput_dataset[i].mean()

    latency_err = np.zeros(N_TEST)
    throughput_err = np.zeros(N_TEST)

    for i in range(0,N_TEST): 
        sum_diff_latency, sum_diff_throughpuy = 0, 0
        for j in range(0,N_EVENTS):
            sum_diff_latency += (latency_dataset[i][j] - latency_mean[i]) **2
            sum_diff_throughpuy += (throughput_dataset[i][j] - throughput_mean[i]) **2
        latency_err[i] = np.sqrt((sum_diff_latency)/N_EVENTS)
        throughput_err[i] = np.sqrt((sum_diff_throughpuy)/N_EVENTS)
    return latency_dataset, throughput_dataset, latency_mean, throughput_mean, latency_err, throughput_err

sonic_latency, sonic_throughput, sonic_latency_mean, sonic_throughput_mean, sonic_latency_err, sonic_throughput_err = parse_data(prefix, 'none')
standardlone_latency, standardlone_throughput, standardlone_latency_mean, standardlone_throughput_mean, standardlone_latency_err, standardlone_throughput_err = parse_data(compare_prefix, "ailab_remote")


x = []
for i in range(N_TEST):
    x.append(i+1)


def plot_figure(x_ax, y_ax, err_array, mode, line_label, figname):
    plt.figure(figsize=(9,6))
    plt.errorbar(x_ax, y_ax, yerr=err_array, label=line_label)
    plt.xlabel('Batch size')
    if ( mode =='Latency' ):
        plt.ylabel('Latency (s)')
    else :
        plt.ylabel('Throughput (image/s)')
    plt.legend(loc="upper right")
    plt.title(mode)
    plt.savefig(figname)
    plt.show()

def compare_plot(x_ax, y_ax1, y_ax2, err_array1, err_array2, mode, line_label1, line_label2, figname):
    plt.figure(figsize=(9,6))
    plt.errorbar(x_ax, y_ax1, yerr=err_array1, label=line_label1)
    #plt.errorbar(x_ax, y_ax2, yerr=err_array2, label=line_label2)
    plt.xlabel('Batch size')
    if ( mode =='Latency' ):
        plt.ylabel('Latency (s)')
    else :
        plt.ylabel('Throughput (image/s)')
    plt.legend(loc="upper right")
    plt.title(mode)
    plt.savefig(figname)
    plt.show()
plot_figure(x, sonic_latency_mean, sonic_latency_err, "Latency", "AWS f1.2xlarge, SONIC client", "Sonic_latency_standlone.png")
plot_figure(x, sonic_throughput_mean, sonic_throughput_mean, "Throughput", "AWS f1.2xlarge, SONIC client", "Sonic_Throughput_standlone.png")
compare_plot(x, sonic_latency_mean, standardlone_latency_mean, sonic_latency_err, standardlone_latency_err, "Latency", "AWS f1.2xlarge, SONIC client", 'AWS f1.2xlarge, ailab_remote', "compare_Latency.png")
compare_plot(x, sonic_throughput_mean, standardlone_throughput_mean, sonic_throughput_err, standardlone_throughput_err, "Throughput", "AWS f1.2xlarge, SONIC client", 'AWS f1.2xlarge, ailab_remote', "compare_Throughput.png")