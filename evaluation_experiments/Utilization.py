import psutil
import time
import os

def get_cpu_utilization():
    return psutil.cpu_percent(interval=1)

def get_memory_usage(pid):
    try:
        # Create a Process object using the current process ID
        process = psutil.Process(pid)

        # Get memory usage in bytes
        memory_info = process.memory_info()

        # Convert memory usage to megabytes for readability
        memory_usage_mb = memory_info.rss / (1024 ** 2)

        return memory_usage_mb

    except Exception as e:
        print(f"Error: {e}")
        return None
    

def main(pid):
    try:
        while True:
            cpu_utilization = get_cpu_utilization()
            #print(f"CPU Utilization: {cpu_utilization}%")
            curr_time = time.strftime("%H:%M:%S", time.localtime())

            memory_usage=get_memory_usage(pid)
            if memory_usage is None:
                memory_usage=0.00

            file1 = open("memory_logs.txt", "a")  # append mode
            file1.write(f"{cpu_utilization}:{curr_time}:{memory_usage:.2f}\n")
            file1.close()

            time.sleep(10)

    except KeyboardInterrupt:
        print("Monitoring stopped.")

if __name__ == "__main__":
    pid = os.getpid()
    main(10556)