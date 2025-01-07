import psutil
import time, os
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS


# InfluxDB configuration
token = "PMs_bIdJc1hNW0IK1AI9m2HZTLDhG9Or9gyMRJzH5UpTP6HQ8_N1i_rk8nyoKDmXeJhmzFAtO5hne5gkn21JPA=="
org = "personal"
bucket = "test-bucket-2"
url = "http://localhost:8086"

# Create InfluxDB client
client = InfluxDBClient(url=url, token=token, org=org)
write_api = client.write_api(write_options=SYNCHRONOUS)

# Function to collect and write CPU usage
def collect_and_write_cpu_usage():
    hostname = os.uname()[1]
    while True:
        cpu_times = psutil.cpu_times_percent(interval=1)
        point = Point("cpu") \
            .tag("hostname", hostname) \
            .field("usage_user", cpu_times.user) \
            .field("usage_system", cpu_times.system) \
            .field("usage_idle", cpu_times.idle) \
            .field("usage_nice", cpu_times.nice) \
            .time(time.time_ns(), WritePrecision.NS)
        write_api.write(bucket=bucket, org=org, record=point)
        time.sleep(5)

if __name__ == "__main__":
    collect_and_write_cpu_usage()