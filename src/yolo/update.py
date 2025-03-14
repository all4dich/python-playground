import csv
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS


# InfluxDB configuration
bucket = "TestBucket3"
org = "KETI"
token = "CPp4oJCt4rnYwHDo"
url = "http://tiburon.keti.re.kr:38086"

# Initialize InfluxDB client
client = InfluxDBClient(url=url, token=token, org=org)
write_api = client.write_api(write_options=SYNCHRONOUS)

# Read the CSV file and write data to InfluxDB
with open("inference_metrics.csv", mode="r") as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        point = Point("inference_metrics") \
            .field("current_time", int(row["current_time"])) \
            .field("inference_time", float(row["inference_time"])) \
            .field("postprocess_time", float(row["postprocess_time"])) \
            .field("preprocess_time", float(row["preprocess_time"])) \
            .time(int(row["current_time"]), WritePrecision.MS)
        write_api.write(bucket=bucket, org=org, record=point)

# Close the client
client.close()