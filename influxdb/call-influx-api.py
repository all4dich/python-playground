import influxdb_client, os, time
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

token = "PMs_bIdJc1hNW0IK1AI9m2HZTLDhG9Or9gyMRJzH5UpTP6HQ8_N1i_rk8nyoKDmXeJhmzFAtO5hne5gkn21JPA=="
org = "personal"
url = "http://localhost:8086"

query_client = InfluxDBClient(url=url, token=token, org=org)
query_api = query_client.query_api()

query = """from(bucket: "home")
 |> range(start: -5y)"""

tables = query_api.query(query, org="personal")
i = 1
for table in tables:
  for record in table.records:
    print(record)