from trino.dbapi import connect
import numpy as np
import random

# Connect to Trino instance
conn = connect(
    host="localhost",
    port=8080,
    user="admin",
    catalog="tpch",
    schema="tiny",
)


def k_means(cur):
    while True:
        # assign clusters to datapoints
        cur.execute("UPDATE km_data d \
                    SET cluster_id = (\
                            SELECT id FROM km_clusters c\
                            ORDER BY POW(d.lat-c.lat,2)+POW(d.lng-c.lng,2)\
                            ASC LIMIT 1);")
        # calculate new cluster center
        cur.execute("UPDATE km_clusters C, (\
                                            SELECT cluster_id, AVG(lat) AS lat, AVG(lng) AS lng\
                                            FROM km_data\
                                            GROUP BY cluster_id) D\
                    SET C.lat=D.lat, C.lng=D.lng\
                    WHERE C.id=D.cluster_id;")


numerical_columns = "orderkey, partkey, suppkey, linenumber, quantity, extendedprice, discount, tax"
cur = conn.cursor()
cur.execute(f"SELECT {numerical_columns} FROM tpch.tiny.lineitem")
