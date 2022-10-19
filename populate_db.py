from trino.dbapi import connect
import numpy as np
import random

# Connect to Trino instance
conn = connect(
    host="localhost",
    port=8080,
    user="postgres",
    catalog="postgresql",
    schema="public",
)

cur = conn.cursor()

cur.execute("SELECT * FROM public.kmeans")
cur.execute("INSERT INTO public.kmeans(\
	id, cluster_id)\
	VALUES (6, 3);")
cur.fetchall()




cur.execute("CREATE TABLE public.test\
(\
    id bigint,\
    cluster_id integer\
)")
cur.fetchall()

cur.execute("UPDATE postgresql.public.kmeans SET cluster_id = 2")


X = cur.fetchall()

cur.execute(
    f"SELECT * FROM postgresql.information_schema.columns WHERE table_schema = 'public' AND table_name = 'kmeans'")
X = cur.fetchall()
