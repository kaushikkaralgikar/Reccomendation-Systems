import operator
import json
import math
import re
import string
import sys
import time
from pyspark import SparkConf, SparkContext

if __name__ == "__main__":
    start_time = time.time()
    model_file_path = sys.argv[1]
    test_file_path = sys.argv[2]

    conf = SparkConf().set("spark.executor.memory", "4g").set("spark.driver.memory", "4g")
    sc = SparkContext(conf=conf)

    model_rdd = sc.textFile(model_file_path).map(lambda row: json.loads(row))
    test_rdd = sc.textFile(test_file_path).map(lambda row: json.loads(row))

    model_business_map = model_rdd.filter(lambda x: x["user_or_b"]=="b")\
        .map(lambda row:{row["business_id"]: row["profile"]})\
            .flatmap(lambda kv_items: kv_items.items()).collectAsMap()

    model_user_map = model_rdd.filter(lambda x: x["user_or_b"]=="u")\
        .map(lambda row:{row["user_id"]: row["profile"]})\
            .flatmap(lambda kv_items: kv_items.items()).collectAsMap()

    test_rdd.map(lambda row:(row["user_id"],row["business_id"]))\
        .map(lambda kv: (kv,model_user_map.get(kv[0],-1),model_business_map.get(kv[1],-1)))\
            .filter(lambda val: (val[0]!=-1) and (val[1]!=-1)).take(1)