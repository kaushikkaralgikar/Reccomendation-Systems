import operator
import json
import math
import re
import string
import sys
import time
from pyspark import SparkConf, SparkContext
from itertools import combinations

NEIGHBORS = 3

def get_business_pairs_list(business_id, business_star_list):
    result_list = []
    if business_star_list is not None:
        for pair in business_star_list:
            result_list.append(tuple((((business_id, pair[0]), pair[1]))))
            result_list.append(tuple((((pair[0], business_id), pair[1]))))
    return result_list

def get_business_weights(business_tuple_star_list):
    result_list = []
    for pair in business_tuple_star_list:
        result_list.append(tuple((business_model_map.get(pair[0],0), pair[1])))
    return result_list

def get_business_rating(business_id, weight_star_tuple_list):

    weight_star_tuple_list = sorted(weight_star_tuple_list, key = lambda item: item[0], reverse = True)[:NEIGHBORS]
    numerator = sum(map(lambda item: item[0] * item[1], weight_star_tuple_list))
    if numerator == 0:
        return bus_avg_dict.get(business_id, AVG_BUSINESS_STAR)
    denominator = sum(map(lambda item: abs(item[0]), weight_star_tuple_list))
    if denominator == 0:
        return bus_avg_dict.get(business_id, AVG_BUSINESS_STAR)

    return float(numerator / denominator)

def get_user_pairs_list(user_id, user_star_list):
    result_list = []
    for pair in user_star_list:
        result_list.append(tuple(((user_id, pair[0]), pair[0], pair[1])))
        result_list.append(tuple(((pair[0], user_id), pair[0], pair[1])))
    return result_list

def get_user_weights(user_star_tuple_list):
    result_list = []
    for pair in user_star_tuple_list:
        result_list.append(tuple((user_model_map.get(pair[0], 0), user_avg_dict.get(pair[1], AVG_USER_STAR), pair[2])))
    return result_list

def get_user_rating(user_id, weight_avg_star_tuple_list):

    numerator = sum(map(lambda item: (item[2] - item[1]) * item[0], weight_avg_star_tuple_list))
    if numerator == 0:
        return user_avg_dict.get(user_id, AVG_USER_STAR)
    denominator = sum(map(lambda item: abs(item[0]), weight_avg_star_tuple_list))
    if denominator == 0:
        return user_avg_dict.get(user_id, AVG_USER_STAR)

    return (user_avg_dict.get(user_id, AVG_USER_STAR) + float(numerator / denominator))

if __name__ == "__main__":
    start_time = time.time()
    input_file_path = sys.argv[1]
    model_type = sys.argv[2]
    model_file_path = sys.argv[3]
    test_file_path = sys.argv[4]
    bus_avg_file_path = "../resource/asnlib/publicdata/business_avg.json"
    user_avg_file_path = "../resource/asnlib/publicdata/user_avg.json"
    #output_file_path = sys.argv[4]

    conf = SparkConf().set("spark.executor.memory", "4g").set("spark.driver.memory", "4g")
    sc = SparkContext(conf=conf)

    train_review_rdd = sc.textFile(input_file_path).map(lambda row: json.loads(row))\
        .map(lambda row:(row["user_id"], row["business_id"],row["stars"]))
    
    bus_avg_dict = sc.textFile(bus_avg_file_path).map(lambda row: json.loads(row)) \
            .map(lambda kv: dict(kv)).flatMap(lambda kv_items: kv_items.items()) \
            .collectAsMap()

    user_avg_dict = sc.textFile(user_avg_file_path).map(lambda row: json.loads(row)) \
            .map(lambda kv: dict(kv)).flatMap(lambda kv_items: kv_items.items()) \
            .collectAsMap()

    AVG_BUSINESS_STAR = bus_avg_dict.get("UNK", 3.89)
    AVG_USER_STAR = user_avg_dict.get("UNK", 3.89)

    if model_type == "item_based":
        
        business_model_map = sc.textFile(model_file_path).map(lambda row: json.loads(row))\
            .map(lambda row:(row["b1"], row["b2"],row["sim"]))\
                .map(lambda kv: {(kv[0],kv[1]):kv[2]})\
                .flatMap(lambda items: items.items()).collectAsMap()\
                
        user_business_star_map = train_review_rdd.map(lambda row: (row[0],(row[1],row[2])))\
            .groupByKey().mapValues(lambda x: list(set(x))).map(lambda kv: {kv[0]:kv[1]})\
                .flatMap(lambda items: items.items()).collectAsMap()\

        ratings_list = sc.textFile(test_file_path).map(lambda row: json.loads(row))\
            .map(lambda row:(row["user_id"],row["business_id"]))\
                .map(lambda kv: (kv[0],kv[1],get_business_pairs_list(kv[1], user_business_star_map.get(kv[0], list()))))\
                    .filter(lambda x: x[2] is not None)\
                        .map(lambda kv: (kv[0], kv[1], get_business_weights(kv[2])))\
                            .map(lambda kv: (kv[0], kv[1], get_business_rating(kv[1],kv[2])))\
                                .map(lambda kv: {"user_id": kv[0], "business_id": kv[1], "stars": kv[2]}).collect()

        with open(output_file_path, 'w+') as output_file:
            for item in ratings_list:
                output_file.writelines(json.dumps(item) + "\n")
            output_file.close()
        print("Duration", time.time()-start_time)
        
    elif model_type == "user_based":

        user_model_map = sc.textFile(model_file_path).map(lambda row: json.loads(row))\
            .map(lambda row:(row["u1"], row["u2"],row["sim"]))\
                .map(lambda kv: {(kv[0],kv[1]):kv[2]})\
                .flatMap(lambda items: items.items()).collectAsMap()\
        
        business_user_star_map = train_review_rdd.map(lambda row: (row[1],(row[0],row[2])))\
            .groupByKey().mapValues(lambda x: list(set(x))).map(lambda kv: {kv[0]:kv[1]})\
                .flatMap(lambda items: items.items()).collectAsMap()\
                
        ratings_list = sc.textFile(test_file_path).map(lambda row: json.loads(row))\
            .map(lambda row:(row["user_id"],row["business_id"]))\
                .map(lambda kv: (kv[0],kv[1],get_user_pairs_list(kv[0], business_user_star_map.get(kv[1], list()))))\
                    .filter(lambda x: x[2] is not None)\
                        .map(lambda kv: (kv[0], kv[1], get_user_weights(kv[2])))\
                            .map(lambda kv : (kv[0], kv[1], get_user_rating(kv[0],kv[2])))\
                                .map(lambda kv: {"user_id": kv[0], "business_id": kv[1], "stars": kv[2]}).collect()

        with open(output_file_path, 'w+') as output_file:
            for item in ratings_list:
                output_file.writelines(json.dumps(item) + "\n")
            output_file.close()
        print("Duration", time.time()-start_time)