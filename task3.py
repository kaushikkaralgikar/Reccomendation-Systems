import operator
import json
import math
import re
import string
import sys
import time
from pyspark import SparkConf, SparkContext
from itertools import combinations

HASH_FUNC_COUNT = 30
ROWS_IN_BAND = 1

def compute_pearson_correlation(dict1, dict2):

    co_rated_user = list(set(dict1.keys()) & (set(dict2.keys())))
    val1_list, val2_list = list(), list()
    [(val1_list.append(dict1[user_id]),val2_list.append(dict2[user_id])) for user_id in co_rated_user]

    avg1 = sum(val1_list) / len(val1_list)
    avg2 = sum(val2_list) / len(val2_list)

    numerator = sum(map(lambda pair: (pair[0] - avg1) * (pair[1] - avg2), zip(val1_list, val2_list)))

    if numerator == 0:
        return 0
    denominator = math.sqrt(sum(map(lambda val: (val - avg1) ** 2, val1_list))) * \
                  math.sqrt(sum(map(lambda val: (val - avg2) ** 2, val2_list)))
    if denominator == 0:
        return 0

    return numerator / denominator

def convert_to_dict(dict_list):
    final_dict = {}
    for dict_item in dict_list:
        final_dict.update(dict_item)
    return final_dict

def exist_3_co_rated_pairs(dict1, dict2):
    if dict1 is not None and dict2 is not None:
        return True if len(set(dict1.keys()) & set(dict2.keys())) >= 3 else False
    return False

def append (a, b):
    a.append(b)
    return a

def get_signature_matrix(uid_bidx_tuple, m):
    #a = random.sample(range(1, 999), HASH_FUNC_COUNT)
    #b = random.sample(range(0, 999), HASH_FUNC_COUNT)
    #b = 563
    a = [500, 6577, 1162, 2881, 9959, 7060, 8420, 1110, 597, 9269, 9882, 707, 9128, 3563, 3600, 5408, 6009, 8921, 294, 1580, 6177, 5545, 2180, 7042, 6620, 3478, 307, 6097, 7618, 2598]                                                                                                                                                       
    b = [7877, 3819, 4383, 5227, 3033, 9555, 2852, 2611, 9019, 7387, 371, 3610, 6481, 4641, 3895, 6275, 747, 1563, 3391, 8897, 2463, 9454, 2658, 7454, 4491, 5607, 7675, 2700, 8586, 8276]
    
    hash_1 = uid_bidx_tuple.map(lambda x : (x[0], (a[0]*x[1]+b[0]) % m)).groupByKey().mapValues(list).map(lambda x : (x[0], min(x[1])))
    hash_2 = uid_bidx_tuple.map(lambda x : (x[0], (a[1]*x[1]+b[1]) % m)).groupByKey().mapValues(list).map(lambda x : (x[0], min(x[1])))
    sm = hash_1.join(hash_2).mapValues(list)

    for i in range(2,HASH_FUNC_COUNT):
        h = uid_bidx_tuple.map(lambda x : (x[0], (a[i]*x[1]+b[i]) % m)).groupByKey().mapValues(list).map(lambda x : (x[0], min(x[1])))
        sm = sm.join(h).mapValues(list).map(lambda x : (x[0], append(x[1][0], x[1][1])))

    signature_matrix = sm.sortByKey()
    return signature_matrix

def split_bands(minhash_list):
    chunks = list()
    for index,start in enumerate(range(0, HASH_FUNC_COUNT, ROWS_IN_BAND)):
        chunks.append((index, hash(tuple(minhash_list[start:start+ROWS_IN_BAND]))))
    return chunks

def compute_jaccard_similarity(u1_business_list, u2_business_list):

    if len(set(u1_business_list) & set(u2_business_list))>=3:
         if float(float(len(set(u1_business_list) & set(u2_business_list))) / float(len(set(u1_business_list) | set(u2_business_list)))) >=0.01:
            return True
    return False

if __name__ == "__main__":
    start_time = time.time()
    input_file_path = sys.argv[1]
    model_type = sys.argv[2]
    output_file_path = sys.argv[3]

    conf = SparkConf().set("spark.executor.memory", "4g").set("spark.driver.memory", "4g")
    sc = SparkContext(conf=conf)

    train_review_rdd = sc.textFile(input_file_path).map(lambda row: json.loads(row))\
        .map(lambda row:(row["user_id"], row["business_id"],row["stars"]))

    if model_type == "item_based":
        
        business_user_star_map = train_review_rdd.map(lambda row: (row[1], (row[0],row[2])))\
            .groupByKey()\
            .map(lambda x: (x[0],list(set(x[1]))))\
            .mapValues(lambda vals: [{uid_score[0]: uid_score[1]} for uid_score in vals]) \
            .mapValues(lambda val_list: convert_to_dict(val_list))\
            .map(lambda kv: {kv[0]:kv[1]})\
            .flatMap(lambda items: items.items()).collectAsMap()\

        business_pairs = train_review_rdd.map(lambda row: (row[1], row[0]))\
            .groupByKey()\
            .map(lambda x: (x[0], list(set(x[1]))))\
            .filter(lambda x: len(x[1])>=3).map(lambda row: (1,row[0])).groupByKey().map(lambda x: x[1])\
            .flatMap(lambda bid_list: [pair for pair in combinations(bid_list, 2)])
            
        business_pairs_list = business_pairs\
            .filter(lambda x: exist_3_co_rated_pairs(business_user_star_map[x[0]],business_user_star_map[x[1]])) \
            .map(lambda kv: (kv[0], kv[1], compute_pearson_correlation(business_user_star_map[kv[0]], business_user_star_map[kv[1]])))\
            .filter(lambda x: x[2]>0).map(lambda row: {"b1": row[0],"b2":row[1],"sim":row[2]}).collect()
            
        with open(output_file_path, 'w+') as output_file:
            for item in business_pairs_list:
                output_file.writelines(json.dumps(item) + "\n")
            output_file.close()
        print("Duration", time.time()-start_time)

    elif model_type == "user_based":

        user_business_star_map = train_review_rdd.map(lambda row: (row[0], (row[1],row[2])))\
            .groupByKey()\
            .map(lambda x: (x[0],list(set(x[1]))))\
            .mapValues(lambda vals: [{bid_score[0]: bid_score[1]} for bid_score in vals]) \
            .mapValues(lambda val_list: convert_to_dict(val_list))\
            .map(lambda kv: {kv[0]:kv[1]})\
            .flatMap(lambda items: items.items()).collectAsMap()\

        business_index_dict = train_review_rdd.map(lambda json_data: json_data[1]).distinct() \
        .sortBy(lambda item: item).zipWithIndex().map(lambda kv: {kv[0]: kv[1]+1}) \
        .flatMap(lambda items: items.items()).collectAsMap()
    
        m = len(business_index_dict)

        uid_bidx_tuple = train_review_rdd.map(lambda kv: (kv[0],business_index_dict[kv[1]]))

        uid_bidx_dict = train_review_rdd.map(lambda kv: (kv[0],business_index_dict[kv[1]]))\
        .groupByKey().map(lambda bidx_uidxs: {bidx_uidxs[0]: list(set(bidx_uidxs[1]))}) \
        .flatMap(lambda kv_items: kv_items.items()).collectAsMap()

        signature_matrix = get_signature_matrix(uid_bidx_tuple, m)

        candidate_pairs = signature_matrix.flatMap(lambda kv: [(chunk, kv[0]) for chunk in split_bands(kv[1])]) \
        .groupByKey().map(lambda kv: list(kv[1])).filter(lambda val: len(val) > 1) \
        .flatMap(lambda uid_list: [pair for pair in combinations(uid_list, 2)])

        user_pairs_list = candidate_pairs.filter(lambda x: compute_jaccard_similarity(uid_bidx_dict[x[0]], uid_bidx_dict[x[1]]))\
            .filter(lambda x: exist_3_co_rated_pairs(user_business_star_map[x[0]],user_business_star_map[x[1]])) \
            .map(lambda kv: (kv[0], kv[1], compute_pearson_correlation(user_business_star_map[kv[0]], user_business_star_map[kv[1]])))\
            .filter(lambda x: x[2]>0).map(lambda row: {"u1": row[0],"u2":row[1],"sim":row[2]}).collect()

