from pyspark import SparkContext, SparkConf
import json
import random
import sys
from itertools import combinations
import time

HASH_FUNC_COUNT = 30
BANDS_COUNT = 30
ROWS_IN_BAND = 1

def append (a, b):
    a.append(b)
    return a

def get_signature_matrix(bidx_uidxs_tuple, m):
    #a = random.sample(range(1, 999), HASH_FUNC_COUNT)
    #b = random.sample(range(0, 999), HASH_FUNC_COUNT)
    #b = 563
    a = [500, 6577, 1162, 2881, 9959, 7060, 8420, 1110, 597, 9269, 9882, 707, 9128, 3563, 3600, 5408, 6009, 8921, 294, 1580, 6177, 5545, 2180, 7042, 6620, 3478, 307, 6097, 7618, 2598]                                                                                                                                                       
    b = [7877, 3819, 4383, 5227, 3033, 9555, 2852, 2611, 9019, 7387, 371, 3610, 6481, 4641, 3895, 6275, 747, 1563, 3391, 8897, 2463, 9454, 2658, 7454, 4491, 5607, 7675, 2700, 8586, 8276]
    
    print("a",a)
    print("b",b)

    hash_1 = bidx_uidxs_tuple.map(lambda x : (x[0], (a[0]*x[1]+b[0]) % m)).groupByKey().mapValues(list).map(lambda x : (x[0], min(x[1])))
    hash_2 = bidx_uidxs_tuple.map(lambda x : (x[0], (a[1]*x[1]+b[1]) % m)).groupByKey().mapValues(list).map(lambda x : (x[0], min(x[1])))
    sm = hash_1.join(hash_2).mapValues(list)

    for i in range(2,HASH_FUNC_COUNT):
        h = bidx_uidxs_tuple.map(lambda x : (x[0], (a[i]*x[1]+b[i]) % m)).groupByKey().mapValues(list).map(lambda x : (x[0], min(x[1])))
        sm = sm.join(h).mapValues(list).map(lambda x : (x[0], append(x[1][0], x[1][1])))

    signature_matrix = sm.sortByKey()
    return signature_matrix

def split_bands(minhash_list):
    chunks = list()
    for index,start in enumerate(range(0, HASH_FUNC_COUNT, ROWS_IN_BAND)):
        chunks.append((index, hash(tuple(minhash_list[start:start+ROWS_IN_BAND]))))
    return chunks

def compute_jaccard_similarity(b1_user_list, b2_user_list):
    return float(float(len(set(b1_user_list) & set(b2_user_list))) / float(len(set(b1_user_list) | set(b2_user_list))))
    
if __name__ == "__main__":
    start_time = time.time()
    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]
    
    configuration = SparkConf()
    configuration.set("spark.driver.memory", "4g")
    configuration.set("spark.executor.memory", "4g")
    sc = SparkContext.getOrCreate(configuration)
    sc.setLogLevel('ERROR')
    
    #sc = SparkContext.getOrCreate()

    train_review_input = sc.textFile(input_file_path).map(lambda row:json.loads(row))\
        .map(lambda json_data:(json_data["user_id"], json_data["business_id"]))

    user_index_dict = train_review_input.map(lambda json_data: json_data[0]).distinct() \
        .sortBy(lambda item: item).zipWithIndex().map(lambda kv: {kv[0]: kv[1]+1}) \
        .flatMap(lambda items: items.items()).collectAsMap()
    
    m = len(user_index_dict)

    #business_index_dict = train_review_input.map(lambda kv: kv[1]).distinct() \
    #    .sortBy(lambda item: item).zipWithIndex().map(lambda kv: {kv[0]: kv[1]+1}) \
    #    .flatMap(lambda kv_items: kv_items.items()).collectAsMap()
    
    #reversed_index_bus_dict = {v: k for k, v in business_index_dict.items()}

    bidx_uidxs_tuple = train_review_input.map(lambda kv: (kv[1],user_index_dict[kv[0]]))#\
        #.groupByKey().mapValues(list)

    bidx_uidxs_dict = train_review_input.map(lambda kv: (kv[1],user_index_dict[kv[0]]))\
        .groupByKey().map(lambda bidx_uidxs: {bidx_uidxs[0]: list(set(bidx_uidxs[1]))}) \
        .flatMap(lambda kv_items: kv_items.items()).collectAsMap()

    signature_matrix = get_signature_matrix(bidx_uidxs_tuple, m)
    #print(signature_matrix)

    candidate_pairs = signature_matrix.flatMap(lambda kv: [(chunk, kv[0]) for chunk in split_bands(kv[1])]) \
        .groupByKey().map(lambda kv: list(kv[1])).filter(lambda val: len(val) > 1) \
        .flatMap(lambda bid_list: [pair for pair in combinations(bid_list, 2)])
    
    jaccard_similarity_list = candidate_pairs.map(lambda x: ((x[0], x[1]),
     compute_jaccard_similarity(bidx_uidxs_dict[x[0]], bidx_uidxs_dict[x[1]]))).filter(lambda val: val[1]>=0.05).distinct().collect()

    print("len",len(jaccard_similarity_list))

    outFile = open(output_file_path, "w")
    jaccard_similarity_list = sorted(jaccard_similarity_list)
    for i in range(1, len(jaccard_similarity_list)):
	    string = '{"b1": "'+str(jaccard_similarity_list[i][0][0]) + '", "b2": "'+str(jaccard_similarity_list[i][0][1]) + '", "sim":' + str(jaccard_similarity_list[i][1]) + "}"+"\n"
	    outFile.write(string)

    print("Duration:", time.time()-start_time)