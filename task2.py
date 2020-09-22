import operator
import json
import math
import re
import string
import sys
import time
from pyspark import SparkConf, SparkContext


def split_text_and_remove_stop_words(texts):
    word_list = list()
    for text in texts:
        text = text.translate(str.maketrans('', '', string.digits + string.punctuation))
        word_list.extend(list(filter(lambda word: word not in stop_words_list and word != '' and word not in string.ascii_lowercase,re.split(r"[~\s\r\n]+", text))))
    return word_list

def compute_tf(words_index_list):
    temp_dict = {}
    for word in words_index_list:
        if word not in temp_dict.keys():
            temp_dict[word] = 1
        else:
            temp_dict[word] += 1
    max_word_count = max(temp_dict.items(), key=operator.itemgetter(1))
    #[tuple((tuple((bid,key)), float(value/max_word_count[1]))) for key, value in temp_dict.items()]
    #[tuple((key, float(value/max_word_count[1]))) for key, value in temp_dict.items()]
    return [tuple((key, float(value/max_word_count[1]))) for key, value in temp_dict.items()]

def combine_lists(list1, list2):
    result = list(list1)
    result.extend(list2)
    return result


def convert_to_json_array(data, key):
    result = list()
    if isinstance(data, dict):
        for key, val in data.items():
            result.append({
                key: key,
                "profile": val
            })
    """
    elif isinstance(data, list):
        for kv in data:
            for key, val in kv.items():
                result.append({
                    "type": type,
                    keys[0]: key,
                    keys[1]: val
                })
    """
    return result


if __name__ == '__main__':
    start = time.time()
    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]
    stop_words_file_path = sys.argv[3]

    stop_words_list = list(word.strip() for word in open(stop_words_file_path))

    conf = SparkConf().set("spark.executor.memory", "4g").set("spark.driver.memory", "4g")
    sc = SparkContext(conf=conf)
    model_content = list()

    train_review_rdd = sc.textFile(input_file_path).map(lambda row: json.loads(row))

    business_words = train_review_rdd.map(lambda kv: (kv["business_id"],str(kv["text"].encode('utf-8')).lower())) \
        .groupByKey().mapValues(lambda texts: split_text_and_remove_stop_words(list(texts)))

    total_business_count = business_words.count()  

    business_words_tf = business_words.map(lambda kv:(kv[0],compute_tf(kv[1])))\
    .flatMap(lambda row: [((row[0], tup[0]), tup[1]) for tup in row[1]]).persist()

    business_words_idf = business_words_tf.map(lambda kv: (kv[0][1], kv[0][0])).groupByKey()\
        .mapValues(lambda bids: list(set(bids))) \
            .flatMap(lambda word_bids: [((bid, word_bids[0]),math.log(total_business_count / len(word_bids[1]), 2)) for bid in word_bids[1]])                   

    business_words_tf_idf = business_words_tf.leftOuterJoin(business_words_idf) \
        .mapValues(lambda tf_idf: tf_idf[0] * tf_idf[1]) \
        .map(lambda x: (x[0][0],(x[0][1], x[1]))) \
        .groupByKey().mapValues(lambda val: sorted(list(val), reverse=True,key=lambda item: item[1])[:200]) \
        .mapValues(lambda row: [item[0] for item in row])

    word_index_dict = business_words_tf_idf.flatMap(lambda kv: [(word, 1) for word in kv[1]]) \
        .groupByKey().map(lambda kv: kv[0]).zipWithIndex().map(lambda kv: {kv[0]: kv[1]}) \
        .flatMap(lambda kv_items: kv_items.items()).collectAsMap()

    business_profile = business_words_tf_idf.mapValues(lambda words: [word_index_dict[word] for word in words]) \
        .map(lambda row: {row[0]: row[1]}).flatMap(lambda kv_items: kv_items.items()).collectAsMap()

    
    user_profile = train_review_rdd.map(lambda kv: (kv["user_id"], kv["business_id"])) \
        .groupByKey().map(lambda kv: (kv[0], list(set(kv[1])))) \
        .flatMapValues(lambda ids: [business_profile[id] for id in ids]) \
        .reduceByKey(combine_lists).filter(lambda uid_bids: len(uid_bids[1]) > 1) \
        .map(lambda row: {row[0]: list(set(row[1]))})\
            .flatMap(lambda kv_items: kv_items.items()).collectAsMap()

    with open(output_file_path, 'w+') as output_file:
        for item in convert_to_json_array(business_profile,"business_id"):
            output_file.writelines(json.dumps(item) + "\n")
        for item in convert_to_json_array(user_profile,"user_id"):
            output_file.writelines(json.dumps(item) + "\n")
        output_file.close()

    print("Duration: %d s." % (time.time() - start))