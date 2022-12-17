# -*- coding: utf-8 -*-

import os
import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
import numpy as np
from datetime import datetime, timedelta
from transformers import logging
from apscheduler.schedulers.background import BlockingScheduler
import pymysql
import re

sched = BlockingScheduler(timezone='Asia/Seoul')
logging.set_verbosity_error()
pd.set_option('mode.chained_assignment',  None)
p = re.compile(r"\[\s*[0-9]{1,2}\s*\:\s*[0-9]{1,2}\s*\].*?(?=\;)", re.DOTALL)
date_pattern = re.compile(r"\[\s*[0-9]{1,2}\s*\:\s*[0-9]{1,2}\s*\]", re.DOTALL)

DATA_TYPE = "emr_NSGREC"
MAX_SEQ_LEN = 650

HUGGINGFACE_MODEL_PATH = os.path.abspath("./model/klue-bert-base")
FINE_TUNED_MODEL_PATH = os.path.abspath(
    "./model/fine-tuned-klue-bert-base-32/fine-tuned-klue-bert-base-32")


class TFBertForSequenceClassification(tf.keras.Model):
    def __init__(self, HUGGINGFACE_MODEL_PATH):
        super(TFBertForSequenceClassification, self).__init__()
        self.bert = TFBertModel.from_pretrained(
            HUGGINGFACE_MODEL_PATH, from_pt=True)
        self.classifier = tf.keras.layers.Dense(1,
                                                kernel_initializer=tf.keras.initializers.TruncatedNormal(
                                                    0.02),
                                                activation='sigmoid',
                                                name='classifier')

    def call(self, inputs):
        input_ids, attention_mask, token_type_ids = inputs
        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_token = outputs[1]
        prediction = self.classifier(cls_token)

        return prediction


def convert_kst(utc_string):
    dt_tm_utc = datetime.strptime(utc_string, '%Y%m%d%H')
    tm_kst = dt_tm_utc + timedelta(hours=9)  # +9 시간
    str_datetime = tm_kst.strftime('%Y%m%d%H')

    return str_datetime


def convert_examples_to_features(examples, max_seq_len, tokenizer):

    input_ids, attention_masks, token_type_ids, data_labels = [], [], [], []

    for example in examples:
        input_id = tokenizer.encode(
            example, truncation=True, max_length=max_seq_len, padding='max_length')

        padding_count = input_id.count(tokenizer.pad_token_id)
        attention_mask = [1] * \
            (max_seq_len - padding_count) + [0] * padding_count

        token_type_id = [0] * max_seq_len

        assert len(input_id) == max_seq_len, "Error with input length {} vs {}".format(
            len(input_id), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(
            len(attention_mask), max_seq_len)
        assert len(token_type_id) == max_seq_len, "Error with token type length {} vs {}".format(
            len(token_type_id), max_seq_len)

        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        token_type_ids.append(token_type_id)

    input_ids = np.array(input_ids, dtype=int)
    attention_masks = np.array(attention_masks, dtype=int)
    token_type_ids = np.array(token_type_ids, dtype=int)

    return (input_ids, attention_masks, token_type_ids)


def inference_from_df(df):
    tokenizer = BertTokenizer.from_pretrained(
        HUGGINGFACE_MODEL_PATH, from_pt=True)
    data_X = convert_examples_to_features(
        df["nsgrec"], max_seq_len=MAX_SEQ_LEN, tokenizer=tokenizer)

    model = TFBertForSequenceClassification(HUGGINGFACE_MODEL_PATH)
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    loss = tf.keras.losses.BinaryCrossentropy()
    model.compile(optimizer=optimizer, loss=loss, metrics=[
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.FalsePositives(name='false_positives'),
        tf.keras.metrics.FalseNegatives(name='false_negatives'),
        'accuracy'],
    )
    model.load_weights(FINE_TUNED_MODEL_PATH)

    predict_y = model.predict(data_X, batch_size=64)
    predict_results = []
    for i in predict_y:
        if i[0] >= 0.5:
            predict_results.append(1)
        else:
            predict_results.append(0)

    df["fallresult"] = predict_results
    now = datetime.now()
    nowdate_kst = now.strftime('%Y%m%d%H')
    # nowdate_kst = convert_kst(now.strftime('%Y%m%d%H')) # 한국 시간이 적용되지 않을 경우, 바로 윗 코드 대신 사용
    df["registerdate"] = [nowdate_kst for _ in range(len(df))]

    return df


def import_from_db(db_conn):
    today = datetime.now()
    today_kst = today.strftime('%Y%m%d%H')
    # today_kst = convert_kst(today.strftime('%Y%m%d%H')) # 한국 시간이 적용되지 않을 경우, 바로 윗 코드 대신 사용
    yesterday = datetime.now() - timedelta(1)
    yesterday_kst = yesterday.strftime('%Y%m%d%H')
    # yesterday_kst = convert_kst(yesterday.strftime('%Y%m%d%H')) # 한국 시간이 적용되지 않을 경우, 바로 윗 코드 대신 사용

    SQL = f"SELECT inputdate, ptid, emr_NSGREC FROM emr_daily_input WHERE CAST(inputdate AS SIGNED)>={yesterday_kst} AND CAST(inputdate AS SIGNED)<{today_kst};"
    dataframe = pd.read_sql(SQL, db_conn)

    return dataframe


def split_record(dataframe):
    dataframe["nsgrec"] = [np.nan for i in range(len(dataframe))]
    dataframe["삭제여부"] = [np.nan for i in range(len(dataframe))]
    dataframe["idx"] = [i for i in range(len(dataframe))]

    length = len(dataframe)
    for i in range(length):
        emr_NSGREC = dataframe[DATA_TYPE][i]

        m = date_pattern.match(emr_NSGREC)
        if m:  # Match found
            splited_record_list = p.findall(emr_NSGREC)
            try:
                splited_record_list.remove('')
            except:
                pass
            for j in range(len(splited_record_list)):  # cleansing date
                splited_record_list[j] = re.sub(
                    r"\[\s*[0-9]{1,2}\s*\:\s*[0-9]{1,2}\s*\]", "", splited_record_list[j]).strip()
            if len(splited_record_list) == 1:  # 한 행에 하나의 레코드만 있는 경우
                dataframe["삭제여부"][i] = 1
                dataframe["nsgrec"][i] = splited_record_list[0]
            elif len(splited_record_list) > 1:  # 한 행에 두 개 이상의 레코드가 있는 경우
                dataframe["삭제여부"][i] = 0
                for idx in range(len(splited_record_list)):
                    rec = splited_record_list[idx]
                    new_row = dataframe.loc[i]
                    new_row['삭제여부'] = idx+1  # 0부터 시작하지 않고, 1부터 시작하도록 1을 더함
                    new_row['nsgrec'] = rec
                    dataframe.loc[len(dataframe)] = new_row
        else:
            dataframe["삭제여부"][i] = 1
            dataframe["nsgrec"][i] = emr_NSGREC

    dataframe = dataframe.drop(dataframe[dataframe['삭제여부'] == 0].index)
    dataframe = dataframe.sort_values(['idx', '삭제여부'])
    dataframe = dataframe.fillna(0)
    return dataframe.drop([DATA_TYPE, 'idx', '삭제여부'], axis=1)


def export_to_db(df, db_conn):
    cursor = db_conn.cursor()
    for i in list(df.index.values):
        # ("nlp_fallresult_id", "ptid", "inputdate", "nsgrec", "registerdate", "fallresult")
        SQL = f'INSERT INTO nlp_fallresult VALUES (NULL, "{df["ptid"][i]}", "{df["inputdate"][i]}", "{df["nsgrec"][i]}", "{df["registerdate"][i]}", "{df["fallresult"][i]}");'
        cursor.execute(SQL)
    db_conn.commit()
    db_conn.close()


# 매일 정해진 hour, minute 마다 실행
@sched.scheduled_job('cron', hour='00', minute='05', id='predict_from_db')
def job():
    host_name = "127.0.0.1"
    host_port = 3306
    username = "root"
    password = "P@$$W0RD"
    database_name = "neticaj_fall"

    db_conn = pymysql.connect(
        host=host_name,     # MySQL Server Address
        port=host_port,     # MySQL Server Port
        user=username,      # MySQL username
        passwd=password,    # password for MySQL username
        db=database_name,   # Database name
        charset='utf8'
    )

    dataframe = import_from_db(db_conn)

    df = split_record(dataframe)
    inferenced_df = inference_from_df(df)

    export_to_db(inferenced_df, db_conn)
    print("\n==============================\nThe job was performed successfully!\n == == == == == == == == == == == == == == ==\n")


def main():
    print("Program Start")
    # job()

    sched.start()


if __name__ == "__main__":
    main()
