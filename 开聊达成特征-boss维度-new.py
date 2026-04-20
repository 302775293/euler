# coding: utf-8
# boss维度的开聊、达成等特征
from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from pyspark.sql.types import *
import datetime as dt
import sys

yesterday = sys.argv[1]

write_path = 'file:///code/lizhibin/contractExtension/produce_cost_gain_ratio_feas/data/'

output_table = "price_calc.lzb_boss_base_cost_gain_features"

app_name = 'produce lzb_boss_cost_gain_ratio_features'

spark = SparkSession.builder.appName(app_name) \
        .config('spark.executor.memory', '10g') \
        .config('spark.executor.memoryOverhead', '2048') \
        .config('spark.driver.memory', '6g') \
        .config('spark.executor.cores', '4') \
        .config('spark.num.executors', '50') \
        .config('spark.default.parallelism', '600') \
        .config('spark.sql.shuffle.partitions', '600') \
        .config('spark.dynamicAllocation.maxExecutors', '50') \
        .config('spark.shuffle.file.buffer', '128k') \
        .enableHiveSupport() \
        .getOrCreate()


def produce_boss_features():

    main_df = spark.sql('''\
    select
        main.boss_id       as boss_id,
        main.boss_name     as boss_name,
        main.company_id    as company_id,

        nvl(online_cost_30, 0)   as online_cost_30,
        nvl(online_cost_90, 0)   as online_cost_90,
        nvl(online_cost_180, 0)  as online_cost_180,

        nvl(item_success_num_30, 0)   as item_success_num_30,
        nvl(item_success_num_90, 0)   as item_success_num_90,
        nvl(item_success_num_180, 0)  as item_success_num_180,

        nvl(detail_num_30, 0)       as detail_num_30,
        nvl(pas_detail_num_30, 0)   as pas_detail_num_30,
        nvl(chat_num_30, 0)         as chat_num_30,
        nvl(pas_chat_num_30, 0)     as pas_chat_num_30,
        nvl(addf_num_30, 0)         as addf_num_30,
        nvl(pas_addf_num_30, 0)     as pas_addf_num_30,
        nvl(success_num_30, 0)      as success_num_30,
        nvl(pas_success_num_30, 0)  as pas_success_num_30,
        nvl(addf_num_30, 0) + nvl(pas_addf_num_30, 0)         as total_addf_num_30,
        nvl(success_num_30, 0) + nvl(pas_success_num_30, 0)   as total_success_num_30,

        nvl(detail_num_90, 0)       as detail_num_90,
        nvl(pas_detail_num_90, 0)   as pas_detail_num_90,
        nvl(chat_num_90, 0)         as chat_num_90,
        nvl(pas_chat_num_90, 0)     as pas_chat_num_90,
        nvl(addf_num_90, 0)         as addf_num_90,
        nvl(pas_addf_num_90, 0)     as pas_addf_num_90,
        nvl(success_num_90, 0)      as success_num_90,
        nvl(pas_success_num_90, 0)  as pas_success_num_90,
        nvl(addf_num_90, 0) + nvl(pas_addf_num_90, 0)         as total_addf_num_90,
        nvl(success_num_90, 0) + nvl(pas_success_num_90, 0)   as total_success_num_90,

        nvl(detail_num_180, 0)       as detail_num_180,
        nvl(pas_detail_num_180, 0)   as pas_detail_num_180,
        nvl(chat_num_180, 0)         as chat_num_180,
        nvl(pas_chat_num_180, 0)     as pas_chat_num_180,
        nvl(addf_num_180, 0)         as addf_num_180,
        nvl(pas_addf_num_180, 0)     as pas_addf_num_180,
        nvl(success_num_180, 0)      as success_num_180,
        nvl(pas_success_num_180, 0)  as pas_success_num_180,
        nvl(addf_num_180, 0) + nvl(pas_addf_num_180, 0)       as total_addf_num_180,
        nvl(success_num_180, 0) + nvl(pas_success_num_180, 0) as total_success_num_180,

        nvl(online_cost_30_chain, 0)   as online_cost_30_chain,
        nvl(online_cost_90_chain, 0)   as online_cost_90_chain,
        nvl(online_cost_180_chain, 0)  as online_cost_180_chain,

        nvl(online_cost_30_last_year, 0)   as online_cost_30_last_year,
        nvl(online_cost_90_last_year, 0)   as online_cost_90_last_year,
        nvl(online_cost_180_last_year, 0)  as online_cost_180_last_year,

        nvl(item_success_num_30_chain, 0)   as item_success_num_30_chain,
        nvl(item_success_num_90_chain, 0)   as item_success_num_90_chain,
        nvl(item_success_num_180_chain, 0)  as item_success_num_180_chain,

        nvl(item_success_num_30_last_year, 0)   as item_success_num_30_last_year,
        nvl(item_success_num_90_last_year, 0)   as item_success_num_90_last_year,
        nvl(item_success_num_180_last_year, 0)  as item_success_num_180_last_year,

        nvl(detail_num_30_chain, 0)       as detail_num_30_chain,
        nvl(pas_detail_num_30_chain, 0)   as pas_detail_num_30_chain,
        nvl(chat_num_30_chain, 0)         as chat_num_30_chain,
        nvl(pas_chat_num_30_chain, 0)     as pas_chat_num_30_chain,
        nvl(addf_num_30_chain, 0)         as addf_num_30_chain,
        nvl(pas_addf_num_30_chain, 0)     as pas_addf_num_30_chain,
        nvl(success_num_30_chain, 0)      as success_num_30_chain,
        nvl(pas_success_num_30_chain, 0)  as pas_success_num_30_chain,
        nvl(addf_num_30_chain, 0) + nvl(pas_addf_num_30_chain, 0)         as total_addf_num_30_chain,
        nvl(success_num_30_chain, 0) + nvl(pas_success_num_30_chain, 0)   as total_success_num_30_chain,

        nvl(detail_num_90_chain, 0)       as detail_num_90_chain,
        nvl(pas_detail_num_90_chain, 0)   as pas_detail_num_90_chain,
        nvl(chat_num_90_chain, 0)         as chat_num_90_chain,
        nvl(pas_chat_num_90_chain, 0)     as pas_chat_num_90_chain,
        nvl(addf_num_90_chain, 0)         as addf_num_90_chain,
        nvl(pas_addf_num_90_chain, 0)     as pas_addf_num_90_chain,
        nvl(success_num_90_chain, 0)      as success_num_90_chain,
        nvl(pas_success_num_90_chain, 0)  as pas_success_num_90_chain,
        nvl(addf_num_90_chain, 0) + nvl(pas_addf_num_90_chain, 0)         as total_addf_num_90_chain,
        nvl(success_num_90_chain, 0) + nvl(pas_success_num_90_chain, 0)   as total_success_num_90_chain,

        nvl(detail_num_180_chain, 0)       as detail_num_180_chain,
        nvl(pas_detail_num_180_chain, 0)   as pas_detail_num_180_chain,
        nvl(chat_num_180_chain, 0)         as chat_num_180_chain,
        nvl(pas_chat_num_180_chain, 0)     as pas_chat_num_180_chain,
        nvl(addf_num_180_chain, 0)         as addf_num_180_chain,
        nvl(pas_addf_num_180_chain, 0)     as pas_addf_num_180_chain,
        nvl(success_num_180_chain, 0)      as success_num_180_chain,
        nvl(pas_success_num_180_chain, 0)  as pas_success_num_180_chain,
        nvl(addf_num_180_chain, 0) + nvl(pas_addf_num_180_chain, 0)       as total_addf_num_180_chain,
        nvl(success_num_180_chain, 0) + nvl(pas_success_num_180_chain, 0) as total_success_num_180_chain,

        nvl(detail_num_30_last_year, 0)       as detail_num_30_last_year,
        nvl(pas_detail_num_30_last_year, 0)   as pas_detail_num_30_last_year,
        nvl(chat_num_30_last_year, 0)         as chat_num_30_last_year,
        nvl(pas_chat_num_30_last_year, 0)     as pas_chat_num_30_last_year,
        nvl(addf_num_30_last_year, 0)         as addf_num_30_last_year,
        nvl(pas_addf_num_30_last_year, 0)     as pas_addf_num_30_last_year,
        nvl(success_num_30_last_year, 0)      as success_num_30_last_year,
        nvl(pas_success_num_30_last_year, 0)  as pas_success_num_30_last_year,
        nvl(addf_num_30_last_year, 0) + nvl(pas_addf_num_30_last_year, 0)         as total_addf_num_30_last_year,
        nvl(success_num_30_last_year, 0) + nvl(pas_success_num_30_last_year, 0)   as total_success_num_30_last_year,

        nvl(detail_num_90_last_year, 0)       as detail_num_90_last_year,
        nvl(pas_detail_num_90_last_year, 0)   as pas_detail_num_90_last_year,
        nvl(chat_num_90_last_year, 0)         as chat_num_90_last_year,
        nvl(pas_chat_num_90_last_year, 0)     as pas_chat_num_90_last_year,
        nvl(addf_num_90_last_year, 0)         as addf_num_90_last_year,
        nvl(pas_addf_num_90_last_year, 0)     as pas_addf_num_90_last_year,
        nvl(success_num_90_last_year, 0)      as success_num_90_last_year,
        nvl(pas_success_num_90_last_year, 0)  as pas_success_num_90_last_year,
        nvl(addf_num_90_last_year, 0) + nvl(pas_addf_num_90_last_year, 0)         as total_addf_num_90_last_year,
        nvl(success_num_90_last_year, 0) + nvl(pas_success_num_90_last_year, 0)   as total_success_num_90_last_year,

        nvl(detail_num_180_last_year, 0)       as detail_num_180_last_year,
        nvl(pas_detail_num_180_last_year, 0)   as pas_detail_num_180_last_year,
        nvl(chat_num_180_last_year, 0)         as chat_num_180_last_year,
        nvl(pas_chat_num_180_last_year, 0)     as pas_chat_num_180_last_year,
        nvl(addf_num_180_last_year, 0)         as addf_num_180_last_year,
        nvl(pas_addf_num_180_last_year, 0)     as pas_addf_num_180_last_year,
        nvl(success_num_180_last_year, 0)      as success_num_180_last_year,
        nvl(pas_success_num_180_last_year, 0)  as pas_success_num_180_last_year,
        nvl(addf_num_180_last_year, 0) + nvl(pas_addf_num_180_last_year, 0)       as total_addf_num_180_last_year,
        nvl(success_num_180_last_year, 0) + nvl(pas_success_num_180_last_year, 0) as total_success_num_180_last_year,

        nvl(item_use_num_7, 0)   as item_use_num_7,
        nvl(item_use_num_14, 0)  as item_use_num_14,
        nvl(item_use_num_30, 0)  as item_use_num_30,
        nvl(item_use_num_60, 0)  as item_use_num_60,

        nvl(item_use_num_7_chain, 0)   as item_use_num_7_chain,
        nvl(item_use_num_14_chain, 0)  as item_use_num_14_chain,
        nvl(item_use_num_30_chain, 0)  as item_use_num_30_chain,
        nvl(item_use_num_60_chain, 0)  as item_use_num_60_chain,

        nvl(item_use_num_7_last_year, 0)   as item_use_num_7_last_year,
        nvl(item_use_num_14_last_year, 0)  as item_use_num_14_last_year,
        nvl(item_use_num_30_last_year, 0)  as item_use_num_30_last_year,
        nvl(item_use_num_60_last_year, 0)  as item_use_num_60_last_year,

        -- 环比趋势特征（当前值 - 环比值）
        nvl(online_cost_30, 0) - nvl(online_cost_30_chain, 0)     as online_cost_30_chain_trend,
        nvl(online_cost_90, 0) - nvl(online_cost_90_chain, 0)     as online_cost_90_chain_trend,
        nvl(online_cost_180, 0) - nvl(online_cost_180_chain, 0)   as online_cost_180_chain_trend,

        nvl(item_success_num_30, 0) - nvl(item_success_num_30_chain, 0)     as item_success_num_30_chain_trend,
        nvl(item_success_num_90, 0) - nvl(item_success_num_90_chain, 0)     as item_success_num_90_chain_trend,
        nvl(item_success_num_180, 0) - nvl(item_success_num_180_chain, 0)   as item_success_num_180_chain_trend,

        nvl(detail_num_30, 0) - nvl(detail_num_30_chain, 0)               as detail_num_30_chain_trend,
        nvl(pas_detail_num_30, 0) - nvl(pas_detail_num_30_chain, 0)       as pas_detail_num_30_chain_trend,
        nvl(chat_num_30, 0) - nvl(chat_num_30_chain, 0)                   as chat_num_30_chain_trend,
        nvl(pas_chat_num_30, 0) - nvl(pas_chat_num_30_chain, 0)           as pas_chat_num_30_chain_trend,
        nvl(addf_num_30, 0) - nvl(addf_num_30_chain, 0)                   as addf_num_30_chain_trend,
        nvl(pas_addf_num_30, 0) - nvl(pas_addf_num_30_chain, 0)           as pas_addf_num_30_chain_trend,
        nvl(success_num_30, 0) - nvl(success_num_30_chain, 0)             as success_num_30_chain_trend,
        nvl(pas_success_num_30, 0) - nvl(pas_success_num_30_chain, 0)     as pas_success_num_30_chain_trend,
        (nvl(addf_num_30, 0) + nvl(pas_addf_num_30, 0)) - (nvl(addf_num_30_chain, 0) + nvl(pas_addf_num_30_chain, 0))         as total_addf_num_30_chain_trend,
        (nvl(success_num_30, 0) + nvl(pas_success_num_30, 0)) - (nvl(success_num_30_chain, 0) + nvl(pas_success_num_30_chain, 0)) as total_success_num_30_chain_trend,

        nvl(detail_num_90, 0) - nvl(detail_num_90_chain, 0)               as detail_num_90_chain_trend,
        nvl(pas_detail_num_90, 0) - nvl(pas_detail_num_90_chain, 0)       as pas_detail_num_90_chain_trend,
        nvl(chat_num_90, 0) - nvl(chat_num_90_chain, 0)                   as chat_num_90_chain_trend,
        nvl(pas_chat_num_90, 0) - nvl(pas_chat_num_90_chain, 0)           as pas_chat_num_90_chain_trend,
        nvl(addf_num_90, 0) - nvl(addf_num_90_chain, 0)                   as addf_num_90_chain_trend,
        nvl(pas_addf_num_90, 0) - nvl(pas_addf_num_90_chain, 0)           as pas_addf_num_90_chain_trend,
        nvl(success_num_90, 0) - nvl(success_num_90_chain, 0)             as success_num_90_chain_trend,
        nvl(pas_success_num_90, 0) - nvl(pas_success_num_90_chain, 0)     as pas_success_num_90_chain_trend,
        (nvl(addf_num_90, 0) + nvl(pas_addf_num_90, 0)) - (nvl(addf_num_90_chain, 0) + nvl(pas_addf_num_90_chain, 0))         as total_addf_num_90_chain_trend,
        (nvl(success_num_90, 0) + nvl(pas_success_num_90, 0)) - (nvl(success_num_90_chain, 0) + nvl(pas_success_num_90_chain, 0)) as total_success_num_90_chain_trend,

        nvl(detail_num_180, 0) - nvl(detail_num_180_chain, 0)               as detail_num_180_chain_trend,
        nvl(pas_detail_num_180, 0) - nvl(pas_detail_num_180_chain, 0)       as pas_detail_num_180_chain_trend,
        nvl(chat_num_180, 0) - nvl(chat_num_180_chain, 0)                   as chat_num_180_chain_trend,
        nvl(pas_chat_num_180, 0) - nvl(pas_chat_num_180_chain, 0)           as pas_chat_num_180_chain_trend,
        nvl(addf_num_180, 0) - nvl(addf_num_180_chain, 0)                   as addf_num_180_chain_trend,
        nvl(pas_addf_num_180, 0) - nvl(pas_addf_num_180_chain, 0)           as pas_addf_num_180_chain_trend,
        nvl(success_num_180, 0) - nvl(success_num_180_chain, 0)             as success_num_180_chain_trend,
        nvl(pas_success_num_180, 0) - nvl(pas_success_num_180_chain, 0)     as pas_success_num_180_chain_trend,
        (nvl(addf_num_180, 0) + nvl(pas_addf_num_180, 0)) - (nvl(addf_num_180_chain, 0) + nvl(pas_addf_num_180_chain, 0))         as total_addf_num_180_chain_trend,
        (nvl(success_num_180, 0) + nvl(pas_success_num_180, 0)) - (nvl(success_num_180_chain, 0) + nvl(pas_success_num_180_chain, 0)) as total_success_num_180_chain_trend,

        nvl(item_use_num_7, 0) - nvl(item_use_num_7_chain, 0)     as item_use_num_7_chain_trend,
        nvl(item_use_num_14, 0) - nvl(item_use_num_14_chain, 0)   as item_use_num_14_chain_trend,
        nvl(item_use_num_30, 0) - nvl(item_use_num_30_chain, 0)   as item_use_num_30_chain_trend,
        nvl(item_use_num_60, 0) - nvl(item_use_num_60_chain, 0)   as item_use_num_60_chain_trend,

        -- 同比趋势特征（当前值 - 同比值）
        nvl(online_cost_30, 0) - nvl(online_cost_30_last_year, 0)     as online_cost_30_last_year_trend,
        nvl(online_cost_90, 0) - nvl(online_cost_90_last_year, 0)     as online_cost_90_last_year_trend,
        nvl(online_cost_180, 0) - nvl(online_cost_180_last_year, 0)   as online_cost_180_last_year_trend,

        nvl(item_success_num_30, 0) - nvl(item_success_num_30_last_year, 0)     as item_success_num_30_last_year_trend,
        nvl(item_success_num_90, 0) - nvl(item_success_num_90_last_year, 0)     as item_success_num_90_last_year_trend,
        nvl(item_success_num_180, 0) - nvl(item_success_num_180_last_year, 0)   as item_success_num_180_last_year_trend,

        nvl(detail_num_30, 0) - nvl(detail_num_30_last_year, 0)               as detail_num_30_last_year_trend,
        nvl(pas_detail_num_30, 0) - nvl(pas_detail_num_30_last_year, 0)       as pas_detail_num_30_last_year_trend,
        nvl(chat_num_30, 0) - nvl(chat_num_30_last_year, 0)                   as chat_num_30_last_year_trend,
        nvl(pas_chat_num_30, 0) - nvl(pas_chat_num_30_last_year, 0)           as pas_chat_num_30_last_year_trend,
        nvl(addf_num_30, 0) - nvl(addf_num_30_last_year, 0)                   as addf_num_30_last_year_trend,
        nvl(pas_addf_num_30, 0) - nvl(pas_addf_num_30_last_year, 0)           as pas_addf_num_30_last_year_trend,
        nvl(success_num_30, 0) - nvl(success_num_30_last_year, 0)             as success_num_30_last_year_trend,
        nvl(pas_success_num_30, 0) - nvl(pas_success_num_30_last_year, 0)     as pas_success_num_30_last_year_trend,
        (nvl(addf_num_30, 0) + nvl(pas_addf_num_30, 0)) - (nvl(addf_num_30_last_year, 0) + nvl(pas_addf_num_30_last_year, 0))         as total_addf_num_30_last_year_trend,
        (nvl(success_num_30, 0) + nvl(pas_success_num_30, 0)) - (nvl(success_num_30_last_year, 0) + nvl(pas_success_num_30_last_year, 0)) as total_success_num_30_last_year_trend,

        nvl(detail_num_90, 0) - nvl(detail_num_90_last_year, 0)               as detail_num_90_last_year_trend,
        nvl(pas_detail_num_90, 0) - nvl(pas_detail_num_90_last_year, 0)       as pas_detail_num_90_last_year_trend,
        nvl(chat_num_90, 0) - nvl(chat_num_90_last_year, 0)                   as chat_num_90_last_year_trend,
        nvl(pas_chat_num_90, 0) - nvl(pas_chat_num_90_last_year, 0)           as pas_chat_num_90_last_year_trend,
        nvl(addf_num_90, 0) - nvl(addf_num_90_last_year, 0)                   as addf_num_90_last_year_trend,
        nvl(pas_addf_num_90, 0) - nvl(pas_addf_num_90_last_year, 0)           as pas_addf_num_90_last_year_trend,
        nvl(success_num_90, 0) - nvl(success_num_90_last_year, 0)             as success_num_90_last_year_trend,
        nvl(pas_success_num_90, 0) - nvl(pas_success_num_90_last_year, 0)     as pas_success_num_90_last_year_trend,
        (nvl(addf_num_90, 0) + nvl(pas_addf_num_90, 0)) - (nvl(addf_num_90_last_year, 0) + nvl(pas_addf_num_90_last_year, 0))         as total_addf_num_90_last_year_trend,
        (nvl(success_num_90, 0) + nvl(pas_success_num_90, 0)) - (nvl(success_num_90_last_year, 0) + nvl(pas_success_num_90_last_year, 0)) as total_success_num_90_last_year_trend,

        nvl(detail_num_180, 0) - nvl(detail_num_180_last_year, 0)               as detail_num_180_last_year_trend,
        nvl(pas_detail_num_180, 0) - nvl(pas_detail_num_180_last_year, 0)       as pas_detail_num_180_last_year_trend,
        nvl(chat_num_180, 0) - nvl(chat_num_180_last_year, 0)                   as chat_num_180_last_year_trend,
        nvl(pas_chat_num_180, 0) - nvl(pas_chat_num_180_last_year, 0)           as pas_chat_num_180_last_year_trend,
        nvl(addf_num_180, 0) - nvl(addf_num_180_last_year, 0)                   as addf_num_180_last_year_trend,
        nvl(pas_addf_num_180, 0) - nvl(pas_addf_num_180_last_year, 0)           as pas_addf_num_180_last_year_trend,
        nvl(success_num_180, 0) - nvl(success_num_180_last_year, 0)             as success_num_180_last_year_trend,
        nvl(pas_success_num_180, 0) - nvl(pas_success_num_180_last_year, 0)     as pas_success_num_180_last_year_trend,
        (nvl(addf_num_180, 0) + nvl(pas_addf_num_180, 0)) - (nvl(addf_num_180_last_year, 0) + nvl(pas_addf_num_180_last_year, 0))         as total_addf_num_180_last_year_trend,
        (nvl(success_num_180, 0) + nvl(pas_success_num_180, 0)) - (nvl(success_num_180_last_year, 0) + nvl(pas_success_num_180_last_year, 0)) as total_success_num_180_last_year_trend,

        nvl(item_use_num_7, 0) - nvl(item_use_num_7_last_year, 0)     as item_use_num_7_last_year_trend,
        nvl(item_use_num_14, 0) - nvl(item_use_num_14_last_year, 0)   as item_use_num_14_last_year_trend,
        nvl(item_use_num_30, 0) - nvl(item_use_num_30_last_year, 0)   as item_use_num_30_last_year_trend,
        nvl(item_use_num_60, 0) - nvl(item_use_num_60_last_year, 0)   as item_use_num_60_last_year_trend
    from
    (
        select distinct boss_id, boss_name, com_id as company_id
        from dwd_boss_user.dwd_boss_info_ss
        where ds = '{yesterday}'
            and boss_id is not null
            and boss_id > 0
    ) main
    -- 获取线上消费
    left join
    (
        select cast(boss_id as bigint) as boss_id,
            sum(if(datediff('{yesterday}', ds) between 0 and 30, pay_amount, 0))  as online_cost_30,
            sum(if(datediff('{yesterday}', ds) between 0 and 90, pay_amount, 0))  as online_cost_90,
            sum(if(datediff('{yesterday}', ds) between 0 and 180, pay_amount, 0)) as online_cost_180,

            sum(if(datediff(date_sub('{yesterday}', 30), ds) between 0 and 30, pay_amount, 0))   as online_cost_30_chain,
            sum(if(datediff(date_sub('{yesterday}', 90), ds) between 0 and 90, pay_amount, 0))   as online_cost_90_chain,
            sum(if(datediff(date_sub('{yesterday}', 180), ds) between 0 and 180, pay_amount, 0)) as online_cost_180_chain,

            sum(if(datediff(add_months('{yesterday}', -12), ds) between 0 and 30, pay_amount, 0))  as online_cost_30_last_year,
            sum(if(datediff(add_months('{yesterday}', -12), ds) between 0 and 90, pay_amount, 0))  as online_cost_90_last_year,
            sum(if(datediff(add_months('{yesterday}', -12), ds) between 0 and 180, pay_amount, 0)) as online_cost_180_last_year
        from price_calc.boss_company_payment_info_daily
        where
            ((ds between date_sub('{yesterday}', 360) and '{yesterday}')
            or (ds between date_sub(add_months('{yesterday}', -12), 180) and add_months('{yesterday}', -12)))
        group by cast(boss_id as bigint)
    ) t_cost
    on main.boss_id = t_cost.boss_id
    -- 获取四项达成
    left join
    (
        select boss_id,
            sum(if(datediff('{yesterday}', ds) between 0 and 30, item_success_num, 0))  as item_success_num_30,
            sum(if(datediff('{yesterday}', ds) between 0 and 90, item_success_num, 0))  as item_success_num_90,
            sum(if(datediff('{yesterday}', ds) between 0 and 180, item_success_num, 0)) as item_success_num_180,

            sum(if(datediff(date_sub('{yesterday}', 30), ds) between 0 and 30, item_success_num, 0))   as item_success_num_30_chain,
            sum(if(datediff(date_sub('{yesterday}', 90), ds) between 0 and 90, item_success_num, 0))   as item_success_num_90_chain,
            sum(if(datediff(date_sub('{yesterday}', 180), ds) between 0 and 180, item_success_num, 0)) as item_success_num_180_chain,

            sum(if(datediff(add_months('{yesterday}', -12), ds) between 0 and 30, item_success_num, 0))  as item_success_num_30_last_year,
            sum(if(datediff(add_months('{yesterday}', -12), ds) between 0 and 90, item_success_num, 0))  as item_success_num_90_last_year,
            sum(if(datediff(add_months('{yesterday}', -12), ds) between 0 and 180, item_success_num, 0)) as item_success_num_180_last_year
        from
        (
            select boss_id, ds,
                sum(if((is_resume_success=1 or is_weixin_success=1 or is_interview_success=1 or is_mobile_success=1), 1, 0)) as item_success_num
            from dm_boss_biz_rec.item_reach_info
            where
                ((ds between date_sub('{yesterday}', 360) and '{yesterday}')
                or (ds between date_sub(add_months('{yesterday}', -12), 180) and add_months('{yesterday}', -12)))
            group by boss_id, ds
        ) t2
        group by boss_id
    ) t_gain
    on main.boss_id = t_gain.boss_id
    -- 获取开聊、简历达成收获特征
    left join
    (
        select cast(boss_id as bigint) as boss_id,
            sum(if(datediff('{yesterday}', ds) between 0 and 180, detail_num, 0))      as detail_num_180,
            sum(if(datediff('{yesterday}', ds) between 0 and 180, pas_detail_num, 0))  as pas_detail_num_180,
            sum(if(datediff('{yesterday}', ds) between 0 and 180, chat_num, 0))        as chat_num_180,
            sum(if(datediff('{yesterday}', ds) between 0 and 180, pas_chat_num, 0))    as pas_chat_num_180,
            sum(if(datediff('{yesterday}', ds) between 0 and 180, addf_num, 0))        as addf_num_180,
            sum(if(datediff('{yesterday}', ds) between 0 and 180, pas_addf_num, 0))    as pas_addf_num_180,
            sum(if(datediff('{yesterday}', ds) between 0 and 180, success_num, 0))     as success_num_180,
            sum(if(datediff('{yesterday}', ds) between 0 and 180, pas_success_num, 0)) as pas_success_num_180,

            sum(if(datediff('{yesterday}', ds) between 0 and 90, detail_num, 0))       as detail_num_90,
            sum(if(datediff('{yesterday}', ds) between 0 and 90, pas_detail_num, 0))   as pas_detail_num_90,
            sum(if(datediff('{yesterday}', ds) between 0 and 90, chat_num, 0))         as chat_num_90,
            sum(if(datediff('{yesterday}', ds) between 0 and 90, pas_chat_num, 0))     as pas_chat_num_90,
            sum(if(datediff('{yesterday}', ds) between 0 and 90, addf_num, 0))         as addf_num_90,
            sum(if(datediff('{yesterday}', ds) between 0 and 90, pas_addf_num, 0))     as pas_addf_num_90,
            sum(if(datediff('{yesterday}', ds) between 0 and 90, success_num, 0))      as success_num_90,
            sum(if(datediff('{yesterday}', ds) between 0 and 90, pas_success_num, 0))  as pas_success_num_90,

            sum(if(datediff('{yesterday}', ds) between 0 and 30, detail_num, 0))       as detail_num_30,
            sum(if(datediff('{yesterday}', ds) between 0 and 30, pas_detail_num, 0))   as pas_detail_num_30,
            sum(if(datediff('{yesterday}', ds) between 0 and 30, chat_num, 0))         as chat_num_30,
            sum(if(datediff('{yesterday}', ds) between 0 and 30, pas_chat_num, 0))     as pas_chat_num_30,
            sum(if(datediff('{yesterday}', ds) between 0 and 30, addf_num, 0))         as addf_num_30,
            sum(if(datediff('{yesterday}', ds) between 0 and 30, pas_addf_num, 0))     as pas_addf_num_30,
            sum(if(datediff('{yesterday}', ds) between 0 and 30, success_num, 0))      as success_num_30,
            sum(if(datediff('{yesterday}', ds) between 0 and 30, pas_success_num, 0))  as pas_success_num_30,

            sum(if(datediff(date_sub('{yesterday}', 180), ds) between 0 and 180, detail_num, 0))      as detail_num_180_chain,
            sum(if(datediff(date_sub('{yesterday}', 180), ds) between 0 and 180, pas_detail_num, 0))  as pas_detail_num_180_chain,
            sum(if(datediff(date_sub('{yesterday}', 180), ds) between 0 and 180, chat_num, 0))        as chat_num_180_chain,
            sum(if(datediff(date_sub('{yesterday}', 180), ds) between 0 and 180, pas_chat_num, 0))    as pas_chat_num_180_chain,
            sum(if(datediff(date_sub('{yesterday}', 180), ds) between 0 and 180, addf_num, 0))        as addf_num_180_chain,
            sum(if(datediff(date_sub('{yesterday}', 180), ds) between 0 and 180, pas_addf_num, 0))    as pas_addf_num_180_chain,
            sum(if(datediff(date_sub('{yesterday}', 180), ds) between 0 and 180, success_num, 0))     as success_num_180_chain,
            sum(if(datediff(date_sub('{yesterday}', 180), ds) between 0 and 180, pas_success_num, 0)) as pas_success_num_180_chain,

            sum(if(datediff(date_sub('{yesterday}', 90), ds) between 0 and 90, detail_num, 0))        as detail_num_90_chain,
            sum(if(datediff(date_sub('{yesterday}', 90), ds) between 0 and 90, pas_detail_num, 0))    as pas_detail_num_90_chain,
            sum(if(datediff(date_sub('{yesterday}', 90), ds) between 0 and 90, chat_num, 0))          as chat_num_90_chain,
            sum(if(datediff(date_sub('{yesterday}', 90), ds) between 0 and 90, pas_chat_num, 0))      as pas_chat_num_90_chain,
            sum(if(datediff(date_sub('{yesterday}', 90), ds) between 0 and 90, addf_num, 0))          as addf_num_90_chain,
            sum(if(datediff(date_sub('{yesterday}', 90), ds) between 0 and 90, pas_addf_num, 0))      as pas_addf_num_90_chain,
            sum(if(datediff(date_sub('{yesterday}', 90), ds) between 0 and 90, success_num, 0))       as success_num_90_chain,
            sum(if(datediff(date_sub('{yesterday}', 90), ds) between 0 and 90, pas_success_num, 0))   as pas_success_num_90_chain,

            sum(if(datediff(date_sub('{yesterday}', 30), ds) between 0 and 30, detail_num, 0))        as detail_num_30_chain,
            sum(if(datediff(date_sub('{yesterday}', 30), ds) between 0 and 30, pas_detail_num, 0))    as pas_detail_num_30_chain,
            sum(if(datediff(date_sub('{yesterday}', 30), ds) between 0 and 30, chat_num, 0))          as chat_num_30_chain,
            sum(if(datediff(date_sub('{yesterday}', 30), ds) between 0 and 30, pas_chat_num, 0))      as pas_chat_num_30_chain,
            sum(if(datediff(date_sub('{yesterday}', 30), ds) between 0 and 30, addf_num, 0))          as addf_num_30_chain,
            sum(if(datediff(date_sub('{yesterday}', 30), ds) between 0 and 30, pas_addf_num, 0))      as pas_addf_num_30_chain,
            sum(if(datediff(date_sub('{yesterday}', 30), ds) between 0 and 30, success_num, 0))       as success_num_30_chain,
            sum(if(datediff(date_sub('{yesterday}', 30), ds) between 0 and 30, pas_success_num, 0))   as pas_success_num_30_chain,

            sum(if(datediff(add_months('{yesterday}', -12), ds) between 0 and 180, detail_num, 0))      as detail_num_180_last_year,
            sum(if(datediff(add_months('{yesterday}', -12), ds) between 0 and 180, pas_detail_num, 0))  as pas_detail_num_180_last_year,
            sum(if(datediff(add_months('{yesterday}', -12), ds) between 0 and 180, chat_num, 0))        as chat_num_180_last_year,
            sum(if(datediff(add_months('{yesterday}', -12), ds) between 0 and 180, pas_chat_num, 0))    as pas_chat_num_180_last_year,
            sum(if(datediff(add_months('{yesterday}', -12), ds) between 0 and 180, addf_num, 0))        as addf_num_180_last_year,
            sum(if(datediff(add_months('{yesterday}', -12), ds) between 0 and 180, pas_addf_num, 0))    as pas_addf_num_180_last_year,
            sum(if(datediff(add_months('{yesterday}', -12), ds) between 0 and 180, success_num, 0))     as success_num_180_last_year,
            sum(if(datediff(add_months('{yesterday}', -12), ds) between 0 and 180, pas_success_num, 0)) as pas_success_num_180_last_year,

            sum(if(datediff(add_months('{yesterday}', -12), ds) between 0 and 90, detail_num, 0))       as detail_num_90_last_year,
            sum(if(datediff(add_months('{yesterday}', -12), ds) between 0 and 90, pas_detail_num, 0))   as pas_detail_num_90_last_year,
            sum(if(datediff(add_months('{yesterday}', -12), ds) between 0 and 90, chat_num, 0))         as chat_num_90_last_year,
            sum(if(datediff(add_months('{yesterday}', -12), ds) between 0 and 90, pas_chat_num, 0))     as pas_chat_num_90_last_year,
            sum(if(datediff(add_months('{yesterday}', -12), ds) between 0 and 90, addf_num, 0))         as addf_num_90_last_year,
            sum(if(datediff(add_months('{yesterday}', -12), ds) between 0 and 90, pas_addf_num, 0))     as pas_addf_num_90_last_year,
            sum(if(datediff(add_months('{yesterday}', -12), ds) between 0 and 90, success_num, 0))      as success_num_90_last_year,
            sum(if(datediff(add_months('{yesterday}', -12), ds) between 0 and 90, pas_success_num, 0))  as pas_success_num_90_last_year,

            sum(if(datediff(add_months('{yesterday}', -12), ds) between 0 and 30, detail_num, 0))       as detail_num_30_last_year,
            sum(if(datediff(add_months('{yesterday}', -12), ds) between 0 and 30, pas_detail_num, 0))   as pas_detail_num_30_last_year,
            sum(if(datediff(add_months('{yesterday}', -12), ds) between 0 and 30, chat_num, 0))         as chat_num_30_last_year,
            sum(if(datediff(add_months('{yesterday}', -12), ds) between 0 and 30, pas_chat_num, 0))     as pas_chat_num_30_last_year,
            sum(if(datediff(add_months('{yesterday}', -12), ds) between 0 and 30, addf_num, 0))         as addf_num_30_last_year,
            sum(if(datediff(add_months('{yesterday}', -12), ds) between 0 and 30, pas_addf_num, 0))     as pas_addf_num_30_last_year,
            sum(if(datediff(add_months('{yesterday}', -12), ds) between 0 and 30, success_num, 0))      as success_num_30_last_year,
            sum(if(datediff(add_months('{yesterday}', -12), ds) between 0 and 30, pas_success_num, 0))  as pas_success_num_30_last_year
        from
        (
            select com_id as company_id, boss_id, ds,
                sum(detail_geek)         as detail_num,
                sum(detail_boss)         as pas_detail_num,
                sum(active_add)          as addf_num,
                sum(passive_add)         as pas_addf_num,
                sum(add_active_reply)    as chat_num,
                sum(passive_add_reply)   as pas_chat_num,
                sum(active_chat_accept)  as success_num,
                sum(passive_chat_accept) as pas_success_num
            from dws_boss_rpc.dws_boss_crm_report_day
            where
                ((ds between date_sub('{yesterday}', 360) and '{yesterday}')
                or (ds between date_sub(add_months('{yesterday}', -12), 180) and add_months('{yesterday}', -12)))
            group by com_id, boss_id, ds
        ) t2
        group by cast(boss_id as bigint)
    ) t_addf_get
    on main.boss_id = t_addf_get.boss_id
    -- 获取道具使用数量
    left join
    (
        select boss_id,
            sum(if(datediff('{yesterday}', ds) between 0 and 7, item_use_nums, 0))   as item_use_num_7,
            sum(if(datediff('{yesterday}', ds) between 0 and 14, item_use_nums, 0))  as item_use_num_14,
            sum(if(datediff('{yesterday}', ds) between 0 and 30, item_use_nums, 0))  as item_use_num_30,
            sum(if(datediff('{yesterday}', ds) between 0 and 60, item_use_nums, 0))  as item_use_num_60,

            sum(if(datediff(date_sub('{yesterday}', 7), ds) between 0 and 7, item_use_nums, 0))    as item_use_num_7_chain,
            sum(if(datediff(date_sub('{yesterday}', 14), ds) between 0 and 14, item_use_nums, 0))  as item_use_num_14_chain,
            sum(if(datediff(date_sub('{yesterday}', 30), ds) between 0 and 30, item_use_nums, 0))  as item_use_num_30_chain,
            sum(if(datediff(date_sub('{yesterday}', 60), ds) between 0 and 60, item_use_nums, 0))  as item_use_num_60_chain,

            sum(if(datediff(add_months('{yesterday}', -12), ds) between 0 and 7, item_use_nums, 0))   as item_use_num_7_last_year,
            sum(if(datediff(add_months('{yesterday}', -12), ds) between 0 and 14, item_use_nums, 0))  as item_use_num_14_last_year,
            sum(if(datediff(add_months('{yesterday}', -12), ds) between 0 and 30, item_use_nums, 0))  as item_use_num_30_last_year,
            sum(if(datediff(add_months('{yesterday}', -12), ds) between 0 and 60, item_use_nums, 0))  as item_use_num_60_last_year
        from
        (
            select user_id as boss_id,
                substr(cast(use_time as string), 1, 10) as ds,
                count(distinct id) as item_use_nums
            from ods_boss_business.ods_user_item
            where deleted = 0
                and status = 1
                and item_use_status != 0
                and substr(cast(add_time as string), 1, 10) <= '{yesterday}'
                and use_time is not null
                and sale_type in (0, 5)
                and ((substr(cast(use_time as string), 1, 10) between date_sub('{yesterday}', 360) and '{yesterday}')
                    or (substr(cast(use_time as string), 1, 10) between date_sub(add_months('{yesterday}', -12), 180) and add_months('{yesterday}', -12)))
            group by user_id, substr(cast(use_time as string), 1, 10)
        ) t2
        group by boss_id
    ) t_item_use
    on main.boss_id = t_item_use.boss_id
        '''.format(yesterday=yesterday)
    ).persist()

    return main_df


# 历史是否有过合作、当前有没有在合作中、历史合作次数、历史总金额、还有多长时间到期
# 是否301期
def produce_base_renew_feas(main_df):
    main_df.createOrReplaceTempView('mains')

    main_df = spark.sql('''\
        select t1.*,
            nvl(t2.coop_cnts, 0)        as coop_cnts,
            nvl(t2.coop_price, 0)       as coop_price,
            if(t2.coop_cnts > 0, 1, 0)  as history_coop_flag,
            if(t2.coop_flag > 0, 1, 0)  as now_coop_flag,
            if(t2.301_flag > 0, 1, 0)   as 301_flag,
            nvl(t2.finish_days, -9999)   as finish_days
        from
        (
            select *
            from mains
        ) t1
        left join
        (
            select t1.boss_id,
                count(distinct t2.order_number)    as coop_cnts,
                cast(sum(t2.price) / 10000 as bigint) as coop_price,
                sum(coop_flag)   as coop_flag,
                sum(301_flag)    as 301_flag,
                max(finish_days) as finish_days
            from
            (
                select *
                from mains
            ) t1
            left join
            (
                select *,
                    if(contract_end >= '{yesterday}', 1, 0) as coop_flag,
                    if(
                        months_between(
                            concat(substring('{yesterday}', 1, 7), '-01'),
                            concat(substring(contract_end, 1, 7), '-01')
                        ) in (1, 0, -1, -2, -3), 1, 0
                    ) as 301_flag,
                    datediff(contract_end, '{yesterday}') as finish_days
                from price_calc.lzb_company_base_contract_order_final
                where ds = '{yesterday}'
            ) t2
            on t1.company_id = t2.company_id
            group by t1.boss_id
        ) t2
        on t1.boss_id = t2.boss_id
        '''.format(yesterday=yesterday)).persist()

    return main_df


def insert_table(main_df):

    main_df = main_df.repartition(10)

    main_df.createOrReplaceTempView('main')

    spark.sql('''\
        insert overwrite table {output_table} partition(ds='{yesterday}')
        select
            boss_id,
            boss_name,
            company_id,

            online_cost_30,
            online_cost_90,
            online_cost_180,

            item_success_num_30,
            item_success_num_90,
            item_success_num_180,

            detail_num_30,
            pas_detail_num_30,
            chat_num_30,
            pas_chat_num_30,
            addf_num_30,
            pas_addf_num_30,
            success_num_30,
            pas_success_num_30,
            total_addf_num_30,
            total_success_num_30,

            detail_num_90,
            pas_detail_num_90,
            chat_num_90,
            pas_chat_num_90,
            addf_num_90,
            pas_addf_num_90,
            success_num_90,
            pas_success_num_90,
            total_addf_num_90,
            total_success_num_90,

            detail_num_180,
            pas_detail_num_180,
            chat_num_180,
            pas_chat_num_180,
            addf_num_180,
            pas_addf_num_180,
            success_num_180,
            pas_success_num_180,
            total_addf_num_180,
            total_success_num_180,

            coop_cnts,
            coop_price,
            history_coop_flag,
            now_coop_flag,
            301_flag,
            finish_days,

            online_cost_30_chain,
            online_cost_90_chain,
            online_cost_180_chain,
            online_cost_30_last_year,
            online_cost_90_last_year,
            online_cost_180_last_year,

            item_success_num_30_chain,
            item_success_num_90_chain,
            item_success_num_180_chain,
            item_success_num_30_last_year,
            item_success_num_90_last_year,
            item_success_num_180_last_year,

            detail_num_30_chain,
            pas_detail_num_30_chain,
            chat_num_30_chain,
            pas_chat_num_30_chain,
            addf_num_30_chain,
            pas_addf_num_30_chain,
            success_num_30_chain,
            pas_success_num_30_chain,
            total_addf_num_30_chain,
            total_success_num_30_chain,

            detail_num_90_chain,
            pas_detail_num_90_chain,
            chat_num_90_chain,
            pas_chat_num_90_chain,
            addf_num_90_chain,
            pas_addf_num_90_chain,
            success_num_90_chain,
            pas_success_num_90_chain,
            total_addf_num_90_chain,
            total_success_num_90_chain,

            detail_num_180_chain,
            pas_detail_num_180_chain,
            chat_num_180_chain,
            pas_chat_num_180_chain,
            addf_num_180_chain,
            pas_addf_num_180_chain,
            success_num_180_chain,
            pas_success_num_180_chain,
            total_addf_num_180_chain,
            total_success_num_180_chain,

            detail_num_30_last_year,
            pas_detail_num_30_last_year,
            chat_num_30_last_year,
            pas_chat_num_30_last_year,
            addf_num_30_last_year,
            pas_addf_num_30_last_year,
            success_num_30_last_year,
            pas_success_num_30_last_year,
            total_addf_num_30_last_year,
            total_success_num_30_last_year,

            detail_num_90_last_year,
            pas_detail_num_90_last_year,
            chat_num_90_last_year,
            pas_chat_num_90_last_year,
            addf_num_90_last_year,
            pas_addf_num_90_last_year,
            success_num_90_last_year,
            pas_success_num_90_last_year,
            total_addf_num_90_last_year,
            total_success_num_90_last_year,

            detail_num_180_last_year,
            pas_detail_num_180_last_year,
            chat_num_180_last_year,
            pas_chat_num_180_last_year,
            addf_num_180_last_year,
            pas_addf_num_180_last_year,
            success_num_180_last_year,
            pas_success_num_180_last_year,
            total_addf_num_180_last_year,
            total_success_num_180_last_year,

            item_use_num_7,
            item_use_num_14,
            item_use_num_30,
            item_use_num_60,

            item_use_num_7_chain,
            item_use_num_14_chain,
            item_use_num_30_chain,
            item_use_num_60_chain,

            item_use_num_7_last_year,
            item_use_num_14_last_year,
            item_use_num_30_last_year,
            item_use_num_60_last_year,

            -- 环比趋势特征
            online_cost_30_chain_trend,
            online_cost_90_chain_trend,
            online_cost_180_chain_trend,

            item_success_num_30_chain_trend,
            item_success_num_90_chain_trend,
            item_success_num_180_chain_trend,

            detail_num_30_chain_trend,
            pas_detail_num_30_chain_trend,
            chat_num_30_chain_trend,
            pas_chat_num_30_chain_trend,
            addf_num_30_chain_trend,
            pas_addf_num_30_chain_trend,
            success_num_30_chain_trend,
            pas_success_num_30_chain_trend,
            total_addf_num_30_chain_trend,
            total_success_num_30_chain_trend,

            detail_num_90_chain_trend,
            pas_detail_num_90_chain_trend,
            chat_num_90_chain_trend,
            pas_chat_num_90_chain_trend,
            addf_num_90_chain_trend,
            pas_addf_num_90_chain_trend,
            success_num_90_chain_trend,
            pas_success_num_90_chain_trend,
            total_addf_num_90_chain_trend,
            total_success_num_90_chain_trend,

            detail_num_180_chain_trend,
            pas_detail_num_180_chain_trend,
            chat_num_180_chain_trend,
            pas_chat_num_180_chain_trend,
            addf_num_180_chain_trend,
            pas_addf_num_180_chain_trend,
            success_num_180_chain_trend,
            pas_success_num_180_chain_trend,
            total_addf_num_180_chain_trend,
            total_success_num_180_chain_trend,

            item_use_num_7_chain_trend,
            item_use_num_14_chain_trend,
            item_use_num_30_chain_trend,
            item_use_num_60_chain_trend,

            -- 同比趋势特征
            online_cost_30_last_year_trend,
            online_cost_90_last_year_trend,
            online_cost_180_last_year_trend,

            item_success_num_30_last_year_trend,
            item_success_num_90_last_year_trend,
            item_success_num_180_last_year_trend,

            detail_num_30_last_year_trend,
            pas_detail_num_30_last_year_trend,
            chat_num_30_last_year_trend,
            pas_chat_num_30_last_year_trend,
            addf_num_30_last_year_trend,
            pas_addf_num_30_last_year_trend,
            success_num_30_last_year_trend,
            pas_success_num_30_last_year_trend,
            total_addf_num_30_last_year_trend,
            total_success_num_30_last_year_trend,

            detail_num_90_last_year_trend,
            pas_detail_num_90_last_year_trend,
            chat_num_90_last_year_trend,
            pas_chat_num_90_last_year_trend,
            addf_num_90_last_year_trend,
            pas_addf_num_90_last_year_trend,
            success_num_90_last_year_trend,
            pas_success_num_90_last_year_trend,
            total_addf_num_90_last_year_trend,
            total_success_num_90_last_year_trend,

            detail_num_180_last_year_trend,
            pas_detail_num_180_last_year_trend,
            chat_num_180_last_year_trend,
            pas_chat_num_180_last_year_trend,
            addf_num_180_last_year_trend,
            pas_addf_num_180_last_year_trend,
            success_num_180_last_year_trend,
            pas_success_num_180_last_year_trend,
            total_addf_num_180_last_year_trend,
            total_success_num_180_last_year_trend,

            item_use_num_7_last_year_trend,
            item_use_num_14_last_year_trend,
            item_use_num_30_last_year_trend,
            item_use_num_60_last_year_trend
        from main
        '''.format(output_table=output_table, yesterday=yesterday))


if __name__ == '__main__':

    print('get boss features......')
    main_df = produce_boss_features()

    main_df = produce_base_renew_feas(main_df)

    print('insert table...........')
    insert_table(main_df)

    print('all jobs done...........')
