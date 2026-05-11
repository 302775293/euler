# coding: utf-8
# boss维度通话特征
from pyspark.sql import SparkSession
import sys

yesterday = sys.argv[1]

output_table = "dm_boss_offcrm.lzb_boss_online_features"

app_name = 'produce lzb_boss_call_features'

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


def produce_call_features():

    main_df = spark.sql('''\
    select
        main.boss_id,

        -- 通话总量
        nvl(call_num_30, 0)   as call_num_30,
        nvl(call_num_90, 0)   as call_num_90,
        nvl(call_num_180, 0)  as call_num_180,

        -- 打通数量（bridge_duration > 0）
        nvl(connect_num_30, 0)   as connect_num_30,
        nvl(connect_num_90, 0)   as connect_num_90,
        nvl(connect_num_180, 0)  as connect_num_180,

        -- 打通占比
        if(nvl(call_num_30, 0) > 0,  nvl(connect_num_30, 0)  / nvl(call_num_30, 0),  null) as call_connect_rate_30,
        if(nvl(call_num_90, 0) > 0,  nvl(connect_num_90, 0)  / nvl(call_num_90, 0),  null) as call_connect_rate_90,
        if(nvl(call_num_180, 0) > 0, nvl(connect_num_180, 0) / nvl(call_num_180, 0), null) as call_connect_rate_180,

        -- 有效通话数量（bridge_duration >= 45s）
        nvl(effective_call_num_30, 0)   as effective_call_num_30,
        nvl(effective_call_num_90, 0)   as effective_call_num_90,
        nvl(effective_call_num_180, 0)  as effective_call_num_180,

        -- 触发内容风控次数（强拒绝 + 高危）
        nvl(risk_call_num_30, 0)   as risk_call_num_30,
        nvl(risk_call_num_90, 0)   as risk_call_num_90,
        nvl(risk_call_num_180, 0)  as risk_call_num_180,

        -- 各风控类别次数
        nvl(safe_call_num_30, 0)          as safe_call_num_30,
        nvl(safe_call_num_90, 0)          as safe_call_num_90,
        nvl(safe_call_num_180, 0)         as safe_call_num_180,

        nvl(weak_reject_call_num_30, 0)   as weak_reject_call_num_30,
        nvl(weak_reject_call_num_90, 0)   as weak_reject_call_num_90,
        nvl(weak_reject_call_num_180, 0)  as weak_reject_call_num_180,

        nvl(strong_reject_call_num_30, 0)  as strong_reject_call_num_30,
        nvl(strong_reject_call_num_90, 0)  as strong_reject_call_num_90,
        nvl(strong_reject_call_num_180, 0) as strong_reject_call_num_180,

        nvl(high_risk_call_num_30, 0)   as high_risk_call_num_30,
        nvl(high_risk_call_num_90, 0)   as high_risk_call_num_90,
        nvl(high_risk_call_num_180, 0)  as high_risk_call_num_180
    from
    (
        -- 主表：当前所有boss
        select distinct boss_id
        from dwd_boss_user.dwd_boss_info_ss
        where ds = '{yesterday}'
            and boss_id is not null
            and boss_id > 0
    ) main
    -- 通话基础特征：通话量、打通量、有效通话量
    left join
    (
        select boss_id,
            count(if(datediff('{yesterday}', ds) between 0 and 30,  1, null)) as call_num_30,
            count(if(datediff('{yesterday}', ds) between 0 and 90,  1, null)) as call_num_90,
            count(if(datediff('{yesterday}', ds) between 0 and 180, 1, null)) as call_num_180,

            count(if(datediff('{yesterday}', ds) between 0 and 30  and bridge_duration > 0, 1, null)) as connect_num_30,
            count(if(datediff('{yesterday}', ds) between 0 and 90  and bridge_duration > 0, 1, null)) as connect_num_90,
            count(if(datediff('{yesterday}', ds) between 0 and 180 and bridge_duration > 0, 1, null)) as connect_num_180,

            count(if(datediff('{yesterday}', ds) between 0 and 30  and bridge_duration >= 45, 1, null)) as effective_call_num_30,
            count(if(datediff('{yesterday}', ds) between 0 and 90  and bridge_duration >= 45, 1, null)) as effective_call_num_90,
            count(if(datediff('{yesterday}', ds) between 0 and 180 and bridge_duration >= 45, 1, null)) as effective_call_num_180
        from dwd_boss_crm.dwd_call_center_record_md5
        where ds between date_sub('{yesterday}', 180) and '{yesterday}'
        group by boss_id
    ) t_call
    on main.boss_id = t_call.boss_id
    -- 风控特征：基于打通电话关联风控结果
    left join
    (
        select boss_id,
            count(if(datediff('{yesterday}', ds) between 0 and 30  and risk_label in ('强拒绝', '高危'), 1, null)) as risk_call_num_30,
            count(if(datediff('{yesterday}', ds) between 0 and 90  and risk_label in ('强拒绝', '高危'), 1, null)) as risk_call_num_90,
            count(if(datediff('{yesterday}', ds) between 0 and 180 and risk_label in ('强拒绝', '高危'), 1, null)) as risk_call_num_180,

            count(if(datediff('{yesterday}', ds) between 0 and 30  and risk_label = '安全', 1, null)) as safe_call_num_30,
            count(if(datediff('{yesterday}', ds) between 0 and 90  and risk_label = '安全', 1, null)) as safe_call_num_90,
            count(if(datediff('{yesterday}', ds) between 0 and 180 and risk_label = '安全', 1, null)) as safe_call_num_180,

            count(if(datediff('{yesterday}', ds) between 0 and 30  and risk_label = '弱拒绝', 1, null)) as weak_reject_call_num_30,
            count(if(datediff('{yesterday}', ds) between 0 and 90  and risk_label = '弱拒绝', 1, null)) as weak_reject_call_num_90,
            count(if(datediff('{yesterday}', ds) between 0 and 180 and risk_label = '弱拒绝', 1, null)) as weak_reject_call_num_180,

            count(if(datediff('{yesterday}', ds) between 0 and 30  and risk_label = '强拒绝', 1, null)) as strong_reject_call_num_30,
            count(if(datediff('{yesterday}', ds) between 0 and 90  and risk_label = '强拒绝', 1, null)) as strong_reject_call_num_90,
            count(if(datediff('{yesterday}', ds) between 0 and 180 and risk_label = '强拒绝', 1, null)) as strong_reject_call_num_180,

            count(if(datediff('{yesterday}', ds) between 0 and 30  and risk_label = '高危', 1, null)) as high_risk_call_num_30,
            count(if(datediff('{yesterday}', ds) between 0 and 90  and risk_label = '高危', 1, null)) as high_risk_call_num_90,
            count(if(datediff('{yesterday}', ds) between 0 and 180 and risk_label = '高危', 1, null)) as high_risk_call_num_180
        from
        (
            -- 打通的通话记录，关联风控结果
            select t_call.boss_id,
                t_call.ds,
                split(t_risk.cate_13_results, '-')[0] as risk_label
            from
            (
                select boss_id, ds, call_id
                from dwd_boss_crm.dwd_call_center_record_md5
                where ds between date_sub('{yesterday}', 180) and '{yesterday}'
                    and bridge_duration > 0
            ) t_call
            inner join
            (
                select call_id, cate_13_results
                from dm_boss_offcrm.lzb_crm_risk_sample_detail_v3
            ) t_risk
            on t_call.call_id = t_risk.call_id
        ) t
        group by boss_id
    ) t_risk
    on main.boss_id = t_risk.boss_id
        '''.format(yesterday=yesterday)
    ).persist()

    return main_df


def insert_table(main_df):

    main_df = main_df.repartition(10)

    main_df.createOrReplaceTempView('main')

    spark.sql('''\
        insert overwrite table {output_table} partition(ds='{yesterday}')
        select
            boss_id,

            call_num_30,
            call_num_90,
            call_num_180,

            connect_num_30,
            connect_num_90,
            connect_num_180,

            call_connect_rate_30,
            call_connect_rate_90,
            call_connect_rate_180,

            effective_call_num_30,
            effective_call_num_90,
            effective_call_num_180,

            risk_call_num_30,
            risk_call_num_90,
            risk_call_num_180,

            safe_call_num_30,
            safe_call_num_90,
            safe_call_num_180,

            weak_reject_call_num_30,
            weak_reject_call_num_90,
            weak_reject_call_num_180,

            strong_reject_call_num_30,
            strong_reject_call_num_90,
            strong_reject_call_num_180,

            high_risk_call_num_30,
            high_risk_call_num_90,
            high_risk_call_num_180
        from main
        '''.format(output_table=output_table, yesterday=yesterday))


if __name__ == '__main__':

    print('produce call features......')
    main_df = produce_call_features()

    print('insert table...........')
    insert_table(main_df)

    print('all jobs done...........')
