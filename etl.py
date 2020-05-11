import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format


config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID'] = config['Credentials']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY'] = config['Credentials']['AWS_SECRET_ACCESS_KEY']

def create_spark_session():
    """ Creates a connection to the Spark Cluster

        Arguments:
            None
        Returns:
            spark {SparkObject}: The connection to Spark.
    """
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    """ Processes the song dataset of the ETL Pipeline

        Arguments:
            spark {SparkObject}: The Spark connection object.
            input_data {string}: Location of the data sources, typically an S3 bucket.
            output_data {string}: Location of the process outputs, typically an S3 bucket.
        Returns:
            None
    """
    # get filepath to song data file
    song_data = os.path.join(input_data,"song_data/*/*/*/*.json")
    
    # read song data file
    df = spark.read.json(song_data)

    # extract columns to create songs table
    songs_table = df['song_id', 'title', 'artist_id', 'year', 'duration'] \
                  .dropDuplicates(['song_id'])
    
    # write songs table to parquet files partitioned by year and artist
    print("### Writing songs table...")
    songs_table.write.partitionBy('year', 'artist_id').parquet(os.path.join(output_data, 'songs.parquet'), 'overwrite')

    # extract columns to create artists table
    artists_table = df['artist_id', 'artist_name', 'artist_location', 'artist_latitude', 'artist_longitude'] \
                    .dropDuplicates(['artist_id'])
    
    # write artists table to parquet files
    print("### Writing artists table...")
    artists_table.write.parquet(os.path.join(output_data, 'artists.parquet'), 'overwrite')


def process_log_data(spark, input_data, output_data):
    """ Processes the log dataset of the ETL Pipeline

        Arguments:
            spark {SparkObject}: The Spark connection object.
            input_data {string}: Location of the data sources, typically an S3 bucket.
            output_data {string}: Location of the process outputs, typically an S3 bucket.
        Returns:
            None
    """
    # get filepath to log data file
    log_data = os.path.join(input_data,"log_data/*/*/*.json")

    # read log data file
    df = spark.read.json(log_data)
    
    # filter by actions for song plays
    songplays_table = df['ts', 'userId', 'level','sessionId', 'location', 'userAgent']

    # extract columns for users table    
    users_table = df['userId', 'firstName', 'lastName', 'gender', 'level'] \
                  .dropDuplicates(['userId'])
    
    # write users table to parquet files
    print("### Writing users table...")
    users_table.write.parquet(os.path.join(output_data, 'users.parquet'), 'overwrite')

    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: str(int(int(x) / 1000)))
    df = df.withColumn('timestamp', get_timestamp(df.ts))
    
    # create datetime column from original timestamp column
    get_datetime = udf(lambda x: str(datetime.fromtimestamp(int(x) / 1000.0)))
    df = df.withColumn('datetime', get_datetime(df.ts))
    
    # extract columns to create time table
    time_table = df.select(
        col('datetime').alias('start_time'), 
        hour('datetime').alias('hour'),
        dayofmonth('datetime').alias('day'),
        weekofyear('datetime').alias('week'),
        month('datetime').alias('month'), 
        year('datetime').alias('year')) \
        .dropDuplicates(['start_time'])
    
    # write time table to parquet files partitioned by year and month
    print("### Writing time table...")
    time_table.write.partitionBy('year', 'month').parquet(os.path.join(output_data, 'time.parquet'), 'overwrite')

    # read in song data to use for songplays table
    song_df = spark.read.json(os.path.join(input_data, "song-data/*/*/*/*.json"))

    # extract columns from joined song and log datasets to create songplays table 
    df = df.join(song_df, song_df.title == df.song)
    
    songplays_table = df['userId', 'level', 'song_id', 'artist_id', 'sessionId', 'location', 'userAgent']
    songplays_table.select(monotonically_increasing_id().alias('songplay_id')).collect()

    # write songplays table to parquet files partitioned by year and month
    print("### Writing songplays table...")
    songplays_table.write.partitionBy('year', 'month').parquet(os.path.join(output_data, 'songplays.parquet'), 'overwrite')


def main():
    """ Entry point to the application

        Arguments:
            None
        Returns:
            None
        """
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = "s3a://majala-data-eng/"
    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
