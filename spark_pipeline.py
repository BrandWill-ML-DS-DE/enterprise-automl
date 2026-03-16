from pyspark.sql import SparkSession

def load_big_data(path):

    spark = SparkSession.builder \
        .appName("EnterpriseAutoML") \
        .getOrCreate()

    df = spark.read.csv(path, header=True, inferSchema=True)

    return df