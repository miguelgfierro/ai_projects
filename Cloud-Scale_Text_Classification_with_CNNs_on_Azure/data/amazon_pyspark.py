# attribution: https://github.com/ilkarman/NLP-Sentiment/

# PySpark to create amazon data-set
# train_amazon.json and test_amazon.json
# Each have around 5.4 million good and 5.4 million bad reviews

# paths (external blob)
blob = "wasb://newamazon@amazonsentimenik.blob.core.windows.net"
json_dta = blob + "/aggressive_dedup.json"

# load data
jsonFile = sqlContext.read.json(json_dta)
jsonFile.registerTempTable("reviews")

# Attach the summary to the review-text
# Split into {1,2}, {3}, {4,5} stars
reviews = sqlContext.sql(
    "SELECT "
    + "CASE WHEN overall < 3 THEN 0 "
    + "WHEN overall > 3 THEN 1 ELSE -1 END as label, "
    + "CONCAT(summary, ' ', reviewText) as sentences "
    + "FROM reviews"
)

# Some very basic cleaning
from pyspark.sql.functions import UserDefinedFunction, col
from pyspark.sql.types import StringType, BooleanType
from bs4 import BeautifulSoup


def cleanerHTML(line):
    # html formatting
    html_clean = BeautifulSoup(line, "lxml").get_text().lower()
    # remove any double spaces, line-breaks, etc.
    return " ".join(html_clean.split())


cleaner = UserDefinedFunction(cleanerHTML, StringType())
reviews = reviews.select(reviews.label, cleaner(reviews.sentences).alias("sentences"))


def longEnough(line, chars=100):
    return len(line) > chars


minlength = UserDefinedFunction(longEnough, BooleanType())
reviews = reviews.where(minlength(col("sentences")))

tally = reviews.groupBy("label").count()
tally.show()

# +-----+--------+
# |label|   count|
# +-----+--------+
# |   -1| 6936883|
# |    0|10811098|
# |    1|63340775|
# +-----+--------+

# Equalise classes
# Random sample from positive limited by negative
# Split into 50% 50% for train-test
# Save locally
neg_rev_train, neg_rev_test = reviews.filter("label = 0").randomSplit([0.5, 0.5])

neg_rev_train.count()
neg_rev_test.count()  # 5,403,897

sample_ratio = float(10811098) / float(63340775)
print(sample_ratio)  # 0.17
good_reviews_random = reviews.filter("label = 1").sample(False, sample_ratio, 12345)

pos_rev_train, pos_rev_test = good_reviews_random.randomSplit([0.5, 0.5])

pos_rev_train.count()
pos_rev_test.count()  # 5,403,929

# Save train data
pos_rev_train.unionAll(neg_rev_train).write.json("train_amazon.json")

# Save test data
pos_rev_test.unionAll(neg_rev_test).write.json("test_amazon.json")
