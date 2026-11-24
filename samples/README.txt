SAMPLE FILES (all SYNTHETIC / made up - never report their numbers as real findings)
Real data: https://archive.ics.uci.edu/dataset/502/online+retail+ii and https://amazon-reviews-2023.github.io/

Upload each file in My Data. What you should see:
01_good_shop_orders.csv    800 orders. Built so late delivery lowers ratings and one state (BA) is slow. Expect clear advice cards.
02_small_file_60_rows.csv  Only 60 rows. Expect the app to say the evidence is weak (low confidence).
03_messy_file.csv          Missing values, duplicates, negative prices, text in number columns, impossible dates and a rating of 7. Expect warnings and data-quality advice.
04_wrong_file_contacts.csv A contact list. Nothing to analyse: the app should explain what it cannot do, not invent results.
05_header_only.csv         Empty file with only column names: should be rejected with a clear message.
06_reviews_only.csv        Ratings and review text but no sales or delivery columns: only text/rating analyses should run.
07_random_no_pattern.csv   1000 orders where ratings are random. Expect NO strong findings (this shows the app does not invent patterns).
Older files: SYNTHETIC_online_retail_like.xlsx, SYNTHETIC_amazon_appliances_like.jsonl
