# Data Dictionary

File: `data/processed/orders_with_history.parquet` (95,446 rows, 76 columns, one row per delivered order). Descriptions are written from the pipeline code and column names; the source dataset is the public Olist Brazilian E-Commerce data (Kaggle).

Leakage rule: the `*_hist_*` columns use only reviews created strictly before the order's purchase time. The columns `review_score`, `dissatisfied` and review text are outcomes and are never model inputs.

| Column | Type | Missing % | Meaning |
|---|---|---|---|
| `order_id` | str | 0.0 | Unique order id (Olist). |
| `customer_id` | str | 0.0 | Id of the customer record for this order. |
| `order_status` | str | 0.0 | Order status (this file holds delivered orders only). |
| `order_purchase_timestamp` | datetime64[us] | 0.0 | When the customer placed the order. |
| `order_approved_at` | datetime64[us] | 0.0 | When payment was approved. |
| `order_delivered_carrier_date` | datetime64[us] | 0.0 | When the seller handed the parcel to the carrier. |
| `order_delivered_customer_date` | datetime64[us] | 0.0 | When the customer received the parcel. |
| `order_estimated_delivery_date` | datetime64[us] | 0.0 | Delivery date promised at purchase. |
| `customer_unique_id` | str | 0.0 | Real customer identity across orders. |
| `customer_zip_code_prefix` | int64 | 0.0 | Customer zip prefix. |
| `customer_city` | str | 0.0 | Customer city. |
| `customer_state` | str | 0.0 | Customer state (2 letters). |
| `n_items` | float64 | 0.0 | Number of items in the order. |
| `n_sellers` | float64 | 0.0 | Number of sellers in the order. |
| `n_categories` | float64 | 0.0 | Number of product categories in the order. |
| `items_price` | float64 | 0.0 | Total item price. |
| `freight_value` | float64 | 0.0 | Total freight (shipping) cost. |
| `total_weight_g` | float64 | 0.0 | Total weight in grams. |
| `total_volume_cm3` | float64 | 0.0 | Total volume in cm3. |
| `max_shipping_limit` | datetime64[us] | 0.0 | Latest seller shipping deadline. |
| `freight_ratio` | float64 | 0.0 | Freight divided by item price. |
| `primary_seller_id` | str | 0.0 | Seller of the first item. |
| `primary_product_id` | str | 0.0 | First product. |
| `primary_category` | str | 0.0 | Category of the first item. |
| `product_photos_qty` | float64 | 1.4 | Photos on the product page. |
| `product_description_lenght` | float64 | 1.4 | Product description length (column name spelled as in the source data). |
| `product_name_lenght` | float64 | 1.4 | Product name length (source spelling). |
| `payment_value` | float64 | 0.0 | Total amount paid. |
| `n_payment_rows` | float64 | 0.0 | Number of payment records. |
| `max_installments` | float64 | 0.0 | Highest number of instalments. |
| `main_payment_type` | str | 0.0 | Main payment method. |
| `review_id` | str | 0.0 | Review id (latest review kept per order). |
| `review_score` | float64 | 0.0 | Review stars, 1 to 5. |
| `review_comment_title` | str | 88.3 | Review title (Portuguese, often empty). |
| `review_comment_message` | str | 59.4 | Review text (Portuguese, often empty). |
| `review_creation_date` | datetime64[us] | 0.0 | Date the review request was created. |
| `review_answer_timestamp` | datetime64[us] | 0.0 | When the customer answered. |
| `has_comment` | bool | 0.0 | True if the review has text. |
| `seller_zip_code_prefix` | float64 | 0.0 | Seller zip prefix. |
| `seller_city` | str | 0.0 | Seller city. |
| `seller_state` | str | 0.0 | Seller state. |
| `cust_lat` | float64 | 0.3 | Customer latitude (zip centroid). |
| `cust_lng` | float64 | 0.3 | Customer longitude. |
| `sell_lat` | float64 | 0.2 | Seller latitude. |
| `sell_lng` | float64 | 0.2 | Seller longitude. |
| `distance_km` | float64 | 0.5 | Distance seller to customer in km. |
| `customer_region` | str | 0.0 | Customer region of Brazil. |
| `seller_region` | str | 0.0 | Seller region of Brazil. |
| `same_state` | int64 | 0.0 | 1 if seller and customer are in the same state. |
| `purchase_month` | str | 0.0 | Purchase month (YYYY-MM). |
| `purchase_dow` | int32 | 0.0 | Purchase weekday (0=Monday). |
| `purchase_hour` | int32 | 0.0 | Purchase hour. |
| `delivery_days` | float64 | 0.0 | Days from purchase to delivery. |
| `estimated_days` | float64 | 0.0 | Days promised. |
| `delay_days` | float64 | 0.0 | Delivery days minus promised days (positive = late). |
| `is_late` | float64 | 0.0 | 1 if delivered after the promised date. |
| `approval_hours` | float64 | 0.0 | Hours until payment approval. |
| `seller_handling_days` | float64 | 0.0 | Days the seller took to hand to carrier. |
| `carrier_transit_days` | float64 | 0.0 | Days the carrier took. |
| `review_after_delivery` | bool | 0.0 | True if the review was written after the recorded delivery. |
| `order_rank` | int64 | 0.0 | Nth order of this customer. |
| `n_orders_customer` | int64 | 0.0 | Total orders of this customer. |
| `has_later_order` | int64 | 0.0 | 1 if the customer ordered again later. |
| `dissatisfied` | float64 | 0.0 | TARGET: 1 if review is 1 or 2 stars, else 0. |
| `log_items_price` | float64 | 0.0 | log(1+items_price). |
| `log_freight_value` | float64 | 0.0 | log(1+freight). |
| `log_total_weight_g` | float64 | 0.0 | log(1+weight). |
| `log_total_volume_cm3` | float64 | 0.0 | log(1+volume). |
| `log_payment_value` | float64 | 0.0 | log(1+payment). |
| `seller_hist_n` | float64 | 0.0 | Reviews the seller had BEFORE this purchase. |
| `seller_hist_mean` | float64 | 0.0 | Seller average stars before this purchase (shrunk to the training mean). |
| `seller_hist_neg` | float64 | 0.0 | Seller share of 1-2 star reviews before this purchase. |
| `seller_hist_late_cmp` | float64 | 0.0 | Seller share of late-order complaints before this purchase. |
| `cat_hist_mean` | float64 | 0.0 | Category average stars before purchase. |
| `cat_hist_neg` | float64 | 0.0 | Category share of 1-2 star reviews before purchase. |
| `state_hist_neg` | float64 | 0.0 | State share of 1-2 star reviews before purchase. |
