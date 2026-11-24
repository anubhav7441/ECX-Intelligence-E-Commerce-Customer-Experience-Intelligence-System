-- ECX-Intelligence KPI queries (SQLite). Aggregate items/payments/reviews BEFORE joining to avoid row inflation.

-- Q1: orders by status
SELECT order_status, COUNT(*) AS n_orders FROM orders GROUP BY order_status ORDER BY n_orders DESC;

-- Q2: total revenue (item price) of delivered orders  [price only; freight excluded]
SELECT ROUND(SUM(i.price), 2) AS revenue_items, ROUND(SUM(i.freight_value), 2) AS freight_total, COUNT(DISTINCT i.order_id) AS orders
FROM order_items i JOIN orders o ON o.order_id = i.order_id WHERE o.order_status = 'delivered';

-- Q3: monthly revenue and orders (delivered)
SELECT substr(o.order_purchase_timestamp, 1, 7) AS month, COUNT(DISTINCT o.order_id) AS orders, ROUND(SUM(i.price), 2) AS revenue
FROM orders o JOIN order_items i ON i.order_id = o.order_id WHERE o.order_status = 'delivered'
GROUP BY month ORDER BY month;

-- Q4: average review score using ONE review per order (latest answer timestamp) - avoids the 547 multi-review orders
WITH latest AS (
  SELECT order_id, review_score,
         ROW_NUMBER() OVER (PARTITION BY order_id ORDER BY review_answer_timestamp DESC, review_pk DESC) AS rn
  FROM order_reviews)
SELECT ROUND(AVG(review_score), 4) AS avg_score, COUNT(*) AS n_reviewed_orders FROM latest WHERE rn = 1;

-- Q5: late-delivery percentage (date-based) among delivered orders with a delivery date
SELECT ROUND(100.0 * AVG(CASE WHEN date(order_delivered_customer_date) > date(order_estimated_delivery_date) THEN 1 ELSE 0 END), 3) AS pct_late,
       COUNT(*) AS n_delivered
FROM orders WHERE order_status = 'delivered' AND order_delivered_customer_date IS NOT NULL;

-- Q6: average delivery days
SELECT ROUND(AVG(julianday(order_delivered_customer_date) - julianday(order_purchase_timestamp)), 3) AS avg_delivery_days
FROM orders WHERE order_status = 'delivered' AND order_delivered_customer_date IS NOT NULL;

-- Q7: revenue and orders by customer state (delivered)
SELECT c.customer_state, COUNT(DISTINCT o.order_id) AS orders, ROUND(SUM(i.price), 2) AS revenue
FROM orders o JOIN customers c ON c.customer_id = o.customer_id JOIN order_items i ON i.order_id = o.order_id
WHERE o.order_status = 'delivered' GROUP BY c.customer_state ORDER BY revenue DESC;

-- Q8: revenue by category (English names)
SELECT COALESCE(t.product_category_name_english, 'unknown') AS category, COUNT(DISTINCT i.order_id) AS orders, ROUND(SUM(i.price), 2) AS revenue
FROM order_items i JOIN products p ON p.product_id = i.product_id
LEFT JOIN category_translation t ON t.product_category_name = p.product_category_name
JOIN orders o ON o.order_id = i.order_id WHERE o.order_status = 'delivered'
GROUP BY category ORDER BY revenue DESC LIMIT 15;

-- Q9: seller revenue with minimum 50 orders
SELECT i.seller_id, COUNT(DISTINCT i.order_id) AS orders, ROUND(SUM(i.price), 2) AS revenue
FROM order_items i JOIN orders o ON o.order_id = i.order_id WHERE o.order_status = 'delivered'
GROUP BY i.seller_id HAVING orders >= 50 ORDER BY revenue DESC LIMIT 20;
