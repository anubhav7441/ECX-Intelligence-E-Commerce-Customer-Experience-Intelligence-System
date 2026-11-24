-- ECX-Intelligence relational schema (SQLite). Verified against the real Olist CSVs.
-- Notes from data profiling:
--   * reviews.review_id is NOT unique (789 ids repeat) -> surrogate key review_pk
--   * 547 orders have more than one review -> one row per review kept here, deduplicated later in analysis
--   * 2 product categories are missing from the official translation file -> added in loader (documented)

PRAGMA foreign_keys = ON;

CREATE TABLE category_translation (
    product_category_name         TEXT PRIMARY KEY,
    product_category_name_english TEXT NOT NULL
);

CREATE TABLE customers (
    customer_id              TEXT PRIMARY KEY,
    customer_unique_id       TEXT NOT NULL,
    customer_zip_code_prefix INTEGER,
    customer_city            TEXT,
    customer_state           TEXT
);

CREATE TABLE sellers (
    seller_id              TEXT PRIMARY KEY,
    seller_zip_code_prefix INTEGER,
    seller_city            TEXT,
    seller_state           TEXT
);

CREATE TABLE products (
    product_id                 TEXT PRIMARY KEY,
    product_category_name      TEXT,
    product_name_lenght        REAL,
    product_description_lenght REAL,
    product_photos_qty         REAL,
    product_weight_g           REAL,
    product_length_cm          REAL,
    product_height_cm          REAL,
    product_width_cm           REAL
);

CREATE TABLE orders (
    order_id                      TEXT PRIMARY KEY,
    customer_id                   TEXT NOT NULL REFERENCES customers(customer_id),
    order_status                  TEXT NOT NULL,
    order_purchase_timestamp      TEXT,
    order_approved_at             TEXT,
    order_delivered_carrier_date  TEXT,
    order_delivered_customer_date TEXT,
    order_estimated_delivery_date TEXT
);

CREATE TABLE order_items (
    order_id            TEXT    NOT NULL REFERENCES orders(order_id),
    order_item_id       INTEGER NOT NULL,
    product_id          TEXT    NOT NULL REFERENCES products(product_id),
    seller_id           TEXT    NOT NULL REFERENCES sellers(seller_id),
    shipping_limit_date TEXT,
    price               REAL,
    freight_value       REAL,
    PRIMARY KEY (order_id, order_item_id)
);

CREATE TABLE order_payments (
    order_id             TEXT    NOT NULL REFERENCES orders(order_id),
    payment_sequential   INTEGER NOT NULL,
    payment_type         TEXT,
    payment_installments INTEGER,
    payment_value        REAL,
    PRIMARY KEY (order_id, payment_sequential)
);

CREATE TABLE order_reviews (
    review_pk               INTEGER PRIMARY KEY AUTOINCREMENT,
    review_id               TEXT NOT NULL,
    order_id                TEXT NOT NULL REFERENCES orders(order_id),
    review_score            INTEGER NOT NULL CHECK (review_score BETWEEN 1 AND 5),
    review_comment_title    TEXT,
    review_comment_message  TEXT,
    review_creation_date    TEXT,
    review_answer_timestamp TEXT
);

-- Aggregated geolocation: raw table has 1,000,163 rows but only 19,015 distinct zip prefixes.
CREATE TABLE geolocation_zip (
    zip_code_prefix INTEGER PRIMARY KEY,
    lat             REAL NOT NULL,
    lng             REAL NOT NULL,
    city            TEXT,
    state           TEXT,
    raw_row_count   INTEGER
);

CREATE INDEX idx_items_order   ON order_items(order_id);
CREATE INDEX idx_items_seller  ON order_items(seller_id);
CREATE INDEX idx_items_product ON order_items(product_id);
CREATE INDEX idx_orders_cust   ON orders(customer_id);
CREATE INDEX idx_reviews_order ON order_reviews(order_id);
CREATE INDEX idx_pay_order     ON order_payments(order_id);
CREATE INDEX idx_cust_unique   ON customers(customer_unique_id);
