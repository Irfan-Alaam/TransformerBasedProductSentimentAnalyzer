from playwright.sync_api import sync_playwright
from selectolax.parser import HTMLParser
import re
import os
import csv
import time
import random
import hashlib
from datetime import datetime

review_list = []

def generate_review_id(title, body):
    unique_string = (title + body).encode('utf-8')
    return hashlib.md5(unique_string).hexdigest()

def random_delay():
    time.sleep(random.uniform(1.5, 4.0))

def scroll_page(page):
    for _ in range(3):
        page.evaluate("window.scrollBy(0, window.innerHeight * 0.8)")
        random_delay()

def get_reviews(html):
    reviews = html.css('[data-hook="review"]')
    print(f" Found {len(reviews)} reviews on this page.")

    for review in reviews:
        if len(review_list) >= 50:
            break
        try:
            title = review.css_first('[data-hook="review-title"] span:not([class])')
            rating = review.css_first('[data-hook="review-star-rating"] span.a-icon-alt')
            body = review.css_first('[data-hook="review-body"] span')
            date = review.css_first('[data-hook="review-date"]')
            
            if not all([title, rating, body]):
                print(" Missing one or more review elements")
                continue

            clean_title = title.text().strip()
            clean_body = re.sub(r'\s+', ' ', body.text()).strip()
            review_id = generate_review_id(clean_title, clean_body)
            
            # Extract rating
            rating_text = rating.text().strip()
            clean_rating = float(re.search(r'(\d+\.?\d*)', rating_text).group(1))
            
            # Extract date
            clean_date = date.text().strip() if date else "N/A"

            if not any(r['id'] == review_id for r in review_list):
                review_list.append({
                    "id": review_id,
                    "review_title": clean_title,
                    "rating": clean_rating,
                    "body": clean_body,
                    "date": clean_date
                })

        except Exception as e:
            print(f" Error extracting review: {e}")
            continue

def scrape_amazon_reviews(asin, max_pages=5):
    review_list.clear()
    with sync_playwright() as p:
        # Configure browser (remove brave_path if using default Chrome)
        brave_path = r"C:\\Program Files\\BraveSoftware\\Brave-Browser\\Application\\brave.exe"
        user_data_dir = os.path.expandvars(
            r"C:\\Users\\Al Muzdhar Computers\\AppData\\Local\\BraveSoftware\\Brave-Browser\\User Data"
        )

        print("🚀 Launching Brave with your logged-in session...")
        context = p.chromium.launch_persistent_context(
            user_data_dir=user_data_dir,
            executable_path=brave_path,
            headless=False,
            args=["--start-maximized"],
            viewport={'width': 1366, 'height': 768},
            locale='en-US'
        )
        
        page = context.new_page()
        
        initial_url = f"https://www.amazon.com/product-reviews/{asin}?reviewerType=all_reviews&sortBy=recent"
        print(f"\n🔎 Loading initial page: {initial_url}")
        
        try:
            page.goto(initial_url, timeout=60000)
            page.wait_for_selector('[data-hook="review"]', timeout=15000)
            
            # Check for CAPTCHA
            if page.query_selector("#captchacharacters"):
                print(" CAPTCHA detected! Please solve it manually in the browser.")
                input("Press Enter after solving CAPTCHA to continue...")
            
            scroll_page(page)
            html = HTMLParser(page.content())
            get_reviews(html)

            for i in range(1, max_pages):
                if len(review_list) >= 50:
                    break

                print(f"\n⏭ Attempting to go to next page #{i + 1}")
                try:
                    next_btn = page.query_selector("li.a-last a")
                    if not next_btn:
                        print("❌ No more pages available.")
                        break

                    with page.expect_navigation():
                        next_btn.click()
                    
                    # Wait for reviews to load
                    page.wait_for_selector('[data-hook="review"]', timeout=15000)
                    scroll_page(page)
                    
                    # Check for CAPTCHA again
                    if page.query_selector("#captchacharacters"):
                        print("❌ CAPTCHA detected on new page! Please solve it manually.")
                        input("Press Enter after solving CAPTCHA to continue...")
                    
                    html = HTMLParser(page.content())
                    get_reviews(html)

                except Exception as e:
                    print(f"⚠️ Error navigating to next page: {e}")
                    break

        except Exception as e:
            print(f"⚠️ Major error occurred: {e}")
        finally:
            context.close()
            print(f"Done scraping. Collected {len(review_list)} unique reviews.")

def save_reviews_to_csv(filename=None):
    if not filename:
        filename = f"amazon_reviews_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    with open(filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["review_title", "rating", "body", "date"])
        writer.writeheader()
        for review in review_list:
            writer.writerow({
                "review_title": review["review_title"],
                "rating": review["rating"],
                "body": review["body"],
                "date": review["date"]
            })
    print(f"Saved {len(review_list)} reviews to '{filename}'.")

if __name__ == "__main__":
    # Example ASIN for testing (Amazon Echo Dot)
    scrape_amazon_reviews("B0DLKB5V35", max_pages=5)

    print("\n======= Scraped Reviews =======")
    for i, r in enumerate(review_list, 1):
        print(f"\nReview #{i}")
        print(f"Title : {r['review_title']}")
        print(f"Rating: {r['rating']} stars")
        print(f"Date  : {r['date']}")
        print(f"Body  : {r['body'][:100]}...")

    if review_list:
        save_reviews_to_csv()