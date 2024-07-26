from selenium import webdriver 
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By 

import numpy as np 
import pandas as pd
import time
import random

from bs4 import BeautifulSoup


chrome_options = Options()
chrome_options.add_experimental_option("detach", True)
chrome_options.add_experimental_option("excludeSwitches", ['enable-logging'])
chrome_options.add_argument("--disable-blink-features-AutomationControlled")

driver = webdriver.Chrome(options=chrome_options)

html = driver.page_source
soup = BeautifulSoup(html, 'html.parser')
# 브라우저 창 크기 설정
driver.set_window_size(1920, 1080)

Lotion_dataset = pd.DataFrame(columns= ['brand', 'name','price','sale_price','picture','url','volume','skin_type','ingredient','review'])
Lotion_dataset

all_reviews = []

#상품페이지 
for page_num in range(1,5):
    olive_url = f"https://www.oliveyoung.co.kr/store/display/getMCategoryList.do?dispCatNo=100000100010016&fltDispCatNo=&prdSort=01&pageIdx={page_num}&rowsPerPage=48&searchTypeSort=btn_thumb&plusButtonFlag=N&isLoginCnt=0&aShowCnt=0&bShowCnt=0&cShowCnt=0&trackingCd=Cat100000100010016_Small&amplitudePageGubun=&t_page=&t_click=&midCategory=%EB%A1%9C%EC%85%98&smallCategory=%EC%A0%84%EC%B2%B4&checkBrnds=&lastChkBrnd="
    driver.get(olive_url)
    time.sleep(1)


    for i in range(2,14):
        for j in range(1, 5):
            driver.find_element(By.XPATH, f'//*[@id="Contents"]/ul[{i}]/li[{j}]/div/div/a').click()
            
            driver.find_element(By.CLASS_NAME, 'prd_info').click()
            time.sleep(1) 
            brand = driver.find_element(By.CLASS_NAME, "prd_brand").text
            name = driver.find_element(By.CLASS_NAME, "prd_name").text
            price = driver.find_element(By.CLASS_NAME, "price-1").text
            sale_price = driver.find_element(By.CLASS_NAME, "price-2").text
            picture = driver.find_element(By.XPATH, '//*[@id="mainImg"]').get_attribute('src')
            url = driver.find_element(By.XPATH, '/html/head/meta[8]').get_attribute('content')
            
            driver.find_element(By.CSS_SELECTOR, "#buyInfo > a").click()
            time.sleep(1)
            volume = driver.find_element(By.XPATH,'//*[@id="artcInfo"]/dl[2]/dd').text
            skin_type = driver.find_element(By.XPATH,'//*[@id="artcInfo"]/dl[3]/dd').text
            ingredient = driver.find_element(By.XPATH,'//*[@id="artcInfo"]/dl[8]/dd').text
            
            print(brand)
            print(name)
            print(price) 
            print(sale_price)
            print(picture)
            print(url)
            print(volume)
            print(skin_type)
            print(ingredient)

                
            #리뷰 크롤링
            driver.find_element(By.CSS_SELECTOR, "#reviewInfo > a").click() #리뷰버튼 클릭
            time.sleep(2) 
            driver.find_element(By.XPATH, '//*[@id="gdasSort"]/li[3]/a').click() #최신순 클릭
            time.sleep(1) 
            driver.find_element(By.CSS_SELECTOR, "#searchType_3").click() #체험단 체크 해제 
            
            
            k = 1
            while True:
                reviews = [] #수정
                review_box = driver.find_element(By.CLASS_NAME,'review_cont').text
                time.sleep(1)
                for review in review_box:
                    review = driver.find_element(By.CLASS_NAME, 'txt_inner').text 
                    reviews.append([review])
                all_reviews.extend(reviews)
                if k < 10:
                    button = driver.find_element(By.XPATH, f'//*[@id="gdasContentsArea"]/div/div[7]/a[{k}]').click() 
                    k += 1             
                    time.sleep(2)
                else:
                    print(all_reviews)
                    break
                
            # driver.back()
            