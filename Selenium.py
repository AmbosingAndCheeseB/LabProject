from selenium import webdriver

search = '어린이 환자복'
url = "https://www.google.co.in/search?q="+search+"&tbm=isch"

browser = webdriver.Chrome('chromedriver.exe')
browser.get(url)

for i in range(100):
    browser.execute_script('window.scrollBy(0,10000)')

for idx, el in enumerate(browser.find_elements_by_class_name("THL2l")):
    try:
        el.screenshot("image/"+search+"/"+str(idx)+".jpg")
    except:
        print("error")

browser.close()
