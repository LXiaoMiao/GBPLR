import time

from selenium import webdriver
from selenium.webdriver import Keys
from selenium.webdriver.common.by import By


option = webdriver.ChromeOptions()
option.binary_location = r'D:\chrome-win\chrome.exe'
driver_path = r'D:\chromedriver_win32\chromedriver.exe'


def select():
    inputs = driver.find_elements(By.XPATH, '//*[@id="maincontent"]/div/div[5]//input')
    for input_ in inputs:
        input_.click()


driver = webdriver.Chrome(executable_path=driver_path, options=option)
# A.+thaliana  Maize  Cucumber  Tomato  Potato
driver.get('https://www.ncbi.nlm.nih.gov/nuccore/?term=Potato+mRNA')
time.sleep(3)

# 左侧筛选
plants = driver.find_element(By.XPATH, '//*[@id="_sp"]/li/ul/li[2]/a')
plants.click()
mrna = driver.find_element(By.XPATH, '//*[@id="_moltypes"]/li/ul/li[2]/a')
mrna.click()

first = 121

if first == 1:
    next_ = driver.find_element(By.XPATH, '//*[@id="EntrezSystem2.PEntrez.Nuccore.Sequence_ResultsPanel.Entrez_Pager.Page"]')
else:
    page = driver.find_element(By.XPATH, '//*[@id="pageno"]')
    page.clear()
    page.send_keys(str(first) + Keys.RETURN)
    next_ = driver.find_element(By.XPATH, '//*[@id="maincontent"]/div/div[7]/div/a[3]')

select()
next_.click()

for i in range(first+1, first+30):
    print(f"Page: {i}")
    select()
    # next page
    next_ = driver.find_element(By.XPATH, '//*[@id="maincontent"]/div/div[7]/div/a[3]')
    next_.click()
    time.sleep(1)

send_to = driver.find_element(By.XPATH, '//*[@id="seqsendto"]/a')
send_to.click()
# file = driver.find_element(By.XPATH, '//*[@id="submenu_complete_rec"]/fieldset/ul/li[1]/label')
# file.click()
# format_ = driver.find_element(By.XPATH, 'u"selected">Summary</option><option value="genbank" showgi="true">GenBank</option><option value="gbwithparts" showgi="true">GenBank (full)</option><option value="fasta" showgi="true">FASTA</option><option value="asn1">ASN.1</option><option value="xml">XML</option><option value="gbc_xml">INSDSeq XML</option><option value="fasta_xml" showgi="true">TinySeq XML</option><option value="ft">Feature Table</option><option value="accnlist" format="text">Accession List</option><option value="gilist" format="text">GI List</option><option value="gff3" format="text">GFF3</option></select>')
