import requests
import re
from bs4 import BeautifulSoup
import csv
file = "output.csv"
data = [["place", "price", "price per sqmeter"]]
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36'}
page = requests.get("https://www.otodom.pl/pl/wyniki/sprzedaz/mieszkanie/mazowieckie/warszawa/warszawa/warszawa?viewType=listing&page=1", headers=headers)
doc = BeautifulSoup(page.text, "html.parser")
links = doc.find(["div"], class_="css-feokcq egbyzpx4").find(["div"], role="main", class_="css-1ktfa37 e11vu9402").find(["div"], {"data-cy": "search.listing.organic"}).find_all(["li"], {'data-cy': 'listing-item', 'class': 'css-o9b79t e1dfeild0'})
for number_of_page in range(1, 10):
    page = requests.get(f"https://www.otodom.pl/pl/wyniki/sprzedaz/mieszkanie/mazowieckie/warszawa/warszawa/warszawa?viewType=listing&page={number_of_page}", headers=headers)
    links = doc.find(["div"], class_="css-feokcq egbyzpx4").find(["div"], role="main", class_="css-1ktfa37 e11vu9402").find(["div"], {"data-cy": "search.listing.organic"}).find_all(["li"], {'data-cy': 'listing-item', 'class': 'css-o9b79t e1dfeild0'})
    for place in range(len(links)):
        extract = links[place].find("a")
        t = extract["href"]
        subpage = requests.get(f"https://www.otodom.pl{t}", headers=headers)
        subpage1 = BeautifulSoup(subpage.text, "html.parser")
        prices = subpage1.find_all(string=re.compile(r".*zł.*"))[0:2]
        name = subpage1.find_all(["header"], class_="css-1tnueo5 efcnut32")[0].find(class_="css-1wnihf5 efcnut38").string
        c = [name, prices[0], prices[1]]
        data.append(c)
print(data)
with open(file, "w") as f:
    object = csv.writer(f)
    for row in data:
        object.writerow(row)
