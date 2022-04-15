import pathlib
import requests
import csv
import json

csvpath = pathlib.Path('Data/CryptoData.csv').resolve()

def main():
    url = 'https://rest.coinapi.io/v1/ohlcv/BITSTAMP_SPOT_BTC_USD/history?period_id=1DAY&time_start=2019-12-31&time_end=2020-04-30'
    headers = {'X-CoinAPI-Key' : '92EDB9C4-EB57-4A40-A314-49A6332D0F22'}
    response = requests.get(url, headers=headers)
    datas = response.json()
    for data in datas:
        a = [data["time_open"], data["time_close"], data["price_open"], data["price_close"], data["price_high"],
                    data["price_low"], data["volume_traded"]]
        writeCSV(a)



def createCSV():
    header = ["time_open", "time_close", "price_open", "price_close", "price_high", "price_low",  "volume_traded"]
    with open(csvpath, 'w') as csv_file:
        fieldnames = header
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()


def writeCSV(text):   
    with open(csvpath, 'a', encoding='UTF8') as f:
            writer = csv.writer(f)

            # write the text
       
       
main()
