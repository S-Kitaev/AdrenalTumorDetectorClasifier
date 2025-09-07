import pandas as pd
import requests
import csv
from urllib.parse import urlencode
import os

def get_resource_info(public_link):
    """
    Получение информации о ресурсах из API Яндекс.Диска.
    """
    base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources?'
    final_url = base_url + urlencode(dict(public_key=public_link))
    response = requests.get(final_url)

    # Проверка успешности запроса
    if response.status_code != 200:
        raise ValueError(f"Ошибка получения данных: {response.status_code} - {response.text}")

    return response.json()

def extract_links_from_excel(input_excel, sheet_name):
    """
    Извлечение данных из Excel файла.
    """
    try:
        data = pd.read_excel(input_excel, sheet_name=sheet_name)
        # Удаление дубликатов ссылок на Яндекс.Диск
        data = data.drop_duplicates(subset=['Местоположение файлов'])
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Файл '{input_excel}' не найден.")
    except Exception as e:
        raise Exception(f"Произошла ошибка: {e}")

def read_existing_csv(output_csv):
    """
    Чтение существующего CSV файла.
    """
    if not os.path.exists(output_csv):
        return pd.DataFrame(columns=["ID", "phase", "file_name", "link"])

    try:
        return pd.read_csv(output_csv)
    except Exception as e:
        print(f"Ошибка при чтении CSV файла '{output_csv}': {e}")
        return pd.DataFrame(columns=["ID", "phase", "file_name", "link"])

def check_and_update_csv(input_excel, sheet_name, output_csv):
    """
    Проверка и обновление CSV файла с прямыми ссылками на файлы из Excel файла.

    :param input_excel: Путь к Excel файлу.
    :param sheet_name: Название листа в Excel файле.
    :param output_csv: Имя создаваемого или дополняемого файла CSV.
    """
    try:
        excel_data = extract_links_from_excel(input_excel, sheet_name)
        existing_csv_data = read_existing_csv(output_csv)

        # Проверяем последний ID в Excel и CSV
        last_excel_id = excel_data['ID пациента'].max()
        last_csv_id = existing_csv_data['ID'].max() if not existing_csv_data.empty else 0

        if last_csv_id == last_excel_id:
            print("Новых записей в Excel нет.")
            return
        elif last_csv_id > last_excel_id:
            print("Проверьте direct_links.csv, в нем лишние записи.")
            return

        # Отбираем только новые записи
        new_records = excel_data[excel_data['ID пациента'] > last_csv_id]

        print(f"Добавление новых записей в CSV файл '{output_csv}'...")

        with open(output_csv, 'a', encoding="utf-8", newline="") as csv_file:
            writer = csv.writer(csv_file, delimiter=",")

            # Если файл новый, добавляем заголовок
            if os.stat(output_csv).st_size == 0:
                writer.writerow(["ID", "phase", "file_name", "link"])

            for _, row in new_records.iterrows():
                href = row['Местоположение файлов']
                try:
                    resource_info = get_resource_info(href)
                    if '_embedded' in resource_info:
                        items = resource_info['_embedded']['items']
                        for item in items:
                            if item['type'] == 'file':
                                folder = item['name'].split("_")[0][2:] if "_" in item['name'] else "Unknown"
                                if "_" in item['name']:
                                    parts = item['name'].split("_")
                                    if len(parts) > 1:
                                        phase = parts[1].split(".")[0]  # Извлекаем всё до точки
                                    else:
                                        phase = "Unknown"
                                else:
                                    phase = "Unknown"

                                filename = item['name'][:-4]
                                download_link = item['file'] if 'file' in item else None

                                if download_link:
                                    writer.writerow([row['ID пациента'], phase, filename, download_link])
                except Exception as e:
                    print(f"Ошибка обработки ссылки {href}: {e}")

        print(f"Новые записи загружены в '{output_csv}'.")

    except Exception as e:
        print(f"Произошла ошибка: {e}")

if __name__ == "__main__":
    input_excel = 'База данных МСКТ надпочечников_MP4.xlsx'  # Укажите путь к вашему Excel файлу
    sheet_name = 'Лист1'  # Укажите имя листа в Excel
    output_csv = 'direct_links.csv'  # Имя выходного файла CSV

    check_and_update_csv(input_excel, sheet_name, output_csv)

