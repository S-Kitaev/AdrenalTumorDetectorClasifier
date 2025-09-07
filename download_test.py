import os
import csv
import requests


def download_file_from_csv(file_name, csv_file='direct_links.csv', download_folder='ct_download'):
    """
    Скачивает файл по названию из CSV файла.

    :param file_name: Название файла, который нужно найти и скачать.
    :param csv_file: Путь к файлу CSV, где содержатся ссылки.
    :param download_folder: Папка, в которую будут сохраняться скачанные файлы.
    """
    if not os.path.exists(csv_file):
        print(f"Файл {csv_file} не найден!")
        return

    # Проверка наличия папки для скачивания
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    # Чтение CSV и поиск названия файла
    file_found = False
    with open(csv_file, 'r', encoding='utf-8') as csv_read:
        reader = csv.reader(csv_read, delimiter=",")
        for row in reader:
            if len(row) < 4:
                continue  # Пропускаем некорректные строки

            folder, phase, csv_file_name, download_link = row

            if csv_file_name == file_name:
                file_found = True

                if not file_name.endswith('.mp4'):
                    file_name += '.mp4'

                # Скачивание файла
                response = requests.get(download_link)
                if response.status_code == 200:
                    file_path = os.path.join(download_folder, file_name)
                    with open(file_path, 'wb') as output_file:
                        output_file.write(response.content)
                    print(f"Файл {file_name} успешно скачан в папку {download_folder}.")
                else:
                    print(f"Ошибка при скачивании файла {file_name}: {response.status_code} — {response.text}")
                break

    if not file_found:
        print("Нет данного файла.")


def main():
    """
    Основная функция для работы с пользователем.
    """
    file_name = input("Введите название файла для скачивания: ").strip()
    download_file_from_csv(file_name)


if __name__ == "__main__":
    main()