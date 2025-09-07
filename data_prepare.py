import os
import pandas as pd
import requests
import shutil
import cv2
import numpy as np
from pymongo import MongoClient
from linked_csv import *
from download_test import download_file_from_csv

def create_folder_structure():
    """
    Создает иерархию папок на локальной машине для хранения и дальнейших преобразований mp4-файлов

    Структура папок:
    base_path/
        └── data/
            ├── left_adrenal/
            │   ├── class_0_0_0/
            │   ├── class_0_0_1/
            │   ├── class_0_1_0/
                ...
            │   └── class_1_1_1/
            └── right_adrenal/
                ├── class_0_0_0/
                ├── class_0_0_1/
                ├── class_0_1_0/
                ...
                └── class_1_1_1/
    """
    base_path = os.path.dirname(os.path.abspath(__file__))

    class_combinations = [
        'class_0_0_0', 'class_0_0_1', 'class_0_1_0', 'class_0_1_1',
        'class_1_0_0', 'class_1_0_1', 'class_1_1_0', 'class_1_1_1'
    ]
    locations = ['left_adrenal', 'right_adrenal']

    for location in locations:
        for class_combination in class_combinations:
            folder_path = os.path.join(base_path, 'data', location, class_combination)
            os.makedirs(folder_path, exist_ok=True)

    print(f"Структура папок успешно создана в: {os.path.join(base_path, 'data')}")

def excel_to_mongodb_with_processing(excel_file, links_csv_file, database_name, collection_name, mongo_uri="mongodb://localhost:27017/"):
    """
    Перенос данных из Excel-файла в коллекцию MongoDB с добавлением поля с локальным путем в файловой системе, а также ссылкой на скачивание.

    :param excel_file: Путь к Excel-файлу.
    :param links_csv_file: Путь к CSV файлу с прямыми ссылками.
    :param database_name: Название базы данных MongoDB.
    :param collection_name: Название коллекции MongoDB.
    :param mongo_uri: URI для подключения к MongoDB (по умолчанию локальный сервер).
    """
    df = pd.read_excel(excel_file)
    direct_links_csv = pd.read_csv(links_csv_file)

    # Добавление колонки с адресом расположения в файловой системе локальной машины
    def generate_local_path(row):
        side = "left_adrenal" if row['Локализация надпочечника (слева/справа)'] == "слева" else "right_adrenal"
        return f"data/{side}/class_{row['Доброкачественный КТ фенотип']}_{row['Неопределенный КТ фенотип']}_{row['Злокачественный КТ фенотип']}"

    df['Локальный путь'] = df.apply(generate_local_path, axis=1)


    # Добавление ссылок из CSV файла в новые поля
    phase_columns = [
        "Файл c нативной фазой", "Файл с разметкой нативной фазы",
        "Файл c артериальной фазой", "Файл c разметкой артериальной фазы",
        "Файл c венозной фазой", "Файл c разметкой венозной фазы",
        "Файл c отсроченной фазой", "Файл c разметкой отсроченной фазы"
    ]
    for phase_column in phase_columns:
        link_column_name = f"Ссылка на {phase_column}"
        df[link_column_name] = pd.Series(dtype="object") #np.nan


    for idx, row in df.iterrows():
        patient_id = row["ID пациента"]

        for phase_column in phase_columns:
            if pd.notna(row[phase_column]):  # Проверка на непустое значение
                file_name = row[phase_column]

                match = direct_links_csv[    # Ищем совпадение в csv-файле
                    (direct_links_csv["ID"] == patient_id) &
                    (direct_links_csv["file_name"] == file_name)
                    ]
                if not match.empty:
                    link_column_name = f"Ссылка на {phase_column}"
                    df.at[idx, link_column_name] = match.iloc[0]["link"]



    # Подключение к MongoDB
    client = MongoClient(mongo_uri)
    db = client[database_name]
    collection = db[collection_name]

    # Преобразование данных DataFrame в список записей
    data = df.to_dict(orient='records')

    # Проверка наличия записей и добавление новых по полям "ID пациента" + "Локализация надпочечника (слева/справа)"
    new_records_count = 0   # Счётчик добавленных записей
    for record in data:
        query = {
            "ID пациента": record["ID пациента"],
            "Локализация надпочечника (слева/справа)": record["Локализация надпочечника (слева/справа)"]
        }
        if not collection.find_one(query):
            collection.insert_one(record)
            new_records_count += 1
        # else:
        #     collection.update_one(query, {"$set": record})  # Обновление существующих записей


    print(f"Данные успешно загружены в MongoDB.")
    print(f"Добавлено новых записей: {new_records_count}")
    print(f"Общее количество записей: {collection.count_documents({})}")

def display_video(video_path, file_name='', frame_skip=5, wait_key=200):
    """
    Воспроизведение каждого {frame_skip} кадра видео, находящегося по пути {video_path}, с задержкой {wait_key} мс между кадрами.

    :param video_path: Путь к видеофайлу.
    :param file_name: Имя файла (при необходимости).
    :param frame_skip: Количество кадров, которые будут пропускаться между отображаемыми (по умолчанию 5).
    :param wait_key: Время задержки в миллисекундах между отображением кадров (по умолчанию 200 мс).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Не удалось открыть видеофайл: {video_path}")
        return

    frame_count = 0  # Счётчик кадров
    window_name = f'Display_video {file_name}'


    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break  # Конец видео

        if frame_count % frame_skip == 0:
            cv2.imshow(window_name, frame) # Отображение кадра в одном и том же окне

            # Задержка между кадрами и Остановка при нажатии клавиши 'q'
            if cv2.waitKey(wait_key) & 0xFF == ord('q'):
                break

        frame_count += 1

    # Освобождение ресурсов
    cap.release()
    cv2.destroyAllWindows()

def download_file(file_name, download_link, download_folder):
    """
    Скачивает файл по указанной ссылке.

    :param file_name: Название файла для сохранения.
    :param download_link: Ссылка для скачивания файла.
    :param download_folder: Папка, в которую сохраняется файл.
    """
    if not file_name.endswith('.mp4'):
        file_name += '.mp4'

    if not os.path.exists(download_folder):
        # os.makedirs(download_folder)
        print("Такой папки нет!")
        return


    file_path = os.path.join(download_folder, file_name)
    response = requests.get(download_link)
    if response.status_code == 200:
        with open(file_path, 'wb') as output_file:
            output_file.write(response.content)
        # print(f"Файл {file_name} успешно скачан в папку {download_folder}.")
    else:
        print(f"Ошибка при скачивании файла {file_name}: {response.status_code} — {response.text}")

def download_files_from_mongo(db_name, collection_name, columns_to_download, mongo_uri="mongodb://localhost:27017/"):
    """
    Проверяет наличие файлов по локальному пути в MongoDB и скачивает отсутствующие.

    :param db_name: Имя базы данных MongoDB.
    :param collection_name: Имя коллекции MongoDB.
    :param columns_to_download: Список полей MongoDB, являющийся списком с фазами, которые нужно скачать.
    :param mongo_uri: URI для подключения к MongoDB.
    """
    print(f"Скачивание новых файлов начато...")

    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]

    base_path = os.path.dirname(__file__)
    downloaded_count = 0  # Счетчик скачанных файлов

    for record in collection.find():
        for column in columns_to_download:
            relative_local_path = record.get("Локальный путь", "")
            local_path = os.path.join(base_path, relative_local_path)

            file_name = record.get(column, "")
            if local_path and file_name:
                full_file_path = os.path.join(local_path, f"{file_name}.mp4")

                # Проверка наличия файла
                if not os.path.exists(full_file_path):
                    download_link = record.get(f"Ссылка на {column}", "")

                    if download_link and not (isinstance(download_link, float) and np.isnan(download_link)):
                        # print(f"Скачиваем файл {file_name}")
                        download_file(file_name, download_link, local_path)
                        downloaded_count += 1
                    else:
                        print(f"Нет ссылки для скачивания файла {file_name}")

    print(f"Скачивание завершено. Всего скачано файлов: {downloaded_count}")

def load_videos_from_mongo(db_name, collection_name, data_dir, target_size=(240, 240), frame_skip=5, add_third_dimension=False):
    """
    Загружает видео из путей, указанных в MongoDB, обрабатывает их и возвращает массивы видео, меток и имен меток.

    Аргументы:
        db_name (str): Имя базы данных MongoDB.
        collection_name (str): Имя коллекции MongoDB.
        data_dir (str): Корневая директория для хранения данных.
        target_size (tuple): Размер, к которому нужно привести кадры видео.
        frame_skip (int): Количество кадров, которые нужно пропустить.
        add_third_dimension (bool): Флаг для добавления третьего измерения к кадрам.

    Возвращает:
        videos (np.array): Массив обработанных видео.
        labels (np.array): Массив меток.
        formatted_label_names (list): Список имен меток.
    """
    # Подключение к MongoDB локально или через песочницу
    client = MongoClient('mongodb://localhost:27017/')
    db = client[db_name]
    collection = db[collection_name]

    videos = []
    labels = []
    formatted_label_names = []

    # Получаем все документы из коллекции
    for document in collection.find():
        path = document["Локальный путь"]  # Извлекаем локальный путь из документа -> заменить на ссылку в БД S3
        full_path = data_dir + path

        if 'left_adrenal' in path:
            prefix = 'left'
        elif 'right_adrenal' in path:
            prefix = 'right'
        else:
            continue

        # только для локалки
        last_backslash_index = full_path.rfind('\\')
        class_name = full_path[last_backslash_index + 1:] # ['class_0_0_0']
        video_path = full_path + "\\" + document["Файл c нативной фазой"] + ".mp4"


        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0 # для пропуска кадров
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip == 0:
                # Обрезаем изображение в зависимости от надпочечника
                if prefix == 'left':
                    frame = frame[:, :frame.shape[1] // 2]
                else:
                    frame = frame[:, frame.shape[1] // 2:]

                frame = cv2.cvtColor(cv2.resize(frame, target_size), cv2.COLOR_BGR2GRAY)
                # МОЖНО ДОБАВИТЬ ЕЩЕ ОБРАБОТКУ

                if add_third_dimension:
                    frame = np.expand_dims(frame, axis=-1) # Добавление канала для совместимости формы для некоторых моделей

                frames.append(frame)
            frame_count += 1
        cap.release()

        # Генерируем метку в виде массива из трех чисел
        class_parts = class_name.split('_')[1:] # ['0', '0', '0']
        label = [np.uint8(int(class_parts[i])) for i in range(3)] # [np.uint8(0), np.uint8(0), np.uint8(0)]

        # Генерируем имя метки в виде left_001 или right_001
        formatted_label_name = f"{prefix}_{''.join(class_parts)}"

        videos.append(np.array(frames, dtype=np.uint8))
        labels.append(label)
        formatted_label_names.append(formatted_label_name)

    return np.array(videos, dtype=np.uint8), np.array(labels, dtype=np.int64), np.array(formatted_label_names)

def process_videos_from_local_data(db_name, collection_name, target_size=(240, 240), frame_skip=5, add_third_dimension=False):
    """
    Обрабатывает все видео из локальной папки [здесь возможны эксперименты по обработке], сопоставляет данные с MongoDB и формирует метки. Возвращает numpy-массивы видео, меток и имен.

    Аргументы:
        db_name (str): Имя базы данных MongoDB.
        collection_name (str): Имя коллекции MongoDB.
        data_dir (str): Корневая директория для хранения данных.
        target_size (tuple): Размер, к которому нужно привести кадры видео.
        frame_skip (int): Количество кадров, которые нужно пропустить.
        add_third_dimension (bool): Флаг для добавления третьего измерения к кадрам.

    Возвращает:
        videos (np.array): Массив обработанных видео.
        labels (np.array): Массив меток.
        formatted_label_names (list): Список имен меток.
    """
    data_dir = os.path.join(os.path.dirname(__file__), 'data')

    client = MongoClient("mongodb://localhost:27017/")
    db = client[db_name]
    collection = db[collection_name]

    videos = []
    labels = []
    formatted_label_names = []

    for root, _, files in os.walk(data_dir):
        for file in files:  # проходимся по всем скачанным файлам
            if not file.endswith(".mp4"):
                continue

            video_path = os.path.join(root, file)
            relative_subpath = os.path.dirname(os.path.relpath(video_path, data_dir))
            relative_path = os.path.join("data", relative_subpath).replace("\\", "/")   # "data/left_adrenal/class_0_0_1"

            # Извлечение ID из имени файла
            patient_id = None
            if "ID" in file and "_" in file:
                patient_id = int(file.split("ID")[1].split("_")[0])

            if not patient_id:
                print(f"Не удалось извлечь ID пациента из файла {relative_path}")
                continue

            # Поиск записи в MongoDB
            record = collection.find_one({"Локальный путь": relative_path, "ID пациента": patient_id})

            if not record:
                print(f"Не найдена запись в MongoDB для файла с ID {patient_id} и Локальным путем {relative_path}")
                continue


            # Генерируем метку в виде массива из трех чисел
            label = np.array([      # [np.uint8(0), np.uint8(0), np.uint8(0)]
                np.uint8(record.get("Доброкачественный КТ фенотип", 0)),
                np.uint8(record.get("Неопределенный КТ фенотип", 0)),
                np.uint8(record.get("Злокачественный КТ фенотип", 0))
            ], dtype=np.uint8)

            class_parts = [
                str(record.get("Доброкачественный КТ фенотип", 0)),
                str(record.get("Неопределенный КТ фенотип", 0)),
                str(record.get("Злокачественный КТ фенотип", 0))
            ]


            # Генерируем имя метки в виде left_001 или right_001
            if "left_adrenal" in relative_path:
                prefix = "left"
            elif "right_adrenal" in relative_path:
                prefix = "right"
            else:
                continue
            formatted_label_name = f"{prefix}_{''.join(class_parts)}"


            # ОБРАБОТКА ВИДЕО ДЛЯ КЛАССИФИКАЦИИ
            cap = cv2.VideoCapture(video_path)
            frames = []
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_skip == 0:
                    # Обрезаем изображение в зависимости от надпочечника
                    if prefix == "left":
                        frame = frame[:, :frame.shape[1] // 2]
                    else:
                        frame = frame[:, frame.shape[1] // 2:]

                    # Обработка кадра [МОЖНО ДОБАВИТЬ ЕЩЕ]
                    frame = cv2.cvtColor(cv2.resize(frame, target_size), cv2.COLOR_BGR2GRAY)

                    if add_third_dimension:
                        frame = np.expand_dims(frame, axis=-1) # Добавление канала для совместимости формы для некоторых моделей

                    frames.append(frame)
                frame_count += 1
            cap.release()

            videos.append(np.array(frames, dtype=np.uint8))
            labels.append(label)
            formatted_label_names.append(formatted_label_name)

    return np.array(videos, dtype=np.uint8), np.array(labels, dtype=np.int64), np.array(formatted_label_names)

def save_npy_arrays(videos, labels, labels_names):
    """
    Сохраняет массивы videos, labels и labels_names в папку 'npy_data_download'.

    Аргументы:
        videos (np.array): Массив видео.
        labels (np.array): Массив меток.
        labels_names (list): Список имен меток.
    """
    download_folder = os.path.join(os.path.dirname(__file__), 'npy_data_download')

    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    videos_file = os.path.join(download_folder, 'videos.npy')
    labels_file = os.path.join(download_folder, 'labels.npy')
    labels_names_file = os.path.join(download_folder, 'labels_names.npy')

    # Сохранение массивов
    np.save(videos_file, videos)
    np.save(labels_file, labels)
    np.save(labels_names_file, labels_names)

    print(f"Файлы успешно сохранены в папку 'npy_data_download'")

def delete_local_videos(data_dir='data'):
    """
    Удаляет все файлы из папки data, включая файлы в подкаталогах.

    Аргументы:
        data_dir (str): Путь к корневой папке (По умолчанию 'data').
    """
    print("Очистка папки data...")
    if os.path.exists(data_dir):
        for root, dirs, files in os.walk(data_dir, topdown=False):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Ошибка при удалении файла {file_path}: {e}")
    else:
        print(f"Папка {data_dir} не найдена!")

    print("Файлы успешно удалены.")


''' вряд ли понадобится, юзалось для проверки смещения'''
def display_video_with_max_contour(video_path, frame_skip=5, wait_key=400):
    """
    Функция находит максимальный контур на каждом {frame_skip} кадре,
    проводит вертикальную линию через центр контура и выводит кадры в одном окне, чтобы убедиться, что центр найден.
    РАБОТАЕТ ПЛОХО
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Не удалось открыть видеофайл: {video_path}")
        return

    window_name = 'Display_video'
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Применяем размытие
            blurred = cv2.GaussianBlur(gray, (9, 9), 5)

            # Используем Canny для выделения границ
            edges = cv2.Canny(blurred, 50, 100)

            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)

                # Получаем момент для нахождения центра
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    center_x = int(M["m10"] / M["m00"])
                    center_y = int(M["m01"] / M["m00"])

                    # Рисуем контур и линию на кадре
                    cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
                    cv2.line(frame, (center_x, 0), (center_x, frame.shape[0]), (255, 0, 0), 2)

            cv2.imshow(window_name, frame)

            if cv2.waitKey(wait_key) & 0xFF == ord('q'):  # 'q' для выхода
                break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
# проверяем что надпочечники на своих местах
def display_video_with_center(video_path, frame_skip=5, wait_key=200):
    """
    Функция проводит вертикальную линию через центр каждого {frame_skip} кадра для РУЧНОГО контроля того, что пациент не сместился. Выводит название файла.
    """

    file_name = os.path.splitext(os.path.basename(video_path))[0]


    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Не удалось открыть видеофайл: {video_path}")
        return

    window_name = 'Display_video'
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            height, width, _ = frame.shape

            center_x = width // 2
            cv2.line(frame, (center_x, 0), (center_x, height), (255, 0, 0), 2)

            # Выводим название файла на видео
            cv2.putText(frame, file_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow(window_name, frame)

            if cv2.waitKey(wait_key) & 0xFF == ord('q'): # выход
                break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
def directory_check_with_center(videos_dir):
    """
        Функция проверяет каждое видео из {videos_dir} с использованием display_video_with_center()
    """

    for file_name in os.listdir(videos_dir):
        video_path = os.path.join(videos_dir, file_name)

        if os.path.isfile(video_path) and file_name.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            display_video_with_center(video_path)




if __name__ == "__main__":
    ''' 0. Надо прописывать команды поочередно '''

    def user_interface(choice):
        match choice:
            case 'create local structure':
                # Создаем иерархию папок на локальной машине для хранения и дальнейших преобразований mp4-файлов
                create_folder_structure()

            case 'create MongoDB database':
                ''' 1. Предварительно установите MongoDBCompass, создайте БД и коллекцию'''

                excel_base_path = os.path.join(os.path.dirname(__file__), r'База данных МСКТ надпочечников_MP4.xlsx')

                # Создание CSV файла с прямыми ссылками на скачивание файлов из Excel файла. Время формирования = 3.8 записи/сек
                # create_direct_links_csv(excel_base_path, sheet_name='Лист1', output_csv='direct_links.csv')
                links_csv_path = os.path.join(os.path.dirname(__file__), r'direct_links.csv')

                # Преобразовываем данные из сырого ХД (excel-файл) в MongoDB, добавляя поле с путем до файла на локальной машине, а также поле с ссылкой на скачивание каждого файла
                excel_to_mongodb_with_processing(
                    excel_file=excel_base_path,
                    links_csv_file=links_csv_path,
                    database_name="Adrenal_CT",
                    collection_name="Data")

            case 'test download file by name':
                file_name = "ID164_NATIVE_MASK" # название файла для скачивания без типа

                # Проверка того, что прямые ссылки из csv-файла рабочие, а файлы скачиваются корректно
                download_file_from_csv(file_name, download_folder='ct_download')
                links_test_download_file = os.path.join(os.path.dirname(__file__), 'ct_download', f'{file_name}.mp4')
                display_video(links_test_download_file, file_name=file_name) # для закрытия нажимать 'q'

            case 'download files from DB to local PC':
                ''' 2. Для Классификации в columns_to_download прописать интересующие имена полей без разметки'''

                # Скачиваем недостающие файлы в локальную систему. columns_to_download содержит интересующие для скачивания фазы.
                download_files_from_mongo(
                    db_name="Adrenal_CT",
                    collection_name="Data",
                    columns_to_download=["Файл c артериальной фазой"],
                    mongo_uri="mongodb://localhost:27017/"
                    )

            case 'data processing for classification and save':
                # Обрабатываем скачанные данные, преобразуя их в numpy-массивы и генерируем метки для классификации
                videos, labels, labels_names = process_videos_from_local_data(
                    db_name="Adrenal_CT",
                    collection_name="Data",
                    target_size=(224, 224),
                    frame_skip=3,
                    add_third_dimension=True)

                print(f"Данные подготовлены.")
                print(f"Форма массива видео: {videos.shape}")
                assert videos.shape[0] == len(labels) == len(labels_names), "Все массивы должны иметь одинаковое количество элементов по первой оси!"


                # Генерация случайного порядка индексов и перемешивание
                shuffle_indices = np.random.permutation(videos.shape[0])
                videos = videos[shuffle_indices]
                labels = labels[shuffle_indices]
                labels_names = labels_names[shuffle_indices]


                # Проверка, При необходимости визуальный вывод
                Num = 10
                print(f"Метки: {labels[:Num]}")
                print(f"Имена меток: {labels_names[:Num]}")

                UI_test_one_video = False
                if UI_test_one_video:
                    first_video = videos[0]
                    window_name = 'Video Display'
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

                    for i, frame in enumerate(first_video):
                        cv2.imshow(window_name, frame)

                        if cv2.waitKey(200) & 0xFF == ord('q'):
                            break
                    cv2.destroyAllWindows()


                # Сохранение файлов
                UI_save_arrays = True
                if UI_save_arrays:
                    save_npy_arrays(videos, labels, labels_names)

            case 'delete local videos':
                # Очистка всех видео-файлов из папки data
                delete_local_videos()

            case 'load of data sets':
                ''' 3. Для работы с имеющимися массивами данных (обработка)'''

                load_folder = os.path.join(os.path.dirname(__file__), 'npy_data_download')
                assert os.path.exists(load_folder), "Папка npy_data_download не найдена!"

                videos_file = os.path.join(load_folder, 'videos.npy')
                labels_file = os.path.join(load_folder, 'labels.npy')
                labels_names_file = os.path.join(load_folder, 'labels_names.npy')

                videos = np.load(videos_file)
                labels = np.load(labels_file)
                labels_names = np.load(labels_names_file)

                print(f"Форма массива видео: {videos.shape}")

                #  А дальше что-то делаем...

            case _:
                print("Неизвестный выбор.")

    choice = 'create local structure'
    user_interface(choice)
    choice = 'create MongoDB database'
    user_interface(choice)
    choice = 'download files from DB to local PC'
    user_interface(choice)
    choice = 'data processing for classification and save'
    user_interface(choice)

    # для удобства запишу все кейсы тут, чтобы не листать

    # create local structure
    # create MongoDB database
    # test download file by name
    # download files from DB to local PC
    # data processing for classification and save
    # delete local videos
    # load of data sets